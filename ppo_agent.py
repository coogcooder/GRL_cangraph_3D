"""PPO training loop for graph-based actor-critic with global/local actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnn_policies import ActorCriticPolicy, PPOAction
from rl_agent import GraphDesignEnv


@dataclass
class Transition:
    state: Tuple[torch.Tensor, torch.Tensor]
    action: PPOAction
    reward: float
    done: bool
    next_state: Tuple[torch.Tensor, torch.Tensor]


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.actions: List[PPOAction] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.next_states: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def add(self, trans: Transition):
        self.states.append(trans.state)
        self.actions.append(trans.action)
        self.rewards.append(trans.reward)
        self.dones.append(trans.done)
        self.next_states.append(trans.next_state)

    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    def __init__(
        self,
        feature_dim: int,
        num_global_ops: int,
        num_local_sizes: int,
        num_violation_keys: int,
        device: str = "cpu",
        lr: float = 3e-4,
    ):
        self.policy = ActorCriticPolicy(
            feature_dim,
            num_global_ops,
            num_local_sizes,
            num_violation_keys,
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = device
        self.gamma = 0.99
        self.clip_eps = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5

    def compute_returns(self, rewards: List[float], dones: List[bool], last_value: float = 0.0):
        returns: List[float] = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, buffer: RolloutBuffer):
        if len(buffer) == 0:
            return 0.0

        states_x = torch.stack([s[0] for s in buffer.states]).to(self.device)
        states_e = torch.stack([s[1] for s in buffer.states]).to(self.device)
        next_x = torch.stack([s[0] for s in buffer.next_states]).to(self.device)
        next_e = torch.stack([s[1] for s in buffer.next_states]).to(self.device)

        with torch.no_grad():
            _, _, _, _, last_value = self.policy(next_x[-1], next_e[-1])

        returns = self.compute_returns(buffer.rewards, buffer.dones, float(last_value))

        logps = torch.stack([a.logp for a in buffer.actions]).to(self.device)
        actions_g = torch.stack([a.global_action for a in buffer.actions]).to(self.device)
        actions_n = torch.stack([a.local_node for a in buffer.actions]).to(self.device)
        actions_s = torch.stack([a.local_size for a in buffer.actions]).to(self.device)
        values = torch.stack([a.value for a in buffer.actions]).to(self.device)
        advantages = returns - values.detach()

        # Collect per-step losses then average to ensure we always backprop a
        # scalar. Using a list + stack avoids accidental broadcasting that could
        # yield multi-element tensors and trip autograd.
        losses = []
        for idx in range(len(buffer)):
            logits_g, logits_n, logits_s, _, value_pred = self.policy(states_x[idx], states_e[idx])
            dist_g = torch.distributions.Categorical(logits=logits_g)
            dist_n = torch.distributions.Categorical(logits=logits_n)
            dist_s = torch.distributions.Categorical(logits=logits_s)
            new_logp = dist_g.log_prob(actions_g[idx]) + dist_n.log_prob(actions_n[idx]) + dist_s.log_prob(actions_s[idx])

            ratio = torch.exp(new_logp - logps[idx])
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
            policy_loss = -torch.min(ratio * advantages[idx], clipped)
            value_loss = F.mse_loss(value_pred, returns[idx])
            entropy = dist_g.entropy().mean() + dist_n.entropy().mean() + dist_s.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        buffer.clear()
        return float(total_loss.detach().cpu().item())


def collect_episode(
    env: GraphDesignEnv,
    trainer: PPOTrainer,
    *,
    steps_per_episode: int = 10,
    local_per_global: int = 30,
    ur_keys: List[str],
    verbose: bool = True,
    warm_start: bool = False,
) -> Dict:
    buffer = RolloutBuffer()
    obs_x, obs_e = env.observe_tensors()
    done = False
    step = 0
    rewards = []
    local_logs: List[Dict[str, float]] = []
    macro_logs: List[Dict[str, float]] = []

    if verbose:
        print(
            f"[episode] running up to {steps_per_episode} macro-steps "
            f"(local_per_global={local_per_global}); initial obs nodes={obs_x.shape[0]} edges={obs_e.shape[1]}"
        )

    while step < steps_per_episode and not done:
        # Local actions loop
        prev_snapshot = env._last_snapshot or env._evaluate_frame()[0]
        prev_v = env.violation_sum(prev_snapshot, ur_keys)
        prev_w = env._last_weight
        for local_step in range(local_per_global):
            masks = env.action_masks()
            masks["meta"]["violation_tensor"] = env.violation_tensor(obs_x, ur_keys)
            act = trainer.policy.act(
                obs_x,
                obs_e,
                masks,
                verbose=False,
            )
            next_obs, _, done = env.step(act, mode="local")
            next_x, next_e = next_obs
            snapshot = env._last_snapshot or env._evaluate_frame()[0]
            v_now = env.violation_sum(snapshot, ur_keys)
            w_now = env._last_weight
            reward = (prev_v - v_now) * 1.0 - max(0.0, w_now - prev_w) * 0.001
            if env._node_index_map:
                node_idx = int(getattr(act, "local_node", 0).item()) % len(env._node_index_map)
                node_id = list(env._node_index_map.keys())[node_idx]
            else:
                node_id = "NA"
            delta = int(getattr(act, "local_size", 1).item()) - 1
            local_logs.append(
                {
                    "macro_step": step,
                    "local_step": local_step,
                    "node_id": node_id,
                    "action_delta": delta,
                    "violation": v_now,
                    "weight": w_now,
                    "reward_local": reward,
                }
            )
            buffer.add(Transition((obs_x, obs_e), act, reward, done, (next_x, next_e)))
            obs_x, obs_e = next_x, next_e
            rewards.append(reward)
            prev_v, prev_w = v_now, w_now
            if done:
                break
            if v_now <= 0.0:
                break
            env.analyze_and_update(env.prefix)
        if done:
            break
        if warm_start:
            step += 1
            continue
        # Global action
        masks = env.action_masks(global_only=True)
        masks["meta"]["violation_tensor"] = env.violation_tensor(obs_x, ur_keys)
        act = trainer.policy.act(
            obs_x,
            obs_e,
            masks,
            verbose=verbose,
            step_label=f"ep_step{step}_global",
        )
        next_obs, _, done = env.step(act, mode="global")
        next_x, next_e = next_obs
        snapshot = env._last_snapshot or env._evaluate_frame()[0]
        v_now = env.violation_sum(snapshot, ur_keys)
        w_now = env._last_weight
        if v_now <= 0.0:
            reward = (prev_w - w_now) / max(prev_w, 1.0)
        else:
            reward = -v_now
        global_choice = int(getattr(act, "global_action", 0).item())
        chosen = None
        if env._global_action_map:
            chosen = env._global_action_map[global_choice % len(env._global_action_map)]
        macro_logs.append(
            {
                "macro_step": step,
                "chosen_op": chosen[1] if chosen else "",
                "chosen_candidate": chosen[0] if chosen else "",
                "reward_global": reward,
                "violation": v_now,
                "weight": w_now,
            }
        )
        buffer.add(Transition((obs_x, obs_e), act, reward, done, (next_x, next_e)))
        obs_x, obs_e = next_x, next_e
        rewards.append(reward)
        step += 1
        env.analyze_and_update(env.prefix)

    loss = trainer.update(buffer)
    return {
        "episode_reward": sum(rewards),
        "loss": loss,
        "local_logs": local_logs,
        "macro_logs": macro_logs,
    }