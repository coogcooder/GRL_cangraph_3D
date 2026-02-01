"""Graph-based DQN policy used by the reinforcement design agent.

The implementation uses PyTorch (and optionally PyTorch Geometric when
available) to encode graph observations and score discrete sizing actions.
When the deep-learning stack is not installed, the policy gracefully falls
back to lightweight numpy/random logic so the rest of the pipeline can run.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency; we still want the file to import without torch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:  # pragma: no cover - soft dependency
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

try:  # Optional PyG encoder
    from torch_geometric.nn import GCNConv
    HAS_PYG = True
except Exception:  # pragma: no cover
    GCNConv = None
    HAS_PYG = False


# ---------------------------------------------------------------------------
# Data helpers


ACTION_VOCAB_BEAM = [
    "inc_bf",
    "dec_bf",
    "inc_tf",
    "dec_tf",
    "inc_hw",
    "dec_hw",
    "inc_tw",
    "dec_tw",
]

ACTION_VOCAB_COL = ["inc_b", "dec_b", "inc_t", "dec_t"]


@dataclass
class CandidateAction:
    member_id: str
    action: str
    member_index: int
    action_index: int
    group: str


def _numeric_features(node_rows: List[dict]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract a numeric feature matrix and member-id index mapping."""

    numeric_keys = []
    sample = node_rows[0] if node_rows else {}
    for k, v in sample.items():
        if k in {"node_id", "member_id", "group"}:
            continue
        if isinstance(v, (int, float, np.floating)):
            numeric_keys.append(k)

    features = []
    member_index = {}
    for idx, row in enumerate(node_rows):
        member_index[row.get("node_id") or row.get("member_id") or f"n{idx}"] = idx
        feats = [float(row.get(k, 0.0) or 0.0) for k in numeric_keys]
        features.append(feats)

    if not features:
        return np.zeros((0, 0), dtype=float), {}
    return np.asarray(features, dtype=float), member_index


def _edge_index(edge_rows: List[dict], member_index: Dict[str, int]) -> np.ndarray:
    pairs = []
    for row in edge_rows:
        u = row.get("src") or row.get("u") or row.get("from") or row.get("source")
        v = row.get("dst") or row.get("v") or row.get("to") or row.get("target")
        if u in member_index and v in member_index:
            pairs.append((member_index[u], member_index[v]))
            pairs.append((member_index[v], member_index[u]))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).T


# ---------------------------------------------------------------------------
# Network building blocks


class _FallbackNet:
    """Random scorer used when torch is unavailable."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score(self, features: np.ndarray, edge_index: np.ndarray, candidates: List[CandidateAction]):
        return [self.rng.random() for _ in candidates]

    def update(self, *args, **kwargs):  # pragma: no cover - noop
        return None


class GraphEncoder(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        if HAS_PYG:
            self.conv1 = GCNConv(in_features, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        else:
            self.conv1 = nn.Linear(in_features, hidden)
            self.conv2 = nn.Linear(hidden, hidden)
        self.use_pyg = HAS_PYG

    def forward(self, x, edge_index):
        if self.use_pyg:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
        else:
            # Simple mean aggregation when PyG is absent
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
        return F.relu(x)


class LocalQNet(nn.Module):
    def __init__(self, in_features: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = GraphEncoder(in_features, hidden)
        self.action_embed = nn.Embedding(action_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, candidates: List[CandidateAction]):
        node_latent = self.encoder(x, edge_index)
        q_values = []
        for cand in candidates:
            node_vec = node_latent[cand.member_index]
            act_vec = self.action_embed(torch.tensor(cand.action_index, device=x.device))
            q = self.head(torch.cat([node_vec, act_vec], dim=-1))
            q_values.append(q)
        if not q_values:
            return torch.zeros(0, device=x.device)
        return torch.cat(q_values, dim=0)


# ---------------------------------------------------------------------------
# Replay buffer + policy


class ReplayBuffer:
    def __init__(self, capacity: int = 5000, seed: int = 0):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.rng = random.Random(seed)

    def push(self, *transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return self.rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class LocalDQNPolicy:
    """Epsilon-greedy DQN policy with a graph encoder."""

    def __init__(
        self,
        *,
        epsilon: float = 0.25,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,
        lr: float = 1e-3,
        replay_capacity: int = 8000,
        batch_size: int = 16,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(replay_capacity)
        self.global_step = 0

        self._fallback = None
        if HAS_TORCH:
            # Lazily initialised once an observation arrives (so we know feature dims).
            self.q_net: Optional[LocalQNet] = None
            self.target_net: Optional[LocalQNet] = None
            self.optimizer: Optional[torch.optim.Optimizer] = None
        else:
            self._fallback = _FallbackNet()
            self.q_net = None
            self.target_net = None
            self.optimizer = None

    # ------------------------------------------------------------------
    def _maybe_init_network(self, feat_dim: int):
        if self._fallback or self.q_net is not None:
            return
        action_dim = max(len(ACTION_VOCAB_BEAM), len(ACTION_VOCAB_COL))
        self.q_net = LocalQNet(feat_dim, action_dim)
        self.target_net = LocalQNet(feat_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-3)

    def _to_tensors(self, node_rows: List[dict], edge_rows: List[dict]):
        feats, member_index = _numeric_features(node_rows)
        edges = _edge_index(edge_rows, member_index)
        if not HAS_TORCH:
            return feats, edges, member_index
        x = torch.tensor(feats, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long)
        return x, edge_index, member_index

    def select_action(
        self,
        node_rows: List[dict],
        edge_rows: List[dict],
        candidates: List[CandidateAction],
        training: bool = True,
    ) -> CandidateAction:
        if not candidates:
            raise ValueError("No candidate actions provided to the DQN policy")

        feats, edges, member_index = self._to_tensors(node_rows, edge_rows)
        if not HAS_TORCH:
            scores = self._fallback.score(feats, edges, candidates)
            return candidates[int(np.argmax(scores))]

        if feats.shape[0] == 0:
            return random.choice(candidates)

        # Update candidate member indices now that we have the mapping
        for cand in candidates:
            cand.member_index = member_index.get(cand.member_id, 0)

        self._maybe_init_network(feats.shape[1])
        device = next(self.q_net.parameters()).device
        feats = feats.to(device)
        edges = edges.to(device)

        explore = training and (random.random() < self.epsilon)
        with torch.no_grad():
            q_values = self.q_net(feats, edges, candidates)

        if explore:
            choice = random.randrange(len(candidates))
        else:
            choice = int(torch.argmax(q_values).item())
        return candidates[choice]

    # ------------------------------------------------------------------
    def record_transition(
        self,
        state_nodes: List[dict],
        state_edges: List[dict],
        candidates: List[CandidateAction],
        action_idx: int,
        reward: float,
        next_nodes: List[dict],
        next_edges: List[dict],
        next_candidates: List[CandidateAction],
        done: bool,
    ):
        self.buffer.push(
            state_nodes,
            state_edges,
            candidates,
            action_idx,
            reward,
            next_nodes,
            next_edges,
            next_candidates,
            done,
        )
        self._train_step()

    # ------------------------------------------------------------------
    def _train_step(self):
        if self._fallback or not HAS_TORCH:
            return
        if len(self.buffer) < self.batch_size or self.q_net is None or self.target_net is None:
            return

        batch = self.buffer.sample(self.batch_size)
        device = next(self.q_net.parameters()).device

        loss_accum = 0.0
        for transition in batch:
            (s_nodes, s_edges, s_cands, a_idx, r, ns_nodes, ns_edges, ns_cands, done) = transition
            s_feats, s_edge_index, _ = self._to_tensors(s_nodes, s_edges)
            ns_feats, ns_edge_index, _ = self._to_tensors(ns_nodes, ns_edges)

            # _to_tensors already returns torch tensors when PyTorch is available;
            # avoid wrapping them again to silence unnecessary copy warnings.
            s_feats = s_feats.to(device)
            s_edge_index = s_edge_index.to(device)
            ns_feats = ns_feats.to(device)
            ns_edge_index = ns_edge_index.to(device)

            for cand in s_cands:
                if cand.member_index is None:
                    cand.member_index = 0
            for cand in ns_cands:
                if cand.member_index is None:
                    cand.member_index = 0

            q_values = self.q_net(s_feats, s_edge_index, s_cands)
            q_sa = q_values[a_idx]

            with torch.no_grad():
                next_q = self.target_net(ns_feats, ns_edge_index, ns_cands)
                target = r + (0.0 if done else self.gamma * torch.max(next_q))

            loss = F.mse_loss(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_accum += loss.item()

        self.global_step += 1
        if self.global_step % 20 == 0 and self.target_net is not None:
            self.target_net.load_state_dict(self.q_net.state_dict())
        # decay epsilon after each batch
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


__all__ = [
    "LocalDQNPolicy",
    "CandidateAction",
    "ACTION_VOCAB_BEAM",
    "ACTION_VOCAB_COL",
]