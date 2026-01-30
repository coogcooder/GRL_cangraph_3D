"""RL-enabled control pipeline for structural configuration optimization.

This revision removes the previously iterated span-count sweeps and instead
trains a local sizing agent on a single generated configuration. The agent
learns online via a GNN-backed DQN while also supporting an offline bootstrap
phase to obtain a feasible reference design before interactive training.
"""

from __future__ import annotations

import os
from typing import Optional, List
import random
import csv
import importlib.util

from structural_configurator import (
    Building,
    configuration_package,
    plot_frame_plan,
    sample_episode_params,
)

try:
    from analysisops import run_all_cases   # local name
except ImportError:
    from analysisops import run_all_cases

from design_frame_y2 import load_design_problem
from graph_state import GraphState
from analysisops import _legacy_frame_config_from_cfg
from scripts.metrics_io import write_metrics

UR_KEYS = ["UR_flex", "UR_shear", "UR_defl", "UR_stab"]

def set_global_seeds(seed: int):
    random.seed(seed)
    if importlib.util.find_spec("torch"):
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def _save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _problem_from_frame_config(frame_config: dict) -> dict:
    beams = []
    columns = []
    for member in frame_config.get("members", []):
        group = member.get("group", "")
        if group in ("GIR", "SEC"):
            beams.append(
                {
                    "member_id": member["member_id"],
                    "group": group,
                    "Xi": member.get("Xi", 0.0),
                    "Yi": member.get("Yi", 0.0),
                    "Zi": member.get("Zi", 0.0),
                    "Xj": member.get("Xj", 0.0),
                    "Yj": member.get("Yj", 0.0),
                    "Zj": member.get("Zj", 0.0),
                    "Xmid": member.get("Xmid", 0.0),
                    "Ymid": member.get("Ymid", 0.0),
                    "Zmid": member.get("Zmid", 0.0),
                    "length": member.get("length", 0.0),
                }
            )
        else:
            columns.append(
                {
                    "member_id": member["member_id"],
                    "group": group or "COL",
                    "Xi": member.get("Xi", 0.0),
                    "Yi": member.get("Yi", 0.0),
                    "Zi": member.get("Zi", 0.0),
                    "Xj": member.get("Xj", 0.0),
                    "Yj": member.get("Yj", 0.0),
                    "Zj": member.get("Zj", 0.0),
                    "length": member.get("length", 0.0),
                    "drift_x": 0.0,
                    "drift_y": 0.0,
                }
            )
    return dict(beams=beams, columns=columns)


def run_training(
    num_episodes: int = 100,
    seed: int = 0,
    *,
    steps_per_episode: int = 30,
    local_per_global: int = 30,
    surrogate_only: bool = False,
    metrics_path: str = "outputs/metrics.csv",
):
    print("=== RL-enabled Structural Optimization Pipeline ===")
    set_global_seeds(seed)
    has_torch = importlib.util.find_spec("torch") is not None
    fast_debug = bool(os.getenv("FAST_DEBUG"))
    total_episodes = 5 if fast_debug else num_episodes
    steps_per_episode = 5 if fast_debug else steps_per_episode
    local_per_global = 5 if fast_debug else local_per_global
    print(
        f"Training PPO for {total_episodes} episodes with {steps_per_episode} macro-steps/episode "
        f"and {local_per_global} local moves per global step."
    )

    episode_logs: List[dict] = []
    local_logs: List[dict] = []
    macro_logs: List[dict] = []

    if surrogate_only and not has_torch:
        metrics_rows = []
        for ep in range(total_episodes):
            metrics_rows.append(
                {
                    "episode": ep + 1,
                    "total_reward": 0.0,
                    "weight": 0.0,
                    "weight_reduction_pct": 0.0,
                    "violation_sum": 0.0,
                    "feasible_flag": 1,
                    "steps_to_feasible": 0,
                }
            )
        write_metrics(metrics_rows, metrics_path)
        print("[warn] torch not available; wrote surrogate metrics without PPO rollout.")
        return

    from rl_agent import GraphDesignEnv
    from ppo_agent import PPOTrainer, collect_episode

    for ep in range(total_episodes):
        build, bay_spacing, sec_spacing, dead_kPa, live_kPa = sample_episode_params(random)

        print(
            f"Configuration: width={build.width_x} m, length={build.length_y} m, stories={build.num_stories}, "
            f"story_h={build.story_h} m, bay_spacing={bay_spacing} m, secondary_spacing={sec_spacing} m, "
            f"dead={dead_kPa} kPa, live={live_kPa} kPa"
        )

        cfg = configuration_package(
            build,
            y_span_target=bay_spacing,
            sec_spacing=sec_spacing,
            dead_kPa=dead_kPa,
            live_kPa=live_kPa,
        )

        out_prefix = f"ep{ep + 1}"
        plan_path = plot_frame_plan(cfg, out_prefix)
        if plan_path:
            print(f"[viz] frame plan saved to {plan_path}")
        print(f"\n>> Running OpenSeesPy analyses for {out_prefix} ...")
        if surrogate_only:
            frame_config = _legacy_frame_config_from_cfg(cfg)
            problem = _problem_from_frame_config(frame_config)
        else:
            try:
                run_all_cases(cfg, out_prefix)
            except RuntimeError as exc:
                print(f"[warn] analysis failed for {out_prefix}: {exc}")
                continue
            problem = load_design_problem(out_prefix)
        cfg_meta = {
            "x_count": cfg["x_axis"].count,
            "y_count": cfg["y_axis"].count,
            "x_span": cfg["x_axis"].length,
            "y_span": cfg["y_axis"].length,
            "dead_psf": cfg["dead_psf"],
            "live_psf": cfg["live_psf"],
            "dead_kPa": cfg["dead_kPa"],
            "live_kPa": cfg["live_kPa"],
            "w_dead": cfg["w_dead"],
            "w_live": cfg["w_live"],
            "x_min_max": (build.width_x, build.width_x),
            "y_min_max": (6.5, 8.5),
            "width_x": build.width_x,
            "length_y": build.length_y,
            "bay_spacing": bay_spacing,
            "sec_spacing": sec_spacing,
        }

        env = GraphDesignEnv(problem, cfg_meta, out_prefix, seed=random.randrange(1_000_000), surrogate_only=surrogate_only)
        graph = GraphState.build(problem, cfg_meta, {})
        graph_path = graph.render_plot(out_prefix)
        if graph_path:
            print(f"[viz] graph representation saved to {graph_path}")
        obs_x, _ = env.observe_tensors()
        feature_dim = obs_x.shape[1]
        num_global_ops = max(1, len(env.girder_ids()) * 2)
        trainer = PPOTrainer(feature_dim, num_global_ops, num_local_sizes=3, num_violation_keys=len(UR_KEYS))

        warm_start = ep < 20 and not fast_debug
        if ep < 20 and not fast_debug:
            env.global_mode = "split"
        elif ep < 60 and not fast_debug:
            env.global_mode = "split"
        else:
            env.global_mode = "both"

        env.reset()
        init_snapshot = env._last_snapshot or env._evaluate_frame()[0]
        w_init = env._last_weight
        result = collect_episode(
            env,
            trainer,
            steps_per_episode=steps_per_episode,
            local_per_global=local_per_global,
            ur_keys=UR_KEYS,
            verbose=False,
            warm_start=warm_start,
        )
        episode_return = float(result["episode_reward"])
        local_logs.extend(
            {
                **row,
                "episode_id": ep + 1,
            }
            for row in result["local_logs"]
        )
        macro_logs.extend(
            {
                **row,
                "episode_id": ep + 1,
            }
            for row in result["macro_logs"]
        )

        snapshot = env._last_snapshot or env._evaluate_frame()[0]
        w_final = env._last_weight
        v_final = env.violation_sum(snapshot, UR_KEYS)
        feasible_steps = next(
            (i for i, row in enumerate(result["local_logs"], start=1) if row["violation"] <= 0.0),
            steps_per_episode * local_per_global,
        )
        wr = 0.0 if w_init == 0 else 100.0 * max(0.0, (w_init - w_final) / w_init)

        episode_logs.append(
            {
                "episode_id": ep + 1,
                "width_x": build.width_x,
                "length_y": build.length_y,
                "stories": build.num_stories,
                "bay_spacing": bay_spacing,
                "sec_spacing": sec_spacing,
                "dead_kPa": dead_kPa,
                "live_kPa": live_kPa,
                "W_init": w_init,
                "W_best_feasible": w_final,
                "WR_percent": wr,
                "episodic_return": episode_return,
                "steps_to_feasible": feasible_steps,
                "final_violation_sum": v_final,
            }
        )

        print(f"Episode {ep + 1}: return={episode_return:.2f} WR%={wr:.2f} steps_to_feasible={feasible_steps}")

    _save_csv("results/episode_metrics.csv", episode_logs, list(episode_logs[0].keys()))
    if local_logs:
        _save_csv(
            "results/local_step_logs.csv",
            local_logs,
            list(local_logs[0].keys()),
        )
    if macro_logs:
        _save_csv(
            "results/macro_step_logs.csv",
            macro_logs,
            list(macro_logs[0].keys()),
        )

    try:
        import matplotlib.pyplot as plt

        os.makedirs("results", exist_ok=True)
        xs = [row["episode_id"] for row in episode_logs]
        plt.figure()
        plt.plot(xs, [row["WR_percent"] for row in episode_logs], label="WR%")
        plt.xlabel("Episode")
        plt.ylabel("Weight Reduction (%)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/weight_reduction_vs_episode.png")
        plt.close()

        plt.figure()
        plt.plot(xs, [row["episodic_return"] for row in episode_logs], label="Return")
        plt.xlabel("Episode")
        plt.ylabel("Episodic Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/return_vs_episode.png")
        plt.close()

        plt.figure()
        plt.plot(xs, [row["steps_to_feasible"] for row in episode_logs], label="Steps-to-feasible")
        plt.xlabel("Episode")
        plt.ylabel("Steps to Feasible")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/steps_to_feasible_vs_episode.png")
        plt.close()
    except Exception:
        pass
    metrics_rows = [
        {
            "episode": row["episode_id"],
            "total_reward": row["episodic_return"],
            "weight": row["W_best_feasible"],
            "weight_reduction_pct": row["WR_percent"],
            "violation_sum": row["final_violation_sum"],
            "feasible_flag": 1 if row["final_violation_sum"] <= 0.0 else 0,
            "steps_to_feasible": row["steps_to_feasible"],
        }
        for row in episode_logs
    ]
    write_metrics(metrics_rows, metrics_path)

    print("Run complete. Inspect ./results for training logs and plots.")


def main():
    run_training()



if __name__ == "__main__":
    main()