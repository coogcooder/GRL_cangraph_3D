"""Minimal surrogate run to validate the pipeline."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from control import run_training, set_global_seeds


def main():
    set_global_seeds(0)
    run_training(
        num_episodes=5,
        seed=0,
        steps_per_episode=5,
        local_per_global=5,
        surrogate_only=True,
        metrics_path="outputs/metrics.csv",
    )


if __name__ == "__main__":
    main()