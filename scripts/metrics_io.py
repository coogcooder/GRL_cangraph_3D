"""Metrics IO helpers for training runs."""
from __future__ import annotations

import csv
import os
from typing import List, Dict


METRIC_FIELDS = [
    "episode",
    "total_reward",
    "weight",
    "weight_reduction_pct",
    "violation_sum",
    "feasible_flag",
    "steps_to_feasible",
]


def write_metrics(rows: List[Dict[str, float]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})
