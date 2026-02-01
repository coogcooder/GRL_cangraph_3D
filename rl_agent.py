

"""Reinforcement-learning design agent with a GNN-backed DQN policy.

The environment is shaped like a lightweight Gym interface so the local sizing
agent can operate in both an offline bootstrap mode (to create a feasible
reference design) and an online mode where a graph encoder + DQN learns to
select member sizing actions.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

import numpy as np

from design_frame_y2 import (
    HSS_THK,
    HSS_WIDTHS,
    I_FLANGE_THK,
    I_FLANGE_WIDTHS,
    I_WEB_HEIGHTS,
    I_WEB_THK,
    hss_square_props,
    i_section_props,
)
from graph_state import DesignGraph, GraphState, graph_records
from analysisops import run_all_cases_from_frame_config, SECTIONS
from gnn_state import build_violation_tensor, enumerate_local_candidates
from structural_configurator import apply_local_resize
from design_frame_y2 import analysis_results_by_member
from gnn_policy import (
    ACTION_VOCAB_BEAM,
    ACTION_VOCAB_COL,
    CandidateAction,
    LocalDQNPolicy,
)


STEEL_DENSITY = 7850.0  # kg/m^3

# Reward shaping constants
FRAME_PASS_BONUS = 6000.0
MEMBER_PASS_BONUS = 2400.0
FAIL_PENALTY = 1800.0
WEIGHT_SCALE = 15.0


@dataclass
class DesignStepResult:
    reward: float
    constraints_passed: bool
    weight: float
    member_id: str
    action: str
    utilization: Dict[str, float]


@dataclass
class EpisodeSummary:
    best_weight: float
    best_state: Dict[str, dict]
    rewards: List[DesignStepResult]
    global_rewards: List[DesignStepResult]
    weight_trace: List[float]
    batch_reports: List["BatchReport"]


@dataclass
class BatchReport:
    batch_index: int
    total: int
    passed: int
    failed: int
    weight: float
    frame_pass: bool


# ---------------------------------------------------------------------------
# Gym-like environment for local actions


class GraphDesignEnv:
    def __init__(
        self,
        problem: dict,
        cfg_meta: dict,
        prefix: str,
        *,
        seed: int = 0,
        surrogate_only: bool = False,
    ):
        self.problem = problem
        self.members = problem["beams"] + problem["columns"]
        self.member_map = {m["member_id"]: m for m in self.members}
        self.section_state: Dict[str, dict] = {}
        self.cfg_meta = cfg_meta
        self.prefix = prefix
        self.rng = random.Random(seed)
        self.member_pass: Dict[str, bool] = {}
        self._graph_state: Optional[GraphState] = None
        self._design_graph: Optional[DesignGraph] = None
        self._graph_rendered = False
        self._last_snapshot: Dict[str, dict] = {}
        self._node_index_map: Dict[str, int] = {}
        self._feature_keys: List[str] = []
        self._ur_indices: Dict[str, int] = {}
        self._last_weight: float = 0.0
        self._last_violation: float = 0.0
        self.surrogate_only = surrogate_only
        self.global_id_counter = 0
        self._global_action_map: List[Tuple[str, str]] = []
        self.global_mode = "both"
        self.span_limits_x = cfg_meta.get("span_limits_x", cfg_meta.get("x_min_max", (8.0, 30.0)))
        self.span_limits_y = cfg_meta.get("y_min_max", (6.0, 12.0))
        self._last_global_span_reward: float = 0.0
        self.reset()

    def reset(self):
        self.section_state = {}
        for m in self.problem["beams"]:
            self.section_state[m["member_id"]] = {"bf_idx": 0, "tf_idx": 0, "hw_idx": 0, "tw_idx": 0}
        for m in self.problem["columns"]:
            self.section_state[m["member_id"]] = {"b_idx": 0, "t_idx": 0}
        self.member_pass = {m["member_id"]: False for m in self.members}
        self._graph_state = None
        if self._design_graph is None:
            self._design_graph = DesignGraph.from_problem(self.problem, self.cfg_meta, {})
        else:
            self._design_graph.reset_analysis()
        self._graph_rendered = False
        return self._observe()

    # ------------------------------------------------------------------
    # Public helpers
    def girder_ids(self) -> List[str]:
        """Return known girder member IDs.

        A few call sites (including the PPO controller) previously reached for
        a private ``_girder_ids`` helper that might not exist on older
        instances. Keeping a public accessor here guarantees callers can size
        global action spaces without AttributeError.
        """

        if hasattr(self, "_girder_ids"):
            try:
                return self._girder_ids()  # type: ignore[attr-defined]
            except Exception:
                pass
        return [m["member_id"] for m in self.problem.get("beams", []) if m.get("group") == "GIR"]

    def section_size_rows(self) -> List[dict]:
        """Return section sizes plus constraint status for the latest snapshot.

        Each entry contains the member id, group (GIR/SEC/COL), translated
        geometric sizes (flange/web for beams, width/thickness for columns) and
        the most recent constraint information such as utilization ratios and
        pass/fail flag. This makes end-of-episode reporting more informative
        without requiring another full evaluation pass from callers.
        """

        # Ensure we have a fresh snapshot with utilization information
        if not self._last_snapshot:
            self._last_snapshot, _, _ = self._evaluate_frame()
        snapshot = self._last_snapshot

        rows: List[dict] = []
        for mid, state in self.section_state.items():
            member = self.member_map.get(mid, {})
            group = member.get("group", "")
            util = snapshot.get(mid, {})
            base = {
                "member_id": mid,
                "group": group,
                "UR_shear": float(util.get("UR_shear", 0.0) or 0.0),
                "UR_flex": float(util.get("UR_flex", 0.0) or 0.0),
                "UR_defl": float(util.get("UR_defl", util.get("UR_deflX", 0.0) or 0.0) or 0.0),
                "constraints_passed": bool(util.get("constraints_passed", False)),
            }

            if "bf_idx" in state:
                bf = I_FLANGE_WIDTHS[state.get("bf_idx", 0)]
                tf = I_FLANGE_THK[state.get("tf_idx", 0)]
                hw = I_WEB_HEIGHTS[state.get("hw_idx", 0)]
                tw = I_WEB_THK[state.get("tw_idx", 0)]
                base.update({"bf_mm": bf, "tf_mm": tf, "hw_mm": hw, "tw_mm": tw})
            else:
                b = HSS_WIDTHS[state.get("b_idx", 0)]
                t = HSS_THK[state.get("t_idx", 0)]
                base.update({"b_mm": b, "t_mm": t})
            rows.append(base)

        rows.sort(key=lambda r: r["member_id"])
        return rows

    def action_masks(self, global_only: bool = False):
        """Return simple logits masks for PPO heads plus action metadata.

        The masks are zero-filled (no exclusions) but sized to the current graph
        so downstream policies can apply additive masking. Calling this helper
        forces a tensor observation refresh when node indices are missing,
        preventing AttributeError during early rollout collection. Additional
        metadata describes how many actions are available and labels each global
        choice (``<girder_id>:add`` / ``<girder_id>:remove``) so logging can
        explain the size of the global logits vector.
        """

        if not getattr(self, "_node_index_map", None):
            # Seed node indices by rebuilding the tensor observation
            self.observe_tensors()

        snapshot = self._last_snapshot or self._evaluate_frame()[0]
        if hasattr(self, "_rank_spans"):
            add_candidates, remove_candidates = self._rank_spans(snapshot)  # type: ignore[attr-defined]
        else:
            # Backward compatibility: fall back to simple ordering
            girder_ids = self.girder_ids()
            add_candidates, remove_candidates = girder_ids, girder_ids
        if self.global_mode == "split":
            remove_candidates = []
        elif self.global_mode == "merge":
            add_candidates = []
        self._global_action_map = [(gid, "add") for gid in add_candidates] + [(gid, "remove") for gid in remove_candidates]
        num_global = max(1, len(self._global_action_map))
        num_nodes = max(1, len(self._node_index_map))

        global_labels: List[str] = []
        for gid, op in self._global_action_map:
            global_labels.append(f"{gid}:{op}")
        global_scores = torch.zeros(num_global)
        snapshot = self._last_snapshot or self._evaluate_frame()[0]
        for idx, (gid, op) in enumerate(self._global_action_map):
            util = snapshot.get(gid, {})
            score = self._max_ur(util)
            if op == "remove":
                score = -score
            global_scores[idx] = float(score)

        meta = {
            "num_global": num_global,
            "num_nodes": num_nodes,
            "global_labels": global_labels,
        }
        if not global_only:
            graph = self.graph()
            violation_tensor = build_violation_tensor(graph)
            local_candidates, local_feats = enumerate_local_candidates(
                graph,
                violation_tensor,
                threshold=1.0,
                top_k=40,
            )
            meta["violation_tensor"] = violation_tensor
            meta["local_candidate_indices"] = local_candidates
            meta["local_candidate_feats"] = local_feats

        masks = {
            "global": torch.zeros(num_global),
            "global_scores": global_scores,
            "node": None if global_only else torch.zeros(num_nodes),
            "size": None if global_only else torch.zeros(3),
            "meta": meta,
        }
        return masks

    def violation_tensor(self, node_tensor: torch.Tensor, ur_keys: List[str]) -> torch.Tensor:
        if not self._ur_indices:
            return torch.zeros((node_tensor.size(0), len(ur_keys)), device=node_tensor.device)
        cols = []
        for key in ur_keys:
            idx = self._ur_indices.get(key)
            if idx is None:
                cols.append(torch.zeros(node_tensor.size(0), device=node_tensor.device))
            else:
                cols.append(node_tensor[:, idx])
        ur = torch.stack(cols, dim=-1)
        return torch.relu(ur - 1.0)

    def violation_sum(self, snapshot: Dict[str, dict], ur_keys: List[str]) -> float:
        total = 0.0
        for util in snapshot.values():
            for key in ur_keys:
                val = float(util.get(key, 0.0) or 0.0)
                if val > 1.0:
                    total += val - 1.0
        return total

    # ------------------------------------------------------------------
    def initial_batch_scan(self, max_batches: int = 15) -> List[BatchReport]:
        """Strengthen members in manageable batches and record pass/fail counts.

        This stage keeps the environment moving beyond the first loop by
        touching a limited number of batches (10–15) before any global or
        local optimization. Each batch nudges members toward a reference size,
        evaluates constraints, and records weight plus pass/fail counts so the
        controller can visualize progress.
        """

        reports: List[BatchReport] = []
        snapshot: Dict[str, dict] = {}
        batch_size = max(1, math.ceil(len(self.members) / max_batches))
        for batch_idx, start in enumerate(range(0, len(self.members), batch_size), start=1):
            if batch_idx > max_batches:
                break
            batch = self.members[start : start + batch_size]
            for member in batch:
                state = self.section_state[member["member_id"]]
                if member["group"] in ("GIR", "SEC"):
                    state["bf_idx"] = max(state["bf_idx"], 1)
                    state["tf_idx"] = max(state["tf_idx"], 1)
                    state["hw_idx"] = max(state["hw_idx"], 1)
                    state["tw_idx"] = max(state["tw_idx"], 1)
                else:
                    state["b_idx"] = max(state["b_idx"], 1)
                    state["t_idx"] = max(state["t_idx"], 1)

            snapshot, weight, frame_pass = self._evaluate_frame()
            passed = 0
            failed = 0
            for member in batch:
                util = snapshot.get(member["member_id"], {})
                if util.get("constraints_passed"):
                    passed += 1
                else:
                    failed += 1
            reports.append(
                BatchReport(
                    batch_index=batch_idx,
                    total=len(batch),
                    passed=passed,
                    failed=failed,
                    weight=weight,
                    frame_pass=frame_pass,
                )
            )

        self._last_snapshot = snapshot if reports else {}
        return reports

    # ------------------------------------------------------------------
    def bootstrap_reference_design(self, batch_size: int = 8):
        """Offline phase: batch-up members and bump their sizes upward."""

        beams = [m for m in self.problem["beams"] if m["group"] in ("GIR", "SEC")]
        cols = [m for m in self.problem["columns"]]
        candidates = beams + cols
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            for member in batch:
                state = self.section_state[member["member_id"]]
                if member["group"] in ("GIR", "SEC"):
                    state["bf_idx"] = min(len(I_FLANGE_WIDTHS) - 1, 2)
                    state["tf_idx"] = min(len(I_FLANGE_THK) - 1, 2)
                    state["hw_idx"] = min(len(I_WEB_HEIGHTS) - 1, 2)
                    state["tw_idx"] = min(len(I_WEB_THK) - 1, 2)
                else:
                    state["b_idx"] = min(len(HSS_WIDTHS) - 1, 2)
                    state["t_idx"] = min(len(HSS_THK) - 1, 2)
            # evaluate to update pass states
            self._evaluate_frame()

    def observe(self):
        return self._observe()

    # ------------------------------------------------------------------
    def _observe(self):
        snapshot, _, _ = self._evaluate_frame()
        self._last_snapshot = snapshot
        node_rows, edge_rows = graph_records(self.graph())
        return snapshot, node_rows, edge_rows

    def step(self, action, action_str: str = None, *, mode: str = None):
        """Apply either a legacy local action or a PPO global/local action.

        - Legacy/DQN callers invoke ``env.step(member_id, action_str)`` and
          expect a 4-tuple ``(obs, reward, done, info)``.
        - PPO callers invoke ``env.step(ppo_action, mode="local"|"global")``
          and expect a 3-tuple ``(obs, reward, done)``.
        """

        # ------------------------------------------------------------------
        # Legacy/local two-argument path (member_id + action string)
        if mode is None:
            member_id = action if isinstance(action, str) else getattr(action, "member_id", None)
            action_choice = action_str if action_str is not None else getattr(action, "action", None)
            if not member_id or not action_choice:
                raise ValueError("Legacy step requires a member_id and action string")

            member = self.member_map[member_id]
            prev_snapshot, _, _ = self._evaluate_frame()
            prev_util = prev_snapshot.get(member_id, {})
            prev_pass = bool(prev_util.get("constraints_passed", False))

            if member["group"] in ("GIR", "SEC"):
                self._mutate_beam_state(self.section_state[member_id], action_choice)
            else:
                self._mutate_column_state(self.section_state[member_id], action_choice)

            snapshot, weight, frame_pass = self._evaluate_frame()
            util = snapshot.get(member_id, {})
            now_pass = bool(util.get("constraints_passed", False))
            self.member_pass[member_id] = now_pass

            reward = self._reward(prev_util, util, prev_pass, now_pass, weight, frame_pass)
            done = all(self.member_pass.values())
            node_rows, edge_rows = graph_records(self.graph())
            self._last_snapshot = snapshot
            return (snapshot, node_rows, edge_rows), reward, done, {"weight": weight, "frame_pass": frame_pass, "util": util}

        # ------------------------------------------------------------------
        # PPO path using combined global/local PPOAction
        delta = int(getattr(action, "local_size", 1).item() if hasattr(action, "local_size") else 1) - 1
        if mode == "local":
            graph = self.graph()
            violation_tensor = build_violation_tensor(graph)
            candidates, _ = enumerate_local_candidates(
                graph,
                violation_tensor,
                threshold=1.0,
                top_k=40,
            )
            if not candidates:
                obs = self.observe_tensors()
                return obs, 0.0, False
            node_rows = graph.node_records()
            cand_idx = int(getattr(action, "local_node", 0).item()) % len(candidates)
            node_idx = candidates[cand_idx]
            node_id = node_rows[node_idx]["node_id"]
            self._apply_local_resize_member(node_id, delta)
        elif mode == "global":
            if not self._global_action_map:
                self.action_masks(global_only=True)
            choice = int(getattr(action, "global_action", 0).item()) % max(1, len(self._global_action_map))
            girder_id, op = self._global_action_map[choice]
            add_col = getattr(self, "_global_add_column", None)
            rem_col = getattr(self, "_global_remove_column", None)
            if op == "add" and add_col:
                add_col(girder_id)
            elif op == "remove" and rem_col:
                rem_col(girder_id)
        else:
            raise ValueError(f"Unknown mode '{mode}' for PPO step")

        snapshot, weight, frame_pass = self._evaluate_frame()
        self._last_snapshot = snapshot
        reward = -self._max_ur(snapshot) * 500.0 - weight / WEIGHT_SCALE
        if frame_pass:
            reward += FRAME_PASS_BONUS
        reward += self._last_global_span_reward
        self._last_global_span_reward = 0.0
        obs = self.observe_tensors()
        return obs, reward, frame_pass

    def graph(self) -> GraphState:
        if self._design_graph is None:
            self._design_graph = DesignGraph.from_problem(self.problem, self.cfg_meta, {})
        if self._graph_state is None:
            self._graph_state = self._design_graph.graph
            if not self._graph_rendered:
                self._graph_state.render_plot(self.prefix)
                self._graph_rendered = True
        return self._graph_state

    # ------------------------------------------------------------------
    # Tensor-friendly observation helper (used by PPO policies)

    def observe_tensors(self):
        """Return node feature and edge index tensors for the current graph state.

        This helper rebuilds/updates the cached :class:`GraphState`, converts the
        node records into a dense feature tensor, and maps edge endpoints to the
        corresponding node indices. It also seeds ``_node_index_map`` so action
        mask generation can safely size distributions even before any steps have
        been taken.
        """

        graph = self.graph()
        node_rows, edge_rows = graph_records(graph)
        if not node_rows:
            # Fallback shape that keeps the PPO loop from crashing on empty graphs
            self._node_index_map = {}
            self._feature_keys = []
            return (
                torch.zeros((1, 1), dtype=torch.float32),
                torch.zeros((2, 0), dtype=torch.long),
            )

        feature_keys = [k for k in sorted(node_rows[0].keys()) if k not in {"node_id", "group"}]
        feats = []
        for row in node_rows:
            feats.append([float(row.get(k, 0.0) or 0.0) for k in feature_keys])
        node_tensor = torch.tensor(feats, dtype=torch.float32)

        node_ids = [row["node_id"] for row in node_rows]
        index_map = {nid: idx for idx, nid in enumerate(node_ids)}
        edges = torch.zeros((2, len(edge_rows)), dtype=torch.long)
        for idx, edge in enumerate(edge_rows):
            edges[0, idx] = index_map.get(edge.get("source"), 0)
            edges[1, idx] = index_map.get(edge.get("target"), 0)

        self._node_index_map = index_map
        self._feature_keys = feature_keys
        self._ur_indices = {k: i for i, k in enumerate(feature_keys) if k.startswith("UR_")}
        return node_tensor, edges

    # ------------------------------------------------------------------
    # Global topology actions (split/merge spans) --------------------------------

    def apply_global_strategy(self, snapshot: Dict[str, dict], max_actions: int = 4) -> List[DesignStepResult]:
        """Run a lightweight batch of global actions before local optimization.

        The strategy is simple: split the worst-performing girders to shorten
        spans, then merge the most over-designed girders to reduce weight. Each
        action records a synthetic reward so global/local plots remain
        comparable.
        """

        results: List[DesignStepResult] = []
        failing = sorted(
            ((self._max_ur(util), mid) for mid, util in snapshot.items() if not util.get("constraints_passed")),
            reverse=True,
        )
        for _, mid in failing[:max_actions]:
            prev_snapshot, _, _ = self._evaluate_frame()
            self.split_span(mid)
            post_snapshot, weight, frame_pass = self._evaluate_frame()
            util = post_snapshot.get(mid, {})
            reward = self._reward(
                prev_snapshot.get(mid, {}),
                util,
                bool(prev_snapshot.get(mid, {}).get("constraints_passed", False)),
                bool(util.get("constraints_passed", False)),
                weight,
                frame_pass,
            )
            results.append(
                DesignStepResult(
                    reward=reward,
                    constraints_passed=frame_pass,
                    weight=weight,
                    member_id=mid,
                    action="split_span",
                    utilization=util,
                )
            )

        passing = sorted(
            ((self._max_ur(util), mid) for mid, util in snapshot.items() if util.get("constraints_passed")),
        )
        for _, mid in passing[:max_actions]:
            prev_snapshot, _, _ = self._evaluate_frame()
            merged_id = self.merge_span(mid)
            if not merged_id:
                continue
            post_snapshot, weight, frame_pass = self._evaluate_frame()
            util = post_snapshot.get(merged_id, post_snapshot.get(mid, {}))
            reward = self._reward(
                prev_snapshot.get(mid, {}),
                util,
                bool(prev_snapshot.get(mid, {}).get("constraints_passed", False)),
                bool(util.get("constraints_passed", False)),
                weight,
                frame_pass,
            )
            results.append(
                DesignStepResult(
                    reward=reward,
                    constraints_passed=frame_pass,
                    weight=weight,
                    member_id=merged_id,
                    action="merge_span",
                    utilization=util,
                )
            )

        self._last_snapshot = snapshot
        return results

    def split_span(self, span_id: str) -> Optional[str]:
        beam = self.member_map.get(span_id)
        if not beam or beam.get("group") not in ("GIR", "SEC"):
            return None

        xi = float(beam.get("Xi", beam.get("Xmid", 0.0)))
        xj = float(beam.get("Xj", beam.get("Xmid", 0.0) + max(beam.get("length", 1.0), 1.0)))
        yi = float(beam.get("Yi", beam.get("Ymid", 0.0)))
        yj = float(beam.get("Yj", beam.get("Ymid", 0.0)))
        ymid = float(beam.get("Ymid", yi))

        total_len = self._beam_length(beam, xi, yi, xj, yj)
        dir_sign = 1.0 if xj >= xi else -1.0
        max_len = self.span_limits_x[1]
        min_len = self.span_limits_x[0]

        seg_len = max(min_len, min(total_len / 2.0, max_len))
        other_len = total_len - seg_len
        if other_len > max_len:
            other_len = max_len
            seg_len = max(min_len, total_len - other_len)

        mid_x = xi + dir_sign * seg_len

        new_col_id = f"COL_SPLIT_{self.global_id_counter}"
        self.global_id_counter += 1
        story_h = self.cfg_meta.get("story_h", 3.0)
        new_col = {
            "member_id": new_col_id,
            "group": "COL",
            "Xi": mid_x,
            "Yi": yi,
            "Zi": 0.0,
            "Xj": mid_x,
            "Yj": yi,
            "Zj": story_h,
            "length": story_h,
        }

        beam_a = {
            **beam,
            "member_id": f"{span_id}_a",
            "Xi": xi,
            "Yi": yi,
            "Xj": mid_x,
            "Yj": yi,
            "Xmid": (xi + mid_x) / 2.0,
            "Ymid": ymid,
            "length": abs(mid_x - xi),
        }
        beam_b = {
            **beam,
            "member_id": f"{span_id}_b",
            "Xi": mid_x,
            "Yi": yi,
            "Xj": xj,
            "Yj": yi,
            "Xmid": (mid_x + xj) / 2.0,
            "Ymid": ymid,
            "length": abs(xj - mid_x),
        }

        # Replace members
        self.problem["beams"] = [b for b in self.problem["beams"] if b["member_id"] != span_id]
        self.problem["beams"].extend([beam_a, beam_b])
        self.problem["columns"].append(new_col)
        self._refresh_members()
        return new_col_id

    def merge_span(self, span_id: str) -> Optional[str]:
        beam = self.member_map.get(span_id)
        if not beam or beam.get("group") not in ("GIR", "SEC"):
            return None

        ymid = float(beam.get("Ymid", beam.get("Yi", 0.0)))
        same_line = [b for b in self.problem["beams"] if b.get("group") in ("GIR", "SEC") and abs(float(b.get("Ymid", b.get("Yi", 0.0))) - ymid) < 1e-3]
        same_line.sort(key=lambda b: float(b.get("Xmid", b.get("Xi", 0.0))))
        idx = next((i for i, b in enumerate(same_line) if b["member_id"] == span_id), None)
        if idx is None:
            return None

        neighbor = None
        if idx > 0:
            neighbor = same_line[idx - 1]
        if idx < len(same_line) - 1:
            right = same_line[idx + 1]
            if neighbor is None:
                neighbor = right
            else:
                # prefer the shorter neighbor to keep merged length reasonable
                if self._beam_length(right) < self._beam_length(neighbor):
                    neighbor = right

        if not neighbor:
            return None

        xi = float(min(beam.get("Xi", beam.get("Xmid", 0.0)), neighbor.get("Xi", neighbor.get("Xmid", 0.0))))
        xj = float(max(beam.get("Xj", beam.get("Xmid", 0.0)), neighbor.get("Xj", neighbor.get("Xmid", 0.0))))
        yi = float(beam.get("Yi", beam.get("Ymid", 0.0)))
        length = abs(xj - xi)
        if length > self.span_limits_x[1]:
            # refuse merge if it violates span limits
            return None

        merged_id = f"{span_id}_merged_{self.global_id_counter}"
        self.global_id_counter += 1
        merged_beam = {
            **beam,
            "member_id": merged_id,
            "Xi": xi,
            "Yi": yi,
            "Xj": xj,
            "Yj": yi,
            "Xmid": (xi + xj) / 2.0,
            "Ymid": ymid,
            "length": length,
        }

        # Remove the interior column nearest the shared boundary
        boundary_x = (float(beam.get("Xmid", beam.get("Xi", 0.0))) + float(neighbor.get("Xmid", neighbor.get("Xi", 0.0)))) / 2.0
        cols = []
        removed_col = None
        for col in self.problem["columns"]:
            if removed_col is None and abs(float(col.get("Xi", 0.0)) - boundary_x) < 0.75 and abs(float(col.get("Yi", 0.0)) - ymid) < 1e-3:
                removed_col = col
                continue
            cols.append(col)
        if removed_col is not None:
            self.problem["columns"] = cols

        self.problem["beams"] = [b for b in self.problem["beams"] if b["member_id"] not in (span_id, neighbor["member_id"])]
        self.problem["beams"].append(merged_beam)
        self._refresh_members()
        return merged_id

    # ------------------------------------------------------------------
    def _reward(self, prev_util, util, prev_pass, now_pass, weight, frame_pass):
        reward = 0.0
        prev_max = self._max_ur(prev_util)
        now_max = self._max_ur(util)
        reward += (prev_max - now_max) * 2500.0
        if now_pass and not prev_pass:
            reward += MEMBER_PASS_BONUS
        if not now_pass:
            reward -= FAIL_PENALTY
        if frame_pass:
            reward += FRAME_PASS_BONUS
        reward -= weight / WEIGHT_SCALE
        return reward

    # ------------------------------------------------------------------
    def _evaluate_frame(self):
        # Build design_state for evaluation
        design_state = {}
        for member in self.members:
            mid = member["member_id"]
            if member["group"] in ("GIR", "SEC"):
                props = self._beam_properties(self.section_state[mid])
            else:
                props = self._column_properties(self.section_state[mid])
            length = float(member.get("length", 1.0) or 1.0)
            props["weight_kg"] = float(props.get("A", 0.0) or 0.0) * length * STEEL_DENSITY
            props["area"] = props.get("A", 0.0)
            design_state[mid] = props

        if self._design_graph is None:
            self._design_graph = DesignGraph.from_problem(self.problem, self.cfg_meta, design_state)
        else:
            self._design_graph.update_from_design_state(self.problem, self.cfg_meta, design_state)
        node_rows, edge_rows = graph_records(self._design_graph.graph)
        analysis_rows = {row["node_id"]: row for row in node_rows}
        weight = sum(float(r.get("weight_kg", r.get("weight", 0.0)) or 0.0) for r in node_rows)
        frame_pass = all(r.get("constraints_passed", False) for r in node_rows)
        return analysis_rows, weight, frame_pass
    
    def _frame_config_from_problem(self) -> dict:
        nodes = []
        node_ids = {}
        def _node_id(x: float, y: float, z: float) -> str:
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key in node_ids:
                return node_ids[key]
            nid = f"N_{len(node_ids)}"
            node_ids[key] = nid
            nodes.append({"node_id": nid, "x": x, "y": y, "z": z})
            return nid

        members = []
        for member in self.problem.get("beams", []) + self.problem.get("columns", []):
            Xi = float(member.get("Xi", member.get("Xmid", 0.0)))
            Yi = float(member.get("Yi", member.get("Ymid", 0.0)))
            Zi = float(member.get("Zi", 0.0))
            Xj = float(member.get("Xj", member.get("Xmid", Xi)))
            Yj = float(member.get("Yj", member.get("Ymid", Yi)))
            Zj = float(member.get("Zj", member.get("Zmid", Zi)))
            i_node_id = _node_id(Xi, Yi, Zi)
            j_node_id = _node_id(Xj, Yj, Zj)
            group = member.get("group", "")
            props = SECTIONS.get(group, SECTIONS.get("GIR", {}))
            members.append(
                {
                    "member_id": member["member_id"],
                    "group": group,
                    "i_node_id": i_node_id,
                    "j_node_id": j_node_id,
                    "story_index": member.get("story_index"),
                    "frame_index": member.get("frame_index"),
                    "span_index": member.get("span_index"),
                    "bay_index": member.get("bay_index"),
                    "E": props.get("E", 2.0e8),
                    "G": props.get("G", props.get("E", 2.0e8) / 2.6),
                    "A": props.get("A", 0.01),
                    "Iy": props.get("Iy", 0.01),
                    "Iz": props.get("Iz", 0.01),
                    "J": props.get("J", 0.01),
                    "analysis": group in {"SEC", "SEC_ANALYSIS"},
                }
            )
        return {
            "nodes": nodes,
            "members": members,
            "loads": {"w_dead": self.cfg_meta.get("w_dead", 0.0), "w_live": self.cfg_meta.get("w_live", 0.0)},
            "meta": {
                "story_h": self.cfg_meta.get("story_h"),
                "num_stories": self.cfg_meta.get("stories", self.cfg_meta.get("num_stories")),
                "bay_spacing": self.cfg_meta.get("bay_spacing"),
                "sec_spacing": self.cfg_meta.get("sec_spacing"),
            },
        }

    def analyze_and_update(self, out_prefix: str):
        if self.surrogate_only:
            return
        frame_config = self._frame_config_from_problem()
        run_all_cases_from_frame_config(frame_config, out_prefix)
        analysis_rows = analysis_results_by_member(out_prefix, case="DL")
        if self._design_graph is None:
            self._design_graph = DesignGraph.from_problem(self.problem, self.cfg_meta, analysis_rows)
        self._design_graph.update_from_analysis(analysis_rows)
        self._last_snapshot = analysis_rows

    def _refresh_members(self):
        self.members = self.problem["beams"] + self.problem["columns"]
        self.member_map = {m["member_id"]: m for m in self.members}
        for mid in list(self.section_state.keys()):
            if mid not in self.member_map:
                del self.section_state[mid]
        for m in self.members:
            if m["member_id"] not in self.section_state:
                if m.get("group") in ("GIR", "SEC"):
                    self.section_state[m["member_id"]] = {"bf_idx": 0, "tf_idx": 0, "hw_idx": 0, "tw_idx": 0}
                else:
                    self.section_state[m["member_id"]] = {"b_idx": 0, "t_idx": 0}
        self.member_pass = {mid: self.member_pass.get(mid, False) for mid in self.section_state.keys()}

    def _beam_length(self, member: dict, xi=None, yi=None, xj=None, yj=None) -> float:
        xi = float(member.get("Xi", xi if xi is not None else 0.0))
        yi = float(member.get("Yi", yi if yi is not None else 0.0))
        xj = float(member.get("Xj", xj if xj is not None else 0.0))
        yj = float(member.get("Yj", yj if yj is not None else 0.0))
        return math.hypot(xj - xi, yj - yi)

    # ------------------------------------------------------------------
    @staticmethod
    def _max_ur(util: Dict[str, float]) -> float:
        return max(
            float(util.get("UR_shear", 0) or 0),
            float(util.get("UR_flex", 0) or 0),
            float(util.get("UR_defl", 0) or 0),
            float(util.get("UR_deflX", 0) or 0),
            float(util.get("UR_deflY", 0) or 0),
        )

    # ------------------------------------------------------------------
    def _mutate_beam_state(self, state: dict, action: str):
        if action == "inc_bf":
            state["bf_idx"] = min(state["bf_idx"] + 1, len(I_FLANGE_WIDTHS) - 1)
        elif action == "dec_bf":
            state["bf_idx"] = max(0, state["bf_idx"] - 1)
        elif action == "inc_tf":
            state["tf_idx"] = min(state["tf_idx"] + 1, len(I_FLANGE_THK) - 1)
        elif action == "dec_tf":
            state["tf_idx"] = max(0, state["tf_idx"] - 1)
        elif action == "inc_hw":
            state["hw_idx"] = min(state["hw_idx"] + 1, len(I_WEB_HEIGHTS) - 1)
        elif action == "dec_hw":
            state["hw_idx"] = max(0, state["hw_idx"] - 1)
        elif action == "inc_tw":
            state["tw_idx"] = min(state["tw_idx"] + 1, len(I_WEB_THK) - 1)
        elif action == "dec_tw":
            state["tw_idx"] = max(0, state["tw_idx"] - 1)

    def _mutate_column_state(self, state: dict, action: str):
        if action == "inc_b":
            state["b_idx"] = min(state["b_idx"] + 1, len(HSS_WIDTHS) - 1)
        elif action == "dec_b":
            state["b_idx"] = max(0, state["b_idx"] - 1)
        elif action == "inc_t":
            state["t_idx"] = min(state["t_idx"] + 1, len(HSS_THK) - 1)
        elif action == "dec_t":
            state["t_idx"] = max(0, state["t_idx"] - 1)

    def _apply_local_delta_safe(self, member_id: str, delta: int):
        """Robust wrapper around local sizing tweaks for PPO actions.

        Some environments loaded from older checkpoints may miss the internal
        ``_apply_local_delta`` helper. This wrapper keeps the new implementation
        but falls back to an inline version if the attribute is absent so PPO
        rollouts never fail with ``AttributeError``.
        """

        impl = getattr(self, "_apply_local_delta_impl", None)
        if callable(impl):
            try:
                impl(member_id, delta)
                return
            except AttributeError:
                # fall through to the inline implementation
                pass

        if member_id not in self.section_state:
            return
        state = self.section_state[member_id]
        if "bf_idx" in state:
            state["bf_idx"] = max(0, min(state.get("bf_idx", 0) + delta, len(I_FLANGE_WIDTHS) - 1))
            state["tf_idx"] = max(0, min(state.get("tf_idx", 0) + delta, len(I_FLANGE_THK) - 1))
            state["hw_idx"] = max(0, min(state.get("hw_idx", 0) + delta, len(I_WEB_HEIGHTS) - 1))
            state["tw_idx"] = max(0, min(state.get("tw_idx", 0) + delta, len(I_WEB_THK) - 1))
        else:
            state["b_idx"] = max(0, min(state.get("b_idx", 0) + delta, len(HSS_WIDTHS) - 1))
            state["t_idx"] = max(0, min(state.get("t_idx", 0) + delta, len(HSS_THK) - 1))

    def _apply_local_resize_member(self, member_id: str, delta: int):
        if member_id not in self.section_state:
            return
        state = self.section_state[member_id]
        member = self.member_map.get(member_id, {})
        temp = {"group": member.get("group", "")}
        temp.update(state)
        updated = apply_local_resize(temp, delta)
        for key in ("bf_idx", "tf_idx", "hw_idx", "tw_idx", "b_idx", "t_idx"):
            if key in updated:
                state[key] = updated[key]
        if self._design_graph is not None and member_id in self._design_graph.graph.nodes:
            feats = self._design_graph.graph.nodes[member_id].features
            for key in ("bf", "tf", "hw", "tw", "b", "t"):
                if key in updated:
                    feats[key] = updated[key]

    # Backward-compatible alias retained for older callers
    def _apply_local_delta(self, member_id: str, delta: int):
        self._apply_local_delta_safe(member_id, delta)

    # Original implementation kept for environments that still call it directly
    def _apply_local_delta_impl(self, member_id: str, delta: int):
        if member_id not in self.section_state:
            return
        state = self.section_state[member_id]
        if "bf_idx" in state:
            state["bf_idx"] = max(0, min(state.get("bf_idx", 0) + delta, len(I_FLANGE_WIDTHS) - 1))
            state["tf_idx"] = max(0, min(state.get("tf_idx", 0) + delta, len(I_FLANGE_THK) - 1))
            state["hw_idx"] = max(0, min(state.get("hw_idx", 0) + delta, len(I_WEB_HEIGHTS) - 1))
            state["tw_idx"] = max(0, min(state.get("tw_idx", 0) + delta, len(I_WEB_THK) - 1))
        else:
            state["b_idx"] = max(0, min(state.get("b_idx", 0) + delta, len(HSS_WIDTHS) - 1))
            state["t_idx"] = max(0, min(state.get("t_idx", 0) + delta, len(HSS_THK) - 1))

    # ------------------------------------------------------------------
    def _beam_properties(self, state: dict):
        """Return a dict of beam properties and a constraints flag.

        This routine synthesizes utilization ratios from the relative sizing
        indices so the RL loop has meaningful feedback even without the full
        code-check backend. Larger indices lower the URs; smaller sizes push
        them above 1.0 until the member is strengthened.
        """

        bf_idx = state.get("bf_idx", 0)
        tf_idx = state.get("tf_idx", 0)
        hw_idx = state.get("hw_idx", 0)
        tw_idx = state.get("tw_idx", 0)

        bf = I_FLANGE_WIDTHS[bf_idx]
        tf = I_FLANGE_THK[tf_idx]
        hw = I_WEB_HEIGHTS[hw_idx]
        tw = I_WEB_THK[tw_idx]
        A, Ixx, S, Aw, d = i_section_props(bf, tf, hw, tw)

        size_score = (
            bf_idx / max(len(I_FLANGE_WIDTHS) - 1, 1)
            + tf_idx / max(len(I_FLANGE_THK) - 1, 1)
            + hw_idx / max(len(I_WEB_HEIGHTS) - 1, 1)
            + tw_idx / max(len(I_WEB_THK) - 1, 1)
        ) / 4.0
        # Higher size_score -> lower utilization; clamp to a realistic band.
        ur_base = max(0.35, 1.4 - 1.2 * size_score)

        props = {
            "A": A,
            "Ixx": Ixx,
            "S": S,
            "Aw": Aw,
            "d": d,
            "UR_shear": ur_base * 0.9,
            "UR_flex": ur_base,
            "UR_defl": ur_base * 1.05,
        }
        props["constraints_passed"] = self._pass_check(props)
        return props

    def _column_properties(self, state: dict):
        """Return a dict of column properties with a constraints flag."""

        b_idx = state.get("b_idx", 0)
        t_idx = state.get("t_idx", 0)
        b = HSS_WIDTHS[b_idx]
        t = HSS_THK[t_idx]
        A, Ixx, S, Av, d = hss_square_props(b, t)

        size_score = (
            b_idx / max(len(HSS_WIDTHS) - 1, 1)
            + t_idx / max(len(HSS_THK) - 1, 1)
        ) / 2.0
        ur_base = max(0.35, 1.35 - 1.15 * size_score)

        props = {
            "A": A,
            "Ixx": Ixx,
            "S": S,
            "Av": Av,
            "d": d,
            "UR_shear": ur_base * 0.95,
            "UR_flex": ur_base,
            "UR_defl": ur_base * 1.02,
        }
        props["constraints_passed"] = self._pass_check(props)
        return props

    def _pass_check(self, props: dict) -> bool:
        ur_shear = float(props.get("UR_shear", 0) or 0)
        ur_flex = float(props.get("UR_flex", 0) or 0)
        ur_defl = float(props.get("UR_defl", 0) or 0)
        return max(ur_shear, ur_flex, ur_defl) <= 1.0

    # ------------------------------------------------------------------
    def candidate_actions(self, snapshot: Dict[str, dict], batch_k: int = 10) -> List[CandidateAction]:
        """Build a focused candidate set (top-k UR members with inc/dec)."""

        scored = []
        for mid, util in snapshot.items():
            max_ur = self._max_ur(util)
            scored.append((max_ur, mid))
        scored.sort(reverse=True)
        top = [mid for _, mid in scored[:batch_k]]
        candidates: List[CandidateAction] = []
        for mid in top:
            member = self.member_map[mid]
            if member["group"] in ("GIR", "SEC"):
                actions = ACTION_VOCAB_BEAM
            else:
                actions = ACTION_VOCAB_COL
            for act in actions:
                candidates.append(
                    CandidateAction(
                        member_id=mid,
                        action=act,
                        member_index=0,
                        action_index=(actions.index(act)),
                        group="beam" if member["group"] in ("GIR", "SEC") else "column",
                    )
                )
        if not candidates:
            # fallback to random member
            m = random.choice(self.members)
            actions = ACTION_VOCAB_BEAM if m["group"] in ("GIR", "SEC") else ACTION_VOCAB_COL
            for act in actions:
                candidates.append(CandidateAction(m["member_id"], act, 0, actions.index(act),
                                                  "beam" if m["group"] in ("GIR", "SEC") else "column"))
        return candidates


# ---------------------------------------------------------------------------
# Agent wrapper


class ReinforcementDesignAgent:
    def __init__(self, *, steps_per_iteration: Optional[int] = 50, seed: int = 0):
        # ``steps_per_iteration`` can be None to let the agent compute a
        # problem-aware cap instead of requiring user input.
        self.steps_per_iteration = steps_per_iteration
        self.policy = LocalDQNPolicy()
        self.rng = random.Random(seed)
        self.rewards_history: List[float] = []
        self.weight_history: List[float] = []
        self.global_reward_history: List[float] = []
        self.local_reward_history: List[float] = []
        self.global_weight_history: List[float] = []
        self.weight_traces: List[List[float]] = []

    # ------------------------------------------------------------------
    def run_iteration(self, prefix: str, problem: dict, cfg_meta: dict) -> EpisodeSummary:
        env = GraphDesignEnv(problem, cfg_meta, prefix, seed=self.rng.randrange(10_000))
        # Offline bootstrap to start near a feasible region
        env.bootstrap_reference_design()
        batch_reports = env.initial_batch_scan(max_batches=15)
        snapshot, node_rows, edge_rows = env.observe()

        # Track initial state weight for decay visualization, including
        # batch-level evaluations so users see progress even before global/local
        # moves.
        weight_trace: List[float] = [r.weight for r in batch_reports] if batch_reports else []
        if not weight_trace:
            _, initial_weight, _ = env._evaluate_frame()
            weight_trace = [initial_weight]

        # Global adjustments on batches of members before local search
        global_results = env.apply_global_strategy(snapshot)
        weight_trace.extend([res.weight for res in global_results])
        snapshot, node_rows, edge_rows = env.observe()

        best_weight = math.inf
        best_snapshot: Dict[str, dict] = {}
        rewards: List[DesignStepResult] = []
        local_reward_total = 0.0

        member_count = len(problem.get("beams", [])) + len(problem.get("columns", []))
        max_steps = self.steps_per_iteration or max(1, member_count * 8)

        for step in range(max_steps):
            candidates = env.candidate_actions(snapshot)
            choice = self.policy.select_action(node_rows, edge_rows, candidates, training=True)
            next_obs, reward, done, info = env.step(choice.member_id, choice.action)
            next_snapshot, next_node_rows, next_edge_rows = next_obs

            action_idx = candidates.index(choice)
            next_candidates = env.candidate_actions(next_snapshot)
            self.policy.record_transition(
                node_rows,
                edge_rows,
                candidates,
                action_idx,
                reward,
                next_node_rows,
                next_edge_rows,
                next_candidates,
                done,
            )

            rewards.append(
                DesignStepResult(
                    reward=reward,
                    constraints_passed=info.get("frame_pass", False),
                    weight=info.get("weight", 0.0),
                    member_id=choice.member_id,
                    action=choice.action,
                    utilization=info.get("util", {}),
                )
            )
            weight_trace.append(info.get("weight", 0.0))
            local_reward_total += reward

            snapshot, node_rows, edge_rows = next_snapshot, next_node_rows, next_edge_rows

            if info.get("frame_pass", False) and info.get("weight", math.inf) < best_weight:
                best_weight = info.get("weight", math.inf)
                best_snapshot = snapshot
            if done:
                break

        if not best_snapshot:
            best_snapshot = snapshot
            best_weight = min(r.weight for r in rewards) if rewards else (min(weight_trace) if weight_trace else math.inf)

        episode_reward = sum(r.reward for r in rewards)
        self.rewards_history.append(episode_reward)
        self.weight_history.append(best_weight if best_weight < math.inf else 0.0)
        self.global_reward_history.append(sum(r.reward for r in global_results))
        self.local_reward_history.append(local_reward_total)
        self.global_weight_history.append(weight_trace[0] if weight_trace else 0.0)
        self.weight_traces.append(weight_trace)

        return EpisodeSummary(
            best_weight=best_weight,
            best_state=best_snapshot,
            rewards=rewards,
            global_rewards=global_results,
            weight_trace=weight_trace,
            batch_reports=batch_reports,
        )

    # ------------------------------------------------------------------
    def plot_rewards(self, path: str):
        try:
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            plt.figure()
            plt.plot(self.global_reward_history, label="global reward")
            plt.plot(self.local_reward_history, label="local reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception:
            return None

    def plot_weight_history(self, path: str):
        try:
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            plt.figure()
            plt.plot(self.global_weight_history, label="post-global weight (kg)")
            plt.plot(self.weight_history, label="best local weight (kg)")
            plt.xlabel("Episode")
            plt.ylabel("Weight (kg)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception:
            return None

    def plot_weight_decay(self, path: str):
        try:
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            plt.figure()
            for idx, trace in enumerate(self.weight_traces):
                if not trace:
                    continue
                plt.plot(trace, label=f"Episode {idx+1} weight")
            plt.xlabel("Step (global + local)")
            plt.ylabel("Weight (kg)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception:
            return None

    def _girder_ids(self) -> List[str]:
        return [m["member_id"] for m in self.problem.get("beams", []) if m.get("group") == "GIR"]

    def _global_add_column(self, girder_id: str):
        """Insert a column at the girder mid-point and split the girder span.

        The global action space should grow/shrink as spans are subdivided or
        merged. To reflect that, we replace the target girder with two shorter
        segments and register the new column through the full story height.
        """

        girder = self.member_map.get(girder_id)
        if not girder:
            return

        # Midpoint where the new column will be placed/split
        xi = float(girder.get("Xmid", girder.get("Xi", 0.0)))
        yi = float(girder.get("Ymid", girder.get("Yi", 0.0)))
        zi = float(girder.get("Zi", 0.0))
        zj = float(girder.get("Zj", 3.0))
        left_len = abs(xi - float(girder.get("Xi", girder.get("Xmid", 0.0))))
        right_len = abs(float(girder.get("Xj", girder.get("Xmid", 0.0))) - xi)
        self._last_global_span_reward = self._span_limit_reward([left_len, right_len])

        # Add column through the full height at the split location
        new_col_id = f"GCOL_{self.global_id_counter}"; self.global_id_counter += 1
        col = {
            "member_id": new_col_id,
            "group": "COL",
            "Xi": xi,
            "Yi": yi,
            "Zi": zi,
            "Xj": xi,
            "Yj": yi,
            "Zj": zj,
        }
        self.problem.setdefault("columns", []).append(col)
        self.member_map[new_col_id] = col
        self.section_state[new_col_id] = {"b_idx": 0, "t_idx": 0}
        self.members.append(col)

        # Split the girder into two spans about the new column
        self._split_girder(girder_id, xi, yi, zi, zj)
        self._graph_state = None  # force rebuild so node/edge counts reflect the new topology
        self._node_index_map = {}
        self._refresh_members()
        if self._design_graph is not None:
            self._design_graph.apply_global_action(self.problem, self.cfg_meta, {})

    def _global_remove_column(self, girder_id: str):
        """Remove a column near the girder midpoint and merge split spans.

        When a column is removed, any girder segments that were created by a
        previous split (and tagged with ``parent_gid``) are merged back into a
        single span so the global action space shrinks accordingly.
        """

        target = self.member_map.get(girder_id)
        if not target:
            return
        root_gid = target.get("parent_gid") or girder_id
        xi = float(target.get("Xmid", target.get("Xi", 0.0)))
        yi = float(target.get("Ymid", target.get("Yi", 0.0)))

        # Remove columns located at the target midpoint
        keep_cols = []
        removed_any = False
        for col in self.problem.get("columns", []):
            if abs(float(col.get("Xi", 0.0)) - xi) < 0.5 and abs(float(col.get("Yi", 0.0)) - yi) < 0.5:
                self.section_state.pop(col["member_id"], None)
                self.member_map.pop(col["member_id"], None)
                removed_any = True
                continue
            keep_cols.append(col)
        self.problem["columns"] = keep_cols
        self.members = [m for m in self.members if m.get("group") != "COL" or m in keep_cols]

        # Merge split girder segments if they exist
        if removed_any:
            merged_len = self._merge_girder_segments(root_gid, xi, yi)
            if merged_len is None:
                self._last_global_span_reward = self._span_limit_reward([])
            else:
                self._last_global_span_reward = self._span_limit_reward([merged_len])
            self._graph_state = None
            self._node_index_map = {}
            if self._design_graph is not None:
                self._design_graph.apply_global_action(self.problem, self.cfg_meta, {})
            
            self._refresh_members()
        else:
            self._last_global_span_reward = self._span_limit_reward([])

    def _split_girder(self, girder_id: str, xi: float, yi: float, zi: float, zj: float):
        """Replace a girder with two segments split about (xi, yi)."""

        girder = self.member_map.get(girder_id)
        if not girder:
            return

        # Remove the original girder
        self.problem["beams"] = [b for b in self.problem.get("beams", []) if b.get("member_id") != girder_id]
        self.members = [m for m in self.members if m.get("member_id") != girder_id]
        self.member_pass.pop(girder_id, None)
        base_state = self.section_state.pop(girder_id, {"bf_idx": 0, "tf_idx": 0, "hw_idx": 0, "tw_idx": 0})
        self.member_map.pop(girder_id, None)

        # Build two new girder segments
        gid_a = f"{girder_id}_a{self.global_id_counter}"; self.global_id_counter += 1
        gid_b = f"{girder_id}_b{self.global_id_counter}"; self.global_id_counter += 1

        girder_a = {
            **girder,
            "member_id": gid_a,
            "Xj": xi,
            "Yj": yi,
            "Zj": zj,
            "parent_gid": girder_id,
        }
        girder_b = {
            **girder,
            "member_id": gid_b,
            "Xi": xi,
            "Yi": yi,
            "Zi": zi,
            "parent_gid": girder_id,
        }

        for g in (girder_a, girder_b):
            self.problem.setdefault("beams", []).append(g)
            self.member_map[g["member_id"]] = g
            self.section_state[g["member_id"]] = dict(base_state)
            self.members.append(g)
            self.member_pass[g["member_id"]] = False

    def _merge_girder_segments(self, girder_id: str, xi: float, yi: float) -> Optional[float]:
        """Merge split girder segments back into a single span."""

        # Find split segments created from this girder
        segments = [b for b in self.problem.get("beams", []) if b.get("parent_gid") == girder_id]
        if len(segments) < 2:
            return None

        # Sort by Xi to keep ordering deterministic
        segments.sort(key=lambda b: (float(b.get("Xi", 0.0)), float(b.get("Yi", 0.0))))
        first, last = segments[0], segments[-1]

        merged = {
            **first,
            "member_id": girder_id,
            "parent_gid": None,
            "Xi": first.get("Xi", first.get("Xmid", 0.0)),
            "Yi": first.get("Yi", first.get("Ymid", 0.0)),
            "Zi": first.get("Zi", 0.0),
            "Xj": last.get("Xj", last.get("Xmid", xi)),
            "Yj": last.get("Yj", last.get("Ymid", yi)),
            "Zj": last.get("Zj", last.get("Zmid", 3.0)),
        }

        # Remove the segments
        keep_beams = []
        for b in self.problem.get("beams", []):
            if b in segments:
                self.section_state.pop(b["member_id"], None)
                self.member_map.pop(b["member_id"], None)
                self.member_pass.pop(b["member_id"], None)
                self.members = [m for m in self.members if m.get("member_id") != b["member_id"]]
                continue
            keep_beams.append(b)
        self.problem["beams"] = keep_beams

        # Add back a merged girder
        self.problem.setdefault("beams", []).append(merged)
        self.member_map[merged["member_id"]] = merged
        self.section_state[merged["member_id"]] = {"bf_idx": 0, "tf_idx": 0, "hw_idx": 0, "tw_idx": 0}
        self.members.append(merged)
        self.member_pass[merged["member_id"]] = False
        length = abs(float(merged.get("Xj", merged.get("Xmid", 0.0))) - float(merged.get("Xi", merged.get("Xmid", 0.0))))
        return length

    def _span_limit_reward(self, lengths: List[float]) -> float:
        min_len, max_len = self.span_limits_x
        if not lengths:
            return -0.1
        within = all(min_len <= length <= max_len for length in lengths)
        return 0.1 if within else -0.1

    # ------------------------------------------------------------------
    def _rank_spans(self, snapshot: Dict[str, dict]) -> Tuple[List[str], List[str]]:
        """Rank girders for add/remove actions based on constraint severity.

        - Add/split: girders that fail constraints, sorted by highest UR first.
        - Remove/merge: girders that pass constraints, sorted by lowest UR first.
        """

        girders = [m for m in self.problem.get("beams", []) if m.get("group") == "GIR"]
        failing: List[Tuple[float, str]] = []
        passing: List[Tuple[float, str]] = []

        for girder in girders:
            mid = girder["member_id"]
            util = snapshot.get(mid, {})
            score = self._max_ur(util)
            if util.get("constraints_passed"):
                passing.append((score, mid))
            else:
                failing.append((score, mid))

        failing.sort(key=lambda t: t[0], reverse=True)
        passing.sort(key=lambda t: t[0])
        add_candidates = [mid for _, mid in failing]
        remove_candidates = [mid for _, mid in passing]
        return add_candidates, remove_candidates

__all__ = [
    "ReinforcementDesignAgent",
    "GraphDesignEnv",
]