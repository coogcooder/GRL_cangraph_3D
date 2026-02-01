"""Canonical GNN state utilities and compatibility exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Sequence, Union

import torch

from canonical_mapping import CanonicalMapping, build_canonical_mapping
from structural_configurator import apply_global_action as apply_global_action_cfg
from structural_configurator import frame_gridlines_from_config

from graph_state import (  # noqa: F401
    GraphNode,
    GraphState,
    DesignGraph,
    export_graph_csv,
    graph_records,
)
DEFLECTION_LIMIT_DENOM = 240.0


@dataclass
class CanonicalGraph:
    nodes: Dict[str, Dict[str, Any]]
    edges: List[Dict[str, Any]]
    mapping: CanonicalMapping

GLOBAL_CANDIDATE_FEATURES = [
    "x_left",
    "x_right",
    "x_new",
    "x_remove",
    "span_left",
    "span_right",
    "span_merge",
    "action_add",
    "action_remove",
    "story_count",
    "current_num_columns",
]

LOCAL_CANDIDATE_FEATURES = [
    "violation_max",
    "violation_defl",
    "violation_drift",
    "section_index_norm",
    "is_girder",
    "is_secondary",
    "is_column",
    "shear_demand",
    "moment_demand",
    "deflection",
]

def build_canonical_graph(
    frame_config_3d: dict,
    cfg_meta: Optional[dict] = None,
    design_state: Optional[Dict[str, dict]] = None,
    surrogate_only: bool = False,
) -> CanonicalGraph:
    cfg_meta = cfg_meta or {}
    design_state = design_state or {}
    mapping = build_canonical_mapping(frame_config_3d)
    member_map = {m.get("member_id"): m for m in frame_config_3d.get("members", []) if m.get("member_id")}

    x_gridlines, y_gridlines = frame_gridlines_from_config(frame_config_3d)
    nodes: Dict[str, Dict[str, Any]] = {}
    for canonical_id, (_, member_id) in mapping.canonical_members.items():
        member = member_map.get(member_id)
        if not member:
            continue
        group = _member_group(member)
        features = _canonical_node_features(
            member,
            frame_config_3d,
            cfg_meta,
            design_state.get(member_id, {}),
            x_gridlines,
            y_gridlines,
            surrogate_only=surrogate_only,
        )
        nodes[canonical_id] = {
            "node_id": canonical_id,
            "group": group,
            "features": features,
        }

    edges = _canonical_edges(
        [member_map[member_id] for _, member_id in mapping.canonical_members.values() if member_id in member_map],
        frame_config_3d,
    )
    return CanonicalGraph(nodes=nodes, edges=edges, mapping=mapping)

def enumerate_global_candidates(graph: GraphState) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    node_rows = graph.node_records()
    candidates: List[Dict[str, Any]] = []
    feature_rows: List[List[float]] = []
    num_columns = sum(1 for row in node_rows if row.get("group") == "COL")

    for row in node_rows:
        if row.get("group") != "GIR":
            continue
        length = float(row.get("member_span_length", row.get("length", 0.0)) or 0.0)
        pos_x = float(row.get("pos_x", 0.0) or 0.0)
        x_left = pos_x - length / 2.0
        x_right = pos_x + length / 2.0
        story_count = float(row.get("story_count", 0.0) or 0.0)

        base = dict(
            x_left=x_left,
            x_right=x_right,
            x_new=pos_x,
            x_remove=0.0,
            span_left=length / 2.0,
            span_right=length / 2.0,
            span_merge=0.0,
            action_add=1.0,
            action_remove=0.0,
            story_count=story_count,
            current_num_columns=float(num_columns),
        )
        candidates.append({"member_id": row.get("node_id"), "action": "add", **base})
        feature_rows.append([float(base[key]) for key in GLOBAL_CANDIDATE_FEATURES])

        remove = dict(
            x_left=x_left,
            x_right=x_right,
            x_new=0.0,
            x_remove=pos_x,
            span_left=0.0,
            span_right=0.0,
            span_merge=length,
            action_add=0.0,
            action_remove=1.0,
            story_count=story_count,
            current_num_columns=float(num_columns),
        )
        candidates.append({"member_id": row.get("node_id"), "action": "remove", **remove})
        feature_rows.append([float(remove[key]) for key in GLOBAL_CANDIDATE_FEATURES])

    if not feature_rows:
        return candidates, torch.zeros((0, len(GLOBAL_CANDIDATE_FEATURES)), dtype=torch.float32)
    return candidates, torch.tensor(feature_rows, dtype=torch.float32)

def build_violation_tensor(graph: GraphState) -> torch.Tensor:
    node_rows = graph.node_records()
    rows: List[List[float]] = []
    for row in node_rows:
        max_ur = _violation_value(row, None)
        defl_ur = max(
            float(row.get("UR_defl", 0.0) or 0.0),
            float(row.get("UR_deflX", 0.0) or 0.0),
            float(row.get("UR_deflY", 0.0) or 0.0),
        )
        drift_ur = max(
            float(row.get("drift_x", 0.0) or 0.0),
            float(row.get("drift_y", 0.0) or 0.0),
        )
        rows.append([float(max_ur), float(defl_ur), float(drift_ur)])
    if not rows:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def enumerate_local_candidates(
    graph: GraphState,
    violation_field: Optional[Union[str, torch.Tensor, Sequence[Dict[str, float]]]] = None,
    *,
    threshold: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor]:
    node_rows = graph.node_records()
    indices: List[int] = []
    feature_rows: List[List[float]] = []
    violation_tensor = None
    if isinstance(violation_field, torch.Tensor):
        violation_tensor = violation_field
    elif isinstance(violation_field, Sequence):
        try:
            violation_tensor = torch.tensor(
                [[d.get("max_UR", 0.0), d.get("deflection_UR", 0.0), d.get("drift_UR", 0.0)] for d in violation_field],
                dtype=torch.float32,
            )
        except Exception:
            violation_tensor = None
    for idx, row in enumerate(node_rows):
        if isinstance(violation_field, str):
            violation_max = float(row.get(violation_field, 0.0) or 0.0)
        elif violation_tensor is not None and idx < violation_tensor.shape[0]:
            violation_max = float(violation_tensor[idx, 0].item())
        else:
            violation_max = _violation_value(row, None)
        if violation_max <= threshold:
            continue
        if violation_tensor is not None and idx < violation_tensor.shape[0]:
            defl_violation = float(violation_tensor[idx, 1].item())
            drift_violation = float(violation_tensor[idx, 2].item())
        else:
            defl_violation = max(
                float(row.get("UR_defl", 0.0) or 0.0),
                float(row.get("UR_deflX", 0.0) or 0.0),
                float(row.get("UR_deflY", 0.0) or 0.0),
            )
            drift_violation = max(
                float(row.get("drift_x", 0.0) or 0.0),
                float(row.get("drift_y", 0.0) or 0.0),
            )
        group = row.get("group", "")
        section_index = _normalized_section_index(row)
        features = [
            float(violation_max),
            float(defl_violation),
            float(drift_violation),
            float(section_index),
            1.0 if group == "GIR" else 0.0,
            1.0 if group == "SEC" else 0.0,
            1.0 if group == "COL" else 0.0,
            float(row.get("shear_demand", 0.0) or 0.0),
            float(row.get("moment_demand", 0.0) or 0.0),
            float(row.get("deflection", 0.0) or 0.0),
        ]
        indices.append(idx)
        feature_rows.append(features)

    if top_k is not None and feature_rows:
        ranked = sorted(
            zip(indices, feature_rows),
            key=lambda item: item[1][0],
            reverse=True,
        )[:top_k]
        indices = [item[0] for item in ranked]
        feature_rows = [item[1] for item in ranked]

    if not feature_rows:
        return indices, torch.zeros((0, len(LOCAL_CANDIDATE_FEATURES)), dtype=torch.float32)
    return indices, torch.tensor(feature_rows, dtype=torch.float32)


def _violation_value(row: Dict[str, Any], violation_field: Optional[str]) -> float:
    if violation_field:
        return float(row.get(violation_field, 0.0) or 0.0)
    vals = [
        float(row.get("UR_shear", 0.0) or 0.0),
        float(row.get("UR_flex", 0.0) or 0.0),
        float(row.get("UR_defl", 0.0) or 0.0),
        float(row.get("UR_deflX", 0.0) or 0.0),
        float(row.get("UR_deflY", 0.0) or 0.0),
    ]
    return max(vals) if vals else 0.0


def _normalized_section_index(row: Dict[str, Any]) -> float:
    for key in ("bf_idx", "b_idx", "tf_idx", "t_idx", "hw_idx", "tw_idx"):
        if key in row:
            try:
                return float(row[key]) / 10.0
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _graph_enumerate_global_candidates(self: GraphState) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    return enumerate_global_candidates(self)


def _graph_enumerate_local_candidates(
    self: GraphState,
    violation_field: Optional[str] = None,
    *,
    threshold: float = 1.0,
) -> Tuple[List[int], torch.Tensor]:
    return enumerate_local_candidates(self, violation_field, threshold=threshold)


if not hasattr(GraphState, "enumerate_global_candidates"):
    GraphState.enumerate_global_candidates = _graph_enumerate_global_candidates  # type: ignore[assignment]
if not hasattr(GraphState, "enumerate_local_candidates"):
    GraphState.enumerate_local_candidates = _graph_enumerate_local_candidates  # type: ignore[assignment]

def apply_global_action(frame_config: dict, action: dict, span_limits: Tuple[float, float]) -> Tuple[bool, str]:
    return apply_global_action_cfg(frame_config, action, span_limits)


def _canonical_edges(canonical_members: List[dict], frame_config_3d: dict) -> List[Dict[str, Any]]:
    node_to_members: Dict[str, List[str]] = {}
    for member in canonical_members:
        member_id = member.get("member_id")
        if not member_id:
            continue
        for node_id in (member.get("i_node_id"), member.get("j_node_id")):
            if node_id:
                node_to_members.setdefault(node_id, []).append(member_id)

    edge_map: Dict[Tuple[str, str], str] = {}
    for node_id, member_ids in node_to_members.items():
        if len(member_ids) < 2:
            continue
        member_ids = sorted(set(member_ids))
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                u, v = member_ids[i], member_ids[j]
                connection = _edge_connection_type(
                    _member_by_id(u, canonical_members),
                    _member_by_id(v, canonical_members),
                )
                key = (u, v)
                if key not in edge_map:
                    edge_map[key] = connection
                elif edge_map[key] != "pinned" and connection == "pinned":
                    edge_map[key] = connection

    return [
        {"source": u, "target": v, "connection_type": conn}
        for (u, v), conn in sorted(edge_map.items())
    ]


def _canonical_node_features(
    member: dict,
    frame_config_3d: dict,
    cfg_meta: dict,
    design: dict,
    x_gridlines: List[float],
    y_gridlines: List[float],
    *,
    surrogate_only: bool,
) -> Dict[str, float]:
    building = frame_config_3d.get("building")
    width_x = cfg_meta.get("width_x") or getattr(building, "width_x", 0.0) or 0.0
    length_y = cfg_meta.get("length_y") or getattr(building, "length_y", 0.0) or 0.0
    story_h = cfg_meta.get("story_h") or getattr(building, "story_h", 0.0) or 0.0
    story_count = cfg_meta.get("stories") or cfg_meta.get("num_stories") or getattr(building, "num_stories", 0) or 0
    x_span_count = max(len(x_gridlines) - 1, 0)
    y_span_count = max(len(y_gridlines) - 1, 0)
    x_span_length = cfg_meta.get("x_span") or _average_spacing(x_gridlines)
    y_span_length = cfg_meta.get("y_span") or _average_spacing(y_gridlines)
    bay_spacing = cfg_meta.get("bay_spacing") or y_span_length

    length = float(member.get("length", 0.0) or 0.0)
    story_number = _member_story(member, story_h)
    deflection_limit = (length / DEFLECTION_LIMIT_DENOM) if length else 0.0

    shear_demand = 0.0 if surrogate_only else float(member.get("shear_demand", 0.0) or 0.0)
    moment_demand = 0.0 if surrogate_only else float(member.get("moment_demand", 0.0) or 0.0)
    deflection = 0.0 if surrogate_only else float(member.get("deflection", 0.0) or 0.0)
    drift_x = 0.0 if surrogate_only else float(member.get("drift_x", 0.0) or 0.0)
    drift_y = 0.0 if surrogate_only else float(member.get("drift_y", 0.0) or 0.0)

    features = {
        "building_width_x": float(width_x),
        "building_length_y": float(length_y),
        "story_count": float(story_count),
        "story_height": float(story_h),
        "bay_spacing": float(bay_spacing),
        "x_span_count": float(x_span_count),
        "y_span_count": float(y_span_count),
        "x_span_length": float(x_span_length or 0.0),
        "y_span_length": float(y_span_length or 0.0),
        "member_span_length": float(length),
        "story_number": float(story_number),
        "shear_demand": float(shear_demand),
        "moment_demand": float(moment_demand),
        "deflection": float(deflection),
        "deflection_limit": float(deflection_limit),
        "UR_shear": float(0.0 if surrogate_only else design.get("UR_shear", 0.0) or 0.0),
        "UR_flex": float(0.0 if surrogate_only else design.get("UR_flex", 0.0) or 0.0),
        "UR_defl": float(0.0 if surrogate_only else design.get("UR_defl", 0.0) or 0.0),
        "UR_deflX": float(0.0 if surrogate_only else design.get("UR_deflX", 0.0) or 0.0),
        "UR_deflY": float(0.0 if surrogate_only else design.get("UR_deflY", 0.0) or 0.0),
        "drift_x": float(drift_x),
        "drift_y": float(drift_y),
        "bf": float(0.0 if surrogate_only else design.get("bf", 0.0) or 0.0),
        "tf": float(0.0 if surrogate_only else design.get("tf", 0.0) or 0.0),
        "hw": float(0.0 if surrogate_only else design.get("hw", 0.0) or 0.0),
        "tw": float(0.0 if surrogate_only else design.get("tw", 0.0) or 0.0),
        "b": float(0.0 if surrogate_only else design.get("b", 0.0) or 0.0),
        "t": float(0.0 if surrogate_only else design.get("t", 0.0) or 0.0),
    }
    return features


def _member_story(member: dict, story_h: float) -> int:
    if member.get("story_index") is not None:
        return int(member.get("story_index"))
    zi = float(member.get("Zi", 0.0) or 0.0)
    zj = float(member.get("Zj", 0.0) or 0.0)
    if story_h:
        return int(round(min(zi, zj) / story_h))
    return 0


def _average_spacing(lines: List[float]) -> float:
    if not lines or len(lines) < 2:
        return 0.0
    spans = [abs(lines[i + 1] - lines[i]) for i in range(len(lines) - 1)]
    return sum(spans) / len(spans) if spans else 0.0


def _edge_connection_type(member_a: Optional[dict], member_b: Optional[dict]) -> str:
    types = {
        _connection_type(member_a),
        _connection_type(member_b),
    }
    if "pinned" in types or "pin" in types:
        return "pinned"
    return "fixed"


def _connection_type(member: Optional[dict]) -> str:
    if not member:
        return "fixed"
    for key in ("connection_type", "connection", "conn_type", "conn"):
        val = member.get(key)
        if isinstance(val, str) and val:
            return val.strip().lower()
    return "fixed"


def _member_group(member: dict) -> str:
    group = str(member.get("group") or member.get("section_id") or "").upper()
    if not group:
        member_id = str(member.get("member_id", "")).upper()
        for prefix in ("COL", "GIR", "SEC"):
            if member_id.startswith(prefix):
                return prefix
    return group


def _member_by_id(member_id: str, members: List[dict]) -> Optional[dict]:
    for member in members:
        if member.get("member_id") == member_id:
            return member
    return None