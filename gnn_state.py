"""Canonical GNN state utilities and compatibility exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

from canonical_mapping import CanonicalMapping, build_canonical_mapping
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