"""Canonical mapping utilities for 3D frame-to-graph translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

from structural_configurator import frame_gridlines_from_config


@dataclass
class CanonicalMapping:
    canonical_members: Dict[str, Tuple[str, str]]
    replication_map: Dict[str, List[Tuple[str, str]]]
    joint_map: Dict[Tuple[float, float, float], Dict[str, str]]
    canonical_frame_id: str
    frame_y_map: Dict[str, float] = field(default_factory=dict)


def select_canonical_frame(frame_config_3d: dict) -> str:
    """Select a deterministic representative frame by Y coordinate."""
    frames = _frames_from_config(frame_config_3d)
    if not frames:
        raise ValueError("No frames found in frame_config_3d")
    frames_sorted = sorted(frames, key=lambda f: f[1])
    if len(frames_sorted) >= 3:
        return frames_sorted[len(frames_sorted) // 2][0]
    return frames_sorted[0][0]


def build_canonical_mapping(frame_config_3d: dict, tol: float = 1e-3) -> CanonicalMapping:
    nodes = _node_lookup(frame_config_3d)
    frames = _frames_from_config(frame_config_3d)
    if not frames:
        raise ValueError("No frames found for canonical mapping")
    frame_y_map = {frame_id: y for frame_id, y in frames}
    canonical_frame_id = select_canonical_frame(frame_config_3d)
    canonical_y = frame_y_map[canonical_frame_id]
    x_gridlines, _ = frame_gridlines_from_config(frame_config_3d)

    members = frame_config_3d.get("members", [])
    canonical_members = []
    for member in members:
        group = _member_group(member)
        if group not in {"COL", "GIR", "SEC"}:
            continue
        xi, yi, zi, xj, yj, zj = _member_endpoints(member, nodes)
        if group in {"COL", "GIR"}:
            if abs(yi - canonical_y) <= tol and abs(yj - canonical_y) <= tol:
                canonical_members.append(member)
        elif group == "SEC":
            if not _secondary_aligned_with_grid(xi, xj, x_gridlines, tol):
                continue
            if abs(yi - canonical_y) <= tol or abs(yj - canonical_y) <= tol:
                canonical_members.append(member)

    canonical_map: Dict[str, Tuple[str, str]] = {}
    replication_map: Dict[str, List[Tuple[str, str]]] = {}
    signatures = {}
    for member in members:
        group = _member_group(member)
        if group not in {"COL", "GIR", "SEC"}:
            continue
        if group == "SEC":
            xi, yi, zi, xj, yj, zj = _member_endpoints(member, nodes)
            if not _secondary_aligned_with_grid(xi, xj, x_gridlines, tol):
                continue
        signature = _member_signature(member, nodes, frame_config_3d)
        signatures.setdefault(signature, []).append(member)

    for member in canonical_members:
        node_id = member.get("member_id")
        if not node_id:
            continue
        canonical_map[node_id] = (canonical_frame_id, node_id)
        signature = _member_signature(member, nodes, frame_config_3d)
        matching = []
        for other in signatures.get(signature, []):
            other_id = other.get("member_id")
            if not other_id:
                continue
            frame_id = _member_frame_id(other, nodes, frame_y_map, tol=tol)
            matching.append((frame_id, other_id))
        matching.sort(key=lambda item: (frame_y_map.get(item[0], 0.0), item[1]))
        replication_map[node_id] = matching

    joint_map: Dict[Tuple[float, float, float], Dict[str, str]] = {}
    coord_to_node = _coord_node_lookup(frame_config_3d)
    for member in canonical_members:
        i_node = member.get("i_node_id")
        j_node = member.get("j_node_id")
        if i_node and i_node in nodes:
            coords = nodes[i_node]
            _populate_joint_map(joint_map, coord_to_node, coords, frame_y_map, tol)
        if j_node and j_node in nodes:
            coords = nodes[j_node]
            _populate_joint_map(joint_map, coord_to_node, coords, frame_y_map, tol)

    return CanonicalMapping(
        canonical_members=canonical_map,
        replication_map=replication_map,
        joint_map=joint_map,
        canonical_frame_id=canonical_frame_id,
        frame_y_map=frame_y_map,
    )


def _frames_from_config(frame_config_3d: dict) -> List[Tuple[str, float]]:
    if "frames" in frame_config_3d and frame_config_3d["frames"]:
        frames = []
        for frame in frame_config_3d["frames"]:
            frame_id = str(frame.get("frame_id", frame.get("id", "")))
            y_val = frame.get("y")
            if frame_id and y_val is not None:
                frames.append((frame_id, float(y_val)))
        if frames:
            return frames
    _, y_gridlines = frame_gridlines_from_config(frame_config_3d)
    return [(f"frame_{idx}", y) for idx, y in enumerate(sorted(y_gridlines))]


def _node_lookup(frame_config_3d: dict) -> Dict[str, Tuple[float, float, float]]:
    nodes = {}
    for node in frame_config_3d.get("nodes", []):
        nid = node.get("node_id")
        if not nid:
            continue
        nodes[nid] = (float(node.get("x", 0.0)), float(node.get("y", 0.0)), float(node.get("z", 0.0)))
    return nodes


def _coord_node_lookup(frame_config_3d: dict, precision: int = 6) -> Dict[Tuple[float, float, float], str]:
    lookup = {}
    for node in frame_config_3d.get("nodes", []):
        nid = node.get("node_id")
        if not nid:
            continue
        key = (
            round(float(node.get("x", 0.0)), precision),
            round(float(node.get("y", 0.0)), precision),
            round(float(node.get("z", 0.0)), precision),
        )
        lookup[key] = nid
    return lookup


def _member_group(member: dict) -> str:
    group = str(member.get("group") or member.get("section_id") or "").upper()
    if not group:
        member_id = str(member.get("member_id", "")).upper()
        for prefix in ("COL", "GIR", "SEC"):
            if member_id.startswith(prefix):
                return prefix
    return group


def _member_endpoints(member: dict, nodes: Dict[str, Tuple[float, float, float]]) -> Tuple[float, float, float, float, float, float]:
    i_node = member.get("i_node_id")
    j_node = member.get("j_node_id")
    if i_node in nodes and j_node in nodes:
        xi, yi, zi = nodes[i_node]
        xj, yj, zj = nodes[j_node]
        return xi, yi, zi, xj, yj, zj
    xi = float(member.get("Xi", member.get("Xmid", 0.0)))
    yi = float(member.get("Yi", member.get("Ymid", 0.0)))
    zi = float(member.get("Zi", 0.0))
    xj = float(member.get("Xj", member.get("Xmid", xi)))
    yj = float(member.get("Yj", member.get("Ymid", yi)))
    zj = float(member.get("Zj", member.get("Zmid", zi)))
    return xi, yi, zi, xj, yj, zj


def _member_signature(member: dict, nodes: Dict[str, Tuple[float, float, float]], frame_config_3d: dict) -> Tuple:
    xi, yi, zi, xj, yj, zj = _member_endpoints(member, nodes)
    dx, dy, dz = abs(xj - xi), abs(yj - yi), abs(zj - zi)
    axis = "Z" if dz >= max(dx, dy) else ("X" if dx >= dy else "Y")
    story_h = _story_height(frame_config_3d)
    story_index = member.get("story_index")
    if story_index is None and story_h:
        story_index = int(round(min(zi, zj) / story_h))
    length = round(math.sqrt(dx * dx + dy * dy + dz * dz), 3)
    x_key = (round(min(xi, xj), 3), round(max(xi, xj), 3))
    z_key = (round(min(zi, zj), 3), round(max(zi, zj), 3))
    return (_member_group(member), axis, story_index, length, x_key, z_key)


def _member_frame_id(
    member: dict,
    nodes: Dict[str, Tuple[float, float, float]],
    frame_y_map: Dict[str, float],
    tol: float = 1e-3,
) -> str:
    if "frame_id" in member and member["frame_id"] is not None:
        return str(member["frame_id"])
    frame_index = member.get("frame_index")
    if frame_index is not None:
        ordered = sorted(frame_y_map.items(), key=lambda item: item[1])
        if 0 <= int(frame_index) < len(ordered):
            return ordered[int(frame_index)][0]
    xi, yi, zi, xj, yj, zj = _member_endpoints(member, nodes)
    y_candidates = [yi, yj]
    for frame_id, y_val in frame_y_map.items():
        for y in y_candidates:
            if abs(y - y_val) <= tol:
                return frame_id
    y_ref = min(yi, yj) if _member_group(member) == "SEC" else yi
    closest = min(frame_y_map.items(), key=lambda item: abs(item[1] - y_ref))
    return closest[0]


def _secondary_aligned_with_grid(xi: float, xj: float, x_gridlines: List[float], tol: float) -> bool:
    for x_line in x_gridlines:
        if abs(xi - x_line) <= tol and abs(xj - x_line) <= tol:
            return True
    return False


def _story_height(frame_config_3d: dict) -> Optional[float]:
    meta = frame_config_3d.get("meta", {})
    if "story_h" in meta and meta["story_h"] is not None:
        return float(meta["story_h"])
    building = frame_config_3d.get("building")
    if building and hasattr(building, "story_h"):
        return float(building.story_h)
    return None


def _populate_joint_map(
    joint_map: Dict[Tuple[float, float, float], Dict[str, str]],
    coord_to_node: Dict[Tuple[float, float, float], str],
    coords: Tuple[float, float, float],
    frame_y_map: Dict[str, float],
    tol: float,
):
    x, y, z = coords
    canonical_key = (round(x, 6), round(y, 6), round(z, 6))
    if canonical_key not in joint_map:
        joint_map[canonical_key] = {}
    for frame_id, frame_y in frame_y_map.items():
        coord_key = (round(x, 6), round(frame_y, 6), round(z, 6))
        node_id = coord_to_node.get(coord_key)
        if node_id:
            joint_map[canonical_key][frame_id] = node_id