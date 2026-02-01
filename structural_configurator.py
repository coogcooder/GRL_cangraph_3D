# structural_configurator.py
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import os
# import numpy as np


from design_frame_y2 import (
    I_FLANGE_WIDTHS,
    I_FLANGE_THK,
    I_WEB_HEIGHTS,
    I_WEB_THK,
    HSS_WIDTHS,
    HSS_THK,
)

# ===================== Data Models =====================
@dataclass
class Building:
    width_x: float      # m
    length_y: float     # m
    num_stories: int    # floors
    story_h: float      # m

@dataclass
class AxisSpans:
    total: float
    min_span: float
    max_span: float
    count: int
    length: float
    lines: List[float]  # 0..total


def frame_gridlines_from_config(frame_config_3d: Dict) -> Tuple[List[float], List[float]]:
    """Return sorted X/Y gridlines inferred from column nodes."""
    nodes = {n.get("node_id"): n for n in frame_config_3d.get("nodes", []) if n.get("node_id")}
    x_vals = set()
    y_vals = set()

    def _coords_from_node(node_id: str) -> Optional[Tuple[float, float, float]]:
        node = nodes.get(node_id)
        if not node:
            return None
        return float(node.get("x", 0.0)), float(node.get("y", 0.0)), float(node.get("z", 0.0))

    def _member_endpoints(member: Dict) -> Tuple[float, float, float, float, float, float]:
        i_node = member.get("i_node_id")
        j_node = member.get("j_node_id")
        if i_node and j_node:
            i_coords = _coords_from_node(i_node)
            j_coords = _coords_from_node(j_node)
            if i_coords and j_coords:
                xi, yi, zi = i_coords
                xj, yj, zj = j_coords
                return xi, yi, zi, xj, yj, zj
        xi = float(member.get("Xi", member.get("Xmid", 0.0)))
        yi = float(member.get("Yi", member.get("Ymid", 0.0)))
        zi = float(member.get("Zi", 0.0))
        xj = float(member.get("Xj", member.get("Xmid", xi)))
        yj = float(member.get("Yj", member.get("Ymid", yi)))
        zj = float(member.get("Zj", member.get("Zmid", zi)))
        return xi, yi, zi, xj, yj, zj

    for member in frame_config_3d.get("members", []):
        group = str(member.get("group") or member.get("section_id") or "").upper()
        if group != "COL":
            continue
        xi, yi, zi, xj, yj, zj = _member_endpoints(member)
        x_vals.add(round(xi, 6))
        x_vals.add(round(xj, 6))
        y_vals.add(round(yi, 6))
        y_vals.add(round(yj, 6))

    if not x_vals or not y_vals:
        for node in nodes.values():
            if abs(float(node.get("z", 0.0))) > 1e-6:
                continue
            x_vals.add(round(float(node.get("x", 0.0)), 6))
            y_vals.add(round(float(node.get("y", 0.0)), 6))

    return sorted(x_vals), sorted(y_vals)

# ===================== Core Generator =====================
class GridGen:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def rand_building(self) -> Building:
        """Random building envelope per episode."""
        w = round(random.uniform(60, 120), 1)
        L = round(random.uniform(40, 80), 1)
        stories = random.randint(3, 6)
        h = 3.0
        return Building(w, L, stories, h)

    @staticmethod
    def valid_counts(total: float, min_span: float, max_span: float) -> List[int]:
        """All integer span counts c where total/c in [min_span, max_span], densest→sparsest."""
        max_c = int(math.floor(total / min_span))  # densest
        min_c = int(math.ceil(total / max_span))   # sparsest
        out = []
        for c in range(max_c, min_c - 1, -1):
            if c > 0:
                s = total / c
                if min_span <= s <= max_span:
                    out.append(c)
        return out

    def make_axis(self, total: float, min_span: float, max_span: float, want: Optional[int] = None) -> AxisSpans:
        """Pick densest or nearest-sparser valid if 'want' invalid."""
        vals = self.valid_counts(total, min_span, max_span)
        if not vals:
            c = 1
        else:
            if want is None:
                c = vals[0]
            else:
                cands = [v for v in vals if v <= want]
                c = cands[0] if cands else vals[-1]
        s = total / c
        lines = [round(i * s, 3) for i in range(c + 1)]
        return AxisSpans(total=total, min_span=min_span, max_span=max_span, count=c, length=round(s, 3), lines=lines)

    @staticmethod
    def secondary_beams(x_axis: AxisSpans, y_axis: AxisSpans, sec_spacing: float) -> List[Tuple[float, float, float, float]]:
        """
        Secondary beams run in Y between Y-grid lines, located every 'sec_spacing' within each X bay.
        Returns list of segments (x1, y1, x2, y2) at z = floor top.
        """
        beams = []
        for i in range(len(x_axis.lines) - 1):
            x0, x1 = x_axis.lines[i], x_axis.lines[i + 1]
            span = x1 - x0
            if sec_spacing <= 0:
                continue
            n = int(math.floor(span / sec_spacing))
            for j in range(1, n + 1):
                xp = x0 + j * sec_spacing
                if xp < x1:  # strictly inside the bay
                    for k in range(len(y_axis.lines) - 1):
                        y0, y1 = y_axis.lines[k], y_axis.lines[k + 1]
                        beams.append((xp, y0, xp, y1))
        return beams
def sample_episode_params(rng: random.Random) -> Tuple[Building, float, float, float, float]:
    """Sample a randomized building + spacing/load parameters for an episode."""
    gen = GridGen()
    build = gen.rand_building()
    bay_spacing = round(rng.uniform(6.5, 8.5), 2)
    sec_spacing = round(rng.uniform(2.0, 2.5), 2)
    load_choices = [1.0 + 0.5 * i for i in range(9)]  # 1.0 .. 5.0 kPa
    dead_kPa = rng.choice(load_choices)
    live_kPa = rng.choice(load_choices)
    return build, bay_spacing, sec_spacing, dead_kPa, live_kPa
# ===================== Load Utilities =====================
def psf_to_kPa(psf: float) -> float:
    """1 psf = 47.880259 N/m² = 0.047880259 kPa."""
    return psf * 0.04788025898

def area_to_line_load_kN_per_m(q_kPa: float, tributary_m: float) -> float:
    """q in kPa == kN/m²; line load w = q * tributary width (m) => kN/m."""
    return q_kPa * tributary_m

# ===================== Public API for pipeline =====================
def configuration_package(
    build: Building,
    y_span_target: float,
    sec_spacing: float,
    dead_kPa: float,
    live_kPa: float,
) -> Dict:
    """
    Produce a complete configuration dict (no plotting):
      - x_axis (edge columns only), y_axis (frames along length)
      - secondary beams on edge column lines (sparser graph)
      - line loads (kN/m) for dead & live on secondary beams
    """
    # Only two X grid lines (columns at edges)
    x_axis = AxisSpans(
        total=build.width_x,
        min_span=build.width_x,
        max_span=build.width_x,
        count=1,
        length=build.width_x,
        lines=[0.0, round(build.width_x, 3)],
    )

    # Frames along building length using a single target span in [6.5, 8.5] m
    span = y_span_target
    n_spans = max(1, int(math.floor(build.length_y / span)))
    span = build.length_y / n_spans
    y_lines = [round(i * span, 3) for i in range(n_spans + 1)]
    y_axis = AxisSpans(
        total=build.length_y,
        min_span=6.5,
        max_span=8.5,
        count=n_spans,
        length=round(span, 3),
        lines=y_lines,
    )

    # Secondary beams at every sec_spacing across the width (including column lines)
    sec = []
    n_sec = max(1, int(math.floor(build.width_x / sec_spacing)))
    sec_positions = [round(i * sec_spacing, 3) for i in range(n_sec + 1)]
    if sec_positions[-1] < build.width_x:
        sec_positions.append(round(build.width_x, 3))
    sec_positions = sorted(set(sec_positions))
    for xp in sec_positions:
        for i in range(len(y_axis.lines) - 1):
            y0, y1 = y_axis.lines[i], y_axis.lines[i + 1]
            sec.append((xp, y0, xp, y1))

    w_dead = area_to_line_load_kN_per_m(dead_kPa, sec_spacing)
    w_live = area_to_line_load_kN_per_m(live_kPa, sec_spacing)
    dead_psf = dead_kPa / 0.04788025898
    live_psf = live_kPa / 0.04788025898

    return dict(
        building=build,
        x_axis=x_axis,
        y_axis=y_axis,
        secondary_beams=sec,
        sec_spacing=sec_spacing,
        w_dead=w_dead,
        w_live=w_live,
        dead_kPa=dead_kPa,
        live_kPa=live_kPa,
        dead_psf=dead_psf,
        live_psf=live_psf,
    )

def span_ladders_for(build: Building, x_min_max=(10.0, 30.0), y_min_max=(6.0, 8.5)):
    """Legacy helper retained for compatibility (no longer used)."""
    gen = GridGen()
    return (
        gen.valid_counts(build.width_x, *x_min_max),
        gen.valid_counts(build.length_y, *y_min_max)
    )

# ---------------------------------------------------------------------------
# Visualization helpers
def plot_frame_plan(cfg: Dict, prefix: str):
    """Plan-view visualization of the sparse frame (columns, girders, edge secondaries)."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    build = cfg["building"]
    y_lines = cfg["y_axis"].lines
    sec = cfg.get("secondary_beams", [])
    plt.figure(figsize=(6, 4))

    # Columns at edges for each y grid line
    col_x = [0.0, build.width_x]
    col_y = y_lines
    for x in col_x:
        plt.scatter([x] * len(col_y), col_y, c="tab:orange", marker="^", label="Columns" if x == 0.0 else None)

    # Girders along width at each bay
    for i in range(len(y_lines)):
        y0 = y_lines[i]
        plt.plot([0.0, build.width_x], [y0, y0], c="tab:blue", lw=2, label="Girders" if i == 0 else None)

    # Edge secondary beams
    for idx, (x0, y0, x1, y1) in enumerate(sec):
        plt.plot([x0, x1], [y0, y1], c="tab:green", lw=1.0, alpha=0.7, label="Edge secondary" if idx == 0 else None)

    plt.title(
        f"Frame plan: W={build.width_x}m L={build.length_y}m stories={build.num_stories} "
        f"bay={cfg['y_axis'].length}m sec={cfg['sec_spacing']}m"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(loc="best")
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"frame_plan_{prefix}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_frame_3d(cfg: Dict, prefix: str):
    """3D visualization of the sparse frame with all stories and edge secondaries."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return None

    build = cfg["building"]
    y_lines = cfg["y_axis"].lines
    sec = cfg.get("secondary_beams", [])
    h = build.story_h
    stories = build.num_stories

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Columns (x=0 and x=width) for each bay and story
    for y in y_lines:
        for x in (0.0, build.width_x):
            for s in range(stories):
                z0, z1 = s * h, (s + 1) * h
                ax.plot([x, x], [y, y], [z0, z1], c="tab:orange", linewidth=2, label="Columns" if x == 0.0 and s == 0 and y == y_lines[0] else None)

    # Girders along width at each story level and frame line
    for y in y_lines:
        for s in range(1, stories + 1):
            z = s * h
            ax.plot([0.0, build.width_x], [y, y], [z, z], c="tab:blue", linewidth=2, label="Girders" if s == 1 and y == y_lines[0] else None)

    # Secondary beams at each story
    for idx_sec, (x0, y0, x1, y1) in enumerate(sec):
        for s in range(1, stories + 1):
            z = s * h
            ax.plot([x0, x1], [y0, y1], [z, z], c="tab:green", linewidth=1.0, alpha=0.7, label="Secondary" if idx_sec == 0 and s == 1 else None)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"3D Frame: W={build.width_x}m L={build.length_y}m stories={stories} "
        f"bay={cfg['y_axis'].length}m sec={cfg['sec_spacing']}m"
    )
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"frame_3d_{prefix}.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    return path
    
SPAN_LENGTH_KEYS = ("length", "member_span_length", "span_len")
TOTAL_KEYS = ("weight", "weight_kg", "shear_demand", "moment_demand")


def get_column_lines_x(frame_config: Dict) -> List[float]:
    meta = frame_config.get("meta", {})
    if meta.get("column_lines_x"):
        return sorted(float(x) for x in meta["column_lines_x"])
    node_lookup = {n.get("node_id"): n for n in frame_config.get("nodes", []) if n.get("node_id")}
    x_vals = set()
    for member in frame_config.get("members", []):
        if str(member.get("group", "")).upper() != "COL":
            continue
        xi = member.get("Xi")
        xj = member.get("Xj")
        if xi is None or xj is None:
            coords = _member_coords(member, node_lookup)
            if coords:
                xi, _, _, xj, _, _ = coords
        if xi is None or xj is None:
            continue
        x_vals.add(round(float(xi), 6))
        x_vals.add(round(float(xj), 6))
    return sorted(x_vals)


def apply_global_action(frame_config: Dict, action: Dict, span_limits: Tuple[float, float]) -> Tuple[bool, str]:
    action_type = action.get("type")
    if action_type not in {"ADD_COLUMN_LINE", "REMOVE_COLUMN_LINE"}:
        return False, f"unknown action '{action_type}'"
    if action_type == "ADD_COLUMN_LINE":
        x_new = float(action.get("x"))
        return _apply_add_column_line(frame_config, x_new, span_limits)
    x_remove = float(action.get("x"))
    return _apply_remove_column_line(frame_config, x_remove, span_limits)


def _apply_add_column_line(frame_config: Dict, x_new: float, span_limits: Tuple[float, float]) -> Tuple[bool, str]:
    x_lines = get_column_lines_x(frame_config)
    if not x_lines:
        return False, "no column lines available"
    for idx in range(len(x_lines) - 1):
        if x_lines[idx] < x_new < x_lines[idx + 1]:
            left = x_lines[idx]
            right = x_lines[idx + 1]
            span_left = x_new - left
            span_right = right - x_new
            if not _span_within_limits(span_left, span_limits) or not _span_within_limits(span_right, span_limits):
                return False, "span limits violated"
            new_lines = x_lines[: idx + 1] + [x_new] + x_lines[idx + 1 :]
            _rewrite_frame_config(frame_config, x_lines, new_lines, action_type="ADD", x_value=x_new)
            return True, "ok"
    return False, "x_new not inside any span"


def _apply_remove_column_line(frame_config: Dict, x_remove: float, span_limits: Tuple[float, float]) -> Tuple[bool, str]:
    x_lines = get_column_lines_x(frame_config)
    if x_remove not in x_lines:
        return False, "x_remove not in column lines"
    if x_remove == x_lines[0] or x_remove == x_lines[-1]:
        return False, "cannot remove end column line"
    idx = x_lines.index(x_remove)
    merged_span = x_lines[idx + 1] - x_lines[idx - 1]
    if not _span_within_limits(merged_span, span_limits):
        return False, "span limits violated"
    new_lines = [x for x in x_lines if x != x_remove]
    _rewrite_frame_config(frame_config, x_lines, new_lines, action_type="REMOVE", x_value=x_remove)
    return True, "ok"


def _rewrite_frame_config(
    frame_config: Dict,
    old_lines: List[float],
    new_lines: List[float],
    *,
    action_type: str,
    x_value: float,
):
    story_h, num_stories = _frame_story_meta(frame_config)
    y_lines = _frame_y_lines(frame_config)
    girder_map = _girder_feature_map(frame_config)
    meta = frame_config.setdefault("meta", {})
    meta["column_lines_x"] = list(new_lines)

    nodes = _build_nodes(new_lines, y_lines, num_stories, story_h)
    members: List[Dict] = []

    for s in range(num_stories):
        z0 = s * story_h
        z1 = (s + 1) * story_h
        for y_idx, y in enumerate(y_lines):
            for i in range(len(new_lines) - 1):
                x_left = new_lines[i]
                x_right = new_lines[i + 1]
                parent_left, parent_right = _parent_interval(old_lines, new_lines, x_left, x_right, action_type, x_value)
                key_left = (s + 1, y, parent_left[0], parent_left[1])
                key_right = (s + 1, y, parent_right[0], parent_right[1]) if parent_right else None
                parent = girder_map.get(key_left) if key_left in girder_map else {}
                parent_right_feat = girder_map.get(key_right, {}) if key_right else {}
                length = abs(x_right - x_left)
                if action_type == "ADD" and parent:
                    left_span = abs(x_value - parent_left[0])
                    right_span = abs(parent_left[1] - x_value)
                    new_features = _split_features(parent, length, left_span, right_span, x_left < x_value)
                elif action_type == "REMOVE" and parent_right_feat:
                    new_features = _merge_features(parent, parent_right_feat, length)
                else:
                    new_features = _update_span_features(dict(parent), length)
                members.append(
                    _girder_member(
                        x_left,
                        x_right,
                        y,
                        z1,
                        s + 1,
                        y_idx,
                        new_features,
                    )
                )
        for y in y_lines:
            for x in new_lines:
                members.append(_column_member(x, y, z0, z1, s))

    members.extend(_secondary_members(new_lines, y_lines, num_stories, story_h))
    frame_config["nodes"] = nodes
    frame_config["members"] = members


def _frame_story_meta(frame_config: Dict) -> Tuple[float, int]:
    meta = frame_config.get("meta", {})
    story_h = float(meta.get("story_h", 3.0))
    num_stories = int(meta.get("num_stories", 1))
    return story_h, num_stories


def _frame_y_lines(frame_config: Dict) -> List[float]:
    y_vals = set()
    node_lookup = {n.get("node_id"): n for n in frame_config.get("nodes", []) if n.get("node_id")}
    for member in frame_config.get("members", []):
        if str(member.get("group", "")).upper() != "COL":
            continue
        yi = member.get("Yi")
        yj = member.get("Yj")
        if yi is None or yj is None:
            coords = _member_coords(member, node_lookup)
            if coords:
                _, yi, _, _, yj, _ = coords
        if yi is None or yj is None:
            continue
        y_vals.add(round(float(yi), 6))
        y_vals.add(round(float(yj), 6))
    if not y_vals:
        for node in frame_config.get("nodes", []):
            if abs(float(node.get("z", 0.0))) < 1e-6:
                y_vals.add(round(float(node.get("y", 0.0)), 6))
    return sorted(y_vals)


def _build_nodes(x_lines: List[float], y_lines: List[float], num_stories: int, story_h: float) -> List[Dict]:
    nodes = []
    node_id = 0
    for k in range(num_stories + 1):
        z = k * story_h
        for y in y_lines:
            for x in x_lines:
                nodes.append({"node_id": f"N_{node_id}", "x": x, "y": y, "z": z})
                node_id += 1
    return nodes


def _girder_feature_map(frame_config: Dict) -> Dict[Tuple[int, float, float, float], Dict]:
    girder_map: Dict[Tuple[int, float, float, float], Dict] = {}
    node_lookup = {n.get("node_id"): n for n in frame_config.get("nodes", []) if n.get("node_id")}
    for member in frame_config.get("members", []):
        if str(member.get("group", "")).upper() != "GIR":
            continue
        story_index = int(member.get("story_index", 0) or 0)
        coords = _member_coords(member, node_lookup)
        if coords:
            xi, yi, _, xj, yj, _ = coords
            y = yi
        else:
            y = float(member.get("Yi", member.get("Ymid", 0.0)))
            xi = float(member.get("Xi", 0.0))
            xj = float(member.get("Xj", 0.0))
        key = (story_index, y, min(xi, xj), max(xi, xj))
        girder_map[key] = dict(member)
    return girder_map


def _girder_member(
    x_left: float,
    x_right: float,
    y: float,
    z: float,
    story_index: int,
    frame_index: int,
    features: Dict,
) -> Dict:
    member = dict(features)
    length = abs(x_right - x_left)
    member.update(
        {
            "member_id": f"GIR_{frame_index}_{story_index}_{x_left:.3f}_{x_right:.3f}",
            "group": "GIR",
            "Xi": x_left,
            "Yi": y,
            "Zi": z,
            "Xj": x_right,
            "Yj": y,
            "Zj": z,
            "Xmid": (x_left + x_right) / 2.0,
            "Ymid": y,
            "Zmid": z,
            "length": length,
            "story_index": story_index,
            "frame_index": frame_index,
        }
    )
    return member


def _column_member(x: float, y: float, z0: float, z1: float, story_index: int) -> Dict:
    return {
        "member_id": f"COL_{story_index}_{x:.3f}_{y:.3f}",
        "group": "COL",
        "Xi": x,
        "Yi": y,
        "Zi": z0,
        "Xj": x,
        "Yj": y,
        "Zj": z1,
        "story_index": story_index,
        "frame_index": None,
    }


def _secondary_members(x_lines: List[float], y_lines: List[float], num_stories: int, story_h: float) -> List[Dict]:
    members = []
    for s in range(1, num_stories + 1):
        z = s * story_h
        for x in x_lines:
            for y_idx in range(len(y_lines) - 1):
                y0 = y_lines[y_idx]
                y1 = y_lines[y_idx + 1]
                members.append(
                    {
                        "member_id": f"SEC_{x:.3f}_{y0:.3f}_{s}",
                        "group": "SEC",
                        "Xi": x,
                        "Yi": y0,
                        "Zi": z,
                        "Xj": x,
                        "Yj": y1,
                        "Zj": z,
                        "story_index": s,
                        "frame_index": None,
                    }
                )
    return members


def _span_within_limits(span: float, span_limits: Tuple[float, float]) -> bool:
    min_span, max_span = span_limits
    return min_span <= span <= max_span


def _update_span_features(features: Dict, length: float) -> Dict:
    for key in SPAN_LENGTH_KEYS:
        if key in features:
            features[key] = length
    return features


def _split_features(parent: Dict, length: float, left_span: float, right_span: float, is_left: bool) -> Dict:
    parent_length = float(parent.get("length", left_span + right_span) or (left_span + right_span))
    ratio = length / parent_length if parent_length else 1.0
    features = dict(parent)
    for key in TOTAL_KEYS:
        if key in features:
            features[key] = float(features[key] or 0.0) * ratio
    return _update_span_features(features, length)


def _merge_features(left: Dict, right: Dict, length: float) -> Dict:
    features = dict(left)
    for key in TOTAL_KEYS:
        if key in left or key in right:
            features[key] = float(left.get(key, 0.0) or 0.0) + float(right.get(key, 0.0) or 0.0)
    return _update_span_features(features, length)


def _parent_interval(
    old_lines: List[float],
    new_lines: List[float],
    x_left: float,
    x_right: float,
    action_type: str,
    x_value: float,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
    if action_type == "ADD":
        for idx in range(len(old_lines) - 1):
            if old_lines[idx] <= x_left and old_lines[idx + 1] >= x_right:
                return (old_lines[idx], old_lines[idx + 1]), None
    else:
        if x_right <= x_value:
            return (x_left, x_value), None
        if x_left >= x_value:
            return (x_value, x_right), None
        left = max(line for line in old_lines if line < x_value)
        right = min(line for line in old_lines if line > x_value)
        return (left, x_value), (x_value, right)
    return (x_left, x_right), None


def _member_coords(member: Dict, node_lookup: Dict[str, Dict]) -> Optional[Tuple[float, float, float, float, float, float]]:
    i_node = member.get("i_node_id")
    j_node = member.get("j_node_id")
    if i_node in node_lookup and j_node in node_lookup:
        i = node_lookup[i_node]
        j = node_lookup[j_node]
        return (
            float(i.get("x", 0.0)),
            float(i.get("y", 0.0)),
            float(i.get("z", 0.0)),
            float(j.get("x", 0.0)),
            float(j.get("y", 0.0)),
            float(j.get("z", 0.0)),
        )
    return None

def apply_local_resize(member: Dict, delta: int) -> Dict:
    """Apply a sizing delta to the member and update section dimension fields."""
    group = str(member.get("group", "")).upper()
    if group in {"GIR", "SEC"}:
        member["bf_idx"] = _clamp_index(member.get("bf_idx", 0) + delta, len(I_FLANGE_WIDTHS))
        member["tf_idx"] = _clamp_index(member.get("tf_idx", 0) + delta, len(I_FLANGE_THK))
        member["hw_idx"] = _clamp_index(member.get("hw_idx", 0) + delta, len(I_WEB_HEIGHTS))
        member["tw_idx"] = _clamp_index(member.get("tw_idx", 0) + delta, len(I_WEB_THK))
        member["bf"] = I_FLANGE_WIDTHS[member["bf_idx"]]
        member["tf"] = I_FLANGE_THK[member["tf_idx"]]
        member["hw"] = I_WEB_HEIGHTS[member["hw_idx"]]
        member["tw"] = I_WEB_THK[member["tw_idx"]]
    else:
        member["b_idx"] = _clamp_index(member.get("b_idx", 0) + delta, len(HSS_WIDTHS))
        member["t_idx"] = _clamp_index(member.get("t_idx", 0) + delta, len(HSS_THK))
        member["b"] = HSS_WIDTHS[member["b_idx"]]
        member["t"] = HSS_THK[member["t_idx"]]
    return member


def _clamp_index(value: int, length: int) -> int:
    if length <= 0:
        return 0
    return max(0, min(int(value), length - 1))