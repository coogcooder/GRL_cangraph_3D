# structural_configurator.py
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import os
# import numpy as np

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
    
