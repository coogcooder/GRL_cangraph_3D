"""Canonical GNN state utilities and compatibility exports."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Any, Optional, Callable
import csv

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional plotting dependency
    plt = None
    HAS_MATPLOTLIB = False


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class GraphNode:
    node_id: str
    group: str
    features: Dict[str, float]
    pos: Optional[Tuple[float, float, float]] = None


@dataclass
class GraphState:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)

    def to_dict(self):
        return {
            "nodes": {
                nid: dict(group=node.group, features=node.features)
                for nid, node in self.nodes.items()
            },
            "edges": self.edges,
        }

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, problem: dict, cfg_meta: dict, design_state: Dict[str, dict]):
        """Primary factory: assembles nodes/edges from a problem description."""

        # This is the single entry point that constructs the graph representation
        # used by the RL environment and exported by the control pipeline.
        graph = cls()
        graph.update(problem, cfg_meta, design_state)
        return graph

    def update(self, problem: dict, cfg_meta: dict, design_state: Dict[str, dict]):
        """Update nodes and edges in-place for a new structural configuration."""

        # Restrict secondary beams to column lines for graph sparsity
        width_x = cfg_meta.get("width_x")
        beams: List[dict] = []
        for member in problem.get("beams", []):
            if member.get("group") == "SEC" and width_x is not None:
                cx = float(member.get("Xmid", member.get("Xi", 0.0)))
                if not (abs(cx) < 0.25 or abs(cx - width_x) < 0.25):
                    continue
            beams.append(member)

        # Update nodes with fresh features while keeping identifiers stable.
        for member in beams + problem["columns"]:
            node_id = member["member_id"]
            base_features = _node_features_from_member(member, cfg_meta)
            if design_state and node_id in design_state:
                base_features.update(_features_from_design(design_state[node_id]))
            if node_id in self.nodes:
                self.nodes[node_id].features = base_features
                self.nodes[node_id].group = member["group"]
                self.nodes[node_id].pos = _member_position(member)
            else:
                self.nodes[node_id] = GraphNode(
                    node_id=node_id,
                    group=member["group"],
                    features=base_features,
                    pos=_member_position(member),
                )

        # Remove nodes no longer present.
        valid_ids = {m["member_id"] for m in (beams + problem["columns"])}
        for obsolete in list(self.nodes.keys()):
            if obsolete not in valid_ids:
                del self.nodes[obsolete]

        # Recompute edges according to the updated layout.
        filtered_problem = {"beams": beams, "columns": problem.get("columns", [])}
        self.edges = _build_edges(filtered_problem)

    def save(self, prefix: str):
        os.makedirs("results", exist_ok=True)
        json_path = os.path.join("results", f"graph_state_{prefix}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        self.to_csv(prefix)
        self.render_plot(prefix)
        return json_path

    def to_csv(self, prefix: str):
        return _write_graph_csv(prefix, self.nodes, self.edges)

    def render_plot(self, prefix: str):
        """3D visualization of the graph; falls back to 2D if matplotlib 3D unavailable."""
        if not HAS_MATPLOTLIB:
            return None
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", f"graph_state_{prefix}.png")
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        beam_pts, col_pts = [], []
        for node in self.nodes.values():
            pos = node.pos or (
                node.features.get("pos_x", 0.0),
                node.features.get("pos_y", 0.0),
                node.features.get("pos_z", 0.0),
            )
            if node.group in ("GIR", "SEC"):
                beam_pts.append(pos)
            else:
                col_pts.append(pos)
        if beam_pts:
            xs, ys, zs = zip(*beam_pts)
            ax.scatter(xs, ys, zs, c="tab:blue", marker="s", label="Beams")
        if col_pts:
            xs, ys, zs = zip(*col_pts)
            ax.scatter(xs, ys, zs, c="tab:orange", marker="^", label="Columns")
        for u, v in self.edges:
            u_node = self.nodes.get(u)
            v_node = self.nodes.get(v)
            if not u_node or not v_node:
                continue
            ux, uy, uz = u_node.pos or (
                u_node.features.get("pos_x", 0.0),
                u_node.features.get("pos_y", 0.0),
                u_node.features.get("pos_z", 0.0),
            )
            vx, vy, vz = v_node.pos or (
                v_node.features.get("pos_x", 0.0),
                v_node.features.get("pos_y", 0.0),
                v_node.features.get("pos_z", 0.0),
            )
            ax.plot([ux, vx], [uy, vy], [uz, vz], color="0.5", linewidth=2.0, alpha=0.8)
        ax.set_title(f"Graph representation: {prefix}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def node_records(self) -> List[Dict[str, Any]]:
        """Return node feature records without writing to disk."""
        records: List[Dict[str, Any]] = []
        for node in sorted(self.nodes.values(), key=lambda n: n.node_id):
            rec: Dict[str, Any] = {"node_id": node.node_id, "group": node.group}
            rec.update(node.features)
            records.append(rec)
        return records

    def edge_records(self) -> List[Dict[str, Any]]:
        """Return edge connectivity records without writing to disk."""
        return [
            {"source": u, "target": v}
            for u, v in sorted(self.edges)
        ]


@dataclass
class DesignGraph:
    """Persistent canonical graph that stores topology + design variables."""

    graph: GraphState
    design_vars: Dict[str, dict] = field(default_factory=dict)
    replication_map: Dict[str, List[str]] = field(default_factory=dict)
    sec_family_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_problem(cls, problem: dict, cfg_meta: dict, design_state: Dict[str, dict]):
        graph = GraphState.build(problem, cfg_meta, design_state)
        design_vars = {node_id: {} for node_id in graph.nodes.keys()}
        return cls(graph=graph, design_vars=design_vars)

    def node_records(self) -> List[Dict[str, Any]]:
        return self.graph.node_records()

    def edge_records(self) -> List[Dict[str, Any]]:
        return self.graph.edge_records()

    def update_from_analysis(self, analysis_rows: Dict[str, Dict[str, Any]]):
        for node_id, node in self.graph.nodes.items():
            update = analysis_rows.get(node_id)
            if not update:
                continue
            node.features.update(update)

    def update_from_design_state(self, problem: dict, cfg_meta: dict, design_state: Dict[str, dict]):
        for member in problem["beams"] + problem["columns"]:
            node_id = member["member_id"]
            if node_id not in self.graph.nodes:
                continue
            base_features = _node_features_from_member(member, cfg_meta)
            if design_state and node_id in design_state:
                base_features.update(_features_from_design(design_state[node_id]))
            self.graph.nodes[node_id].features.update(base_features)

    def apply_local_action(self, node_id: str, resize_action: Callable[[dict], None]):
        state = self.design_vars.setdefault(node_id, {})
        resize_action(state)

    def apply_global_action(self, problem: dict, cfg_meta: dict, design_state: Dict[str, dict]):
        """Mutate topology after a global edit by syncing to the updated problem."""
        new_graph = GraphState.build(problem, cfg_meta, design_state)
        existing = set(self.graph.nodes.keys())
        new_ids = set(new_graph.nodes.keys())
        for mid in new_ids - existing:
            self.design_vars.setdefault(mid, {})
        for mid in existing - new_ids:
            self.design_vars.pop(mid, None)
        self.graph = new_graph

    def reset_analysis(self):
        for node in self.graph.nodes.values():
            for key in list(node.features.keys()):
                if key.startswith("UR_") or key in {"area", "weight", "weight_kg", "constraints_passed"}:
                    node.features[key] = 0.0


def export_graph_csv(graph: Any, prefix: str):
    """Export node/edge tables even if the caller passes a legacy object."""

    if hasattr(graph, "to_csv"):
        return graph.to_csv(prefix)

    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", [])
    return _write_graph_csv(prefix, nodes, edges)


def graph_records(graph: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return node/edge tables for GraphState or compatible legacy objects."""

    if hasattr(graph, "node_records") and hasattr(graph, "edge_records"):
        return graph.node_records(), graph.edge_records()

    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", [])

    node_rows: List[Dict[str, Any]] = []
    if isinstance(nodes, dict):
        for key in sorted(nodes.keys()):
            node = nodes[key]
            if isinstance(node, GraphNode):
                record = {"node_id": node.node_id, "group": node.group}
                record.update(node.features)
                node_rows.append(record)
            elif isinstance(node, dict):
                record = {"node_id": node.get("node_id", key), "group": node.get("group", "")}
                features = node.get("features", {})
                if isinstance(features, dict):
                    record.update(features)
                node_rows.append(record)

    edge_rows: List[Dict[str, Any]] = []
    for pair in edges:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            edge_rows.append({"source": pair[0], "target": pair[1]})

    return node_rows, edge_rows


def _node_features_from_member(member: dict, cfg_meta: dict):
    pos_x, pos_y, pos_z = _member_position(member)
    feats = dict(
        length=member.get("length", 0.0),
        shear_demand=member.get("shear_demand", 0.0),
        moment_demand=member.get("moment_demand", 0.0),
        deflection=member.get("deflection", 0.0),
        drift_x=member.get("drift_x", 0.0),
        drift_y=member.get("drift_y", 0.0),
        cfg_x_count=cfg_meta.get("x_count", 0),
        cfg_y_count=cfg_meta.get("y_count", 0),
        cfg_span_x=cfg_meta.get("x_span", 0.0),
        cfg_span_y=cfg_meta.get("y_span", 0.0),
        dead_psf=cfg_meta.get("dead_psf", 0.0),
        live_psf=cfg_meta.get("live_psf", 0.0),
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
    )
    for key, value in list(feats.items()):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            feats[key] = 0.0
    return feats


def _features_from_design(design: dict):
    feats = {
        "area": design.get("area", design.get("A", 0.0)),
        "weight": design.get("weight", design.get("weight_kg", 0.0)),
        "weight_kg": design.get("weight_kg", design.get("weight", 0.0)),
        "UR_shear": design.get("UR_shear", 0.0),
        "UR_flex": design.get("UR_flex", 0.0),
        "UR_defl": design.get("UR_defl", 0.0),
        "UR_deflX": design.get("UR_deflX", 0.0),
        "UR_deflY": design.get("UR_deflY", 0.0),
        "constraints_passed": 1.0 if design.get("constraints_passed") else 0.0,
    }
    for param in ("bf", "tf", "hw", "tw", "b", "t"):
        if param in design:
            feats[param] = design[param]
    return feats


def _build_edges(problem: dict) -> List[Tuple[str, str]]:
    beams = problem["beams"]
    cols = problem["columns"]
    edges = set()
    tol = 1e-3

    # Index columns by (x,y) and sort by elevation for vertical stacking
    col_map: Dict[Tuple[float, float], List[dict]] = {}
    for col in cols:
        key = (round(col.get("Xi", 0.0), 3), round(col.get("Yi", 0.0), 3))
        col_map.setdefault(key, []).append(col)
    for key, arr in col_map.items():
        arr.sort(key=lambda c: float(c.get("Zi", 0.0)))
        for i in range(len(arr) - 1):
            edges.add(tuple(sorted((arr[i]["member_id"], arr[i + 1]["member_id"]))))

    def _column_spans_floor(col: dict, z_floor: float) -> bool:
        zi = float(col.get("Zi", 0.0))
        zj = float(col.get("Zj", zi))
        z_min, z_max = (zi, zj) if zi <= zj else (zj, zi)
        return z_min - tol <= z_floor <= z_max + tol

    def _columns_at_floor(cols_list: List[dict], z_floor: float) -> List[dict]:
        return [col for col in cols_list if _column_spans_floor(col, z_floor)]

    def _nearest_column_same_x(cols_list: List[dict], x: float, y: float) -> Optional[str]:
        candidates = []
        for col in cols_list:
            cx = float(col.get("Xi", 0.0))
            if abs(cx - x) > tol:
                continue
            cy = float(col.get("Yi", 0.0))
            candidates.append((col["member_id"], abs(cy - y)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1])
        return candidates[0][0]

    def _girder_columns_same_y(cols_list: List[dict], x: float, y: float) -> List[str]:
        left: List[Tuple[float, str]] = []
        right: List[Tuple[float, str]] = []
        for col in cols_list:
            cy = float(col.get("Yi", 0.0))
            if abs(cy - y) > tol:
                continue
            cx = float(col.get("Xi", 0.0))
            if cx <= x:
                left.append((cx, col["member_id"]))
            else:
                right.append((cx, col["member_id"]))
        left_id = max(left, key=lambda item: item[0])[1] if left else None
        right_id = min(right, key=lambda item: item[0])[1] if right else None
        return [cid for cid in (left_id, right_id) if cid]

    # Connect beams to columns at the same floor elevation.
    for beam in beams:
        beam_x = float(beam.get("Xmid", beam.get("Xi", 0.0)))
        beam_y = float(beam.get("Ymid", beam.get("Yi", 0.0)))
        beam_z = float(beam.get("Zmid", beam.get("Zi", beam.get("Zj", 0.0))))
        floor_cols = _columns_at_floor(cols, beam_z)
        if beam.get("group") == "SEC":
            col_id = _nearest_column_same_x(floor_cols, beam_x, beam_y)
            if col_id is None:
                col_id = _nearest_column_same_x(cols, beam_x, beam_y)
            if col_id is not None:
                edges.add(tuple(sorted((beam["member_id"], col_id))))
        else:
            girder_cols = _girder_columns_same_y(floor_cols, beam_x, beam_y)
            if not girder_cols:
                girder_cols = _girder_columns_same_y(cols, beam_x, beam_y)
            for col_id in girder_cols:
                edges.add(tuple(sorted((beam["member_id"], col_id))))

    return sorted(list(edges))


def _member_position(member: dict) -> Tuple[float, float, float]:
    if member.get("group") in ("GIR", "SEC"):
        x = float(member.get("Xmid", member.get("Xi", 0.0)))
        y = float(member.get("Ymid", member.get("Yi", 0.0)))
        z = float(member.get("Zmid", (member.get("Zi", 0.0) + member.get("Zj", 0.0)) / 2.0))
    else:
        x = float(member.get("Xi", member.get("Xmid", 0.0)))
        y = float(member.get("Yi", member.get("Ymid", 0.0)))
        z = float((member.get("Zi", 0.0) + member.get("Zj", 0.0)) / 2.0)
    return x, y, z



def _write_graph_csv(prefix: str, nodes: Dict[str, Any], edges: Iterable[Tuple[str, str]]):
    os.makedirs("results", exist_ok=True)
    node_path = os.path.join("results", f"graph_state_{prefix}_nodes.csv")
    edge_path = os.path.join("results", f"graph_state_{prefix}_edges.csv")

    feature_keys = set()
    rows = []
    for key, node in (nodes.items() if isinstance(nodes, dict) else []):
        if isinstance(node, GraphNode):
            node_id = node.node_id
            group = node.group
            features = dict(node.features)
            if node.pos:
                features.setdefault("pos_x", node.pos[0])
                features.setdefault("pos_y", node.pos[1])
                features.setdefault("pos_z", node.pos[2])
        elif isinstance(node, dict):
            node_id = node.get("node_id", key)
            group = node.get("group", "")
            features = node.get("features", {})
        else:
            continue
        feature_keys.update(features.keys())
        rows.append((node_id, group, features))

    fieldnames = ["node_id", "group"] + sorted(feature_keys)

    with open(node_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for node_id, group, features in sorted(rows, key=lambda r: r[0]):
            row = {"node_id": node_id, "group": group}
            row.update({k: features.get(k, 0.0) for k in feature_keys})
            writer.writerow(row)

    with open(edge_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target"])
        for u, v in sorted(edges):
            writer.writerow([u, v])

    return node_path, edge_path