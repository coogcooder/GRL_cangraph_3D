"""OpenSees model builder for GRL steel optimization.

Builds the full building model from an explicit frame configuration that is
already broadcast from the persistent canonical graph.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple, Iterable

try:
    import openseespy.opensees as ops
    HAS_OPENSEES = True
except ImportError:  # pragma: no cover - handled at runtime when dependency missing
    ops = None
    HAS_OPENSEES = False

# Legacy stiffness defaults used only when converting older cfg inputs.
SECTIONS = {
    "COL": dict(E=2.0e8, A=0.020, Iz=0.12, Iy=0.08, J=0.02, G=2.0e8 / 2.6),
    "GIR": dict(E=2.0e8, A=0.015, Iz=0.06, Iy=0.03, J=0.01, G=2.0e8 / 2.6),
    "FRM": dict(E=2.0e8, A=0.014, Iz=0.04, Iy=0.025, J=0.008, G=2.0e8 / 2.6),
    "SEC": dict(E=2.0e8, A=0.010, Iz=0.02, Iy=0.015, J=0.006, G=2.0e8 / 2.6),
}


def build_model_from_frame_config(
    frame_config: dict,
    case: str,
    out_prefix: str,
) -> dict:
    """Build and analyze the OpenSees model from an explicit frame config."""
    if not HAS_OPENSEES:
        raise RuntimeError("openseespy is required for the real analysis engine")

    nodes: List[dict] = frame_config.get("nodes", [])
    members: List[dict] = frame_config.get("members", [])
    loads: dict = frame_config.get("loads", {})

    node_ids = [n["node_id"] for n in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node_id entries detected in frame_config")
    member_ids = [m["member_id"] for m in members]
    if len(member_ids) != len(set(member_ids)):
        raise ValueError("Duplicate member_id entries detected in frame_config")

    node_id_to_tag = {nid: idx + 1 for idx, nid in enumerate(sorted(node_ids))}
    tag_to_node_id = {tag: nid for nid, tag in node_id_to_tag.items()}

    ops.wipe()
    ops.model("Basic", "-ndm", 3, "-ndf", 6)

    for node in nodes:
        tag = node_id_to_tag[node["node_id"]]
        ops.node(tag, float(node["x"]), float(node["y"]), float(node["z"]))

    for node in nodes:
        if abs(float(node["z"])) < 1e-6:
            tag = node_id_to_tag[node["node_id"]]
            ops.fix(tag, 1, 1, 1, 1, 1, 1)

    ops.geomTransf("Linear", 1, 1, 0, 0)  # columns
    ops.geomTransf("Linear", 2, 0, 0, 1)  # beams

    member_to_eleTags: Dict[str, List[int]] = {}
    eleTag_to_member: Dict[int, str] = {}
    ele_conn: Dict[int, Tuple[int, int]] = {}

    etag = 1
    for member in members:
        mid = member["member_id"]
        i_id = member["i_node_id"]
        j_id = member["j_node_id"]
        if i_id not in node_id_to_tag or j_id not in node_id_to_tag:
            raise ValueError(f"Member {mid} references missing node id(s): {i_id}, {j_id}")
        i_tag = node_id_to_tag[i_id]
        j_tag = node_id_to_tag[j_id]
        props = {
            "E": float(member["E"]),
            "A": float(member["A"]),
            "Iy": float(member["Iy"]),
            "Iz": float(member["Iz"]),
            "J": float(member["J"]),
            "G": float(member.get("G", member["E"] / 2.6)),
        }
        transf = 1 if member.get("group") == "COL" else 2
        ops.element(
            "elasticBeamColumn",
            etag,
            i_tag,
            j_tag,
            props["A"],
            props["E"],
            props["G"],
            props["J"],
            props["Iy"],
            props["Iz"],
            transf,
        )
        member_to_eleTags.setdefault(mid, []).append(etag)
        eleTag_to_member[etag] = mid
        ele_conn[etag] = (i_tag, j_tag)
        etag += 1

    load_kNm = 0.0
    if case == "D":
        load_kNm = float(loads.get("w_dead", 0.0))
    elif case == "L":
        load_kNm = float(loads.get("w_live", 0.0))
    elif case == "DL":
        load_kNm = float(loads.get("w_dead", 0.0)) + float(loads.get("w_live", 0.0))
    else:
        raise ValueError("case must be 'D', 'L', or 'DL'")

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    loaded_members = set()
    for member in members:
        group = member.get("group")
        wants_load = member.get("analysis", group in {"SEC", "SEC_ANALYSIS"})
        if not wants_load:
            continue
        if group not in {"SEC", "SEC_ANALYSIS"}:
            continue
        for e in member_to_eleTags.get(member["member_id"], []):
            ops.eleLoad("-ele", e, "-type", "-beamUniform", 0.0, -load_kNm)
        loaded_members.add(member["member_id"])
    for member in members:
        if member.get("analysis", member.get("group") in {"SEC", "SEC_ANALYSIS"}):
            if member["member_id"] not in loaded_members:
                raise ValueError(f"Secondary member missing load assignment: {member['member_id']}")

    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ok = ops.analyze(1)
    if ok != 0:
        raise RuntimeError(
            f"Analysis failed for case {case} (code={ok}). "
            f"members={len(members)} nodes={len(nodes)}"
        )

    os.makedirs("results", exist_ok=True)
    elem_csv = os.path.join("results", f"{out_prefix}_elements_{case}.csv")
    node_csv = os.path.join("results", f"{out_prefix}_nodes_{case}.csv")
    mid_csv = os.path.join("results", f"{out_prefix}_beam_mid_deflections_{case}.csv")

    with open(elem_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "member_id",
                "group",
                "eleTag",
                "iNode",
                "jNode",
                "length(m)",
                "FxI",
                "FyI",
                "FzI",
                "MxI",
                "MyI",
                "MzI",
                "FxJ",
                "FyJ",
                "FzJ",
                "MxJ",
                "MyJ",
                "MzJ",
                "story_index",
                "frame_index",
                "span_index",
                "bay_index",
            ]
        )
        for e, (i_tag, j_tag) in ele_conn.items():
            mid = eleTag_to_member[e]
            member = next(m for m in members if m["member_id"] == mid)
            xi, yi, zi = ops.nodeCoord(i_tag)
            xj, yj, zj = ops.nodeCoord(j_tag)
            length = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
            forces = ops.eleResponse(e, "force")
            w.writerow(
                [
                    mid,
                    member.get("group", ""),
                    e,
                    i_tag,
                    j_tag,
                    f"{length:.6f}",
                    *[f"{v:.6f}" for v in forces],
                    member.get("story_index", ""),
                    member.get("frame_index", ""),
                    member.get("span_index", ""),
                    member.get("bay_index", ""),
                ]
            )

    with open(node_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "nodeTag", "X", "Y", "Z", "Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])
        for node in nodes:
            tag = node_id_to_tag[node["node_id"]]
            x, y, z = ops.nodeCoord(tag)
            ux, uy, uz, rx, ry, rz = ops.nodeDisp(tag)
            w.writerow(
                [
                    node["node_id"],
                    tag,
                    f"{x:.6f}",
                    f"{y:.6f}",
                    f"{z:.6f}",
                    f"{ux:.9e}",
                    f"{uy:.9e}",
                    f"{uz:.9e}",
                    f"{rx:.9e}",
                    f"{ry:.9e}",
                    f"{rz:.9e}",
                ]
            )

    with open(mid_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["member_id", "mid_node_id", "Ux", "Uy", "Uz"])

    return {
        "member_to_eleTags": member_to_eleTags,
        "eleTag_to_member": eleTag_to_member,
        "node_id_to_tag": node_id_to_tag,
        "tag_to_node_id": tag_to_node_id,
    }


def _legacy_frame_config_from_cfg(cfg: dict) -> dict:
    """Convert legacy cfg (AxisSpans-based) into a frame_config."""
    b = cfg["building"]
    x_lines = cfg["x_axis"].lines
    y_lines = cfg["y_axis"].lines
    nodes: List[dict] = []
    node_lookup: Dict[Tuple[float, float, float], str] = {}
    node_counter = 0

    def _ensure_node(x: float, y: float, z: float) -> str:
        nonlocal node_counter
        key = (x, y, z)
        if key in node_lookup:
            return node_lookup[key]
        node_id = f"N_{node_counter}"
        node_counter += 1
        node_lookup[key] = node_id
        nodes.append({"node_id": node_id, "x": x, "y": y, "z": z})
        return node_id
    for k in range(b.num_stories + 1):
        z = k * b.story_h
        for j, y in enumerate(y_lines):
            for i, x in enumerate(x_lines):
                _ensure_node(x, y, z)

    members: List[dict] = []
    for k in range(b.num_stories):
        z0 = k * b.story_h
        z1 = (k + 1) * b.story_h
        for j, y in enumerate(y_lines):
            for i, x in enumerate(x_lines):
                i_node_id = node_lookup[(x, y, z0)]
                j_node_id = node_lookup[(x, y, z1)]
                props = SECTIONS["COL"]
                members.append(
                    {
                        "member_id": f"COL_{i}_{j}_{k}",
                        "group": "COL",
                        "i_node_id": i_node_id,
                        "j_node_id": j_node_id,
                        "story_index": k,
                        "frame_index": j,
                        "E": props["E"],
                        "G": props["G"],
                        "A": props["A"],
                        "Iy": props["Iy"],
                        "Iz": props["Iz"],
                        "J": props["J"],
                    }
                )

    for k in range(1, b.num_stories + 1):
        z = k * b.story_h
        for j, y in enumerate(y_lines):
            for i in range(len(x_lines) - 1):
                x0, x1 = x_lines[i], x_lines[i + 1]
                i_node_id = node_lookup[(x0, y, z)]
                j_node_id = node_lookup[(x1, y, z)]
                props = SECTIONS["GIR"]
                members.append(
                    {
                        "member_id": f"GIR_{i}_{j}_{k}",
                        "group": "GIR",
                        "i_node_id": i_node_id,
                        "j_node_id": j_node_id,
                        "story_index": k,
                        "frame_index": j,
                        "span_index": f"a{i + 1}",
                        "E": props["E"],
                        "G": props["G"],
                        "A": props["A"],
                        "Iy": props["Iy"],
                        "Iz": props["Iz"],
                        "J": props["J"],
                    }
                )

    for k in range(1, b.num_stories + 1):
        z = k * b.story_h
        for (x1, y1, x2, y2) in cfg["secondary_beams"]:
            i_node_id = _ensure_node(x1, y1, z)
            j_node_id = _ensure_node(x2, y2, z)
            props = SECTIONS["SEC"]
            members.append(
                {
                    "member_id": f"SEC_{x1}_{y1}_{k}",
                    "group": "SEC",
                    "i_node_id": i_node_id,
                    "j_node_id": j_node_id,
                    "story_index": k,
                    "frame_index": None,
                    "bay_index": "",
                    "E": props["E"],
                    "G": props["G"],
                    "A": props["A"],
                    "Iy": props["Iy"],
                    "Iz": props["Iz"],
                    "J": props["J"],
                    "analysis": True,
                }
            )

    return {
        "nodes": nodes,
        "members": members,
        "loads": {"w_dead": cfg["w_dead"], "w_live": cfg["w_live"]},
        "meta": {
            "story_h": b.story_h,
            "num_stories": b.num_stories,
            "bay_spacing": cfg.get("y_span_target", cfg.get("bay_spacing")),
            "sec_spacing": cfg.get("sec_spacing"),
        },
    }


def run_all_cases_from_frame_config(frame_config: dict, out_prefix: str):
    """Run D, L, DL cases from an explicit frame_config."""
    if HAS_OPENSEES:
        for case in ["D", "L", "DL"]:
            build_model_from_frame_config(frame_config, case, out_prefix)
    else:
        _fake_run_all_cases_from_frame_config(frame_config, out_prefix)


def run_all_cases(cfg: Dict, out_prefix: str):
    """Run D, L, DL cases; accepts legacy cfg or a frame_config dict."""
    frame_config = cfg if "nodes" in cfg and "members" in cfg else _legacy_frame_config_from_cfg(cfg)
    run_all_cases_from_frame_config(frame_config, out_prefix)


def _fake_run_all_cases_from_frame_config(frame_config: dict, out_prefix: str):
    print("[analysis] openseespy not found; generating synthetic analysis CSVs.")
    nodes = frame_config["nodes"]
    members = frame_config["members"]
    node_id_to_tag = {n["node_id"]: idx + 1 for idx, n in enumerate(sorted(nodes, key=lambda n: n["node_id"]))}

    os.makedirs("results", exist_ok=True)
    for case in ["D", "L", "DL"]:
        elem_csv = os.path.join("results", f"{out_prefix}_elements_{case}.csv")
        node_csv = os.path.join("results", f"{out_prefix}_nodes_{case}.csv")
        mid_csv = os.path.join("results", f"{out_prefix}_beam_mid_deflections_{case}.csv")

        with open(elem_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "member_id",
                    "group",
                    "eleTag",
                    "iNode",
                    "jNode",
                    "length(m)",
                    "FxI",
                    "FyI",
                    "FzI",
                    "MxI",
                    "MyI",
                    "MzI",
                    "FxJ",
                    "FyJ",
                    "FzJ",
                    "MxJ",
                    "MyJ",
                    "MzJ",
                    "story_index",
                    "frame_index",
                    "span_index",
                    "bay_index",
                ]
            )
            for etag, member in enumerate(members, start=1):
                i_tag = node_id_to_tag[member["i_node_id"]]
                j_tag = node_id_to_tag[member["j_node_id"]]
                i_node = next(n for n in nodes if n["node_id"] == member["i_node_id"])
                j_node = next(n for n in nodes if n["node_id"] == member["j_node_id"])
                length = math.sqrt(
                    (j_node["x"] - i_node["x"]) ** 2
                    + (j_node["y"] - i_node["y"]) ** 2
                    + (j_node["z"] - i_node["z"]) ** 2
                )
                w.writerow(
                    [
                        member["member_id"],
                        member.get("group", ""),
                        etag,
                        i_tag,
                        j_tag,
                        f"{length:.6f}",
                        *["0.000000"] * 12,
                        member.get("story_index", ""),
                        member.get("frame_index", ""),
                        member.get("span_index", ""),
                        member.get("bay_index", ""),
                    ]
                )

        with open(node_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node_id", "nodeTag", "X", "Y", "Z", "Ux", "Uy", "Uz", "Rx", "Ry", "Rz"])
            for node in nodes:
                tag = node_id_to_tag[node["node_id"]]
                w.writerow(
                    [
                        node["node_id"],
                        tag,
                        f"{node['x']:.6f}",
                        f"{node['y']:.6f}",
                        f"{node['z']:.6f}",
                        "0.0",
                        "0.0",
                        "0.0",
                        "0.0",
                        "0.0",
                        "0.0",
                    ]
                )

        with open(mid_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["member_id", "mid_node_id", "Ux", "Uy", "Uz"])