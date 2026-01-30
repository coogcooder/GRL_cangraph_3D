# design_frame_y2.py — robust Y2 frame extraction using COLUMN CSV for Y-gridlines
import os
import math
import csv
from typing import Dict, List, Callable, Tuple


_ELEMENT_SCHEMA = {
    "member_id": str,
    "group": str,
    "eleTag": int,
    "iNode": int,
    "jNode": int,
    "length(m)": float,
    "FxI": float,
    "FyI": float,
    "FzI": float,
    "MxI": float,
    "MyI": float,
    "MzI": float,
    "FxJ": float,
    "FyJ": float,
    "FzJ": float,
    "MxJ": float,
    "MyJ": float,
    "MzJ": float,
}

_NODE_SCHEMA = {
    "node_id": str,
    "nodeTag": int,
    "X": float,
    "Y": float,
    "Z": float,
    "Ux": float,
    "Uy": float,
    "Uz": float,
    "Rx": float,
    "Ry": float,
    "Rz": float,
}

_MID_SCHEMA = {
    "rep_eleTag": int,
    "mid_node": int,
    "Xmid": float,
    "Ymid": float,
    "Zmid": float,
    "Ux": float,
    "Uy": float,
    "Uz": float,
}

_COLUMN_SCHEMA = {
    "eleTag": int,
    "bottomNode": int,
    "topNode": int,
    "Xb": float,
    "Yb": float,
    "Zb": float,
    "Ux_b": float,
    "Uy_b": float,
    "Xt": float,
    "Yt": float,
    "Zt": float,
    "Ux_t": float,
    "Uy_t": float,
}

# ----------------------------- Settings & Material -----------------------------
FY = 345_000.0         # kN/m^2  (≈ 345 MPa)
E  = 200_000_000.0     # kN/m^2  (≈ 200 GPa)
DEFLECTION_LIMIT_DENOM = 240.0   # L/240 for DL case

# I-shape (beams): discrete catalog (mm)
I_FLANGE_WIDTHS = [200, 250, 300, 350, 400, 450, 500]
I_FLANGE_THK    = [5, 6, 8, 10, 12, 16, 20, 22]
I_WEB_HEIGHTS   = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
I_WEB_THK       = [5, 6, 8, 10, 12, 16, 20, 22]

# HSS square (columns): outer width & thickness (mm)
HSS_WIDTHS = [150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
HSS_THK    = [5, 6, 8, 10, 12, 16, 20, 22]

# ----------------------------- Utilities -----------------------------
def mm_to_m(x): return x / 1000.0
def unique_sorted(vals): return sorted(set(round(float(v), 6) for v in vals))

def read_case(prefix_base: str, case: str = "DL"):
    base = os.path.join("results", prefix_base)
    req = [
        f"{base}_elements_{case}.csv",
        f"{base}_nodes_{case}.csv",
        f"{base}_beam_mid_deflections_{case}.csv",
    ]
    for p in req:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required analysis output not found: {p}")
    col_path = f"{base}_column_end_disp_{case}.csv"

    elems = _read_csv_with_schema(req[0], _ELEMENT_SCHEMA)
    nodes_list = _read_csv_with_schema(req[1], _NODE_SCHEMA)
    mids = _read_csv_with_schema(req[2], _MID_SCHEMA)
    nodes = {int(row["nodeTag"]): row for row in nodes_list}
    if os.path.isfile(col_path):
        cols = _read_csv_with_schema(col_path, _COLUMN_SCHEMA)
    else:
        cols = []
        for row in elems:
            if str(row.get("group", "")).upper() != "COL":
                continue
            iN = int(row.get("iNode", 0) or 0)
            jN = int(row.get("jNode", 0) or 0)
            if iN == 0 or jN == 0:
                continue
            xb, yb, zb = node_coords(nodes, iN)
            xt, yt, zt = node_coords(nodes, jN)
            uxb, uyb, *_ = node_disp(nodes, iN)
            uxt, uyt, *_ = node_disp(nodes, jN)
            cols.append(
                dict(
                    eleTag=int(row.get("eleTag", 0) or 0),
                    bottomNode=iN,
                    topNode=jN,
                    Xb=xb,
                    Yb=yb,
                    Zb=zb,
                    Ux_b=uxb,
                    Uy_b=uyb,
                    Xt=xt,
                    Yt=yt,
                    Zt=zt,
                    Ux_t=uxt,
                    Uy_t=uyt,
                )
            )
    return elems, nodes, mids, cols

def _members_from_elements(elems: List[Dict], nodes: Dict[int, Dict[str, float]]) -> Tuple[List[Dict], List[Dict]]:
    beams: List[Dict] = []
    cols: List[Dict] = []
    for row in elems:
        group = str(row.get("group") or row.get("section_id") or "").upper()
        if not group:
            member_id = str(row.get("member_id", "")).upper()
            for prefix in ("COL", "GIR", "SEC"):
                if member_id.startswith(prefix):
                    group = prefix
                    break
        iN = int(row.get("iNode", 0) or 0)
        jN = int(row.get("jNode", 0) or 0)
        if iN == 0 or jN == 0:
            continue
        Xi, Yi, Zi = node_coords(nodes, iN)
        Xj, Yj, Zj = node_coords(nodes, jN)
        length = math.sqrt((Xj - Xi) ** 2 + (Yj - Yi) ** 2 + (Zj - Zi) ** 2)
        base = dict(
            member_id=str(row.get("member_id", f"{group}_{row.get('eleTag', '')}")),
            group=group,
            iNode=iN,
            jNode=jN,
            length=length,
            shear_demand=0.0,
            moment_demand=0.0,
            deflection=0.0,
            Xi=Xi,
            Yi=Yi,
            Zi=Zi,
            Xj=Xj,
            Yj=Yj,
            Zj=Zj,
        )
        if group == "COL":
            base.update(drift_x=0.0, drift_y=0.0)
            cols.append(base)
        elif group in ("GIR", "SEC"):
            base.update(
                Xmid=0.5 * (Xi + Xj),
                Ymid=0.5 * (Yi + Yj),
                Zmid=0.5 * (Zi + Zj),
            )
            beams.append(base)
    return beams, cols

def _read_csv_with_schema(path: str, schema: Dict[str, Callable[[str], object]]):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                caster = schema.get(key)
                if caster:
                    try:
                        parsed[key] = caster(value)
                    except Exception:
                        parsed[key] = caster(float(value)) if caster is int else float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows

def node_coords(nodes: Dict[int, Dict[str, float]], tag: int):
    row = nodes.get(tag)
    if not row:
        return None
    return float(row.get("X", 0.0) or 0.0), float(row.get("Y", 0.0) or 0.0), float(row.get("Z", 0.0) or 0.0)


def node_disp(nodes: Dict[int, Dict[str, float]], tag: int):
    row = nodes.get(tag)
    if not row:
        return None
    return (
        float(row.get("Ux", 0.0) or 0.0),
        float(row.get("Uy", 0.0) or 0.0),
        float(row.get("Uz", 0.0) or 0.0),
        float(row.get("Rx", 0.0) or 0.0),
        float(row.get("Ry", 0.0) or 0.0),
        float(row.get("Rz", 0.0) or 0.0),
    )


def element_rows_with_node(elems: List[Dict], node_tag: int):
    return [r for r in elems if int(r.get("iNode", -1)) == node_tag or int(r.get("jNode", -1)) == node_tag]

# ----------------------------- Section properties -----------------------------
def i_section_props(bf, tf, hw, tw):
    A = 2*bf*tf + tw*hw
    d = hw + 2*tf
    Ifl = (bf*tf**3)/12.0 + bf*tf*((d/2 - tf/2)**2)
    Ixx = 2*Ifl + (tw*hw**3)/12.0
    S = Ixx / (d/2.0)
    Aw = tw * hw
    return A, Ixx, S, Aw, d

def hss_square_props(b, t):
    bi = b - 2*t
    if bi <= 0: return 0,0,0,0,b
    A = b*b - bi*bi
    I = (b**4 - bi**4)/12.0
    S = I / (b/2.0)
    Av = 4.0 * t * (b - t)
    return A, I, S, Av, b

# ----------------------------- Design checks -----------------------------
DEF_DEN = DEFLECTION_LIMIT_DENOM
def ur_shear(V_req, V_cap):     return 0.0 if V_cap <= 0 else abs(V_req) / V_cap
def ur_flexure(M_req, M_cap):   return 0.0 if M_cap <= 0 else abs(M_req) / M_cap
def ur_deflection(delta, L):    return 0.0 if L <= 0 else abs(delta) / (L / DEF_DEN)

# ----------------------------- Demand extraction -----------------------------
def beam_demands_from_halves(elems: List[Dict], nodes: Dict[int, Dict[str, float]], mid_row: Dict):
    nM = int(mid_row.get("mid_node", 0))
    member_type = str(mid_row.get("member_type", "")).strip()
    relevant = [
        e for e in elems
        if str(e.get("section_id", "")).strip() == member_type
    ]
    halves = element_rows_with_node(relevant, nM)
    if len(halves) < 2 and mid_row.get("rep_eleTag") is not None:
        rep = int(mid_row.get("rep_eleTag"))
        halves = [e for e in elems if int(e.get("eleTag", -1)) in (rep, rep + 1)]
    Vc, Mc, Ltot = [], [], 0.0
    for e in halves:
        Vc += [abs(float(e.get("FzI", 0.0) or 0.0)), abs(float(e.get("FzJ", 0.0) or 0.0))]
        Mc += [abs(float(e.get("MyI", 0.0) or 0.0)), abs(float(e.get("MyJ", 0.0) or 0.0))]
        Ltot += float(e.get("length(m)", 0.0) or 0.0)
    Uz = 0.0
    nd = nodes.get(nM)
    if nd and nd.get("Uz") is not None:
        Uz = float(nd.get("Uz", 0.0) or 0.0)
    return (
        max(Vc) if Vc else 0.0,
        max(Mc) if Mc else 0.0,
        Ltot,
        abs(Uz),
    )

def column_story_demand(e_row: Dict, nodes: Dict[int, Dict[str, float]]):
    iN, jN = int(e_row.get("iNode", 0)), int(e_row.get("jNode", 0))
    FyI = float(e_row.get("FyI", 0.0) or 0.0)
    FzI = float(e_row.get("FzI", 0.0) or 0.0)
    MyI = float(e_row.get("MyI", 0.0) or 0.0)
    MzI = float(e_row.get("MzI", 0.0) or 0.0)
    FyJ = float(e_row.get("FyJ", 0.0) or 0.0)
    FzJ = float(e_row.get("FzJ", 0.0) or 0.0)
    MyJ = float(e_row.get("MyJ", 0.0) or 0.0)
    MzJ = float(e_row.get("MzJ", 0.0) or 0.0)
    V_req = max(math.hypot(FyI, FzI), math.hypot(FyJ, FzJ))
    M_req = max(math.hypot(MyI, MzI), math.hypot(MyJ, MzJ))
    Uxb, Uyb, *_ = node_disp(nodes, iN)
    Uxt, Uyt, *_ = node_disp(nodes, jN)
    drift_x = abs(Uxt - Uxb)
    drift_y = abs(Uyt - Uyb)
    L = float(e_row["length(m)"])
    return V_req, M_req, L, drift_x, drift_y

# ----------------------------- Catalog builders -----------------------------
def build_i_catalog():
    cat = []
    for bf_mm in I_FLANGE_WIDTHS:
        for tf_mm in I_FLANGE_THK:
            for hw_mm in I_WEB_HEIGHTS:
                for tw_mm in I_WEB_THK:
                    bf = mm_to_m(bf_mm); tf = mm_to_m(tf_mm)
                    hw = mm_to_m(hw_mm); tw = mm_to_m(tw_mm)
                    A, Imaj, Smaj, Aw, d = i_section_props(bf, tf, hw, tw)
                    Vcap = 0.6 * FY * Aw
                    Mcap = FY * Smaj
                    cat.append(dict(type="I", bf=bf_mm, tf=tf_mm, hw=hw_mm, tw=tw_mm,
                                    A=A, I=Imaj, S=Smaj, Aw=Aw, d=d, Vcap=Vcap, Mcap=Mcap))
    cat.sort(key=lambda r: (r["A"], r["d"]))
    return cat

def build_hss_catalog():
    cat = []
    for b_mm in HSS_WIDTHS:
        for t_mm in HSS_THK:
            b = mm_to_m(b_mm); t = mm_to_m(t_mm)
            if 2*t >= b:   # impossible
                continue
            A, I, S, Av, depth = hss_square_props(b, t)
            Vcap = 0.6 * FY * Av
            Mcap = FY * S
            cat.append(dict(type="HSS", b=b_mm, t=t_mm,
                            A=A, I=I, S=S, Av=Av, d=depth, Vcap=Vcap, Mcap=Mcap))
    cat.sort(key=lambda r: (r["A"], r["d"]))
    return cat

def pick_beam_size(V_req, M_req, L, delta_mid, catalog):
    for sec in catalog:
        urV = ur_shear(V_req, sec["Vcap"])
        urM = ur_flexure(M_req, sec["Mcap"])
        urD = ur_deflection(delta_mid, L)
        if max(urV, urM, urD) <= 1.0:
            out = sec.copy()
            out.update(UR_shear=urV, UR_flex=urM, UR_defl=urD)
            return out
    return None

def pick_column_size(V_req, M_req, L, drift_x, drift_y, catalog):
    for sec in catalog:
        urV = ur_shear(V_req, sec["Vcap"])
        urM = ur_flexure(M_req, sec["Mcap"])
        urDx = ur_deflection(drift_x, L)
        urDy = ur_deflection(drift_y, L)
        if max(urV, urM, urDx, urDy) <= 1.0:
            out = sec.copy()
            out.update(UR_shear=urV, UR_flex=urM, UR_deflX=urDx, UR_deflY=urDy)
            return out
    return None

# ----------------------------- Build Y-gridlines from COLUMN CSV -----------------------------
def y_gridlines_from_columns(cols_rows: List[Dict]):
    """
    Use only column end coordinates to build true Y-gridlines.
    These nodes are exactly at grid intersections (base/top), so no mid-node noise.
    """
    ys = []
    for row in cols_rows:
        if row.get("Yb") is not None:
            ys.append(float(row.get("Yb") or 0.0))
        if row.get("Yt") is not None:
            ys.append(float(row.get("Yt") or 0.0))
    ys_unique = unique_sorted(ys)
    return ys_unique

def nearest_y_index(y: float, ys: list) -> int:
    if not ys:
        return 0
    return min(range(len(ys)), key=lambda idx: abs(y - ys[idx]))

# ----------------------------- Robust frame extraction -----------------------------
def extract_frame_and_secondaries(elems: List[Dict], nodes: Dict[int, Dict[str, float]],
                                  mids: List[Dict], cols: List[Dict]):
    def _row_group(row: Dict) -> str:
        raw = str(row.get("group") or row.get("section_id") or "").strip().upper()
        if raw:
            return raw
        member_id = str(row.get("member_id", "")).upper()
        for prefix in ("COL", "GIR", "SEC"):
            if member_id.startswith(prefix):
                return prefix
        return ""

    def _mid_rows_from_elements() -> List[Dict]:
        out: List[Dict] = []
        for row in elems:
            group = _row_group(row)
            if group not in {"GIR", "SEC"}:
                continue
            iN, jN = int(row.get("iNode", 0) or 0), int(row.get("jNode", 0) or 0)
            ci = node_coords(nodes, iN)
            cj = node_coords(nodes, jN)
            if not ci or not cj:
                continue
            xm = 0.5 * (ci[0] + cj[0])
            ym = 0.5 * (ci[1] + cj[1])
            zm = 0.5 * (ci[2] + cj[2])
            out.append(
                dict(
                    member_type=group,
                    rep_eleTag=int(row.get("eleTag", 0) or 0),
                    mid_node=int(row.get("eleTag", 0) or 0),
                    Xmid=xm,
                    Ymid=ym,
                    Zmid=zm,
                )
            )
        return out

    ys = y_gridlines_from_columns(cols)
    if len(ys) < 3:
        raise RuntimeError(f"Need at least 3 Y gridlines from column file (found {len(ys)}).")
    y1, y2, y3 = ys[0], ys[1], ys[2]

    # ---- Columns at Y2: classify by nearest Y index of BOTH ends using nodes coords ----
    column_rows = []
    for row in elems:
        if str(row.get("section_id", "")).strip() != "COL":
            continue
        iN, jN = int(row.get("iNode", 0)), int(row.get("jNode", 0))
        ci = node_coords(nodes, iN)
        cj = node_coords(nodes, jN)
        if not ci or not cj:
            continue
        yi, yj = ci[1], cj[1]
        if nearest_y_index(yi, ys) == 1 and nearest_y_index(yj, ys) == 1:
            column_rows.append(row.copy())

    # ---- Girders at Y2: mid Y snaps to Y2 ----
    girders_mids = []
    for row in mids:
        if str(row.get("member_type", "")).strip() != "GIR":
            continue
        ymid = float(row.get("Ymid", 0.0) or 0.0)
        if nearest_y_index(ymid, ys) == 1:
            girders_mids.append(row.copy())

    # ---- Secondaries adjacent to Y2: Ymid strictly between (y1,y2) or (y2,y3) ----
    seconds_mids = []
    for row in mids:
        if str(row.get("member_type", "")).strip() != "SEC":
            continue
        ymid = float(row.get("Ymid", 0.0) or 0.0)
        for k in range(len(ys) - 1):
            if ys[k] < ymid < ys[k + 1]:
                if k in (0, 1):
                    seconds_mids.append(row.copy())
                break

    print(f"[design] Y gridlines (from columns): y1={y1:.3f}, y2={y2:.3f}, y3={y3:.3f}")
    print(f"[design] Classified → columns@Y2: {len(column_rows)}, girders@Y2(mids): {len(girders_mids)}, adj.SEC(mids): {len(seconds_mids)}")

    # Normalize the collections so downstream code can safely call pandas-like helpers
    # (``iterrows``/``to_dict``) even when pandas is not installed.  Converting here keeps
    # a single representation regardless of who consumes the frame dictionary and avoids
    # the AttributeError surfaced when plain lists reach RL loaders.
    column_rows = _coerce_records(column_rows)
    girders_mids = _coerce_records(girders_mids)
    seconds_mids = _coerce_records(seconds_mids)

    return dict(
        ys=ys, y1=y1, y2=y2, y3=y3,
        columns_rows=column_rows,
        girders_mids=girders_mids,
        seconds_mids=seconds_mids
    )


# ----------------------------- Controller entry -----------------------------
def run_design_for_prefix(prefix: str) -> str:
    elems, nodes, mids, cols = read_case(prefix, case="DL")
    frame = extract_frame_and_secondaries(elems, nodes, mids, cols)
    i_catalog   = build_i_catalog()
    hss_catalog = build_hss_catalog()

    # --- Girders on Y2 ---
    beam_rows = []
    for mr in frame["girders_mids"]:
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        pick = pick_beam_size(V_req, M_req, L, Uz_mid, i_catalog)
        rec = dict(member_group="GIR", mid_node=int(mr["mid_node"]),
                   Xmid=float(mr["Xmid"]), Ymid=float(mr["Ymid"]), Zmid=float(mr["Zmid"]),
                   L=L, V_req=V_req, M_req=M_req, Uz_mid=Uz_mid)
        if pick:
            rec.update(section=f"I {pick['bf']}x{pick['hw']}x{pick['tw']}/{pick['tf']} (mm)",
                       A=pick["A"], S=pick["S"], Vcap=pick["Vcap"], Mcap=pick["Mcap"],
                       UR_shear=pick["UR_shear"], UR_flex=pick["UR_flex"], UR_defl=pick["UR_defl"])
        else:
           rec.update(section="NO PASS", A=float('nan'), S=float('nan'), Vcap=float('nan'), Mcap=float('nan'),
                       UR_shear=float('inf'), UR_flex=float('inf'), UR_defl=float('inf'))
        beam_rows.append(rec)

    # --- Adjacent secondary beams ---
    for mr in frame["seconds_mids"]:
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        pick = pick_beam_size(V_req, M_req, L, Uz_mid, i_catalog)
        rec = dict(member_group="SEC", mid_node=int(mr["mid_node"]),
                   Xmid=float(mr["Xmid"]), Ymid=float(mr["Ymid"]), Zmid=float(mr["Zmid"]),
                   L=L, V_req=V_req, M_req=M_req, Uz_mid=Uz_mid)
        if pick:
            rec.update(section=f"I {pick['bf']}x{pick['hw']}x{pick['tw']}/{pick['tf']} (mm)",
                       A=pick["A"], S=pick["S"], Vcap=pick["Vcap"], Mcap=pick["Mcap"],
                       UR_shear=pick["UR_shear"], UR_flex=pick["UR_flex"], UR_defl=pick["UR_defl"])
        else:
            rec.update(section="NO PASS", A=float('nan'), S=float('nan'), Vcap=float('nan'), Mcap=float('nan'),
                       UR_shear=float('inf'), UR_flex=float('inf'), UR_defl=float('inf'))
        beam_rows.append(rec)

    # --- Columns at Y2 (story-by-story) ---
    col_rows = []
    for er in frame["columns_rows"]:
        V_req, M_req, L, drift_x, drift_y = column_story_demand(er, nodes)
        pick = pick_column_size(V_req, M_req, L, drift_x, drift_y, hss_catalog)
        iN, jN = int(er["iNode"]), int(er["jNode"])
        Xi, Yi, Zi = node_coords(nodes, iN)
        Xj, Yj, Zj = node_coords(nodes, jN)
        rec = dict(member_group="COL", eleTag=int(er["eleTag"]),
                   iNode=iN, jNode=jN, Xi=Xi, Yi=Yi, Zi=Zi, Xj=Xj, Yj=Yj, Zj=Zj,
                   L=L, V_req=V_req, M_req=M_req, drift_x=drift_x, drift_y=drift_y)
        if pick:
            rec.update(section=f"HSS {pick['b']}x{pick['t']} (mm)",
                       A=pick["A"], S=pick["S"], Vcap=pick["Vcap"], Mcap=pick["Mcap"],
                       UR_shear=pick["UR_shear"], UR_flex=pick["UR_flex"],
                       UR_deflX=pick["UR_deflX"], UR_deflY=pick["UR_deflY"])
        else:
            rec.update(section="NO PASS", A=float('nan'), S=float('nan'), Vcap=float('nan'), Mcap=float('nan'),
                       UR_shear=float('inf'), UR_flex=float('inf'), UR_deflX=float('inf'), UR_deflY=float('inf'))
        col_rows.append(rec)

    # ---- Write output (always with headers) ----
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"design_{prefix}_frame_y2.csv")
    col_schema = [
        "member_group","mid_node","eleTag","iNode","jNode",
        "Xmid","Ymid","Zmid","Xi","Yi","Zi","Xj","Yj","Zj",
        "L","V_req","M_req","Uz_mid","drift_x","drift_y",
        "section","A","S","Vcap","Mcap","UR_shear","UR_flex","UR_defl","UR_deflX","UR_deflY"
    ]
    rows = beam_rows + col_rows
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_schema)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in col_schema})

    total_rows = len(rows)
    print(f"[design] Wrote {total_rows} row(s) to {out_path}")
    if total_rows == 0:
        print("[design][WARN] No members classified; check files/prefix.")
    return out_path

# ----------------------------- RL helper exports -----------------------------
class RecordList(list):
    """List subclass that mimics minimal pandas-style helpers used downstream."""

    __slots__ = ()

    def iterrows(self):
        for idx, rec in enumerate(self):
            yield idx, rec

    def to_dict(self, orient="records"):
        if orient == "records":
            return [dict(rec) for rec in self]
        raise ValueError(f"Unsupported orient: {orient}")


def _coerce_records(rows):
    """Return a RecordList of plain dict records from diverse iterable inputs."""
    if rows is None:
        return RecordList()
    if isinstance(rows, list):
        out = RecordList()
        for item in rows:
            if isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "to_dict"):
                out.append(item.to_dict())
        return out
    if hasattr(rows, "to_dict"):
        try:
            recs = rows.to_dict(orient="records")
            if isinstance(recs, list):
                return RecordList([dict(rec) for rec in recs])
        except TypeError:
            pass
    if hasattr(rows, "iterrows"):
        out = RecordList()
        for _, rec in rows.iterrows():
            if isinstance(rec, dict):
                out.append(rec)
            elif hasattr(rec, "to_dict"):
                out.append(rec.to_dict())
        return out
    out = RecordList()
    for item in rows:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            out.append(item[1])
        elif hasattr(item, "to_dict"):
            out.append(item.to_dict())
    return out


def _beam_problem_rows(frame, elems, nodes):
    beams = []
    girders = _coerce_records(frame.get("girders_mids"))
    seconds = _coerce_records(frame.get("seconds_mids"))
    frame["girders_mids"] = girders
    frame["seconds_mids"] = seconds
    for mr in girders:
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        beams.append(dict(
            member_id=f"GIR_{int(mr['mid_node'])}",
            group="GIR",
            mid_node=int(mr["mid_node"]),
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            deflection=Uz_mid,
            Xmid=float(mr["Xmid"]),
            Ymid=float(mr["Ymid"]),
            Zmid=float(mr["Zmid"])
        ))
    for mr in seconds:
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        beams.append(dict(
            member_id=f"SEC_{int(mr['mid_node'])}",
            group="SEC",
            mid_node=int(mr["mid_node"]),
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            deflection=Uz_mid,
            Xmid=float(mr["Xmid"]),
            Ymid=float(mr["Ymid"]),
            Zmid=float(mr["Zmid"])
        ))
    return beams

def _column_problem_rows(frame, nodes):
    cols = []
    col_rows = _coerce_records(frame.get("columns_rows"))
    frame["columns_rows"] = col_rows
    for er in col_rows:
        V_req, M_req, L, drift_x, drift_y = column_story_demand(er, nodes)
        iN, jN = int(er["iNode"]), int(er["jNode"])
        Xi, Yi, Zi = node_coords(nodes, iN)
        Xj, Yj, Zj = node_coords(nodes, jN)
        cols.append(dict(
            member_id=f"COL_{int(er['eleTag'])}",
            group="COL",
            eleTag=int(er["eleTag"]),
            iNode=iN,
            jNode=jN,
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            drift_x=drift_x,
            drift_y=drift_y,
            Xi=Xi,
            Yi=Yi,
            Zi=Zi,
            Xj=Xj,
            Yj=Yj,
            Zj=Zj
        ))
    return cols

def load_design_problem(prefix: str):
    """Return demand data for RL-based design exploration."""
    elems, nodes, mids, cols = read_case(prefix, case="DL")
    frame = extract_frame_and_secondaries(elems, nodes, mids, cols)
    if not frame.get("girders_mids") or not frame.get("columns_rows"):
        beams, columns = _members_from_elements(elems, nodes)
        frame["girders_mids"] = frame.get("girders_mids", [])
        frame["seconds_mids"] = frame.get("seconds_mids", [])
    else:
        beams = _beam_problem_rows(frame, elems, nodes)
        columns = _column_problem_rows(frame, nodes)
    return dict(frame=frame, beams=beams, columns=columns)


def analysis_results_by_member(prefix: str, case: str = "DL") -> Dict[str, dict]:
    """Return analysis/design rows keyed by member_id for a given case."""
    elems, nodes, mids, cols = read_case(prefix, case=case)
    frame = extract_frame_and_secondaries(elems, nodes, mids, cols)
    beams = _beam_problem_rows(frame, elems, nodes)
    columns = _column_problem_rows(frame, nodes)
    return {row["member_id"]: row for row in beams + columns}
# ----------------------------- RL helper exports -----------------------------
def _beam_problem_rows(frame, elems, nodes):
    beams = []
    for _, mr in frame["girders_mids"].iterrows():
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        beams.append(dict(
            member_id=f"GIR_{int(mr['mid_node'])}",
            group="GIR",
            mid_node=int(mr["mid_node"]),
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            deflection=Uz_mid,
            Xmid=float(mr["Xmid"]),
            Ymid=float(mr["Ymid"]),
            Zmid=float(mr["Zmid"])
        ))
    for _, mr in frame["seconds_mids"].iterrows():
        V_req, M_req, L, Uz_mid = beam_demands_from_halves(elems, nodes, mr)
        beams.append(dict(
            member_id=f"SEC_{int(mr['mid_node'])}",
            group="SEC",
            mid_node=int(mr["mid_node"]),
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            deflection=Uz_mid,
            Xmid=float(mr["Xmid"]),
            Ymid=float(mr["Ymid"]),
            Zmid=float(mr["Zmid"])
        ))
    return beams

def _column_problem_rows(frame, nodes):
    cols = []
    for _, er in frame["columns_rows"].iterrows():
        V_req, M_req, L, drift_x, drift_y = column_story_demand(er, nodes)
        iN, jN = int(er["iNode"]), int(er["jNode"])
        Xi, Yi, Zi = node_coords(nodes, iN)
        Xj, Yj, Zj = node_coords(nodes, jN)
        cols.append(dict(
            member_id=f"COL_{int(er['eleTag'])}",
            group="COL",
            eleTag=int(er["eleTag"]),
            iNode=iN,
            jNode=jN,
            length=L,
            shear_demand=V_req,
            moment_demand=M_req,
            drift_x=drift_x,
            drift_y=drift_y,
            Xi=Xi,
            Yi=Yi,
            Zi=Zi,
            Xj=Xj,
            Yj=Yj,
            Zj=Zj
        ))
    return cols

def load_design_problem(prefix: str):
    """Return demand data for RL-based design exploration."""
    elems, nodes, mids, cols = read_case(prefix, case="DL")
    frame = extract_frame_and_secondaries(elems, nodes, mids, cols)
    beams = _beam_problem_rows(frame, elems, nodes)
    columns = _column_problem_rows(frame, nodes)
    return dict(frame=frame, beams=beams, columns=columns)

# ----------------------------- CLI mode -----------------------------
def main():
    prefix = input("Enter results prefix (e.g., cfg_0_densest_X7_Y7): ").strip()
    path = run_design_for_prefix(prefix)
    print(f"Design results saved: {path}")

if __name__ == "__main__":
    main()
