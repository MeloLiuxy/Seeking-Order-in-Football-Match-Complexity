# -*- coding: utf-8 -*-
"""
为 Shot 计算围绕 L/E 两点的一组空间指标（含质心距离）：
- 基于同一帧 freeze_frame（teammate=True/False 且 keeper=False 为外场球员）
- L 点：location_x, location_y
- E 点：end_location_x, end_location_y  （来自该帧 keeper=True 的坐标）
- 指标：
  * Adv_5(*) / Adv_10(*)
  * Avg_k_Att(*) / Avg_k_Def(*)  (k∈{1,2,3,5})
  * Area_Att(*) / Area_Def(*)
  * Spr_Att(*)  / Spr_Def(*)
  * DistToAttCentroid(*) / DistToDefCentroid(*)
"""

import os
import ast
import math
import numpy as np
import pandas as pd

# ======== 配置 ========
INPUT_XLS  = r""
SHEET_NAME = 0
OUTPUT_XLS = r""

# ======== 解析工具 ========
def _maybe_eval(x):
    if isinstance(x, (dict, list, tuple)):
        return x
    if isinstance(x, str):
        sx = x.strip()
        if (sx.startswith("{") and sx.endswith("}")) or (sx.startswith("[") and sx.endswith("]")):
            try:
                return ast.literal_eval(sx)
            except Exception:
                return x
    return x

def _ensure_eval(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(_maybe_eval)

# ======== 几何工具 ========
def _dist(p, q):
    if p is None or q is None:
        return np.nan
    if any(pd.isna(v) for v in p) or any(pd.isna(v) for v in q):
        return np.nan
    return math.hypot(p[0]-q[0], p[1]-q[1])

def _centroid(points):
    if not points:
        return None
    sx = sum(p[0] for p in points); sy = sum(p[1] for p in points)
    return (sx/len(points), sy/len(points))

def _mean_dist_to_centroid(points):
    if not points:
        return np.nan
    c = _centroid(points)
    if c is None:
        return np.nan
    d = [_dist(p, c) for p in points]
    d = [x for x in d if not pd.isna(x)]
    return np.mean(d) if d else np.nan

# ---- 凸包（单调链）与多边形面积 ----
def _convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-a[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % len(poly)]
        s += x1*y2 - x2*y1
    return abs(s) / 2.0

def _frame_points(frame_list):
    """从 freeze_frame 中抽取攻/防外场球员坐标（排除 keeper=True）"""
    att, deff = [], []
    if not isinstance(frame_list, (list, tuple)):
        return att, deff
    for pl in frame_list:
        try:
            if pl.get("keeper", False):
                continue
            loc = pl.get("location", None)
            if not (isinstance(loc, (list, tuple)) and len(loc) >= 2):
                continue
            xy = (float(loc[0]), float(loc[1]))
            if pl.get("teammate", False):
                att.append(xy)
            else:
                deff.append(xy)
        except Exception:
            continue
    return att, deff

def _advantage(points_att, points_def, center_xy, radius):
    if center_xy is None or any(pd.isna(v) for v in center_xy):
        return np.nan
    a = sum(1 for p in points_att if _dist(p, center_xy) <= radius)
    d = sum(1 for p in points_def if _dist(p, center_xy) <= radius)
    return a - d

def _avg_k_dist(points, center_xy, k):
    if center_xy is None or any(pd.isna(v) for v in center_xy):
        return np.nan
    if not points:
        return np.nan
    ds = sorted(_dist(center_xy, p) for p in points)
    use = ds[:min(k, len(ds))]
    return float(np.mean(use)) if use else np.nan

# ======== 指标主函数 ========
def _compute_metrics_one_row(row):
    """
    输入：一行（需含 freeze_frame, location_x/y, end_location_x/y）
    输出：字典（L/E 全量指标）
    """
    frame = row.get("freeze_frame", None)
    att, deff = _frame_points(frame)

    # 固定帧整体（与 L/E 无关，但复制到 L/E 两套字段以保持命名一致）
    area_att = _polygon_area(_convex_hull(att))
    area_def = _polygon_area(_convex_hull(deff))
    spr_att  = _mean_dist_to_centroid(att)
    spr_def  = _mean_dist_to_centroid(deff)

    # 质心
    c_att = _centroid(att)
    c_def = _centroid(deff)

    L = (row.get("location_x", np.nan), row.get("location_y", np.nan))
    E = (row.get("end_location_x", np.nan), row.get("end_location_y", np.nan))

    out = {}

    # ---- L 点 ----
    out["Adv_5(L)"]   = _advantage(att, deff, L, radius=5.0)
    out["Adv_10(L)"]  = _advantage(att, deff, L, radius=10.0)
    out["Area_Att(L)"] = area_att
    out["Area_Def(L)"] = area_def
    out["Spr_Att(L)"]  = spr_att
    out["Spr_Def(L)"]  = spr_def
    for k in (1, 2, 3, 5):
        out[f"Avg_{k}_Att(L)"] = _avg_k_dist(att, L, k)
        out[f"Avg_{k}_Def(L)"] = _avg_k_dist(deff, L, k)
    out["DistToAttCentroid(L)"] = _dist(L, c_att)
    out["DistToDefCentroid(L)"] = _dist(L, c_def)

    # ---- E 点 ----
    out["Adv_5(E)"]   = _advantage(att, deff, E, radius=5.0)
    out["Adv_10(E)"]  = _advantage(att, deff, E, radius=10.0)
    out["Area_Att(E)"] = area_att
    out["Area_Def(E)"] = area_def
    out["Spr_Att(E)"]  = spr_att
    out["Spr_Def(E)"]  = spr_def
    for k in (1, 2, 3, 5):
        out[f"Avg_{k}_Att(E)"] = _avg_k_dist(att, E, k)
        out[f"Avg_{k}_Def(E)"] = _avg_k_dist(deff, E, k)
    out["DistToAttCentroid(E)"] = _dist(E, c_att)
    out["DistToDefCentroid(E)"] = _dist(E, c_def)

    return out

# ======== 主流程 ========
def main():
    df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)

    needed = ["freeze_frame", "location_x", "location_y", "end_location_x", "end_location_y"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列：{miss}")

    _ensure_eval(df, ["freeze_frame"])  # 解析帧

    # 逐行计算
    metrics = df.apply(_compute_metrics_one_row, axis=1, result_type="expand")

    # 合并写出
    final = pd.concat([df, metrics], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLS)), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as w:
        final.to_excel(w, index=False, sheet_name="shots_with_metrics")

    print(f"[OK] Shot L/E 指标（含质心距离）计算完成：{OUTPUT_XLS}")

if __name__ == "__main__":
    main()
