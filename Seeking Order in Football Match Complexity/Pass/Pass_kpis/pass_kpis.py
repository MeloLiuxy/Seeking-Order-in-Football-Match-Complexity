# -*- coding: utf-8 -*-
import os
import ast
import math
import numpy as np
import pandas as pd

# ========= 路径 =========
INPUT_XLS  = r""
SHEET_NAME = 0
OUTPUT_XLS = r""

# ========= 常用小工具 =========
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def parse_xy(val):
    """把 [x,y] / '(x,y)' / '[x, y]' 解析为 (x,y)，否则返回 (nan,nan)"""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        try:
            return float(val[0]), float(val[1])
        except Exception:
            return np.nan, np.nan
    if isinstance(val, str):
        try:
            obj = ast.literal_eval(val)
            if isinstance(obj, (list, tuple)) and len(obj) >= 2:
                return float(obj[0]), float(obj[1])
        except Exception:
            pass
    return np.nan, np.nan

def parse_ff(val):
    """
    解析 freeze_frame / back_ff：
    返回 (A, D) 两个 numpy 数组，分别是进攻方与防守方的 Nx2 坐标。
    """
    A, D = [], []
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except Exception:
            val = None
    if not isinstance(val, list):
        return np.empty((0,2)), np.empty((0,2))
    for obj in val:
        try:
            loc = obj.get("location", None)
            tm  = bool(obj.get("teammate", False))
        except Exception:
            continue
        x, y = parse_xy(loc)
        if not np.isnan(x) and not np.isnan(y):
            if tm:
                A.append([x, y])
            else:
                D.append([x, y])
    return np.array(A, dtype=float), np.array(D, dtype=float)

def centroid(P):
    """点集质心；空返回 (nan,nan)"""
    if P is None or len(P) == 0:
        return (np.nan, np.nan)
    return (float(np.mean(P[:,0])), float(np.mean(P[:,1])))

def mean_dist_to_centroid(P):
    """离散度：到自身质心的平均距离；空返回 nan"""
    if P is None or len(P) == 0:
        return np.nan
    c = np.array(centroid(P))
    if np.any(np.isnan(c)):
        return np.nan
    d = np.linalg.norm(P - c, axis=1)
    return float(np.mean(d)) if len(d) > 0 else np.nan

def k_avg_dist(focal, P, k):
    """
    焦点到该集合最近 k 人的平均距离；集合人数 < k 时取所有人；空返回 nan
    """
    if P is None or len(P) == 0:
        return np.nan
    f = np.array(focal, dtype=float)
    if np.any(np.isnan(f)):
        return np.nan
    d = np.linalg.norm(P - f, axis=1)
    d = np.sort(d)
    m = min(k, len(d))
    if m <= 0:
        return np.nan
    return float(np.mean(d[:m]))

def count_within_radius(focal, P, r):
    """半径 r 内人数计数"""
    if P is None or len(P) == 0:
        return 0
    f = np.array(focal, dtype=float)
    if np.any(np.isnan(f)):
        return 0
    d = np.linalg.norm(P - f, axis=1)
    return int(np.sum(d <= r))

def convex_hull_area(P):
    """
    单调链算法（不依赖 scipy）：返回点集的凸包面积；少于 3 点返回 0
    """
    if P is None or len(P) < 3:
        return 0.0
    pts = np.unique(P, axis=0)  # 去重
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0

    # 多边形面积
    area = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]
        x2, y2 = hull[(i+1) % len(hull)]
        area += x1*y2 - x2*y1
    return abs(area) * 0.5

def dist(p, q):
    if p is None or q is None:
        return np.nan
    if any(np.isnan(p)) or any(np.isnan(q)):
        return np.nan
    p = np.array(p, dtype=float); q = np.array(q, dtype=float)
    return float(np.linalg.norm(p - q))

# ========= 主流程 =========
def compute_for_frame(focal_xy, ff):
    """
    在某帧（ff）下，以 focal_xy 为中心，计算所有指标，返回 dict
    """
    A, D = parse_ff(ff)
    res = {}

    # 人数优势
    for R in (5.0, 10.0):
        na = count_within_radius(focal_xy, A, R)
        nd = count_within_radius(focal_xy, D, R)
        res[f"Adv_{int(R)}"] = int(na - nd)

    # 面积
    res["Area_Att"] = float(convex_hull_area(A))
    res["Area_Def"] = float(convex_hull_area(D))

    # 离散度（spread）
    res["Spr_Att"] = float(mean_dist_to_centroid(A))
    res["Spr_Def"] = float(mean_dist_to_centroid(D))

    # 最近 k 人平均距离
    for k in (1, 2, 3, 5):
        res[f"Avg_{k}_Att"] = float(k_avg_dist(focal_xy, A, k))
        res[f"Avg_{k}_Def"] = float(k_avg_dist(focal_xy, D, k))

    # 到各方质心距离
    cA = centroid(A)
    cD = centroid(D)
    res["DistToAttCentroid"] = float(dist(focal_xy, cA))
    res["DistToDefCentroid"] = float(dist(focal_xy, cD))

    return res

def main():
    df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)

    # 起点/终点坐标准备
    if "location_x" not in df.columns or "location_y" not in df.columns:
        # 从 location 列解析
        loc_xy = df.get("location")
        if loc_xy is None:
            raise ValueError("缺少 location 列，无法得到 L 的坐标")
        loc_xy = loc_xy.apply(parse_xy)
        df["location_x"] = loc_xy.apply(lambda t: t[0])
        df["location_y"] = loc_xy.apply(lambda t: t[1])

    df["end_location_x"] = to_num(df.get("end_location_x"))
    df["end_location_y"] = to_num(df.get("end_location_y"))

    # 三个帧：L=freeze_frame；E=back_ff（若没有 back_ff 就降级用 freeze_frame）
    ff_L = df.get("freeze_frame")
    ff_E = df.get("back_ff", df.get("freeze_frame"))

    # 结果容器
    out_cols = {}

    # ----- L 帧 -----
    L_focal = list(zip(df["location_x"], df["location_y"]))
    L_res_list = []
    for i, focal in enumerate(L_focal):
        ff = ff_L.iloc[i] if ff_L is not None and i < len(ff_L) else None
        L_res_list.append(compute_for_frame(focal, ff))
    L_res = pd.DataFrame(L_res_list)
    L_res.columns = [f"{c}(L)" for c in L_res.columns]
    out_cols["L"] = L_res

    # ----- E 帧（以落点为焦点，使用 back_ff）-----
    E_focal = list(zip(df["end_location_x"], df["end_location_y"]))
    E_res_list = []
    for i, focal in enumerate(E_focal):
        ff = ff_E.iloc[i] if ff_E is not None and i < len(ff_E) else None
        E_res_list.append(compute_for_frame(focal, ff))
    E_res = pd.DataFrame(E_res_list)
    E_res.columns = [f"{c}(E)" for c in E_res.columns]
    out_cols["E"] = E_res

    # ----- E' 帧（以落点为焦点，但用 freeze_frame）-----
    Eprime_res_list = []
    for i, focal in enumerate(E_focal):  # 焦点仍是 end_location
        ff = ff_L.iloc[i] if ff_L is not None and i < len(ff_L) else None  # 用 freeze_frame
        Eprime_res_list.append(compute_for_frame(focal, ff))
    Eprime_res = pd.DataFrame(Eprime_res_list)
    Eprime_res.columns = [f"{c}(E')" for c in Eprime_res.columns]
    out_cols["E'"] = Eprime_res  # ★ 关键：加到字典里

    # 合并写回
    out = pd.concat(
        [df.reset_index(drop=True), out_cols["L"], out_cols["E"], out_cols["E'"]],
        axis=1
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLS)), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as w:
        out.to_excel(w, index=False, sheet_name="passes_with_metrics")

    print(f"[OK] 指标计算完成，已写出：{OUTPUT_XLS}")

if __name__ == "__main__":
    main()
