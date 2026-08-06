# -*- coding: utf-8 -*-
"""Construct manuscript KPIs for the shot-time configuration.

The internal suffix `(L)` is retained for compatibility. Shot endpoints are not
used to construct KPIs or to cluster shot locations.
"""
from __future__ import annotations

import ast
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

INPUT_XLS = r""
SHEET_NAME = 0
OUTPUT_XLS = r""


def parse_xy(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return np.nan, np.nan
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return float(value[0]), float(value[1])
        except (SyntaxError, ValueError, TypeError):
            pass
    return np.nan, np.nan


def parse_frame(value: Any):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = None
    groups = {"att_all": [], "def_all": [], "att_out": [], "def_out": []}
    if not isinstance(value, list):
        return tuple(np.empty((0, 2), dtype=float) for _ in range(4))
    for player in value:
        if not isinstance(player, dict):
            continue
        x, y = parse_xy(player.get("location"))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        side = "att" if bool(player.get("teammate", False)) else "def"
        groups[f"{side}_all"].append([x, y])
        if not bool(player.get("keeper", False)):
            groups[f"{side}_out"].append([x, y])
    return tuple(np.asarray(groups[key], dtype=float).reshape(-1, 2) for key in
                 ("att_all", "def_all", "att_out", "def_out"))


def centroid(points):
    return np.mean(points, axis=0) if len(points) else np.array([np.nan, np.nan])


def distance(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.linalg.norm(p - q)) if np.all(np.isfinite(p)) and np.all(np.isfinite(q)) else np.nan


def mean_squared_centroid_distance(points):
    if len(points) == 0:
        return np.nan
    centre = centroid(points)
    return float(np.mean(np.sum((points - centre) ** 2, axis=1)))


def convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(tuple(point))
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(tuple(point))
    hull = lower[:-1] + upper[:-1]
    return float(abs(sum(hull[i][0]*hull[(i+1)%len(hull)][1] - hull[(i+1)%len(hull)][0]*hull[i][1] for i in range(len(hull)))) / 2.0)


def average_k_distance(focal, points, k):
    focal = np.asarray(focal, dtype=float)
    if len(points) == 0 or not np.all(np.isfinite(focal)):
        return np.nan
    values = np.sort(np.linalg.norm(points - focal, axis=1))
    return float(np.mean(values[:min(k, len(values))]))


def count_within(focal, points, radius):
    focal = np.asarray(focal, dtype=float)
    if len(points) == 0 or not np.all(np.isfinite(focal)):
        return 0
    return int(np.sum(np.linalg.norm(points - focal, axis=1) <= radius))


def compute_metrics(focal, frame) -> Dict[str, float]:
    att_all, def_all, att_out, def_out = parse_frame(frame)
    result = {}
    for radius in (5.0, 10.0):
        result[f"Adv_{int(radius)}"] = count_within(focal, att_all, radius) - count_within(focal, def_all, radius)
    result["Area_Att"] = convex_hull_area(att_out)
    result["Area_Def"] = convex_hull_area(def_out)
    result["Spr_Att"] = mean_squared_centroid_distance(att_out)
    result["Spr_Def"] = mean_squared_centroid_distance(def_out)
    for k in (1, 3, 5):
        result[f"Avg_{k}_Att"] = average_k_distance(focal, att_all, k)
        result[f"Avg_{k}_Def"] = average_k_distance(focal, def_all, k)
    result["DistToAttCentroid"] = distance(focal, centroid(att_all))
    result["DistToDefCentroid"] = distance(focal, centroid(def_all))
    return result


def main():
    if not INPUT_XLS or not OUTPUT_XLS:
        raise ValueError("Set INPUT_XLS and OUTPUT_XLS in the user-configuration section.")
    df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)
    if "freeze_frame" not in df:
        raise ValueError("Missing freeze_frame/event-time player-location field.")
    if "location_x" not in df or "location_y" not in df:
        if "location" not in df:
            raise ValueError("The input must contain location or location_x/location_y.")
        parsed = df["location"].apply(parse_xy)
        df["location_x"] = parsed.str[0]
        df["location_y"] = parsed.str[1]
    df["location_x"] = pd.to_numeric(df["location_x"], errors="coerce")
    df["location_y"] = pd.to_numeric(df["location_y"], errors="coerce")
    metrics = [compute_metrics((row.location_x, row.location_y), row.freeze_frame) for row in df.itertuples(index=False)]
    metrics_df = pd.DataFrame(metrics).add_suffix("(L)")
    output = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLS)), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as writer:
        output.to_excel(writer, index=False, sheet_name="shots_with_metrics")
    print(f"[OK] Shot-time KPIs written to {OUTPUT_XLS}")


if __name__ == "__main__":
    main()
