# -*- coding: utf-8 -*-
"""Construct manuscript KPIs for the two Pass focal-location configurations.

Internal suffixes are retained for compatibility:
- (L): pass-origin configuration;
- (E'): pass-destination-referenced configuration.

Both configurations use the pass-initiation player snapshot. The pass destination
is used only as the E' focal coordinate; no receiving-time player snapshot is used.
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


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def parse_xy(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return np.nan, np.nan
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return float(parsed[0]), float(parsed[1])
        except (SyntaxError, ValueError, TypeError):
            pass
    return np.nan, np.nan


def parse_frame(value: Any):
    """Return all and outfield attacker/defender coordinate arrays."""
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


def centroid(points: np.ndarray) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.array([np.nan, np.nan], dtype=float)
    return np.mean(points, axis=0)


def mean_squared_centroid_distance(points: np.ndarray) -> float:
    """Mean squared Euclidean distance from outfield players to their centroid (m²)."""
    if points is None or len(points) == 0:
        return np.nan
    centre = centroid(points)
    return float(np.mean(np.sum((points - centre) ** 2, axis=1)))


def k_average_distance(focal, points: np.ndarray, k: int) -> float:
    focal = np.asarray(focal, dtype=float)
    if points is None or len(points) == 0 or not np.all(np.isfinite(focal)):
        return np.nan
    distances = np.sort(np.linalg.norm(points - focal, axis=1))
    return float(np.mean(distances[: min(k, len(distances))]))


def count_within(focal, points: np.ndarray, radius: float) -> int:
    focal = np.asarray(focal, dtype=float)
    if points is None or len(points) == 0 or not np.all(np.isfinite(focal)):
        return 0
    return int(np.sum(np.linalg.norm(points - focal, axis=1) <= radius))


def convex_hull_area(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(tuple(point))
    upper = []
    for point in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(tuple(point))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    return float(abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1]
        - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )) * 0.5)


def distance(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        return np.nan
    return float(np.linalg.norm(p - q))


def compute_metrics(focal, frame) -> Dict[str, float]:
    att_all, def_all, att_out, def_out = parse_frame(frame)
    result: Dict[str, float] = {}
    for radius in (5.0, 10.0):
        result[f"Adv_{int(radius)}"] = count_within(focal, att_all, radius) - count_within(focal, def_all, radius)
    result["Area_Att"] = convex_hull_area(att_out)
    result["Area_Def"] = convex_hull_area(def_out)
    result["Spr_Att"] = mean_squared_centroid_distance(att_out)
    result["Spr_Def"] = mean_squared_centroid_distance(def_out)
    for k in (1, 3, 5):
        result[f"Avg_{k}_Att"] = k_average_distance(focal, att_all, k)
        result[f"Avg_{k}_Def"] = k_average_distance(focal, def_all, k)
    result["DistToAttCentroid"] = distance(focal, centroid(att_all))
    result["DistToDefCentroid"] = distance(focal, centroid(def_all))
    return result


def main() -> None:
    if not INPUT_XLS or not OUTPUT_XLS:
        raise ValueError("Set INPUT_XLS and OUTPUT_XLS in the user-configuration section.")
    df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)
    if "location_x" not in df or "location_y" not in df:
        if "location" not in df:
            raise ValueError("The input must contain location or location_x/location_y.")
        parsed = df["location"].apply(parse_xy)
        df["location_x"] = parsed.str[0]
        df["location_y"] = parsed.str[1]
    for column in ("location_x", "location_y", "end_location_x", "end_location_y"):
        if column not in df:
            raise ValueError(f"Missing required coordinate column: {column}")
        df[column] = to_num(df[column])
    if "freeze_frame" not in df:
        raise ValueError("Missing freeze_frame/event-time player-location field.")

    origin_rows = []
    destination_rows = []
    for row in df.itertuples(index=False):
        frame = getattr(row, "freeze_frame")
        origin = (getattr(row, "location_x"), getattr(row, "location_y"))
        destination = (getattr(row, "end_location_x"), getattr(row, "end_location_y"))
        origin_rows.append(compute_metrics(origin, frame))
        destination_rows.append(compute_metrics(destination, frame))

    origin_df = pd.DataFrame(origin_rows).add_suffix("(L)")
    destination_df = pd.DataFrame(destination_rows).add_suffix("(E')")
    output = pd.concat([df.reset_index(drop=True), origin_df, destination_df], axis=1)
    PathLike = os.path.dirname(os.path.abspath(OUTPUT_XLS))
    os.makedirs(PathLike, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as writer:
        output.to_excel(writer, index=False, sheet_name="passes_with_metrics")
    print(f"[OK] Pass-origin and pass-destination-referenced KPIs written to {OUTPUT_XLS}")


if __name__ == "__main__":
    main()
