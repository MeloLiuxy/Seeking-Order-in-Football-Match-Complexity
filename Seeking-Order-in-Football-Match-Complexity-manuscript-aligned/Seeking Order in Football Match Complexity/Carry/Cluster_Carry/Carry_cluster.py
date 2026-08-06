# -*- coding: utf-8 -*-
r"""
Carry clustering sensitivity analysis for reviewer revision (Dribble excluded)
===========================================================

This script performs the complete Carry clustering analysis required for revision:

1. Compares K = 2, 3, 4, and 5 using the same four trajectory coordinates.
2. Reports Silhouette, Calinski-Harabasz, Davies-Bouldin, and inertia.
3. Reports the minimum cluster size/proportion and cluster imbalance for every K.
4. Evaluates random-seed stability using ARI across 20 KMeans solutions.
5. Reports original-scale cluster centers, carry length, direction, and spatial zones.
6. Produces an overview plot, single-cluster plots, Silhouette plots, and metric curves.
7. Preserves labels for every K in the full input table.
8. Produces a standalone K=3 dataset with cluster_id for the subsequent DML analysis.
9. Produces a reviewer checklist, K-selection table, missing/exclusion audit,
   feature-resolution audit, scaler parameters, and reproducibility manifest.

Important:
- The input Carry table is clustered directly. Existing labels from another file
  are not merged and no observations receive guessed labels.
- The primary K remains K=3 by design. Statistical ranks are reported as
  sensitivity evidence and are not used as an automatic K-selection rule.
- The code accepts end_location_x/end_location_y and the historical aliases
  end_location_x/end_location_y.
- Only clustering is performed. DML, placebo tests, and policy optimisation are
  intentionally outside this script.
"""

import os
import sys
import platform
import warnings
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps
from matplotlib.lines import Line2D

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

warnings.filterwarnings("ignore")


# =========================================================
# 1. 路径配置
# =========================================================

SCENARIO_NAME = "Carry"

INPUT_XLS = r""
SHEET_NAME = 0

OUTPUT_DIR = r""

OUTPUT_XLS = os.path.join(
    OUTPUT_DIR,
    "",
)

OUTPUT_FIG_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)

OUTPUT_ALL_LABELS_CSV = os.path.join(
    OUTPUT_DIR,
    "Carry_all_K_cluster_labels.csv",
)

OUTPUT_K3_CSV = os.path.join(
    OUTPUT_DIR,
    "Carry_K3_main_for_DML.csv",
)

OUTPUT_K3_XLSX = os.path.join(
    OUTPUT_DIR,
    "Carry_K3_main_for_DML.xlsx",
)

OUTPUT_K3_LABELS_ONLY_CSV = os.path.join(
    OUTPUT_DIR,
    "Carry_K3_labels_only.csv",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


# =========================================================
# 2. 聚类参数
# =========================================================

# 固定比较，不再自动只选一个K
K_VALUES = [2, 3, 4, 5]

# 当前论文主分析使用K=3
MAIN_K = 3

MAIN_RANDOM_SEED = 42
MAIN_N_INIT = 50
MAX_ITER = 500

# 用于判断某个cluster是否足以支持后续DML
MIN_CLUSTER_SIZE = 120

# 随机种子稳定性分析
STABILITY_SEEDS = list(range(20))
STABILITY_N_INIT = 10

# Silhouette样本过多时随机抽样，避免太慢
SILHOUETTE_SAMPLE_SIZE = 5000

# Silhouette分布图最多使用多少样本
SILHOUETTE_PLOT_SAMPLE_SIZE = 3000

# 每个cluster画图时最多使用多少条轨迹
PLOT_SAMPLE_MAX_PER_CLUSTER = 2000
PLOT_RANDOM_SEED = 42

# 线条样式
LINE_ALPHA = 0.30
LINE_WIDTH = 1.35


# =========================================================
# 3. 特征列
# =========================================================

# Canonical feature names used internally.
FEATURE_COLS = [
    "location_x",
    "location_y",
    "end_location_x",
    "end_location_y",
]

# The current Carry table uses end_location_x/end_location_y. Historical files
# may use end_location_x/end_location_y, so aliases are resolved
# explicitly and recorded in the output workbook.
FEATURE_ALIASES = {
    "location_x": [
        "location_x",
        "carry_start_location_x",
        "start_location_x",
    ],
    "location_y": [
        "location_y",
        "carry_start_location_y",
        "start_location_y",
    ],
    "end_location_x": [
        "end_location_x",
        "end_location_x",
        "target_location_x",
    ],
    "end_location_y": [
        "end_location_y",
        "end_location_y",
        "target_location_y",
    ],
}

START_X_COL = "location_x"
START_Y_COL = "location_y"
END_X_COL = "end_location_x"
END_Y_COL = "end_location_y"

MAIN_CLUSTER_COL = "cluster_id"

# Coordinate quality audit. Values outside the football pitch plus this
# tolerance are excluded transparently and written to the exclusion sheet.
CHECK_COORDINATE_RANGE = True
COORDINATE_TOLERANCE = 0.50

# Existing cluster_id is preserved before the new K=3 label is written.
PRESERVE_EXISTING_CLUSTER_ID = True

# Event-type exclusion rule.
# Rows whose type_name equals any value below are excluded before scaling,
# KMeans fitting, metric calculation, plotting, and K=3 DML export.
TYPE_NAME_COL = "type_name"
EXCLUDED_TYPE_NAMES = {
    "dribble",
}
REQUIRE_TYPE_NAME_COLUMN = True


# =========================================================
# 4. 球场参数
# =========================================================

FIELD_LENGTH = 120.0
FIELD_WIDTH = 80.0
BORDER_OFFSET = 5.0

plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


# =========================================================
# 5. 通用函数
# =========================================================

def first_existing_column(columns, candidates):
    """Return the first exact column match after whitespace stripping."""
    col_set = set(columns)
    for candidate in candidates:
        if candidate in col_set:
            return candidate
    return None


def resolve_feature_columns(frame):
    """Resolve source columns and copy them to the canonical feature names."""
    resolved = {}
    rows = []

    for canonical, aliases in FEATURE_ALIASES.items():
        source = first_existing_column(frame.columns, aliases)
        if source is None:
            raise ValueError(
                f"缺少聚类坐标列 {canonical!r}。可识别列名为：{aliases}"
            )

        resolved[canonical] = source
        if canonical != source:
            frame[canonical] = frame[source]

        numeric = pd.to_numeric(frame[canonical], errors="coerce")
        rows.append({
            "canonical_feature": canonical,
            "source_column": source,
            "alias_copy_required": bool(canonical != source),
            "nonmissing_numeric_n": int(numeric.notna().sum()),
            "missing_or_non_numeric_n": int(numeric.isna().sum()),
            "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
            "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
        })

    return frame, resolved, pd.DataFrame(rows)


def build_exclusion_reason_table(frame, numeric_features, valid_mask):
    """Create a row-level audit for observations excluded before clustering."""
    excluded_index = frame.index[~valid_mask]
    if len(excluded_index) == 0:
        return pd.DataFrame(columns=[
            "original_row_index", "exclusion_reason",
            *FEATURE_COLS,
        ])

    rows = []
    for idx in excluded_index:
        reasons = []
        values = {}

        for col in FEATURE_COLS:
            value = numeric_features.loc[idx, col]
            values[col] = value
            if not np.isfinite(value):
                reasons.append(f"{col}:missing_or_non_numeric")

        if (
            TYPE_NAME_COL in frame.columns
            and str(frame.at[idx, TYPE_NAME_COL]).strip().casefold()
            in EXCLUDED_TYPE_NAMES
        ):
            reasons.append(
                f"{TYPE_NAME_COL}:{frame.at[idx, TYPE_NAME_COL]}_excluded"
            )

        if CHECK_COORDINATE_RANGE:
            range_rules = {
                "location_x": (-COORDINATE_TOLERANCE, FIELD_LENGTH + COORDINATE_TOLERANCE),
                "end_location_x": (-COORDINATE_TOLERANCE, FIELD_LENGTH + COORDINATE_TOLERANCE),
                "location_y": (-COORDINATE_TOLERANCE, FIELD_WIDTH + COORDINATE_TOLERANCE),
                "end_location_y": (-COORDINATE_TOLERANCE, FIELD_WIDTH + COORDINATE_TOLERANCE),
            }
            for col, (low, high) in range_rules.items():
                value = numeric_features.loc[idx, col]
                if np.isfinite(value) and not (low <= value <= high):
                    reasons.append(f"{col}:outside_pitch_range")

        row = {
            "original_row_index": int(idx),
            "exclusion_reason": " | ".join(reasons) if reasons else "invalid_coordinate",
        }
        row.update(values)

        for key in [
            "__source_file__", "match_id", "id", "index",
            "period", "timestamp", "minute", "second",
            "player_name", "team", "type_name",
        ]:
            if key in frame.columns:
                row[key] = frame.at[idx, key]

        rows.append(row)

    return pd.DataFrame(rows)


def format_excel_workbook(writer, tables):
    """Apply light reviewer-friendly formatting to every generated sheet."""
    workbook = writer.book
    header_format = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "top",
        "border": 1,
        "bg_color": "#D9EAF7",
    })
    percent_format = workbook.add_format({"num_format": "0.00%"})
    float_format = workbook.add_format({"num_format": "0.0000"})

    for sheet_name, table in tables:
        if table is None:
            continue
        worksheet = writer.sheets.get(sheet_name)
        if worksheet is None:
            continue

        worksheet.freeze_panes(1, 0)
        if len(table.columns) > 0:
            worksheet.autofilter(0, 0, max(len(table), 1), len(table.columns) - 1)

        for col_idx, column in enumerate(table.columns):
            worksheet.write(0, col_idx, str(column), header_format)

            if len(table):
                sample_values = table[column].head(500).tolist()
                value_lengths = [
                    len(str(value))
                    if not pd.isna(value)
                    else 0
                    for value in sample_values
                ]
            else:
                value_lengths = []

            max_len = max(
                [len(str(column))] + value_lengths
            )
            width = min(max(max_len + 2, 11), 45)

            column_lower = str(column).lower()
            cell_format = None
            if "proportion" in column_lower or "rate" in column_lower:
                cell_format = percent_format
            elif pd.api.types.is_float_dtype(table[column]):
                cell_format = float_format

            worksheet.set_column(col_idx, col_idx, width, cell_format)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def classify_longitudinal_zone(x):
    if not np.isfinite(x):
        return ""

    if x < 40:
        return "X_0_40"

    if x < 80:
        return "X_40_80"

    return "X_80_120"


def classify_lateral_zone(y):
    if not np.isfinite(y):
        return ""

    if y < FIELD_WIDTH / 3:
        return "Upper"

    if y < 2 * FIELD_WIDTH / 3:
        return "Central"

    return "Lower"


def draw_pitch_shot_style(ax):
    ax.set_facecolor("white")
    line_color = "#000000"

    # 外框
    field = patches.Rectangle(
        (0, 0),
        FIELD_LENGTH,
        FIELD_WIDTH,
        linewidth=2.8,
        edgecolor=line_color,
        facecolor="none",
        zorder=10,
    )
    ax.add_patch(field)

    # 中线
    ax.plot(
        [FIELD_LENGTH / 2, FIELD_LENGTH / 2],
        [0, FIELD_WIDTH],
        color=line_color,
        lw=2.2,
        zorder=11,
    )

    # 大禁区
    big_box_left = patches.Rectangle(
        (0, (FIELD_WIDTH - 40.3) / 2),
        16.5,
        40.3,
        linewidth=2,
        edgecolor="#6666FF",
        facecolor="none",
        zorder=11,
    )

    big_box_right = patches.Rectangle(
        (FIELD_LENGTH - 16.5, (FIELD_WIDTH - 40.3) / 2),
        16.5,
        40.3,
        linewidth=2,
        edgecolor="#6666FF",
        facecolor="none",
        zorder=11,
    )

    ax.add_patch(big_box_left)
    ax.add_patch(big_box_right)

    # 小禁区
    small_box_left = patches.Rectangle(
        (0, (FIELD_WIDTH - 18.3) / 2),
        5.5,
        18.3,
        linewidth=2,
        edgecolor="#FF6666",
        facecolor="none",
        zorder=11,
    )

    small_box_right = patches.Rectangle(
        (FIELD_LENGTH - 5.5, (FIELD_WIDTH - 18.3) / 2),
        5.5,
        18.3,
        linewidth=2,
        edgecolor="#FF6666",
        facecolor="none",
        zorder=11,
    )

    ax.add_patch(small_box_left)
    ax.add_patch(small_box_right)

    # 球门
    goal_y1 = (FIELD_WIDTH - 7.32) / 2
    goal_y2 = (FIELD_WIDTH + 7.32) / 2

    ax.plot(
        [0, 0],
        [goal_y1, goal_y2],
        color="#FF6666",
        lw=5,
        zorder=11,
    )

    ax.plot(
        [FIELD_LENGTH, FIELD_LENGTH],
        [goal_y1, goal_y2],
        color="#FF6666",
        lw=5,
        zorder=11,
    )

    # 中圈
    center_circle = plt.Circle(
        (FIELD_LENGTH / 2, FIELD_WIDTH / 2),
        9.15,
        color=line_color,
        fill=False,
        lw=2.2,
        zorder=11,
    )
    ax.add_patch(center_circle)

    ax.plot(
        FIELD_LENGTH / 2,
        FIELD_WIDTH / 2,
        marker="o",
        color=line_color,
        markersize=6,
        zorder=12,
    )

    # 罚球点
    ax.add_patch(
        patches.Circle(
            (11, FIELD_WIDTH / 2),
            radius=0.2,
            color=line_color,
            zorder=11,
        )
    )

    ax.add_patch(
        patches.Circle(
            (FIELD_LENGTH - 11, FIELD_WIDTH / 2),
            radius=0.2,
            color=line_color,
            zorder=11,
        )
    )

    # 角球弧
    corner_radius = 1.0

    corners = [
        (0, 0, 0, 90),
        (0, FIELD_WIDTH, 270, 360),
        (FIELD_LENGTH, 0, 90, 180),
        (FIELD_LENGTH, FIELD_WIDTH, 180, 270),
    ]

    for x, y, theta1, theta2 in corners:
        ax.add_patch(
            patches.Arc(
                (x, y),
                2 * corner_radius,
                2 * corner_radius,
                angle=0,
                theta1=theta1,
                theta2=theta2,
                color=line_color,
                lw=2,
                zorder=11,
            )
        )

    ax.set_xlim(
        -BORDER_OFFSET - 1,
        FIELD_LENGTH + BORDER_OFFSET + 1,
    )

    ax.set_ylim(
        -BORDER_OFFSET - 1,
        FIELD_WIDTH + BORDER_OFFSET + 1,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)


def order_cluster_labels(labels, centers_original):
    """
    对cluster重新排序，避免不同KMeans运行产生无规律编号。

    排序依据：
    1. 起点X
    2. 起点Y
    3. 终点X
    4. 终点Y
    """

    k = centers_original.shape[0]

    raw_order = sorted(
        range(k),
        key=lambda raw_id: (
            centers_original[raw_id, 0],
            centers_original[raw_id, 1],
            centers_original[raw_id, 2],
            centers_original[raw_id, 3],
        ),
    )

    raw_to_ordered = {
        raw_id: ordered_id
        for ordered_id, raw_id in enumerate(raw_order)
    }

    ordered_labels = np.array(
        [
            raw_to_ordered[int(raw_label)]
            for raw_label in labels
        ],
        dtype=int,
    )

    ordered_centers = centers_original[
        raw_order,
        :
    ]

    return (
        ordered_labels,
        raw_to_ordered,
        ordered_centers,
        raw_order,
    )


def sample_cluster_rows(cluster_df, max_n, seed):
    if max_n is None:
        return cluster_df.copy()

    if len(cluster_df) <= int(max_n):
        return cluster_df.copy()

    return cluster_df.sample(
        n=int(max_n),
        random_state=seed,
        replace=False,
    ).copy()


def make_cluster_colors(k):
    cmap = colormaps.get_cmap(
        "tab10" if k <= 10 else "tab20"
    )

    return {
        cid: cmap(cid % cmap.N)
        for cid in range(k)
    }


# =========================================================
# 6. 读取数据
# =========================================================

print("[1/8] Reading Carry data...")

df = pd.read_excel(
    INPUT_XLS,
    sheet_name=SHEET_NAME,
)

df.columns = [
    str(col).strip()
    for col in df.columns
]

# Preserve any old labels rather than silently overwriting them.
if (
    PRESERVE_EXISTING_CLUSTER_ID
    and MAIN_CLUSTER_COL in df.columns
):
    previous_col = f"{MAIN_CLUSTER_COL}_before_reclustering"
    suffix = 1
    while previous_col in df.columns:
        previous_col = f"{MAIN_CLUSTER_COL}_before_reclustering_{suffix}"
        suffix += 1
    df = df.rename(columns={MAIN_CLUSTER_COL: previous_col})

df, resolved_feature_map, feature_resolution_df = resolve_feature_columns(df)

if TYPE_NAME_COL not in df.columns:
    if REQUIRE_TYPE_NAME_COLUMN:
        raise ValueError(
            f"缺少用于异常类型排除的字段：{TYPE_NAME_COL!r}。"
        )
    type_name_normalized = pd.Series(
        "",
        index=df.index,
        dtype="string",
    )
else:
    type_name_normalized = (
        df[TYPE_NAME_COL]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

excluded_type_mask = type_name_normalized.isin(EXCLUDED_TYPE_NAMES)

n_original = int(len(df))

numeric_data = df[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    numeric_data[col] = pd.to_numeric(
        numeric_data[col],
        errors="coerce",
    )

complete_coordinate_mask = numeric_data[FEATURE_COLS].notna().all(axis=1)

if CHECK_COORDINATE_RANGE:
    x_low = -COORDINATE_TOLERANCE
    x_high = FIELD_LENGTH + COORDINATE_TOLERANCE
    y_low = -COORDINATE_TOLERANCE
    y_high = FIELD_WIDTH + COORDINATE_TOLERANCE

    in_range_mask = (
        numeric_data["location_x"].between(x_low, x_high, inclusive="both")
        & numeric_data["end_location_x"].between(x_low, x_high, inclusive="both")
        & numeric_data["location_y"].between(y_low, y_high, inclusive="both")
        & numeric_data["end_location_y"].between(y_low, y_high, inclusive="both")
    )
else:
    in_range_mask = pd.Series(True, index=df.index)

valid_cluster_mask = (
    complete_coordinate_mask
    & in_range_mask
    & ~excluded_type_mask
)

n_complete_coordinate = int(complete_coordinate_mask.sum())
n_out_of_range = int((complete_coordinate_mask & ~in_range_mask).sum())
n_excluded_type_name = int(excluded_type_mask.sum())
n_valid = int(valid_cluster_mask.sum())
n_excluded = int(n_original - n_valid)
excluded_rate = n_excluded / n_original if n_original > 0 else np.nan

if n_valid == 0:
    raise RuntimeError(
        "应用坐标质量规则及type_name排除规则后，没有可用于聚类的样本。"
    )

data = numeric_data.loc[valid_cluster_mask].copy()
data["__orig_idx__"] = data.index
data = data.reset_index(drop=True)

orig_indices = data["__orig_idx__"].to_numpy()
X = data[FEATURE_COLS].to_numpy(dtype=float)
n_samples = int(len(data))

excluded_rows_df = build_exclusion_reason_table(
    frame=df,
    numeric_features=numeric_data,
    valid_mask=valid_cluster_mask,
)


# =========================================================
# 7. 缺失和样本审计
# =========================================================

missing_rows = []
for col in FEATURE_COLS:
    missing_n = int(numeric_data[col].isna().sum())
    out_of_range_n = 0

    if CHECK_COORDINATE_RANGE:
        if col.endswith("_x"):
            low, high = -COORDINATE_TOLERANCE, FIELD_LENGTH + COORDINATE_TOLERANCE
        else:
            low, high = -COORDINATE_TOLERANCE, FIELD_WIDTH + COORDINATE_TOLERANCE

        finite = numeric_data[col].notna()
        out_of_range_n = int(
            (finite & ~numeric_data[col].between(low, high, inclusive="both")).sum()
        )

    missing_rows.append({
        "variable": col,
        "source_column": resolved_feature_map[col],
        "initial_n": n_original,
        "missing_or_non_numeric_n": missing_n,
        "missing_or_non_numeric_rate": (
            missing_n / n_original if n_original > 0 else np.nan
        ),
        "out_of_pitch_range_n": out_of_range_n,
        "out_of_pitch_range_rate": (
            out_of_range_n / n_original if n_original > 0 else np.nan
        ),
    })

missing_audit_df = pd.DataFrame(missing_rows)

sample_audit_df = pd.DataFrame([
    {
        "scenario": SCENARIO_NAME,
        "initial_rows": n_original,
        "complete_coordinate_rows": n_complete_coordinate,
        "out_of_pitch_range_rows": n_out_of_range,
        "excluded_type_name_rows": n_excluded_type_name,
        "type_name_filter_column": TYPE_NAME_COL,
        "excluded_type_names": ", ".join(sorted(EXCLUDED_TYPE_NAMES)),
        "type_name_filter_rule": (
            "case-insensitive exact match; excluded before scaling and KMeans"
        ),
        "rows_used_for_clustering": n_valid,
        "excluded_rows": n_excluded,
        "excluded_rate": excluded_rate,
        "all_valid_rows_receive_K3_label": True,
        "features_used": ", ".join(FEATURE_COLS),
        "feature_sources": " | ".join(
            f"{canonical} <- {source}"
            for canonical, source in resolved_feature_map.items()
        ),
        "coordinate_standardisation": "StandardScaler fitted on valid Carry rows",
        "coordinate_direction_note": (
            "No additional coordinate flip is performed in this script; "
            "the input table must already use the intended common direction."
        ),
    }
])


# =========================================================
# 8. 标准化
# =========================================================

print(
    f"[FILTER] Excluded {n_excluded_type_name:,} rows where "
    f"{TYPE_NAME_COL} is in {sorted(EXCLUDED_TYPE_NAMES)}."
)
print(
    f"[FILTER] Remaining rows used for clustering: "
    f"{n_valid:,}/{n_original:,}."
)

print("[2/8] Standardizing Carry coordinates...")

scaler = StandardScaler()

Z = scaler.fit_transform(
    X
)

scaler_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean": scaler.mean_,
    "scale_sd": scaler.scale_,
    "variance": scaler.var_,
})


# =========================================================
# 9. 固定比较K=2、3、4、5
# =========================================================

print("[3/8] Running K=2,3,4,5...")

valid_k_values = [
    int(k)
    for k in K_VALUES
    if 2 <= int(k) < n_samples
]

if not valid_k_values:
    raise RuntimeError(
        "样本量不足，无法运行K=2至K=5。"
    )

metrics_rows = []
cluster_size_rows = []
cluster_center_rows = []
silhouette_cluster_rows = []

main_results = {}

for k in valid_k_values:
    print(f"    Main KMeans: K={k}")

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=MAIN_N_INIT,
        max_iter=MAX_ITER,
        random_state=MAIN_RANDOM_SEED,
        algorithm="lloyd",
    )

    raw_labels = model.fit_predict(
        Z
    )

    if len(np.unique(raw_labels)) < 2:
        print(
            f"    [WARNING] K={k}未产生两个以上cluster。"
        )
        continue

    centers_original_raw = scaler.inverse_transform(
        model.cluster_centers_
    )

    (
        ordered_labels,
        raw_to_ordered,
        ordered_centers,
        raw_order,
    ) = order_cluster_labels(
        labels=raw_labels,
        centers_original=centers_original_raw,
    )

    silhouette_sample_n = min(
        SILHOUETTE_SAMPLE_SIZE,
        n_samples,
    )

    if silhouette_sample_n < n_samples:
        rng = np.random.RandomState(MAIN_RANDOM_SEED + int(k))
        silhouette_indices = np.sort(
            rng.choice(
                np.arange(n_samples),
                size=silhouette_sample_n,
                replace=False,
            )
        )
    else:
        silhouette_indices = np.arange(n_samples)

    silhouette_sample_values = silhouette_samples(
        Z[silhouette_indices],
        raw_labels[silhouette_indices],
        metric="euclidean",
    )

    silhouette_value = float(
        np.mean(silhouette_sample_values)
    )

    ch_value = calinski_harabasz_score(
        Z,
        raw_labels,
    )

    db_value = davies_bouldin_score(
        Z,
        raw_labels,
    )

    counts = pd.Series(
        ordered_labels
    ).value_counts().sort_index()

    min_cluster_n = int(
        counts.min()
    )

    max_cluster_n = int(
        counts.max()
    )

    min_cluster_proportion = float(
        min_cluster_n / n_samples
    )

    max_cluster_proportion = float(
        max_cluster_n / n_samples
    )

    max_min_cluster_ratio = (
        max_cluster_n / min_cluster_n
        if min_cluster_n > 0
        else np.nan
    )

    meets_min_cluster_size = bool(
        min_cluster_n >= MIN_CLUSTER_SIZE
    )

    metrics_rows.append({
        "scenario": SCENARIO_NAME,
        "K": int(k),
        "n_samples": n_samples,

        "silhouette_score": float(
            silhouette_value
        ),

        "silhouette_sample_n": int(
            silhouette_sample_n
        ),

        "calinski_harabasz_score": float(
            ch_value
        ),

        "davies_bouldin_score": float(
            db_value
        ),

        "inertia": float(
            model.inertia_
        ),

        "n_iter": int(
            model.n_iter_
        ),

        "min_cluster_n": min_cluster_n,
        "max_cluster_n": max_cluster_n,

        "min_cluster_proportion": (
            min_cluster_proportion
        ),

        "max_cluster_proportion": (
            max_cluster_proportion
        ),

        "max_min_cluster_ratio": float(
            max_min_cluster_ratio
        ),

        "minimum_required_cluster_n": int(
            MIN_CLUSTER_SIZE
        ),

        "all_clusters_meet_minimum_n": (
            meets_min_cluster_size
        ),
    })

    data[f"cluster_raw_K{k}"] = (
        raw_labels.astype(int)
    )

    data[f"cluster_id_K{k}"] = (
        ordered_labels.astype(int)
    )

    for ordered_id in range(k):
        raw_id = raw_order[
            ordered_id
        ]

        cluster_mask = (
            ordered_labels == ordered_id
        )

        cluster_n = int(
            cluster_mask.sum()
        )

        cluster_proportion = float(
            cluster_n / n_samples
        )

        center_start_x = safe_float(
            ordered_centers[ordered_id, 0]
        )

        center_start_y = safe_float(
            ordered_centers[ordered_id, 1]
        )

        center_end_x = safe_float(
            ordered_centers[ordered_id, 2]
        )

        center_end_y = safe_float(
            ordered_centers[ordered_id, 3]
        )

        center_dx = (
            center_end_x - center_start_x
        )

        center_dy = (
            center_end_y - center_start_y
        )

        center_length = float(
            np.sqrt(
                center_dx ** 2
                + center_dy ** 2
            )
        )

        center_angle = float(
            np.degrees(
                np.arctan2(
                    center_dy,
                    center_dx,
                )
            )
        )

        cluster_x = X[
            cluster_mask
        ]

        cluster_z = Z[
            cluster_mask
        ]

        actual_dx = (
            cluster_x[:, 2]
            - cluster_x[:, 0]
        )

        actual_dy = (
            cluster_x[:, 3]
            - cluster_x[:, 1]
        )

        actual_length = np.sqrt(
            actual_dx ** 2
            + actual_dy ** 2
        )

        actual_angle = np.degrees(
            np.arctan2(
                actual_dy,
                actual_dx,
            )
        )

        center_z_raw = (
            model.cluster_centers_[raw_id]
        )

        distance_to_center_z = np.linalg.norm(
            cluster_z - center_z_raw,
            axis=1,
        )

        cluster_size_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "cluster_id": int(ordered_id),
            "raw_cluster_id": int(raw_id),

            "N": cluster_n,
            "proportion": cluster_proportion,

            "minimum_required_N": int(
                MIN_CLUSTER_SIZE
            ),

            "meets_minimum_N": bool(
                cluster_n >= MIN_CLUSTER_SIZE
            ),

            "mean_distance_to_center_z": float(
                np.mean(distance_to_center_z)
            ),

            "median_distance_to_center_z": float(
                np.median(distance_to_center_z)
            ),

            "p95_distance_to_center_z": float(
                np.percentile(
                    distance_to_center_z,
                    95,
                )
            ),
        })

        cluster_center_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "cluster_id": int(ordered_id),
            "raw_cluster_id": int(raw_id),

            "N": cluster_n,
            "proportion": cluster_proportion,

            "center_start_x": center_start_x,
            "center_start_y": center_start_y,

            "center_end_x": center_end_x,
            "center_end_y": center_end_y,

            "center_dx": float(center_dx),
            "center_dy": float(center_dy),

            "center_length": center_length,
            "center_angle_degree": center_angle,

            "actual_mean_start_x": float(
                np.mean(cluster_x[:, 0])
            ),

            "actual_mean_start_y": float(
                np.mean(cluster_x[:, 1])
            ),

            "actual_mean_end_x": float(
                np.mean(cluster_x[:, 2])
            ),

            "actual_mean_end_y": float(
                np.mean(cluster_x[:, 3])
            ),

            "actual_mean_length": float(
                np.mean(actual_length)
            ),

            "actual_median_length": float(
                np.median(actual_length)
            ),

            "actual_sd_length": float(
                np.std(
                    actual_length,
                    ddof=1,
                )
                if cluster_n > 1
                else 0.0
            ),

            "actual_mean_angle_degree": float(
                np.mean(actual_angle)
            ),

            "actual_median_angle_degree": float(
                np.median(actual_angle)
            ),

            "start_x_zone": (
                classify_longitudinal_zone(
                    center_start_x
                )
            ),

            "start_y_zone": (
                classify_lateral_zone(
                    center_start_y
                )
            ),

            "end_x_zone": (
                classify_longitudinal_zone(
                    center_end_x
                )
            ),

            "end_y_zone": (
                classify_lateral_zone(
                    center_end_y
                )
            ),
        })

    sampled_ordered_labels = ordered_labels[silhouette_indices]
    for ordered_id in range(k):
        cluster_sil = silhouette_sample_values[
            sampled_ordered_labels == ordered_id
        ]
        if len(cluster_sil) == 0:
            continue

        silhouette_cluster_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "cluster_id": int(ordered_id),
            "sample_n": int(len(cluster_sil)),
            "mean_silhouette": float(np.mean(cluster_sil)),
            "median_silhouette": float(np.median(cluster_sil)),
            "p05_silhouette": float(np.quantile(cluster_sil, 0.05)),
            "p25_silhouette": float(np.quantile(cluster_sil, 0.25)),
            "p75_silhouette": float(np.quantile(cluster_sil, 0.75)),
            "p95_silhouette": float(np.quantile(cluster_sil, 0.95)),
            "negative_silhouette_n": int(np.sum(cluster_sil < 0)),
            "negative_silhouette_proportion": float(np.mean(cluster_sil < 0)),
        })

    main_results[k] = {
        "model": model,
        "raw_labels": raw_labels,
        "ordered_labels": ordered_labels,
        "raw_to_ordered": raw_to_ordered,
        "raw_order": raw_order,
        "centers_original_raw": (
            centers_original_raw
        ),
        "centers_original_ordered": (
            ordered_centers
        ),
        "silhouette_indices": silhouette_indices,
        "silhouette_sample_values": silhouette_sample_values,
    }


metrics_df = pd.DataFrame(
    metrics_rows
).sort_values(
    "K"
).reset_index(
    drop=True
)

cluster_sizes_df = pd.DataFrame(
    cluster_size_rows
).sort_values(
    ["K", "cluster_id"]
).reset_index(
    drop=True
)

cluster_centers_df = pd.DataFrame(
    cluster_center_rows
).sort_values(
    ["K", "cluster_id"]
).reset_index(
    drop=True
)


silhouette_by_cluster_df = pd.DataFrame(
    silhouette_cluster_rows
).sort_values(
    ["K", "cluster_id"]
).reset_index(
    drop=True
)


# =========================================================
# 10. Inertia相对改善比例
# =========================================================

metrics_df[
    "inertia_reduction_from_previous_K"
] = np.nan

for i in range(1, len(metrics_df)):
    previous_inertia = safe_float(
        metrics_df.loc[
            i - 1,
            "inertia",
        ]
    )

    current_inertia = safe_float(
        metrics_df.loc[
            i,
            "inertia",
        ]
    )

    if (
        np.isfinite(previous_inertia)
        and previous_inertia > 0
        and np.isfinite(current_inertia)
    ):
        reduction = (
            previous_inertia
            - current_inertia
        ) / previous_inertia

        metrics_df.loc[
            i,
            "inertia_reduction_from_previous_K",
        ] = float(reduction)


# =========================================================
# 11. 随机种子稳定性分析
# =========================================================

print("[4/8] Running seed stability analysis...")

stability_run_rows = []
stability_summary_rows = []

for k in valid_k_values:
    if k not in main_results:
        continue

    print(f"    Seed stability: K={k}")

    reference_labels = main_results[
        k
    ]["raw_labels"]

    seed_label_results = []
    ari_vs_main_values = []

    for seed in STABILITY_SEEDS:
        stability_model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=STABILITY_N_INIT,
            max_iter=MAX_ITER,
            random_state=int(seed),
            algorithm="lloyd",
        )

        seed_labels = (
            stability_model.fit_predict(Z)
        )

        seed_label_results.append(
            seed_labels
        )

        ari_vs_main = adjusted_rand_score(
            reference_labels,
            seed_labels,
        )

        ari_vs_main_values.append(
            float(ari_vs_main)
        )

        seed_counts = np.bincount(
            seed_labels,
            minlength=k,
        )

        stability_run_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "seed": int(seed),

            "ARI_vs_main_seed_42": float(
                ari_vs_main
            ),

            "inertia": float(
                stability_model.inertia_
            ),

            "n_iter": int(
                stability_model.n_iter_
            ),

            "minimum_cluster_n": int(
                seed_counts.min()
            ),
        })

    pairwise_ari_values = []

    for i, j in combinations(
        range(len(seed_label_results)),
        2,
    ):
        pairwise_ari = adjusted_rand_score(
            seed_label_results[i],
            seed_label_results[j],
        )

        pairwise_ari_values.append(
            float(pairwise_ari)
        )

    stability_summary_rows.append({
        "scenario": SCENARIO_NAME,
        "K": int(k),

        "n_seed_runs": int(
            len(STABILITY_SEEDS)
        ),

        "mean_ARI_vs_main": float(
            np.mean(ari_vs_main_values)
        ),

        "sd_ARI_vs_main": float(
            np.std(
                ari_vs_main_values,
                ddof=1,
            )
            if len(ari_vs_main_values) > 1
            else 0.0
        ),

        "min_ARI_vs_main": float(
            np.min(ari_vs_main_values)
        ),

        "max_ARI_vs_main": float(
            np.max(ari_vs_main_values)
        ),

        "mean_pairwise_ARI": float(
            np.mean(pairwise_ari_values)
            if pairwise_ari_values
            else np.nan
        ),

        "sd_pairwise_ARI": float(
            np.std(
                pairwise_ari_values,
                ddof=1,
            )
            if len(pairwise_ari_values) > 1
            else 0.0
        ),

        "min_pairwise_ARI": float(
            np.min(pairwise_ari_values)
            if pairwise_ari_values
            else np.nan
        ),

        "max_pairwise_ARI": float(
            np.max(pairwise_ari_values)
            if pairwise_ari_values
            else np.nan
        ),
    })


stability_runs_df = pd.DataFrame(
    stability_run_rows
).sort_values(
    ["K", "seed"]
).reset_index(
    drop=True
)

stability_summary_df = pd.DataFrame(
    stability_summary_rows
).sort_values(
    "K"
).reset_index(
    drop=True
)


metrics_df = metrics_df.merge(
    stability_summary_df[
        [
            "K",
            "mean_ARI_vs_main",
            "sd_ARI_vs_main",
            "min_ARI_vs_main",
            "mean_pairwise_ARI",
            "sd_pairwise_ARI",
            "min_pairwise_ARI",
        ]
    ],
    on="K",
    how="left",
)


# =========================================================
# 12. 指标排名
# =========================================================

metrics_df["rank_silhouette"] = (
    metrics_df["silhouette_score"]
    .rank(
        ascending=False,
        method="min",
    )
)

metrics_df["rank_calinski_harabasz"] = (
    metrics_df["calinski_harabasz_score"]
    .rank(
        ascending=False,
        method="min",
    )
)

metrics_df["rank_davies_bouldin"] = (
    metrics_df["davies_bouldin_score"]
    .rank(
        ascending=True,
        method="min",
    )
)

metrics_df["rank_seed_stability"] = (
    metrics_df["mean_pairwise_ARI"]
    .rank(
        ascending=False,
        method="min",
    )
)

metrics_df["mean_statistical_rank"] = (
    metrics_df[
        [
            "rank_silhouette",
            "rank_calinski_harabasz",
            "rank_davies_bouldin",
            "rank_seed_stability",
        ]
    ].mean(axis=1)
)

metrics_df["overall_statistical_rank"] = (
    metrics_df["mean_statistical_rank"]
    .rank(
        ascending=True,
        method="min",
    )
)


# =========================================================
# 13. 写回原始表
# =========================================================

print("[5/8] Preparing Carry output tables...")

all_labels_df = df.copy()

for k in valid_k_values:
    ordered_col = f"cluster_id_K{k}"
    raw_col = f"cluster_raw_K{k}"

    all_labels_df[ordered_col] = pd.Series(
        pd.NA,
        index=all_labels_df.index,
        dtype="Int64",
    )
    all_labels_df[raw_col] = pd.Series(
        pd.NA,
        index=all_labels_df.index,
        dtype="Int64",
    )

    all_labels_df.loc[
        orig_indices,
        ordered_col,
    ] = data[ordered_col].astype(int).to_numpy()

    all_labels_df.loc[
        orig_indices,
        raw_col,
    ] = data[raw_col].astype(int).to_numpy()


main_k_df = None
k3_labels_only_df = None
main_ordered_col = f"cluster_id_K{MAIN_K}"

if (
    MAIN_K in valid_k_values
    and main_ordered_col in all_labels_df.columns
):
    # Assignment-distance audit for the main K.
    result_main = main_results[MAIN_K]
    ordered_centers_z = result_main["model"].cluster_centers_[
        result_main["raw_order"]
    ]

    distance_matrix = np.linalg.norm(
        Z[:, None, :] - ordered_centers_z[None, :, :],
        axis=2,
    )
    sorted_distance = np.sort(distance_matrix, axis=1)

    all_labels_df["K3_distance_to_assigned_center_z"] = np.nan
    all_labels_df["K3_second_nearest_center_distance_z"] = np.nan
    all_labels_df["K3_assignment_margin_z"] = np.nan
    all_labels_df["K3_assignment_distance_ratio"] = np.nan

    assigned_distance = distance_matrix[
        np.arange(n_samples),
        data[main_ordered_col].astype(int).to_numpy(),
    ]
    second_distance = sorted_distance[:, 1]
    margin_distance = second_distance - assigned_distance
    distance_ratio = np.divide(
        assigned_distance,
        second_distance,
        out=np.full_like(assigned_distance, np.nan, dtype=float),
        where=second_distance > 1e-12,
    )

    all_labels_df.loc[
        orig_indices,
        "K3_distance_to_assigned_center_z",
    ] = assigned_distance
    all_labels_df.loc[
        orig_indices,
        "K3_second_nearest_center_distance_z",
    ] = second_distance
    all_labels_df.loc[
        orig_indices,
        "K3_assignment_margin_z",
    ] = margin_distance
    all_labels_df.loc[
        orig_indices,
        "K3_assignment_distance_ratio",
    ] = distance_ratio

    # DML receives only rows that actually participated in clustering.
    main_k_df = all_labels_df.loc[
        all_labels_df[main_ordered_col].notna()
    ].copy()

    main_k_df[MAIN_CLUSTER_COL] = (
        main_k_df[main_ordered_col]
        .astype("Int64")
    )

    identifier_candidates = [
        "__source_file__", "match_id", "id", "index",
        "period", "timestamp", "minute", "second",
        "match_second", "player_name", "team",
    ]
    label_cols = [
        c for c in identifier_candidates
        if c in main_k_df.columns
    ] + [
        "cluster_id",
        f"cluster_id_K{MAIN_K}",
        f"cluster_raw_K{MAIN_K}",
        "K3_distance_to_assigned_center_z",
        "K3_second_nearest_center_distance_z",
        "K3_assignment_margin_z",
        "K3_assignment_distance_ratio",
    ]
    k3_labels_only_df = main_k_df[label_cols].copy()


# =========================================================
# 14. 绘图函数
# =========================================================

def plot_k_overview(
    k,
    ordered_labels,
    centers_df_k,
    metrics_row,
):
    colors = make_cluster_colors(k)

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    draw_pitch_shot_style(ax)

    plot_df = data[
        FEATURE_COLS
        + ["__orig_idx__"]
    ].copy()

    plot_df["cluster_id"] = (
        ordered_labels
    )

    legend_handles = []

    for cid in range(k):
        cluster_df = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = int(
            len(cluster_df)
        )

        sampled_df = sample_cluster_rows(
            cluster_df,
            PLOT_SAMPLE_MAX_PER_CLUSTER,
            PLOT_RANDOM_SEED + cid,
        )

        color = colors[cid]

        for row in sampled_df.itertuples(
            index=False
        ):
            x1 = float(
                getattr(row, START_X_COL)
            )

            y1 = float(
                getattr(row, START_Y_COL)
            )

            x2 = float(
                getattr(row, END_X_COL)
            )

            y2 = float(
                getattr(row, END_Y_COL)
            )

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                alpha=0.20,
                lw=1.05,
                zorder=4,
            )

        center_row = centers_df_k[
            centers_df_k["cluster_id"] == cid
        ]

        if not center_row.empty:
            center_row = (
                center_row.iloc[0]
            )

            start_x = float(
                center_row["center_start_x"]
            )

            start_y = float(
                center_row["center_start_y"]
            )

            end_x = float(
                center_row["center_end_x"]
            )

            end_y = float(
                center_row["center_end_y"]
            )

            ax.scatter(
                start_x,
                start_y,
                s=150,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=20,
            )

            ax.scatter(
                end_x,
                end_y,
                s=170,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=20,
            )

            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 4.0,
                },
                zorder=19,
            )

            ax.text(
                start_x,
                start_y + 2.5,
                f"C{cid}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
                zorder=21,
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=4,
                label=(
                    f"Cluster {cid} "
                    f"(N={cluster_n})"
                ),
            )
        )

    silhouette_value = float(
        metrics_row[
            "silhouette_score"
        ]
    )

    ch_value = float(
        metrics_row[
            "calinski_harabasz_score"
        ]
    )

    db_value = float(
        metrics_row[
            "davies_bouldin_score"
        ]
    )

    ax.set_title(
        f"{SCENARIO_NAME}: K={k}\n"
        f"Silhouette={silhouette_value:.3f}, "
        f"CH={ch_value:.1f}, "
        f"DB={db_value:.3f}"
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(k, 4),
        frameon=False,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_FIG_DIR,
        f"{SCENARIO_NAME}_K{k}_overview.png",
    )

    plt.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"    Wrote overview: {output_path}"
    )


def plot_each_cluster(
    k,
    ordered_labels,
    centers_df_k,
):
    colors = make_cluster_colors(k)

    plot_df = data[
        FEATURE_COLS
        + ["__orig_idx__"]
    ].copy()

    plot_df["cluster_id"] = (
        ordered_labels
    )

    k_dir = os.path.join(
        OUTPUT_FIG_DIR,
        f"K{k}_single_clusters",
    )

    os.makedirs(
        k_dir,
        exist_ok=True,
    )

    for cid in range(k):
        cluster_df = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = int(
            len(cluster_df)
        )

        sampled_df = sample_cluster_rows(
            cluster_df,
            PLOT_SAMPLE_MAX_PER_CLUSTER,
            PLOT_RANDOM_SEED + cid,
        )

        color = colors[cid]

        fig, ax = plt.subplots(
            figsize=(14, 9)
        )

        draw_pitch_shot_style(ax)

        for row in sampled_df.itertuples(
            index=False
        ):
            x1 = float(
                getattr(row, START_X_COL)
            )

            y1 = float(
                getattr(row, START_Y_COL)
            )

            x2 = float(
                getattr(row, END_X_COL)
            )

            y2 = float(
                getattr(row, END_Y_COL)
            )

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                alpha=LINE_ALPHA,
                lw=LINE_WIDTH,
                zorder=5,
            )

        center_row = centers_df_k[
            centers_df_k["cluster_id"] == cid
        ]

        if not center_row.empty:
            center_row = (
                center_row.iloc[0]
            )

            start_x = float(
                center_row["center_start_x"]
            )

            start_y = float(
                center_row["center_start_y"]
            )

            end_x = float(
                center_row["center_end_x"]
            )

            end_y = float(
                center_row["center_end_y"]
            )

            ax.scatter(
                start_x,
                start_y,
                s=180,
                color=color,
                edgecolor="black",
                linewidth=1.8,
                zorder=20,
            )

            ax.scatter(
                end_x,
                end_y,
                s=200,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.8,
                zorder=20,
            )

            ax.annotate(
                "",
                xy=(end_x, end_y),
                xytext=(start_x, start_y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 5,
                },
                zorder=19,
            )

        ax.set_title(
            f"{SCENARIO_NAME}: "
            f"K={k}, Cluster {cid}, "
            f"N={cluster_n}"
        )

        plt.tight_layout()

        output_path = os.path.join(
            k_dir,
            f"{SCENARIO_NAME}_K{k}_cluster_{cid}.png",
        )

        plt.savefig(
            output_path,
            dpi=500,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"    Wrote cluster plot: "
            f"{output_path}"
        )


def plot_silhouette_distribution(
    k,
    raw_labels,
    silhouette_average,
):
    rng = np.random.RandomState(
        MAIN_RANDOM_SEED + k
    )

    sample_n = min(
        SILHOUETTE_PLOT_SAMPLE_SIZE,
        n_samples,
    )

    if sample_n < n_samples:
        sample_indices = rng.choice(
            n_samples,
            size=sample_n,
            replace=False,
        )
    else:
        sample_indices = np.arange(
            n_samples
        )

    sampled_z = Z[
        sample_indices
    ]

    sampled_labels = raw_labels[
        sample_indices
    ]

    if len(np.unique(sampled_labels)) < 2:
        return

    silhouette_values = silhouette_samples(
        sampled_z,
        sampled_labels,
        metric="euclidean",
    )

    colors = make_cluster_colors(k)

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    y_lower = 10

    for raw_cluster_id in sorted(
        np.unique(sampled_labels)
    ):
        cluster_values = silhouette_values[
            sampled_labels == raw_cluster_id
        ]

        cluster_values.sort()

        cluster_size = len(
            cluster_values
        )

        y_upper = (
            y_lower + cluster_size
        )

        ax.fill_betweenx(
            np.arange(
                y_lower,
                y_upper,
            ),
            0,
            cluster_values,
            alpha=0.7,
            color=colors[
                int(raw_cluster_id) % k
            ],
        )

        ax.text(
            -0.05,
            y_lower + 0.5 * cluster_size,
            str(raw_cluster_id),
        )

        y_lower = y_upper + 10

    ax.axvline(
        x=silhouette_average,
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean silhouette = "
            f"{silhouette_average:.3f}"
        ),
    )

    ax.set_xlabel(
        "Silhouette coefficient"
    )

    ax.set_ylabel(
        "Clustered samples"
    )

    ax.set_title(
        f"{SCENARIO_NAME}: "
        f"Silhouette profile, K={k}\n"
        f"Sample N={sample_n}"
    )

    ax.set_yticks([])
    ax.legend(frameon=False)
    ax.grid(False)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_FIG_DIR,
        f"{SCENARIO_NAME}_K{k}_silhouette_profile.png",
    )

    plt.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"    Wrote silhouette plot: "
        f"{output_path}"
    )


def plot_metric_curve(
    metrics_table,
    metric_col,
    y_label,
    file_name,
):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        metrics_table["K"],
        metrics_table[metric_col],
        marker="o",
        linewidth=2,
        markersize=8,
    )

    for _, row in metrics_table.iterrows():
        value = safe_float(
            row[metric_col]
        )

        ax.text(
            row["K"],
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(
        "Number of clusters (K)"
    )

    ax.set_ylabel(
        y_label
    )

    ax.set_xticks(
        metrics_table["K"].tolist()
    )

    ax.set_title(
        f"{SCENARIO_NAME}: {y_label}"
    )

    ax.grid(False)

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_FIG_DIR,
        file_name,
    )

    plt.savefig(
        output_path,
        dpi=500,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# 15. 输出所有图片
# =========================================================

print("[6/8] Creating Carry clustering figures...")

for k in valid_k_values:
    if k not in main_results:
        continue

    result = main_results[k]

    metrics_row = metrics_df[
        metrics_df["K"] == k
    ].iloc[0]

    centers_df_k = cluster_centers_df[
        cluster_centers_df["K"] == k
    ].copy()

    plot_k_overview(
        k=k,
        ordered_labels=result[
            "ordered_labels"
        ],
        centers_df_k=centers_df_k,
        metrics_row=metrics_row,
    )

    plot_each_cluster(
        k=k,
        ordered_labels=result[
            "ordered_labels"
        ],
        centers_df_k=centers_df_k,
    )

    plot_silhouette_distribution(
        k=k,
        raw_labels=result[
            "raw_labels"
        ],
        silhouette_average=float(
            metrics_row[
                "silhouette_score"
            ]
        ),
    )


plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="silhouette_score",
    y_label="Silhouette score",
    file_name=(
        f"{SCENARIO_NAME}_metric_"
        f"silhouette.png"
    ),
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="calinski_harabasz_score",
    y_label="Calinski-Harabasz score",
    file_name=(
        f"{SCENARIO_NAME}_metric_"
        f"calinski_harabasz.png"
    ),
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="davies_bouldin_score",
    y_label="Davies-Bouldin score",
    file_name=(
        f"{SCENARIO_NAME}_metric_"
        f"davies_bouldin.png"
    ),
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="inertia",
    y_label="KMeans inertia",
    file_name=(
        f"{SCENARIO_NAME}_metric_"
        f"inertia.png"
    ),
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="mean_pairwise_ARI",
    y_label="Mean pairwise ARI",
    file_name=(
        f"{SCENARIO_NAME}_metric_"
        f"seed_stability_ARI.png"
    ),
)


# =========================================================
# 16. 返修检查表与K选择表
# =========================================================

k_selection_df = metrics_df.copy()
k_selection_df["analysis_role"] = np.where(
    k_selection_df["K"] == MAIN_K,
    "MAIN_ANALYSIS",
    "K_SENSITIVITY",
)
k_selection_df["automatic_K_selection"] = False
k_selection_df["selection_principle"] = (
    "K is evaluated jointly by separation, compactness, seed stability, "
    "minimum cluster size/proportion, trajectory plots, and tactical interpretability."
)
k_selection_df["K3_retention_statement"] = np.where(
    k_selection_df["K"] == MAIN_K,
    (
        "K=3 is retained as the prespecified main solution when it has adequate "
        "sample support and stability; it is not claimed to be selected solely "
        "because it has the best single internal-validity metric."
    ),
    (
        "Reported as a sensitivity solution to show whether lower or higher K "
        "changes separation, stability, and subgroup size."
    ),
)

reviewer_checklist_df = pd.DataFrame([
    {
        "revision_requirement": "Compare K=2,3,4,5",
        "evidence": "01_K_metrics and 02_K_selection",
        "status": "PASS" if set(K_VALUES).issubset(set(valid_k_values)) else "CHECK",
    },
    {
        "revision_requirement": "Silhouette score",
        "evidence": "01_K_metrics, 05_silhouette_by_cluster, figures",
        "status": "PASS" if metrics_df["silhouette_score"].notna().all() else "CHECK",
    },
    {
        "revision_requirement": "Calinski-Harabasz score",
        "evidence": "01_K_metrics",
        "status": "PASS" if metrics_df["calinski_harabasz_score"].notna().all() else "CHECK",
    },
    {
        "revision_requirement": "Davies-Bouldin score",
        "evidence": "01_K_metrics",
        "status": "PASS" if metrics_df["davies_bouldin_score"].notna().all() else "CHECK",
    },
    {
        "revision_requirement": "Inertia and marginal reduction",
        "evidence": "01_K_metrics and metric figure",
        "status": "PASS",
    },
    {
        "revision_requirement": "Minimum cluster N and proportion",
        "evidence": "01_K_metrics and 03_cluster_sizes",
        "status": "PASS",
    },
    {
        "revision_requirement": "Random-seed stability using ARI",
        "evidence": "06_stability_summary and 07_stability_runs",
        "status": "PASS" if not stability_summary_df.empty else "CHECK",
    },
    {
        "revision_requirement": "Exclude anomalous Dribble rows",
        "evidence": "08_sample_audit and 10_excluded_rows",
        "status": (
            "PASS"
            if TYPE_NAME_COL in df.columns
            and int(
                df.loc[valid_cluster_mask, TYPE_NAME_COL]
                .astype("string")
                .str.strip()
                .str.casefold()
                .isin(EXCLUDED_TYPE_NAMES)
                .sum()
            ) == 0
            else "CHECK"
        ),
    },
    {
        "revision_requirement": "Complete sample and coordinate audit",
        "evidence": "08_sample_audit, 09_missing_by_column, 10_excluded_rows",
        "status": "PASS",
    },
    {
        "revision_requirement": "Explicit feature and scaler documentation",
        "evidence": "11_feature_resolution and 12_scaler",
        "status": "PASS",
    },
    {
        "revision_requirement": "K=3 labels directly usable by DML",
        "evidence": "K3_main_for_DML plus standalone CSV/XLSX",
        "status": (
            "PASS"
            if main_k_df is not None
            and len(main_k_df) == n_valid
            and main_k_df[MAIN_CLUSTER_COL].notna().all()
            else "CHECK"
        ),
    },
    {
        "revision_requirement": "No labels guessed from another dataset",
        "evidence": "Input table is clustered directly",
        "status": "PASS",
    },
    {
        "revision_requirement": "Reproducible fixed random seed and n_init",
        "evidence": "14_run_info",
        "status": "PASS",
    },
])


# =========================================================
# 17. 指标说明
# =========================================================

metric_guide_df = pd.DataFrame([
    {
        "indicator": "Silhouette score",
        "preferred_direction": "Higher is better",
        "purpose": (
            "Measures within-cluster cohesion "
            "and between-cluster separation."
        ),
        "use_in_revision": (
            "Evaluate whether K=3 provides "
            "acceptable or near-best separation."
        ),
    },
    {
        "indicator": "Calinski-Harabasz score",
        "preferred_direction": "Higher is better",
        "purpose": (
            "Ratio of between-cluster dispersion "
            "to within-cluster dispersion."
        ),
        "use_in_revision": (
            "Provides an additional internal "
            "cluster-quality criterion."
        ),
    },
    {
        "indicator": "Davies-Bouldin score",
        "preferred_direction": "Lower is better",
        "purpose": (
            "Measures similarity between each "
            "cluster and its closest alternative."
        ),
        "use_in_revision": (
            "Checks whether the spatial clusters "
            "remain distinct."
        ),
    },
    {
        "indicator": "Inertia",
        "preferred_direction": (
            "Lower as K increases; inspect "
            "the marginal reduction"
        ),
        "purpose": (
            "Within-cluster sum of squared distances."
        ),
        "use_in_revision": (
            "Used for elbow-style comparison, "
            "not as the sole K-selection rule."
        ),
    },
    {
        "indicator": "Mean pairwise ARI",
        "preferred_direction": "Closer to 1 is better",
        "purpose": (
            "Measures agreement among clustering "
            "solutions from different random seeds."
        ),
        "use_in_revision": (
            "Shows whether clusters are stable "
            "rather than seed-dependent."
        ),
    },
    {
        "indicator": "Minimum cluster N",
        "preferred_direction": (
            "All clusters should contain "
            "sufficient observations"
        ),
        "purpose": (
            "Checks whether each cluster can "
            "support downstream DML estimation."
        ),
        "use_in_revision": (
            "Identifies whether larger K values "
            "create very small subgroups."
        ),
    },
])


# =========================================================
# 18. 运行信息
# =========================================================

run_info_df = pd.DataFrame([
    {
        "item": "run_datetime",
        "value": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    },
    {
        "item": "scenario",
        "value": SCENARIO_NAME,
    },
    {
        "item": "input_file",
        "value": INPUT_XLS,
    },
    {
        "item": "input_sheet",
        "value": str(SHEET_NAME),
    },
    {
        "item": "output_file",
        "value": OUTPUT_XLS,
    },
    {
        "item": "features",
        "value": ", ".join(FEATURE_COLS),
    },
    {
        "item": "feature_source_mapping",
        "value": " | ".join(
            f"{canonical} <- {source}"
            for canonical, source in resolved_feature_map.items()
        ),
    },
    {
        "item": "type_name_filter",
        "value": (
            f"{TYPE_NAME_COL} not in "
            f"{sorted(EXCLUDED_TYPE_NAMES)}; "
            "case-insensitive exact matching"
        ),
    },
    {
        "item": "excluded_type_name_rows",
        "value": str(n_excluded_type_name),
    },
    {
        "item": "rows_input",
        "value": str(n_original),
    },
    {
        "item": "rows_clustered",
        "value": str(n_valid),
    },
    {
        "item": "rows_excluded",
        "value": str(n_excluded),
    },
    {
        "item": "K3_standalone_xlsx",
        "value": OUTPUT_K3_XLSX,
    },
    {
        "item": "K3_standalone_csv",
        "value": OUTPUT_K3_CSV,
    },
    {
        "item": "K_values",
        "value": str(valid_k_values),
    },
    {
        "item": "main_K",
        "value": str(MAIN_K),
    },
    {
        "item": "minimum_cluster_size",
        "value": str(MIN_CLUSTER_SIZE),
    },
    {
        "item": "standardization",
        "value": "StandardScaler Z-score",
    },
    {
        "item": "KMeans_initialization",
        "value": "k-means++",
    },
    {
        "item": "main_n_init",
        "value": str(MAIN_N_INIT),
    },
    {
        "item": "main_random_seed",
        "value": str(MAIN_RANDOM_SEED),
    },
    {
        "item": "max_iter",
        "value": str(MAX_ITER),
    },
    {
        "item": "stability_seeds",
        "value": str(STABILITY_SEEDS),
    },
    {
        "item": "stability_n_init",
        "value": str(STABILITY_N_INIT),
    },
    {
        "item": "silhouette_sample_size",
        "value": str(SILHOUETTE_SAMPLE_SIZE),
    },
    {
        "item": "python_version",
        "value": sys.version,
    },
    {
        "item": "platform",
        "value": platform.platform(),
    },
    {
        "item": "numpy_version",
        "value": np.__version__,
    },
    {
        "item": "pandas_version",
        "value": pd.__version__,
    },
    {
        "item": "matplotlib_version",
        "value": matplotlib.__version__,
    },
    {
        "item": "scikit_learn_version",
        "value": sklearn.__version__,
    },
])


# =========================================================
# 19. 输出Excel
# =========================================================

print("[7/8] Writing Carry clustering Excel...")

excel_tables = [
    ("00_reviewer_checklist", reviewer_checklist_df),
    ("01_K_metrics", metrics_df),
    ("02_K_selection", k_selection_df),
    ("03_cluster_sizes", cluster_sizes_df),
    ("04_cluster_centers", cluster_centers_df),
    ("05_silhouette_by_cluster", silhouette_by_cluster_df),
    ("06_stability_summary", stability_summary_df),
    ("07_stability_runs", stability_runs_df),
    ("08_sample_audit", sample_audit_df),
    ("09_missing_by_column", missing_audit_df),
    ("10_excluded_rows", excluded_rows_df),
    ("11_feature_resolution", feature_resolution_df),
    ("12_scaler", scaler_df),
    ("13_metric_guide", metric_guide_df),
    ("14_run_info", run_info_df),
    ("all_K_labels", all_labels_df),
    (f"K{MAIN_K}_main_for_DML", main_k_df),
    (f"K{MAIN_K}_labels_only", k3_labels_only_df),
]

with pd.ExcelWriter(
    OUTPUT_XLS,
    engine="xlsxwriter",
) as writer:
    for sheet_name, table in excel_tables:
        if table is None:
            continue
        table.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31],
        )

    format_excel_workbook(
        writer,
        [
            (sheet_name[:31], table)
            for sheet_name, table in excel_tables
            if table is not None
        ],
    )


# =========================================================
# 20. 额外输出文件
# =========================================================

all_labels_df.to_csv(
    OUTPUT_ALL_LABELS_CSV,
    index=False,
    encoding="utf-8-sig",
)

if main_k_df is not None:
    main_k_df.to_csv(
        OUTPUT_K3_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        OUTPUT_K3_XLSX,
        engine="xlsxwriter",
    ) as writer:
        main_k_df.to_excel(
            writer,
            index=False,
            sheet_name="K3_main_for_DML",
        )
        format_excel_workbook(
            writer,
            [("K3_main_for_DML", main_k_df)],
        )

if k3_labels_only_df is not None:
    k3_labels_only_df.to_csv(
        OUTPUT_K3_LABELS_ONLY_CSV,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# 21. 控制台输出
# =========================================================

print("[8/8] Finished.")
print()
print("=" * 100)
print("Carry K sensitivity metrics")
print("=" * 100)

display_cols = [
    "K",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "inertia",
    "inertia_reduction_from_previous_K",
    "mean_pairwise_ARI",
    "min_cluster_n",
    "min_cluster_proportion",
    "all_clusters_meet_minimum_n",
    "overall_statistical_rank",
]

print(
    metrics_df[
        display_cols
    ]
    .sort_values("K")
    .to_string(index=False)
)

print()
print(
    f"[AUDIT] Input rows={n_original:,}; "
    f"Dribble/type exclusions={n_excluded_type_name:,}; "
    f"clustered rows={n_valid:,}; excluded rows={n_excluded:,}; "
    f"K3 label coverage among valid rows=100.00%"
)
print()
print(f"[OK] Excel：{OUTPUT_XLS}")
print(f"[OK] 全部标签CSV：{OUTPUT_ALL_LABELS_CSV}")
print(f"[OK] K=3 DML Excel：{OUTPUT_K3_XLSX}")
print(f"[OK] K=3 DML CSV：{OUTPUT_K3_CSV}")
print(f"[OK] K=3标签键表：{OUTPUT_K3_LABELS_ONLY_CSV}")
print(f"[OK] 图片目录：{OUTPUT_FIG_DIR}")

if main_k_df is not None:
    print(
        f"[OK] K={MAIN_K}主分析数据已输出，"
        f"cluster_id列可用于后续DML。"
    )

print()
print(
    "注意：overall_statistical_rank只作为统计参考。"
    "最终选择K时，还需要结合各cluster样本量、"
    "空间轨迹图和战术可解释性。"
)
