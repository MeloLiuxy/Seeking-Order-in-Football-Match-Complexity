# -*- coding: utf-8 -*-
r"""
Shot clustering sensitivity analysis: K = 2, 3, 4, 5
======================================================

输出内容：
1. K=2、3、4、5的聚类质量指标：
   - Silhouette score：越大越好
   - Calinski-Harabasz score：越大越好
   - Davies-Bouldin score：越小越好
   - Inertia：观察继续增加K后改善是否明显
2. 每个K下各cluster的样本量和比例。
3. 不同随机种子下的ARI稳定性。
4. 聚类仅使用location_x/location_y；同时报告每个cluster的起点中心、实际终点均值、射门长度和角度。
5. 每个K的总体聚类图。
6. 每个K、每个cluster的单独轨迹图。
7. 每个K的Silhouette分布图。
8. 各评价指标随K变化的折线图。
9. Excel中写入K=2、3、4、5的全部cluster标签。
10. 单独生成K=3主分析数据，可用于后续DML。

本代码只进行聚类敏感性分析，不运行DML和遗传算法。
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

SCENARIO_NAME = "Shot"

INPUT_XLS = r"D:\pipeline\Shot\起点shots_的聚类 - 副本 - 副本_with_success_def - 副本.xlsx"
SHEET_NAME = 0

OUTPUT_DIR = r"D:\pipeline\返修\聚类\射门cluster_sensitivity_K2_K5_仅location聚类"

OUTPUT_XLS = os.path.join(
    OUTPUT_DIR,
    "Shot_cluster_sensitivity_K2_K5.xlsx"
)

OUTPUT_FIG_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


# =========================================================
# 2. 聚类参数
# =========================================================

# 固定比较K=2、3、4、5
K_VALUES = [2, 3, 4, 5]

# 论文当前主分析使用K=3
MAIN_K = 3

# 主聚类随机种子
MAIN_RANDOM_SEED = 42

# 主聚类初始化次数
MAIN_N_INIT = 50

# 最大迭代次数
MAX_ITER = 500

# 评价每个cluster是否满足后续DML基本样本量
MIN_CLUSTER_SIZE = 120

# 稳定性分析使用的随机种子
STABILITY_SEEDS = list(range(20))

# 稳定性分析每次初始化次数
STABILITY_N_INIT = 10

# 数据量太大时，Silhouette随机抽样
SILHOUETTE_SAMPLE_SIZE = 5000

# Silhouette分布图最多使用多少样本
SILHOUETTE_PLOT_SAMPLE_SIZE = 3000

# 总体图中每个cluster最多画多少条线
PLOT_SAMPLE_MAX_PER_CLUSTER = 2000

# 画图抽样种子
PLOT_RANDOM_SEED = 42


# =========================================================
# 3. 聚类特征列
# =========================================================

# KMeans只使用射门发生位置。
CLUSTER_FEATURE_COLS = [
    "location_x",
    "location_y",
]

# end_location只用于轨迹绘图和cluster描述，不参与聚类。
PLOT_COORD_COLS = [
    "location_x",
    "location_y",
    "end_location_x",
    "end_location_y",
]

START_X_COL = "location_x"
START_Y_COL = "location_y"
END_X_COL = "end_location_x"
END_Y_COL = "end_location_y"

# K=3数据中供后续DML使用的列名
MAIN_CLUSTER_COL = "cluster_id"


# =========================================================
# 4. 球场配置
# =========================================================

PITCH_X_MIN = 0.0
PITCH_X_MAX = 120.0
PITCH_Y_MIN = 0.0
PITCH_Y_MAX = 80.0

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

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def classify_longitudinal_zone(x):
    """
    按120长度将球场分为三段。
    使用中性命名，避免提前绑定攻守方向。
    """
    if not np.isfinite(x):
        return ""

    if x < 40:
        return "X_0_40"

    if x < 80:
        return "X_40_80"

    return "X_80_120"


def classify_lateral_zone(y):
    """
    按80宽度将球场分为上、中、下区域。
    """
    if not np.isfinite(y):
        return ""

    if y < FIELD_WIDTH / 3:
        return "Upper"

    if y < 2 * FIELD_WIDTH / 3:
        return "Central"

    return "Lower"


def draw_pitch(ax):
    """
    绘制120×80球场。
    """

    ax.set_facecolor("white")

    line_color = "#000000"

    # 外框
    field = patches.Rectangle(
        (PITCH_X_MIN, PITCH_Y_MIN),
        PITCH_X_MAX - PITCH_X_MIN,
        PITCH_Y_MAX - PITCH_Y_MIN,
        linewidth=2.8,
        edgecolor=line_color,
        facecolor="none",
        zorder=10,
    )
    ax.add_patch(field)

    # 中线
    mid_x = FIELD_LENGTH / 2
    mid_y = FIELD_WIDTH / 2

    ax.plot(
        [mid_x, mid_x],
        [PITCH_Y_MIN, PITCH_Y_MAX],
        color=line_color,
        lw=2.2,
        zorder=11,
    )

    # 中圈
    center_circle = plt.Circle(
        (mid_x, mid_y),
        9.15,
        color=line_color,
        fill=False,
        lw=2.2,
        zorder=11,
    )
    ax.add_patch(center_circle)

    ax.plot(
        mid_x,
        mid_y,
        marker="o",
        color=line_color,
        markersize=5,
        zorder=12,
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


def order_shot_cluster_labels(labels, centers_original):
    """
    对Shot的cluster重新排序。

    聚类中心只有两个维度：
    - location_x
    - location_y

    优先按照射门起点Y坐标排序；起点Y相近时，再按照起点X排序。
    end_location不参与聚类和标签排序，只用于后续轨迹绘图与cluster描述。
    """

    k = centers_original.shape[0]

    raw_order = sorted(
        range(k),
        key=lambda raw_id: (
            centers_original[raw_id, 1],
            centers_original[raw_id, 0],
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
    """
    每个cluster最多抽取max_n条轨迹用于绘图。
    """

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
    """
    为指定K生成固定颜色。
    """

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

print("[1/8] Reading Shot data...")

df = pd.read_excel(
    INPUT_XLS,
    sheet_name=SHEET_NAME,
)

df.columns = [
    str(col).strip()
    for col in df.columns
]

required_cols = list(dict.fromkeys(
    CLUSTER_FEATURE_COLS + PLOT_COORD_COLS
))

missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"缺少必要坐标列：{missing_cols}"
    )

n_original = int(len(df))

numeric_data = df[
    required_cols
].copy()

for col in required_cols:
    numeric_data[col] = pd.to_numeric(
        numeric_data[col],
        errors="coerce",
    )

# 聚类只要求location_x/location_y完整。
complete_mask = numeric_data[
    CLUSTER_FEATURE_COLS
].notna().all(axis=1)

# end_location缺失的样本仍参与聚类，但不会绘制轨迹。
plot_complete_mask = numeric_data[
    PLOT_COORD_COLS
].notna().all(axis=1)

n_complete = int(
    complete_mask.sum()
)

n_plot_complete = int(
    (complete_mask & plot_complete_mask).sum()
)

n_excluded = int(
    n_original - n_complete
)

excluded_rate = (
    n_excluded / n_original
    if n_original > 0
    else np.nan
)

if n_complete == 0:
    raise RuntimeError(
        "没有具有完整location_x/location_y的射门样本。"
    )

data = numeric_data.loc[
    complete_mask
].copy()

data["__orig_idx__"] = data.index

data = data.reset_index(
    drop=True
)

orig_indices = data[
    "__orig_idx__"
].to_numpy()

# KMeans输入只有两个location维度。
X = data[
    CLUSTER_FEATURE_COLS
].to_numpy(
    dtype=float
)

# 四个坐标仅供轨迹绘图和cluster描述。
plot_coordinates = data[
    PLOT_COORD_COLS
].to_numpy(
    dtype=float
)

n_samples = int(
    len(data)
)


# =========================================================
# 7. 缺失和样本审计
# =========================================================

missing_rows = []

for col in required_cols:

    missing_n = int(
        numeric_data[col].isna().sum()
    )

    missing_rows.append({
        "variable": col,
        "initial_n": n_original,
        "missing_n": missing_n,
        "missing_rate": (
            missing_n / n_original
            if n_original > 0
            else np.nan
        ),
    })

missing_audit_df = pd.DataFrame(
    missing_rows
)

sample_audit_df = pd.DataFrame([
    {
        "scenario": SCENARIO_NAME,
        "initial_rows": n_original,
        "complete_location_rows_for_clustering": n_complete,
        "complete_start_end_rows_for_plotting": n_plot_complete,
        "excluded_missing_location_rows": n_excluded,
        "excluded_missing_location_rate": excluded_rate,
        "clustering_features_used": ", ".join(CLUSTER_FEATURE_COLS),
        "plot_only_coordinates": ", ".join(
            [c for c in PLOT_COORD_COLS if c not in CLUSTER_FEATURE_COLS]
        ),
    }
])


# =========================================================
# 8. 标准化
# =========================================================

print("[2/8] Standardizing Shot location coordinates only...")

scaler = StandardScaler()

Z = scaler.fit_transform(
    X
)

scaler_df = pd.DataFrame({
    "feature": CLUSTER_FEATURE_COLS,
    "mean": scaler.mean_,
    "scale_sd": scaler.scale_,
    "variance": scaler.var_,
})


# =========================================================
# 9. 固定比较K=2、3、4、5
# =========================================================

print("[3/8] Running K=2,3,4,5...")
print("    Clustering features: location_x, location_y only")
print("    end_location_x/end_location_y: plotting and descriptive output only")

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
    ) = order_shot_cluster_labels(
        labels=raw_labels,
        centers_original=centers_original_raw,
    )

    silhouette_sample_n = min(
        SILHOUETTE_SAMPLE_SIZE,
        n_samples,
    )

    silhouette_value = silhouette_score(
        Z,
        raw_labels,
        metric="euclidean",
        sample_size=(
            silhouette_sample_n
            if silhouette_sample_n < n_samples
            else None
        ),
        random_state=MAIN_RANDOM_SEED,
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

        # KMeans中心只表示location中心。
        center_start_x = safe_float(
            ordered_centers[ordered_id, 0]
        )

        center_start_y = safe_float(
            ordered_centers[ordered_id, 1]
        )

        cluster_x = X[
            cluster_mask
        ]

        cluster_z = Z[
            cluster_mask
        ]

        cluster_plot_coordinates = plot_coordinates[
            cluster_mask
        ]

        valid_trajectory_mask = np.isfinite(
            cluster_plot_coordinates
        ).all(axis=1)

        valid_trajectory_coordinates = (
            cluster_plot_coordinates[
                valid_trajectory_mask
            ]
        )

        trajectory_n = int(
            len(valid_trajectory_coordinates)
        )

        if trajectory_n > 0:

            # 仅用于绘图：该location-cluster内实际end_location的均值。
            center_end_x = float(
                np.mean(
                    valid_trajectory_coordinates[:, 2]
                )
            )

            center_end_y = float(
                np.mean(
                    valid_trajectory_coordinates[:, 3]
                )
            )

            actual_dx = (
                valid_trajectory_coordinates[:, 2]
                - valid_trajectory_coordinates[:, 0]
            )

            actual_dy = (
                valid_trajectory_coordinates[:, 3]
                - valid_trajectory_coordinates[:, 1]
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

        else:

            center_end_x = np.nan
            center_end_y = np.nan
            actual_length = np.array([], dtype=float)
            actual_angle = np.array([], dtype=float)

        center_dx = (
            center_end_x - center_start_x
            if np.isfinite(center_end_x)
            else np.nan
        )

        center_dy = (
            center_end_y - center_start_y
            if np.isfinite(center_end_y)
            else np.nan
        )

        center_length = (
            float(np.sqrt(center_dx ** 2 + center_dy ** 2))
            if np.isfinite(center_dx) and np.isfinite(center_dy)
            else np.nan
        )

        center_angle = (
            float(np.degrees(np.arctan2(center_dy, center_dx)))
            if np.isfinite(center_dx) and np.isfinite(center_dy)
            else np.nan
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
            "trajectory_N_with_complete_end": trajectory_n,
            "clustering_features": "location_x | location_y",
            "center_end_source": (
                "mean end_location within location-based cluster; "
                "not used in KMeans"
            ),

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

            "actual_mean_end_x": center_end_x,
            "actual_mean_end_y": center_end_y,

            "actual_mean_length": (
                float(np.mean(actual_length))
                if trajectory_n > 0
                else np.nan
            ),

            "actual_median_length": (
                float(np.median(actual_length))
                if trajectory_n > 0
                else np.nan
            ),

            "actual_sd_length": (
                float(np.std(actual_length, ddof=1))
                if trajectory_n > 1
                else (
                    0.0
                    if trajectory_n == 1
                    else np.nan
                )
            ),

            "actual_mean_angle_degree": (
                float(np.mean(actual_angle))
                if trajectory_n > 0
                else np.nan
            ),

            "actual_median_angle_degree": (
                float(np.median(actual_angle))
                if trajectory_n > 0
                else np.nan
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
    }


metrics_df = pd.DataFrame(
    metrics_rows
).sort_values("K").reset_index(drop=True)

cluster_sizes_df = pd.DataFrame(
    cluster_size_rows
).sort_values(
    ["K", "cluster_id"]
).reset_index(drop=True)

cluster_centers_df = pd.DataFrame(
    cluster_center_rows
).sort_values(
    ["K", "cluster_id"]
).reset_index(drop=True)


# =========================================================
# 10. 计算Inertia相对改善比例
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
# 11. 不同随机种子稳定性分析
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
).reset_index(drop=True)

stability_summary_df = pd.DataFrame(
    stability_summary_rows
).sort_values(
    "K"
).reset_index(drop=True)


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
# 12. 生成评价指标排名
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

print("[5/8] Preparing Shot output tables...")

all_labels_df = df.copy()

for k in valid_k_values:

    ordered_col = f"cluster_id_K{k}"
    raw_col = f"cluster_raw_K{k}"

    all_labels_df[ordered_col] = np.nan
    all_labels_df[raw_col] = np.nan

    all_labels_df.loc[
        orig_indices,
        ordered_col,
    ] = data[
        ordered_col
    ].to_numpy()

    all_labels_df.loc[
        orig_indices,
        raw_col,
    ] = data[
        raw_col
    ].to_numpy()


main_k_df = None

main_ordered_col = (
    f"cluster_id_K{MAIN_K}"
)

if (
    MAIN_K in valid_k_values
    and main_ordered_col
    in all_labels_df.columns
):

    main_k_df = all_labels_df.copy()

    main_k_df[
        MAIN_CLUSTER_COL
    ] = main_k_df[
        main_ordered_col
    ]


# =========================================================
# 14. 绘图函数
# =========================================================

def plot_k_overview(
    k,
    ordered_labels,
    centers_df_k,
    metrics_row,
):
    """
    每个K生成一张总体聚类图。
    """

    colors = make_cluster_colors(k)

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    draw_pitch(ax)

    plot_df = data[
        PLOT_COORD_COLS
        + ["__orig_idx__"]
    ].copy()

    plot_df["cluster_id"] = (
        ordered_labels
    )

    legend_handles = []

    for cid in range(k):

        cluster_df_all = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = int(
            len(cluster_df_all)
        )

        # 终点缺失的行仍参与location聚类，但不绘制轨迹。
        cluster_df = cluster_df_all.dropna(
            subset=PLOT_COORD_COLS
        ).copy()

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
                alpha=0.22,
                lw=1.1,
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

            if np.isfinite(end_x) and np.isfinite(end_y):

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
    """
    每个K下的每个cluster单独绘图。
    """

    colors = make_cluster_colors(k)

    plot_df = data[
        PLOT_COORD_COLS
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

        cluster_df_all = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = int(
            len(cluster_df_all)
        )

        # 终点缺失的行仍参与location聚类，但不绘制轨迹。
        cluster_df = cluster_df_all.dropna(
            subset=PLOT_COORD_COLS
        ).copy()

        sampled_df = sample_cluster_rows(
            cluster_df,
            PLOT_SAMPLE_MAX_PER_CLUSTER,
            PLOT_RANDOM_SEED + cid,
        )

        color = colors[cid]

        fig, ax = plt.subplots(
            figsize=(14, 9)
        )

        draw_pitch(ax)

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
                alpha=0.32,
                lw=1.4,
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

            if np.isfinite(end_x) and np.isfinite(end_y):

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
    """
    绘制每个K的Silhouette分布图。
    """

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

    if len(
        np.unique(sampled_labels)
    ) < 2:
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

        cluster_values = (
            silhouette_values[
                sampled_labels
                == raw_cluster_id
            ]
        )

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
            y_lower
            + 0.5 * cluster_size,
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
    """
    每个聚类评价指标单独输出折线图。
    """

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

print("[6/8] Creating Shot clustering figures...")

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
# 16. 指标说明
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
            "Within-cluster sum of squared "
            "distances."
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
# 17. 运行信息
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
        "item": "clustering_features",
        "value": ", ".join(CLUSTER_FEATURE_COLS),
    },
    {
        "item": "plot_only_coordinates",
        "value": ", ".join(
            [c for c in PLOT_COORD_COLS if c not in CLUSTER_FEATURE_COLS]
        ),
    },
    {
        "item": "clustering_rule",
        "value": (
            "KMeans uses location_x/location_y only; "
            "end_location is used only for trajectory plots and descriptive means"
        ),
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
# 18. 输出Excel
# =========================================================

print("[7/8] Writing Shot clustering Excel...")

with pd.ExcelWriter(
    OUTPUT_XLS,
    engine="xlsxwriter",
) as writer:

    metrics_df.to_excel(
        writer,
        index=False,
        sheet_name="01_K_metrics",
    )

    cluster_sizes_df.to_excel(
        writer,
        index=False,
        sheet_name="02_cluster_sizes",
    )

    cluster_centers_df.to_excel(
        writer,
        index=False,
        sheet_name="03_cluster_centers",
    )

    stability_summary_df.to_excel(
        writer,
        index=False,
        sheet_name="04_stability_summary",
    )

    stability_runs_df.to_excel(
        writer,
        index=False,
        sheet_name="05_stability_runs",
    )

    sample_audit_df.to_excel(
        writer,
        index=False,
        sheet_name="06_sample_audit",
    )

    missing_audit_df.to_excel(
        writer,
        index=False,
        sheet_name="07_missing_by_column",
    )

    scaler_df.to_excel(
        writer,
        index=False,
        sheet_name="08_scaler",
    )

    metric_guide_df.to_excel(
        writer,
        index=False,
        sheet_name="09_metric_guide",
    )

    run_info_df.to_excel(
        writer,
        index=False,
        sheet_name="10_run_info",
    )

    all_labels_df.to_excel(
        writer,
        index=False,
        sheet_name="all_K_labels",
    )

    if main_k_df is not None:

        main_k_df.to_excel(
            writer,
            index=False,
            sheet_name=f"K{MAIN_K}_main_for_DML",
        )


# =========================================================
# 19. 额外输出CSV
# =========================================================

all_labels_csv = os.path.join(
    OUTPUT_DIR,
    "Shot_all_K_cluster_labels.csv",
)

all_labels_df.to_csv(
    all_labels_csv,
    index=False,
    encoding="utf-8-sig",
)

if main_k_df is not None:

    main_k_csv = os.path.join(
        OUTPUT_DIR,
        f"Shot_K{MAIN_K}_main_for_DML.csv",
    )

    main_k_df.to_csv(
        main_k_csv,
        index=False,
        encoding="utf-8-sig",
    )


# =========================================================
# 20. 控制台输出
# =========================================================

print("[8/8] Finished.")
print()
print("=" * 100)
print("Shot K sensitivity metrics")
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
print(f"[OK] Excel：{OUTPUT_XLS}")
print(f"[OK] 全部标签CSV：{all_labels_csv}")
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
