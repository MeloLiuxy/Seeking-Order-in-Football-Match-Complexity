# -*- coding: utf-8 -*-
r"""
Pass clustering sensitivity analysis using end-location-corrected data: K = 2, 3, 4, 5
======================================================

用途：
1. 固定比较 K=2、3、4、5，不再自动只保留一个K。
2. 输出返修所需的聚类质量指标：
   - Silhouette score：越大越好
   - Calinski-Harabasz score：越大越好
   - Davies-Bouldin score：越小越好
   - Inertia：用于观察肘部变化
3. 输出每个K下各cluster的样本量和比例。
4. 输出不同随机种子下的ARI稳定性：
   - ARI越接近1，说明聚类越稳定
5. 输出每个cluster的原始尺度中心：
   - 起点X/Y
   - 终点X/Y
   - 轨迹长度
   - 轨迹角度
6. 输出：
   - 每个K的总体聚类图
   - 每个K、每个cluster的单独轨迹图
   - 每个K的Silhouette分布图
   - 各评价指标随K变化的折线图
7. Excel中保留K=2、3、4、5对应的cluster标签。
8. 另外生成K=3主分析数据表，cluster_id列可直接用于后续DML。

注意：
- 本代码只做聚类敏感性分析。
- 暂时不运行DML、placebo或遗传算法。
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
# 1. 路径与基本配置
# =========================================================

SCENARIO_NAME = "Pass"

INPUT_XLS = r""
SHEET_NAME = 0

OUTPUT_DIR = r""
OUTPUT_XLS = os.path.join(
    OUTPUT_DIR,
    "x"
)

OUTPUT_FIG_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

OUTPUT_K3_XLSX = os.path.join(
    OUTPUT_DIR,
    ""
)

OUTPUT_K3_CSV = os.path.join(
    OUTPUT_DIR,
    ""
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


# =========================================================
# 2. 聚类配置
# =========================================================

# 固定比较，不自动挑选
K_VALUES = [2, 3, 4, 5]

# 最终用于输出标签和图片的主随机种子
MAIN_RANDOM_SEED = 42

# 主聚类初始化次数
MAIN_N_INIT = 50

# KMeans最大迭代次数
MAX_ITER = 500

# 稳定性分析：使用多少个不同随机种子
STABILITY_SEEDS = list(range(20))

# 稳定性分析每次KMeans的初始化次数
STABILITY_N_INIT = 10

# Silhouette全样本过大时随机抽样
# 5000通常已足够，同时避免计算过慢
SILHOUETTE_SAMPLE_SIZE = 5000

# Silhouette分布图最多使用多少样本
SILHOUETTE_PLOT_SAMPLE_SIZE = 3000

# 每个cluster画图时最多抽取多少条轨迹
PLOT_SAMPLE_MAX_PER_CLUSTER = 2000

# 图形抽样随机种子
PLOT_RANDOM_SEED = 42


# =========================================================
# 3. 数据字段
# =========================================================

FEATURE_COLS = [
    "location_x",
    "location_y",
    "end_location_x",
    "end_location_y",
]

START_X_COL = "location_x"
START_Y_COL = "location_y"
END_X_COL = "end_location_x"
END_Y_COL = "end_location_y"

# K=3写回主分析时使用的列名
MAIN_CLUSTER_COL = "cluster_id"


# =========================================================
# 4. 球场与绘图配置
# =========================================================

FIELD_LENGTH = 120
FIELD_WIDTH = 80
BORDER_OFFSET = 5

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
    """安全转换为float。"""
    try:
        return float(value)
    except Exception:
        return np.nan


def classify_longitudinal_zone(x):
    """
    按120长度将X分为三个区域。
    这里使用中性名称，避免提前指定攻守方向。
    """
    if not np.isfinite(x):
        return ""
    if x < 40:
        return "X_0_40"
    if x < 80:
        return "X_40_80"
    return "X_80_120"


def classify_lateral_zone(y):
    """按80宽度将Y分为上、中、下三个区域。"""
    if not np.isfinite(y):
        return ""
    if y < FIELD_WIDTH / 3:
        return "Upper"
    if y < 2 * FIELD_WIDTH / 3:
        return "Central"
    return "Lower"


def draw_pitch_shot_style(ax):
    """绘制120×80球场。"""

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
    big_box_1 = patches.Rectangle(
        (0, (FIELD_WIDTH - 40.3) / 2),
        16.5,
        40.3,
        linewidth=2,
        edgecolor="#6666FF",
        facecolor="none",
        zorder=11,
    )
    big_box_2 = patches.Rectangle(
        (FIELD_LENGTH - 16.5, (FIELD_WIDTH - 40.3) / 2),
        16.5,
        40.3,
        linewidth=2,
        edgecolor="#6666FF",
        facecolor="none",
        zorder=11,
    )
    ax.add_patch(big_box_1)
    ax.add_patch(big_box_2)

    # 小禁区
    small_box_1 = patches.Rectangle(
        (0, (FIELD_WIDTH - 18.3) / 2),
        5.5,
        18.3,
        linewidth=2,
        edgecolor="#FF6666",
        facecolor="none",
        zorder=11,
    )
    small_box_2 = patches.Rectangle(
        (FIELD_LENGTH - 5.5, (FIELD_WIDTH - 18.3) / 2),
        5.5,
        18.3,
        linewidth=2,
        edgecolor="#FF6666",
        facecolor="none",
        zorder=11,
    )
    ax.add_patch(small_box_1)
    ax.add_patch(small_box_2)

    # 球门线
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
        markersize=5,
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
    对cluster编号进行稳定排序，避免KMeans随机给出的编号难以解释。

    排序顺序：
    1. 起点X
    2. 起点Y
    3. 终点X
    4. 终点Y

    返回：
    - ordered_labels
    - raw_to_ordered映射
    - ordered_centers
    - 原始cluster顺序
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
        [raw_to_ordered[int(raw)] for raw in labels],
        dtype=int,
    )

    ordered_centers = centers_original[raw_order, :]

    return (
        ordered_labels,
        raw_to_ordered,
        ordered_centers,
        raw_order,
    )


def sample_cluster_rows(cluster_df, max_n, seed):
    """每个cluster最多抽取max_n行用于绘图。"""

    if max_n is None or len(cluster_df) <= int(max_n):
        return cluster_df.copy()

    return cluster_df.sample(
        n=int(max_n),
        random_state=seed,
        replace=False,
    ).copy()


# =========================================================
# 6. 数据读取与审计
# =========================================================

print("[1/8] Reading data...")

df = pd.read_excel(
    INPUT_XLS,
    sheet_name=SHEET_NAME,
)

df.columns = [
    str(c).strip()
    for c in df.columns
]

missing_required_cols = [
    c for c in FEATURE_COLS
    if c not in df.columns
]

if missing_required_cols:
    raise ValueError(
        f"缺少必要坐标列：{missing_required_cols}"
    )

n_original = int(len(df))

numeric_data = df[FEATURE_COLS].copy()

for col in FEATURE_COLS:
    numeric_data[col] = pd.to_numeric(
        numeric_data[col],
        errors="coerce",
    )

complete_mask = numeric_data[FEATURE_COLS].notna().all(axis=1)

n_complete = int(complete_mask.sum())
n_excluded = int(n_original - n_complete)
excluded_rate = (
    n_excluded / n_original
    if n_original > 0
    else np.nan
)

if n_complete == 0:
    raise RuntimeError(
        "没有同时具有完整起点和终点坐标的样本。"
    )

data = numeric_data.loc[complete_mask].copy()
data["__orig_idx__"] = data.index
data = data.reset_index(drop=True)

# 原始行号，方便写回
orig_indices = data["__orig_idx__"].to_numpy()

# 特征矩阵
X = data[FEATURE_COLS].to_numpy(dtype=float)


# 缺失审计
missing_rows = []

for col in FEATURE_COLS:
    miss_n = int(numeric_data[col].isna().sum())

    missing_rows.append({
        "item": col,
        "initial_n": n_original,
        "missing_n": miss_n,
        "missing_rate": (
            miss_n / n_original
            if n_original > 0
            else np.nan
        ),
    })

missing_audit_df = pd.DataFrame(missing_rows)

overall_audit_df = pd.DataFrame([
    {
        "scenario": SCENARIO_NAME,
        "initial_rows": n_original,
        "complete_coordinate_rows": n_complete,
        "excluded_coordinate_rows": n_excluded,
        "excluded_coordinate_rate": excluded_rate,
        "features_used": ", ".join(FEATURE_COLS),
    }
])


# =========================================================
# 7. 标准化
# =========================================================

print("[2/8] Standardizing features...")

scaler = StandardScaler()
Z = scaler.fit_transform(X)

scaler_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean": scaler.mean_,
    "scale_sd": scaler.scale_,
    "variance": scaler.var_,
})


# =========================================================
# 8. 逐个K运行主聚类
# =========================================================

print("[3/8] Running K=2,3,4,5...")

metrics_rows = []
cluster_size_rows = []
cluster_center_rows = []

main_results = {}

n_samples = int(len(data))

valid_k_values = [
    k for k in K_VALUES
    if 2 <= int(k) < n_samples
]

if not valid_k_values:
    raise RuntimeError(
        "样本数不足，无法运行K=2至K=5。"
    )

for k in valid_k_values:

    print(f"    Running main KMeans: K={k}")

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=MAIN_N_INIT,
        max_iter=MAX_ITER,
        random_state=MAIN_RANDOM_SEED,
        algorithm="lloyd",
    )

    raw_labels = model.fit_predict(Z)

    if len(np.unique(raw_labels)) < 2:
        print(f"    [WARNING] K={k} only produced one cluster.")
        continue

    # 转回球场原始坐标
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

    # Silhouette大样本抽样
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

    # 每簇样本量
    counts = pd.Series(
        ordered_labels
    ).value_counts().sort_index()

    min_cluster_n = int(counts.min())
    max_cluster_n = int(counts.max())

    min_cluster_prop = float(
        min_cluster_n / n_samples
    )
    max_cluster_prop = float(
        max_cluster_n / n_samples
    )

    imbalance_ratio = (
        max_cluster_n / min_cluster_n
        if min_cluster_n > 0
        else np.nan
    )

    metrics_rows.append({
        "scenario": SCENARIO_NAME,
        "K": int(k),
        "n_samples": n_samples,
        "silhouette_score": float(silhouette_value),
        "silhouette_sample_n": int(silhouette_sample_n),
        "calinski_harabasz_score": float(ch_value),
        "davies_bouldin_score": float(db_value),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "min_cluster_n": min_cluster_n,
        "max_cluster_n": max_cluster_n,
        "min_cluster_proportion": min_cluster_prop,
        "max_cluster_proportion": max_cluster_prop,
        "max_min_cluster_ratio": float(imbalance_ratio),
    })

    # 保存标签
    data[f"cluster_raw_K{k}"] = raw_labels.astype(int)
    data[f"cluster_id_K{k}"] = ordered_labels.astype(int)

    # 每个cluster详细信息
    for ordered_id in range(k):

        raw_id = raw_order[ordered_id]
        cluster_mask = ordered_labels == ordered_id
        cluster_n = int(cluster_mask.sum())

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

        dx = center_end_x - center_start_x
        dy = center_end_y - center_start_y

        center_length = float(
            np.sqrt(dx ** 2 + dy ** 2)
        )

        center_angle = float(
            np.degrees(np.arctan2(dy, dx))
        )

        # 实际cluster样本
        cluster_x = X[cluster_mask]
        cluster_z = Z[cluster_mask]

        actual_dx = (
            cluster_x[:, 2] - cluster_x[:, 0]
        )
        actual_dy = (
            cluster_x[:, 3] - cluster_x[:, 1]
        )

        actual_length = np.sqrt(
            actual_dx ** 2 + actual_dy ** 2
        )

        # 到该簇中心的标准化距离
        center_z_raw = model.cluster_centers_[raw_id]

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
            "proportion": float(cluster_n / n_samples),
            "mean_distance_to_center_z": float(
                np.mean(distance_to_center_z)
            ),
            "median_distance_to_center_z": float(
                np.median(distance_to_center_z)
            ),
            "p95_distance_to_center_z": float(
                np.percentile(distance_to_center_z, 95)
            ),
        })

        cluster_center_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "cluster_id": int(ordered_id),
            "raw_cluster_id": int(raw_id),
            "N": cluster_n,
            "proportion": float(cluster_n / n_samples),

            "center_start_x": center_start_x,
            "center_start_y": center_start_y,
            "center_end_x": center_end_x,
            "center_end_y": center_end_y,

            "center_dx": float(dx),
            "center_dy": float(dy),
            "center_length": center_length,
            "center_angle_degree": center_angle,

            "actual_mean_length": float(
                np.mean(actual_length)
            ),
            "actual_median_length": float(
                np.median(actual_length)
            ),
            "actual_sd_length": float(
                np.std(actual_length, ddof=1)
                if cluster_n > 1
                else 0.0
            ),

            "start_x_zone": classify_longitudinal_zone(
                center_start_x
            ),
            "start_y_zone": classify_lateral_zone(
                center_start_y
            ),
            "end_x_zone": classify_longitudinal_zone(
                center_end_x
            ),
            "end_y_zone": classify_lateral_zone(
                center_end_y
            ),
        })

    main_results[k] = {
        "model": model,
        "raw_labels": raw_labels,
        "ordered_labels": ordered_labels,
        "raw_to_ordered": raw_to_ordered,
        "raw_order": raw_order,
        "centers_original_raw": centers_original_raw,
        "centers_original_ordered": ordered_centers,
    }


metrics_df = pd.DataFrame(metrics_rows).sort_values("K")
cluster_sizes_df = pd.DataFrame(
    cluster_size_rows
).sort_values(["K", "cluster_id"])

cluster_centers_df = pd.DataFrame(
    cluster_center_rows
).sort_values(["K", "cluster_id"])


# Inertia相对前一个K的下降比例
metrics_df["inertia_reduction_from_previous_K"] = np.nan

for i in range(1, len(metrics_df)):
    previous_inertia = metrics_df.iloc[i - 1]["inertia"]
    current_inertia = metrics_df.iloc[i]["inertia"]

    if previous_inertia > 0:
        reduction = (
            previous_inertia - current_inertia
        ) / previous_inertia

        metrics_df.loc[
            metrics_df.index[i],
            "inertia_reduction_from_previous_K",
        ] = float(reduction)


# =========================================================
# 9. 不同随机种子的聚类稳定性
# =========================================================

print("[4/8] Running seed stability analysis...")

stability_run_rows = []
stability_summary_rows = []

for k in valid_k_values:

    if k not in main_results:
        continue

    print(f"    Stability analysis: K={k}")

    reference_labels = main_results[k]["raw_labels"]

    seed_label_results = []

    for seed in STABILITY_SEEDS:

        stability_model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=STABILITY_N_INIT,
            max_iter=MAX_ITER,
            random_state=int(seed),
            algorithm="lloyd",
        )

        seed_labels = stability_model.fit_predict(Z)
        seed_label_results.append(seed_labels)

        ari_vs_reference = adjusted_rand_score(
            reference_labels,
            seed_labels,
        )

        stability_run_rows.append({
            "scenario": SCENARIO_NAME,
            "K": int(k),
            "seed": int(seed),
            "ARI_vs_main_seed_42": float(
                ari_vs_reference
            ),
            "inertia": float(
                stability_model.inertia_
            ),
            "n_iter": int(
                stability_model.n_iter_
            ),
        })

    # 所有随机种子结果两两比较
    pairwise_ari_values = []

    for i, j in combinations(
        range(len(seed_label_results)),
        2,
    ):
        ari_value = adjusted_rand_score(
            seed_label_results[i],
            seed_label_results[j],
        )
        pairwise_ari_values.append(
            float(ari_value)
        )

    ari_reference_values = [
        row["ARI_vs_main_seed_42"]
        for row in stability_run_rows
        if row["K"] == k
    ]

    stability_summary_rows.append({
        "scenario": SCENARIO_NAME,
        "K": int(k),
        "n_seed_runs": int(len(STABILITY_SEEDS)),

        "mean_ARI_vs_main": float(
            np.mean(ari_reference_values)
        ),
        "sd_ARI_vs_main": float(
            np.std(ari_reference_values, ddof=1)
            if len(ari_reference_values) > 1
            else 0.0
        ),
        "min_ARI_vs_main": float(
            np.min(ari_reference_values)
        ),
        "max_ARI_vs_main": float(
            np.max(ari_reference_values)
        ),

        "mean_pairwise_ARI": float(
            np.mean(pairwise_ari_values)
            if pairwise_ari_values
            else np.nan
        ),
        "sd_pairwise_ARI": float(
            np.std(pairwise_ari_values, ddof=1)
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
).sort_values(["K", "seed"])

stability_summary_df = pd.DataFrame(
    stability_summary_rows
).sort_values("K")


# 合并稳定性结果
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
# 10. 指标排名
# =========================================================

# Silhouette越大越好
metrics_df["rank_silhouette"] = (
    metrics_df["silhouette_score"]
    .rank(
        ascending=False,
        method="min",
    )
)

# CH越大越好
metrics_df["rank_calinski_harabasz"] = (
    metrics_df["calinski_harabasz_score"]
    .rank(
        ascending=False,
        method="min",
    )
)

# DB越小越好
metrics_df["rank_davies_bouldin"] = (
    metrics_df["davies_bouldin_score"]
    .rank(
        ascending=True,
        method="min",
    )
)

# ARI越大越好
metrics_df["rank_seed_stability"] = (
    metrics_df["mean_pairwise_ARI"]
    .rank(
        ascending=False,
        method="min",
    )
)

# 仅作为统计参考，不自动决定最终K
metrics_df["mean_statistical_rank"] = metrics_df[
    [
        "rank_silhouette",
        "rank_calinski_harabasz",
        "rank_davies_bouldin",
        "rank_seed_stability",
    ]
].mean(axis=1)

metrics_df["overall_statistical_rank"] = (
    metrics_df["mean_statistical_rank"]
    .rank(
        ascending=True,
        method="min",
    )
)


# =========================================================
# 11. 写回完整数据
# =========================================================

print("[5/8] Preparing output datasets...")

all_labels_df = df.copy()

for k in valid_k_values:

    ordered_col = f"cluster_id_K{k}"
    raw_col = f"cluster_raw_K{k}"

    all_labels_df[ordered_col] = np.nan
    all_labels_df[raw_col] = np.nan

    all_labels_df.loc[
        orig_indices,
        ordered_col,
    ] = data[ordered_col].to_numpy()

    all_labels_df.loc[
        orig_indices,
        raw_col,
    ] = data[raw_col].to_numpy()


# K=3主分析数据
k3_main_df = None

if 3 in valid_k_values and "cluster_id_K3" in all_labels_df.columns:

    k3_main_df = all_labels_df.loc[
        all_labels_df["cluster_id_K3"].notna()
    ].copy()

    k3_main_df[MAIN_CLUSTER_COL] = (
        k3_main_df["cluster_id_K3"]
        .astype("Int64")
    )


# =========================================================
# 12. 绘图函数
# =========================================================

def make_cluster_colors(k):
    """为指定K生成颜色。"""

    cmap = colormaps.get_cmap(
        "tab10" if k <= 10 else "tab20"
    )

    return {
        cid: cmap(cid % cmap.N)
        for cid in range(k)
    }


def plot_k_overview(
    k,
    ordered_labels,
    centers_df_k,
    metrics_row,
):
    """每个K生成一张总体轨迹图。"""

    colors = make_cluster_colors(k)

    fig, ax = plt.subplots(
        figsize=(15, 9.5)
    )

    draw_pitch_shot_style(ax)

    plot_df = data[
        FEATURE_COLS + ["__orig_idx__"]
    ].copy()

    plot_df["cluster_id"] = ordered_labels

    legend_handles = []

    for cid in range(k):

        cluster_df = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = len(cluster_df)

        sampled_df = sample_cluster_rows(
            cluster_df,
            PLOT_SAMPLE_MAX_PER_CLUSTER,
            PLOT_RANDOM_SEED + cid,
        )

        color = colors[cid]

        for row in sampled_df.itertuples(index=False):

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
                lw=1.1,
                zorder=4,
            )

        center_row = centers_df_k[
            centers_df_k["cluster_id"] == cid
        ]

        if not center_row.empty:

            center_row = center_row.iloc[0]

            sx = float(
                center_row["center_start_x"]
            )
            sy = float(
                center_row["center_start_y"]
            )
            ex = float(
                center_row["center_end_x"]
            )
            ey = float(
                center_row["center_end_y"]
            )

            ax.scatter(
                sx,
                sy,
                s=130,
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=20,
            )

            ax.scatter(
                ex,
                ey,
                s=150,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.5,
                zorder=20,
            )

            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 4.0,
                },
                zorder=19,
            )

            ax.text(
                sx,
                sy + 2.5,
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
                label=f"Cluster {cid} (N={cluster_n})",
            )
        )

    silhouette_value = float(
        metrics_row["silhouette_score"]
    )
    ch_value = float(
        metrics_row["calinski_harabasz_score"]
    )
    db_value = float(
        metrics_row["davies_bouldin_score"]
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

    print(f"    Wrote overview: {output_path}")


def plot_each_cluster(
    k,
    ordered_labels,
    centers_df_k,
):
    """每个K下，每个cluster单独绘制。"""

    colors = make_cluster_colors(k)

    plot_df = data[
        FEATURE_COLS + ["__orig_idx__"]
    ].copy()

    plot_df["cluster_id"] = ordered_labels

    k_dir = os.path.join(
        OUTPUT_FIG_DIR,
        f"K{k}_single_clusters",
    )

    os.makedirs(k_dir, exist_ok=True)

    for cid in range(k):

        cluster_df = plot_df[
            plot_df["cluster_id"] == cid
        ].copy()

        cluster_n = int(len(cluster_df))

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

        for row in sampled_df.itertuples(index=False):

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
                alpha=0.30,
                lw=1.4,
                zorder=5,
            )

        center_row = centers_df_k[
            centers_df_k["cluster_id"] == cid
        ]

        if not center_row.empty:

            center_row = center_row.iloc[0]

            sx = float(
                center_row["center_start_x"]
            )
            sy = float(
                center_row["center_start_y"]
            )
            ex = float(
                center_row["center_end_x"]
            )
            ey = float(
                center_row["center_end_y"]
            )

            ax.scatter(
                sx,
                sy,
                s=180,
                color=color,
                edgecolor="black",
                linewidth=1.8,
                zorder=20,
                label="Cluster center start",
            )

            ax.scatter(
                ex,
                ey,
                s=200,
                marker="X",
                color=color,
                edgecolor="black",
                linewidth=1.8,
                zorder=20,
                label="Cluster center end",
            )

            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(sx, sy),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 5,
                },
                zorder=19,
            )

        ax.set_title(
            f"{SCENARIO_NAME}: K={k}, "
            f"Cluster {cid}, N={cluster_n}"
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

        print(f"    Wrote cluster plot: {output_path}")


def plot_silhouette_distribution(
    k,
    raw_labels,
    silhouette_average,
):
    """绘制每个K的Silhouette分布图。"""

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

    sampled_z = Z[sample_indices]
    sampled_labels = raw_labels[sample_indices]

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

        y_upper = y_lower + cluster_size

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
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
        f"{SCENARIO_NAME}: Silhouette profile, K={k}\n"
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

    print(f"    Wrote silhouette plot: {output_path}")


def plot_metric_curve(
    metrics_table,
    metric_col,
    y_label,
    file_name,
):
    """绘制单个聚类评价指标随K的变化。"""

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

        ax.text(
            row["K"],
            row[metric_col],
            f'{row[metric_col]:.3f}',
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(
        "Number of clusters (K)"
    )
    ax.set_ylabel(y_label)
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
# 13. 生成全部图片
# =========================================================

print("[6/8] Creating figures...")

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
        ordered_labels=result["ordered_labels"],
        centers_df_k=centers_df_k,
        metrics_row=metrics_row,
    )

    plot_each_cluster(
        k=k,
        ordered_labels=result["ordered_labels"],
        centers_df_k=centers_df_k,
    )

    plot_silhouette_distribution(
        k=k,
        raw_labels=result["raw_labels"],
        silhouette_average=float(
            metrics_row["silhouette_score"]
        ),
    )


# 聚类指标折线图
plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="silhouette_score",
    y_label="Silhouette score",
    file_name=f"{SCENARIO_NAME}_metric_silhouette.png",
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="calinski_harabasz_score",
    y_label="Calinski-Harabasz score",
    file_name=f"{SCENARIO_NAME}_metric_calinski_harabasz.png",
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="davies_bouldin_score",
    y_label="Davies-Bouldin score",
    file_name=f"{SCENARIO_NAME}_metric_davies_bouldin.png",
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="inertia",
    y_label="KMeans inertia",
    file_name=f"{SCENARIO_NAME}_metric_inertia.png",
)

plot_metric_curve(
    metrics_table=metrics_df,
    metric_col="mean_pairwise_ARI",
    y_label="Mean pairwise ARI",
    file_name=f"{SCENARIO_NAME}_metric_seed_stability_ARI.png",
)


# =========================================================
# 14. 指标说明与运行信息
# =========================================================

metric_guide_df = pd.DataFrame([
    {
        "indicator": "Silhouette score",
        "preferred_direction": "Higher is better",
        "purpose": (
            "Measures within-cluster cohesion and "
            "between-cluster separation."
        ),
        "use_in_revision": (
            "Compare whether K=3 has acceptable or "
            "near-best separation."
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
            "Provides a second internal-quality "
            "criterion beyond Silhouette."
        ),
    },
    {
        "indicator": "Davies-Bouldin score",
        "preferred_direction": "Lower is better",
        "purpose": (
            "Measures average similarity between "
            "each cluster and its most similar cluster."
        ),
        "use_in_revision": (
            "Checks whether clusters remain distinct."
        ),
    },
    {
        "indicator": "Inertia",
        "preferred_direction": (
            "Lower as K increases; inspect marginal reduction"
        ),
        "purpose": (
            "Within-cluster sum of squared distances."
        ),
        "use_in_revision": (
            "Used for elbow-style comparison, "
            "not as the only selection criterion."
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
            "Shows whether the cluster structure "
            "is reproducible rather than seed-dependent."
        ),
    },
    {
        "indicator": "Minimum cluster N/proportion",
        "preferred_direction": (
            "No cluster should be extremely small"
        ),
        "purpose": (
            "Evaluates whether each cluster has enough "
            "observations for downstream DML."
        ),
        "use_in_revision": (
            "Checks whether larger K creates small "
            "and analytically unstable subgroups."
        ),
    },
])


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
        "item": "K_values",
        "value": str(valid_k_values),
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
# 15. 输出Excel
# =========================================================

print("[7/8] Writing Excel output...")

with pd.ExcelWriter(
    OUTPUT_XLS,
    engine="xlsxwriter",
) as writer:

    # 核心返修结果
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

    overall_audit_df.to_excel(
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

    # 所有K的标签
    all_labels_df.to_excel(
        writer,
        index=False,
        sheet_name="all_K_labels",
    )

    # K=3主分析表
    if k3_main_df is not None:
        k3_main_df.to_excel(
            writer,
            index=False,
            sheet_name="K3_main_for_DML",
        )


# =========================================================
# 16. 额外保存CSV，避免Excel读取慢
# =========================================================

all_labels_csv = os.path.join(
    OUTPUT_DIR,
    "Pass_all_K_cluster_labels.csv",
)

all_labels_df.to_csv(
    all_labels_csv,
    index=False,
    encoding="utf-8-sig",
)

if k3_main_df is not None:

    k3_main_df.to_csv(
        OUTPUT_K3_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        OUTPUT_K3_XLSX,
        engine="xlsxwriter",
    ) as writer:
        k3_main_df.to_excel(
            writer,
            index=False,
            sheet_name="K3_main_for_DML",
        )


# =========================================================
# 17. 控制台结果
# =========================================================

print("[8/8] Finished.")
print()
print("=" * 80)
print("K sensitivity metrics")
print("=" * 80)

display_cols = [
    "K",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "inertia",
    "mean_pairwise_ARI",
    "min_cluster_n",
    "min_cluster_proportion",
    "overall_statistical_rank",
]

print(
    metrics_df[display_cols]
    .sort_values("K")
    .to_string(index=False)
)

print()
print(f"[OK] Excel: {OUTPUT_XLS}")
print(f"[OK] Labels CSV: {all_labels_csv}")
print(f"[OK] Figures: {OUTPUT_FIG_DIR}")

if k3_main_df is not None:
    print(f"[OK] K=3 DML Excel: {OUTPUT_K3_XLSX}")
    print(f"[OK] K=3 DML CSV: {OUTPUT_K3_CSV}")
    print(
        "[OK] K=3主分析数据已输出，"
        "其中cluster_id列可直接用于后续DML。"
    )

print()
print(
    "注意：overall_statistical_rank只用于统计参考，"
    "最终选择K时还应结合各cluster样本量和轨迹图的战术可解释性。"
)
