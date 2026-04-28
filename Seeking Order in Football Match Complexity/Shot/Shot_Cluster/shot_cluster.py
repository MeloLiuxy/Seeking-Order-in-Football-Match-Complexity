
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib import colormaps


# -----------------------------
# 全局字体大小设置（保持你原风格）
# -----------------------------
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


# -----------------------------
# 配置
# -----------------------------
INPUT_XLS  = r""
SHEET_NAME = 0

OUTPUT_XLS = r""

# ✅ 每个 cluster 输出一张图（目录）
OUTPUT_FIG_DIR = r""
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# 自动选择簇数（silhouette + 最小簇样本量约束）
AUTO_K   = True
K_RANGE  = range(2, 7)   # 2–6
FIXED_K  = 5

# 每簇最小样本量（用于避免太碎）
MIN_CLUSTER_SIZE = 120

# 每簇最多画多少条线（避免太密）
PLOT_SAMPLE_MAX_PER_CLUSTER = 2500

# 线段视觉
LINE_ALPHA = 0.35
LINE_WIDTH = 1.6

# 球场尺寸（120×80）
field_length = 120
field_width = 80
border_offset = 5

# 列名
need_loc = ["location_x", "location_y"]
need_end = ["end_location_x", "end_location_y"]  # 画线用（可缺失）
CLUSTER_COL_OUT = "cluster_id_loc"


# ============================================================================
# 读取数据
# ============================================================================
df = pd.read_excel(INPUT_XLS, sheet_name=SHEET_NAME)
df.columns = [str(c).strip() for c in df.columns]

missing_loc = [c for c in need_loc if c not in df.columns]
if missing_loc:
    raise ValueError(f"缺少必要列（聚类必须）：{missing_loc}")

# 转数值
for c in need_loc + need_end:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 只用 location 完整的行参与聚类
mask_loc = df[need_loc].notna().all(axis=1)
data = df.loc[mask_loc, need_loc].copy().reset_index(drop=False).rename(columns={"index": "__orig_idx__"})

if data.empty:
    raise RuntimeError("没有可用于聚类的完整 location 数据（location_x/location_y 全缺）。")


# ============================================================================
# 聚类只用 location (2D)
# ============================================================================
X = data[need_loc].values.astype(float)

scaler = StandardScaler()
Z = scaler.fit_transform(X)

if AUTO_K:
    n_samples = Z.shape[0]
    max_k_possible = min(max(K_RANGE), n_samples)
    valid_k_range = [k for k in K_RANGE if 2 <= k <= max_k_possible]

    if not valid_k_range:
        k_used = 1
        km = KMeans(n_clusters=k_used, n_init=20, random_state=42).fit(Z)
    else:
        best_model = None
        best_score = -1.0
        tried_candidates = []

        for k in valid_k_range:
            try:
                km_tmp = KMeans(n_clusters=k, n_init=20, random_state=42)
                labels_tmp = km_tmp.fit_predict(Z)
                uniq = np.unique(labels_tmp)
                if len(uniq) < 2:
                    continue

                counts = np.bincount(labels_tmp, minlength=k)
                tried_candidates.append((k, int(counts.min())))

                # 最小簇样本量约束
                if counts.min() < MIN_CLUSTER_SIZE:
                    continue

                score = float(silhouette_score(Z, labels_tmp))
                if score > best_score:
                    best_score = score
                    best_model = km_tmp
            except Exception:
                continue

        if best_model is None:
            # fallback：选 min cluster size 最大且 K 更小的那个
            if tried_candidates:
                tried_candidates_sorted = sorted(tried_candidates, key=lambda x: (-x[1], x[0]))
                fallback_k = int(tried_candidates_sorted[0][0])
                k_used = fallback_k
                km = KMeans(n_clusters=k_used, n_init=20, random_state=42).fit(Z)
            else:
                k_used = int(valid_k_range[0])
                km = KMeans(n_clusters=k_used, n_init=20, random_state=42).fit(Z)
        else:
            km = best_model
            k_used = int(km.n_clusters)
else:
    k_used = int(FIXED_K)
    km = KMeans(n_clusters=k_used, n_init=20, random_state=42).fit(Z)

labels = km.labels_.astype(int)
data[CLUSTER_COL_OUT] = labels

# 写回原表（保持原有顺序）
df[CLUSTER_COL_OUT] = np.nan
df.loc[data["__orig_idx__"].values, CLUSTER_COL_OUT] = data[CLUSTER_COL_OUT].values
df[CLUSTER_COL_OUT] = df[CLUSTER_COL_OUT].astype("Int64")


# ============================================================================
# 保存结果表
# ============================================================================
os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLS)), exist_ok=True)
with pd.ExcelWriter(OUTPUT_XLS, engine="xlsxwriter") as w:
    df.to_excel(w, index=False, sheet_name="shots_with_loc_clusters")

print(f"[OK] location-only 聚类完成，K={k_used}；写出：{OUTPUT_XLS}")


# ============================================================================
# ✅ 球场画法：对标你 shot 那套（白底、无网格、去边框、黑线）
# ============================================================================
def draw_pitch_shot_style(ax):
    ax.set_facecolor("white")
    LINE = "#000000"

    # 外框
    field = patches.Rectangle((0, 0), field_length, field_width,
                              linewidth=2.8, edgecolor=LINE,
                              facecolor="none", zorder=10)
    ax.add_patch(field)

    # 中线
    ax.plot([field_length/2, field_length/2], [0, field_width],
            color=LINE, lw=2.2, zorder=11)

    # 大禁区（蓝）
    big_box_1 = patches.Rectangle((0, (field_width-40.3)/2), 16.5, 40.3,
                                  linewidth=2, edgecolor="#6666FF",
                                  facecolor="none", zorder=11)
    big_box_2 = patches.Rectangle((field_length-16.5, (field_width-40.3)/2), 16.5, 40.3,
                                  linewidth=2, edgecolor="#6666FF",
                                  facecolor="none", zorder=11)
    ax.add_patch(big_box_1); ax.add_patch(big_box_2)

    # 小禁区（红）
    small_box_1 = patches.Rectangle((0, (field_width-18.3)/2), 5.5, 18.3,
                                    linewidth=2, edgecolor="#FF6666",
                                    facecolor="none", zorder=11)
    small_box_2 = patches.Rectangle((field_length-5.5, (field_width-18.3)/2), 5.5, 18.3,
                                    linewidth=2, edgecolor="#FF6666",
                                    facecolor="none", zorder=11)
    ax.add_patch(small_box_1); ax.add_patch(small_box_2)

    # 球门线
    goal_y1 = (field_width-7.32)/2
    goal_y2 = (field_width+7.32)/2
    ax.plot([0, 0], [goal_y1, goal_y2], color="#FF6666", lw=5, zorder=11)
    ax.plot([field_length, field_length], [goal_y1, goal_y2], color="#FF6666", lw=5, zorder=11)

    # 中圈
    center_circle = plt.Circle((field_length/2, field_width/2), 9.15,
                               color=LINE, fill=False, lw=2.2, zorder=11)
    ax.add_patch(center_circle)
    ax.plot(field_length/2, field_width/2, marker="o",
            color=LINE, markersize=6, zorder=12)

    # 罚球点
    ax.add_patch(patches.Circle((11, field_width/2), radius=0.2, color=LINE, zorder=11))
    ax.add_patch(patches.Circle((field_length-11, field_width/2), radius=0.2, color=LINE, zorder=11))

    # 角旗弧线
    corner_radius = 1.0
    corners = [(0, 0, 0, 90), (0, field_width, 270, 360),
               (field_length, 0, 90, 180), (field_length, field_width, 180, 270)]
    for (x, y, t1, t2) in corners:
        ax.add_patch(patches.Arc((x, y), 2*corner_radius, 2*corner_radius,
                                 angle=0, theta1=t1, theta2=t2,
                                 color=LINE, lw=2, zorder=11))

    ax.set_xlim(-border_offset-1, field_length + border_offset+1)
    ax.set_ylim(-border_offset-1, field_width + border_offset+1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


# ============================================================================
# 每簇单独输出图：画线 location -> end_location（若 end_location 缺失则画点）
# ============================================================================
has_end = all(c in df.columns for c in need_end)

# 全量可画数据（必须有 cluster + location）
plot_all = df.loc[
    df[CLUSTER_COL_OUT].notna() &
    df["location_x"].notna() & df["location_y"].notna()
].copy()

if has_end:
    plot_all = plot_all.loc[
        plot_all["end_location_x"].notna() & plot_all["end_location_y"].notna()
    ].copy()

if plot_all.empty:
    raise RuntimeError("没有可画图的数据（可能 end_location 全缺，或 cluster 全缺）。")

# 颜色映射（全簇一致）
all_cids = sorted(df[CLUSTER_COL_OUT].dropna().unique().tolist())
cmap = colormaps.get_cmap("tab20")
cluster_colors = {cid: cmap(i % cmap.N) for i, cid in enumerate(all_cids)}

# 输出每簇图
for cid in all_cids:
    sub_all = plot_all[plot_all[CLUSTER_COL_OUT] == cid].copy()

    if sub_all.empty:
        continue

    # 每簇采样
    plot_sub = sub_all
    if (PLOT_SAMPLE_MAX_PER_CLUSTER is not None) and (len(plot_sub) > int(PLOT_SAMPLE_MAX_PER_CLUSTER)):
        plot_sub = plot_sub.sample(int(PLOT_SAMPLE_MAX_PER_CLUSTER), random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch_shot_style(ax)

    col = cluster_colors[cid]

    if has_end:
        for _, r in plot_sub.iterrows():
            x1, y1 = float(r["location_x"]), float(r["location_y"])
            x2, y2 = float(r["end_location_x"]), float(r["end_location_y"])
            ax.plot([x1, x2], [y1, y2], color=col, alpha=LINE_ALPHA, lw=LINE_WIDTH, zorder=5)
    else:
        ax.scatter(plot_sub["location_x"].values, plot_sub["location_y"].values,
                   s=18, color=col, alpha=0.55, zorder=6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FIG_DIR, f"shot_cluster_{int(cid)}.png")
    plt.savefig(out_path, dpi=500)
    plt.close(fig)

    print(f"[OK] Cluster {int(cid)} 图写出：{out_path}")

print(f"\n[DONE] 每簇单图输出完成：{OUTPUT_FIG_DIR}")
