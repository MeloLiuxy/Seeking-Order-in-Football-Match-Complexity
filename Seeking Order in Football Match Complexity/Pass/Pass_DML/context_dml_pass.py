# -*- coding: utf-8 -*-
"""
Pass · Cluster-wise DML with Full Paper-grade Diagnostics
========================================================

改动点：
1) Att 相关 KPI 不再作为 treatment（不进 D）
2) 但 Att KPI 仍作为协变量进入 X（即：att 始终在 X）
3) 新增 minute_bin / score_state 的 dummy 列作为额外情境控制变量进入 X：
   - minute_bin_31-60
   - minute_bin_61-90+
   - score_state_leading
   - score_state_trailing

其余逻辑不变：清理/overlap/multiD/robustness/oneD/stability/placebo/输出结构均保留。
"""

import os
import re
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.utils import resample


# =========================
# 路径与基本配置
# =========================
INPUT_PATH   = r""
SHEET_NAME   = 0
OUTPUT_XLSX  = r""

OUTCOME_COL  = "success_def"
CLUSTER_COL  = "cluster_id"

RANDOM_SEED    = 42
BASE_SPLITS    = 5

# 训练折下采样（用于拟合 nuisance；不会改变第二阶段回归样本）
DO_UNDERSAMPLE_DEFAULT = True
MAJ_KEEP_RATIO_DEFAULT = 0.5

# 数据门槛
MIN_SAMPLES   = 80
MIN_MINOR     = 12

# trimming（主结果默认 0.02；稳健性会跑 0）
TRIM_TOP_FRAC_DEFAULT = 0.02
MIN_RESID_SD  = 1e-10

# FDR
DO_FDR    = True
FDR_ALPHA = 0.10

# near-constant 判定：有效样本中 unique 数很少 或方差太小
NEAR_CONST_VAR_EPS = 1e-12
NEAR_CONST_UNIQUE_MAX = 2

# overlap flag 阈值（可调得更严）
FLAG_R2_GE = 0.90
FLAG_RESID_RATIO_LE = 0.30

# gps_bin_range：按 D_hat 分箱数量
GPS_BINS = 5

# 稳定性重复次数
STAB_REPS = 20
STAB_PVAL_CUT = 0.05

# placebo 次数（y 置换次数；越大越慢）
PLACEBO_Y_REPS = 50

# 可选：只对通过筛选的 KPI 做 D-置换 placebo（否则很慢）
PLACEBO_D_ONLY_ON_CANDIDATES = True


# =========================
# 新增：强制纳入的情境控制变量
# =========================
FORCED_CONTEXT_COLS = [
    "minute_bin_31-60",
    "minute_bin_61-90+",
    "score_state_leading",
    "score_state_trailing",
]

# 原始字符串标签列不进模型
RAW_CONTEXT_LABEL_COLS = [
    "minute_bin",
    "score_state",
]


# =========================
# 排除 meta 字段
# =========================
EXCLUDE_PATTERNS = [
    r"match_id", r"game_id", r"player_id", r"team_id", r"event_id",
    r"frame", r"second", r"minute", r"period", r"half", r"timestamp",
    r"^x$|^y$|^z$"
]
EXCLUDE_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in EXCLUDE_PATTERNS]

def looks_like_meta(name: str):
    name = str(name)

    # 白名单：这些虽然包含 minute，但必须进 X
    if name in FORCED_CONTEXT_COLS:
        return False

    # 原始标签列不进模型
    if name in RAW_CONTEXT_LABEL_COLS:
        return True

    for rgx in EXCLUDE_REGEXES:
        if rgx.search(name):
            return True
    return False

def is_L_col(c): return str(c).strip().endswith("(L)")
def is_E_col(c): return str(c).strip().endswith("(E)")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def uniq_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# =========================
# ✅ 识别 Att KPI（从 D 剔除，加入 X）
# =========================
ATT_HINT_REGEXES = [
    re.compile(r"DistToAttCentroid", re.IGNORECASE),
    re.compile(r"Area[_\s\-]*Att", re.IGNORECASE),
    re.compile(r"Spr[_\s\-]*Att", re.IGNORECASE),
    re.compile(r"Avg[_\s\-]*\d+[_\s\-]*Att", re.IGNORECASE),
    re.compile(r"Pre[_\s\-]*Att|Pressure[_\s\-]*Att", re.IGNORECASE),
]

def is_att_kpi(colname: str) -> bool:
    s = str(colname)
    for rgx in ATT_HINT_REGEXES:
        if rgx.search(s):
            return True
    sl = s.lower()
    # 兜底：含 att 但不含 def（避免把“def_*_att_speed”这类误判为进攻 KPI）
    if ("att" in sl) and ("def" not in sl):
        return True
    return False


# =========================
# FDR(BH)
# =========================
def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n); m = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        if val < m:
            m = val
        q[i] = min(m, 1.0)
    qvals = np.empty(n); qvals[order] = q
    return qvals, (qvals <= alpha)


# =========================
# 学习器
# =========================
def make_y_model(seed):
    base = RandomForestClassifier(
        n_estimators=600, min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed, n_jobs=-1
    )
    try:
        return CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=3)

def make_d_model(seed):
    return RandomForestRegressor(
        n_estimators=600, min_samples_leaf=5,
        random_state=seed, n_jobs=-1
    )

def choose_splits(n):
    if n < 50:
        return max(2, min(3, n-1))
    return max(2, min(BASE_SPLITS, n // 5, n-1))

def stratified_undersample(y, tr_idx, do_undersample, maj_keep_ratio, seed):
    if not do_undersample:
        return tr_idx
    y_tr = y[tr_idx]
    cls, counts = np.unique(y_tr, return_counts=True)
    if len(cls) < 2:
        return tr_idx
    maj = cls[np.argmax(counts)]
    maj_mask = (y_tr == maj)
    maj_idx = tr_idx[maj_mask]
    min_idx = tr_idx[~maj_mask]
    keep_maj = int(len(maj_idx) * maj_keep_ratio)
    if keep_maj < 1:
        return tr_idx
    maj_idx_kept = resample(
        maj_idx, replace=False, n_samples=keep_maj, random_state=seed
    )
    tr_eff = np.concatenate([maj_idx_kept, min_idx])
    if len(np.unique(y[tr_eff])) < 2:
        return tr_idx
    return tr_eff


# =========================
# 清理：重复列（按内容）/ 常数列
# =========================
def series_digest(s: pd.Series) -> str:
    hv = pd.util.hash_pandas_object(s, index=False).values
    return hashlib.md5(hv.tobytes()).hexdigest()

def drop_duplicate_by_content(df: pd.DataFrame):
    dig2cols = {}
    for c in df.columns:
        dig2cols.setdefault(series_digest(df[c]), []).append(c)

    dup_groups = [cols for cols in dig2cols.values() if len(cols) >= 2]
    drop_cols = []
    report = []
    for g in dup_groups:
        keep = g[0]
        drops = g[1:]
        drop_cols.extend(drops)
        report.append({
            "group_size": len(g),
            "keep": keep,
            "drop": ", ".join(drops),
            "all_cols_in_group": ", ".join(g)
        })

    df2 = df.drop(columns=list(set(drop_cols)), errors="ignore").copy()
    rep_df = pd.DataFrame(report) if report else pd.DataFrame([{"msg": "No duplicate-by-content columns found."}])
    return df2, rep_df

def find_near_constant_cols(sub_df: pd.DataFrame, cols: list):
    bad = []
    for c in cols:
        s = to_num(sub_df[c])
        s = s.dropna()
        if len(s) == 0:
            bad.append((c, "all_nan"))
            continue
        if s.nunique() <= NEAR_CONST_UNIQUE_MAX:
            bad.append((c, f"nunique<= {NEAR_CONST_UNIQUE_MAX}"))
            continue
        if np.nanvar(s.values) <= NEAR_CONST_VAR_EPS:
            bad.append((c, f"var<= {NEAR_CONST_VAR_EPS}"))
            continue
    return bad


# =========================
# DML：核心函数（支持变体）
# =========================
def dml_multi_once(
    sub: pd.DataFrame,
    y_col: str,
    D_cols: list,
    X_cols: list,
    block_label: str,
    seed: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    if not D_cols:
        return None, None, {"err": f"{block_label}: no D_cols"}

    cols = [y_col] + D_cols + X_cols
    cols = uniq_keep_order(cols)
    tmp = sub[cols].copy()

    tmp[y_col] = to_num(tmp[y_col])
    for c in D_cols:
        tmp[c] = to_num(tmp[c])

    tmp = tmp.dropna(subset=[y_col] + D_cols)

    # X 数值化 + 中位数填补；剔除全 NaN 的 X
    kept_x = []
    for c in X_cols:
        tmp[c] = to_num(tmp[c])
        med = np.nanmedian(tmp[c].values)
        if np.isnan(med):
            continue
        tmp[c] = tmp[c].fillna(med)
        kept_x.append(c)
    X_cols = kept_x

    n = len(tmp)
    if n < MIN_SAMPLES:
        return None, None, {"err": f"{block_label}: n={n} too small"}

    y = (tmp[y_col].values > 0.5).astype(int)
    pos = int(y.sum()); neg = n - pos
    if min(pos, neg) < MIN_MINOR:
        return None, None, {"err": f"{block_label}: minority={min(pos,neg)} too small"}

    D = tmp[D_cols].values.astype(float)
    X = tmp[X_cols].values.astype(float) if X_cols else np.zeros((n, 1))
    K = D.shape[1]

    # cross-fitting
    k = choose_splits(n)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    y_hat = np.full(n, np.nan)
    D_hat = np.full((n, K), np.nan)
    r2y_list = []
    r2d_list_each = [[] for _ in range(K)]

    for tr, te in skf.split(X, y):
        tr_eff = stratified_undersample(y, tr, do_undersample, maj_keep_ratio, seed)
        if len(np.unique(y[tr_eff])) < 2:
            tr_eff = tr
            if len(np.unique(y[tr_eff])) < 2:
                continue

        Xtr, ytr, Dtr = X[tr_eff], y[tr_eff], D[tr_eff]
        Xte, yte, Dte = X[te], y[te], D[te]

        my = make_y_model(seed).fit(Xtr, ytr)
        y_hat_te = my.predict_proba(Xte)[:, 1]
        y_hat[te] = y_hat_te
        try:
            r2y_list.append(r2_score(yte, y_hat_te))
        except Exception:
            r2y_list.append(np.nan)

        for j in range(K):
            md = make_d_model(seed).fit(Xtr, Dtr[:, j])
            d_hat_te = md.predict(Xte)
            D_hat[te, j] = d_hat_te
            try:
                r2d_list_each[j].append(r2_score(Dte[:, j], d_hat_te))
            except Exception:
                r2d_list_each[j].append(np.nan)

    y_res = y - y_hat
    D_res = D - D_hat

    ok = np.isfinite(y_res) & np.all(np.isfinite(D_res), axis=1)
    if not np.any(ok):
        return None, None, {"err": f"{block_label}: no finite residuals"}

    # trimming：按每行 max|D_res|
    if trim_top_frac and trim_top_frac > 0:
        max_abs = np.max(np.abs(D_res[ok, :]), axis=1)
        thr = np.quantile(max_abs, 1 - trim_top_frac)
        ok2 = ok.copy()
        ok2[np.where(ok)[0]] = (max_abs <= thr)
        ok = ok2

    n_used = int(ok.sum())
    if n_used < max(30, int(0.3 * n)):
        return None, None, {"err": f"{block_label}: usable={n_used} too small"}

    sd_d_raw = np.nanstd(D, axis=0, ddof=0)
    sd_d_res = np.nanstd(D_res[ok, :], axis=0, ddof=1)
    if np.any(sd_d_res < MIN_RESID_SD):
        return None, None, {"err": f"{block_label}: some resid var too small"}

    # second stage
    Xols = sm.add_constant(D_res[ok, :], has_constant='add')

    w = None
    if use_wls_weights:
        pr = float(y[ok].mean())
        if 0 < pr < 1:
            w = np.where(y[ok] == 1, 1.0 / max(pr, 1e-6), 1.0 / max(1 - pr, 1e-6))
            w = w / np.mean(w)

    model = sm.WLS(y_res[ok], Xols, weights=w) if use_wls_weights else sm.OLS(y_res[ok], Xols)
    res_ols = model.fit(cov_type="HC3")

    params = res_ols.params[1:]
    ses    = res_ols.bse[1:]
    ci     = res_ols.conf_int(alpha=0.05)[1:, :]
    pvals  = res_ols.pvalues[1:]

    # overlap diagnostics table (+ gps_bin_range)
    overlap_rows = []
    for j, name in enumerate(D_cols):
        r2d_j = float(np.nanmean(r2d_list_each[j])) if r2d_list_each[j] else np.nan
        sd_raw_j = float(sd_d_raw[j])
        sd_res_j = float(sd_d_res[j])
        resid_ratio = (sd_res_j / sd_raw_j) if (sd_raw_j > 0 and np.isfinite(sd_raw_j)) else np.nan
        flag = bool((np.isfinite(r2d_j) and r2d_j >= FLAG_R2_GE) and (np.isfinite(resid_ratio) and resid_ratio <= FLAG_RESID_RATIO_LE))

        dh = D_hat[:, j]
        dj = D[:, j]
        gps_txt = ""
        try:
            mask = np.isfinite(dh) & np.isfinite(dj)
            if mask.sum() >= 20:
                q = np.quantile(dh[mask], np.linspace(0, 1, GPS_BINS + 1))
                q[0] -= 1e-12
                parts = []
                for b in range(GPS_BINS):
                    m2 = mask & (dh >= q[b]) & (dh <= q[b+1])
                    if m2.sum() < 5:
                        parts.append(f"bin{b+1}:n<5")
                    else:
                        lo = float(np.nanmin(dj[m2])); hi = float(np.nanmax(dj[m2]))
                        parts.append(f"bin{b+1}:[{lo:.3g},{hi:.3g}] n={int(m2.sum())}")
                gps_txt = " | ".join(parts)
        except Exception:
            gps_txt = "gps_failed"

        overlap_rows.append({
            "block": block_label,
            "treatment": name,
            "r2_d_hat": r2d_j,
            "sd_d_raw": sd_raw_j,
            "sd_d_resid_used": sd_res_j,
            "resid_sd_ratio": resid_ratio,
            "overlap_flag": flag,
            "gps_bin_range": gps_txt,
            "n_total": int(n),
            "n_used": int(n_used),
            "pos_rate": float(pos / n),
            "r2_y_hat": float(np.nanmean(r2y_list)),
        })
    overlap_df = pd.DataFrame(overlap_rows)

    rows = []
    for j, name in enumerate(D_cols):
        theta_unit = float(params[j])
        se = float(ses[j])
        p = float(pvals[j])
        ci_low = float(ci[j, 0])
        ci_high = float(ci[j, 1])

        sd_raw_j = float(sd_d_raw[j])
        sd_res_j = float(sd_d_res[j])
        theta_1sd = theta_unit * sd_raw_j if sd_raw_j > 0 else np.nan

        df_approx = max(1, n_used - (K + 1))
        tval = theta_unit / se if se > 0 else np.nan
        rv_like = (tval**2) / (tval**2 + df_approx) if np.isfinite(tval) else np.nan

        r2d_j = float(overlap_df.loc[overlap_df["treatment"] == name, "r2_d_hat"].values[0])

        rows.append({
            "block": block_label,
            "treatment": name,
            "theta_per_unit": theta_unit,
            "se": se,
            "pval": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "theta_per_1sd": theta_1sd,
            "t_stat": tval,
            "rv_like_min_strength": rv_like,
            "n_total": int(n),
            "n_used": int(n_used),
            "n_pos": int(pos),
            "n_neg": int(neg),
            "pos_rate": float(pos / n),
            "r2_y_hat": float(np.nanmean(r2y_list)),
            "r2_d_hat": r2d_j,
            "sd_d_raw": sd_raw_j,
            "sd_d_resid_used": sd_res_j,
            "resid_sd_ratio": float(overlap_df.loc[overlap_df["treatment"] == name, "resid_sd_ratio"].values[0]),
            "overlap_flag": bool(overlap_df.loc[overlap_df["treatment"] == name, "overlap_flag"].values[0]),
            "trim_top_frac": float(trim_top_frac),
            "undersample": bool(do_undersample),
            "maj_keep_ratio": float(maj_keep_ratio),
            "second_stage": "WLS_HC3" if use_wls_weights else "OLS_HC3",
            "learners": "Y|X: RFc+cal; D|X: RFr"
        })

    results_df = pd.DataFrame(rows)
    meta = {
        "err": None,
        "n": int(n),
        "n_used": int(n_used),
        "K": int(K),
        "splits": int(k),
        "seed": int(seed),
        "trim": float(trim_top_frac),
        "undersample": bool(do_undersample),
        "wls": bool(use_wls_weights)
    }
    return results_df, overlap_df, meta


# =========================
# one-D DML：每次只估一个 KPI
# =========================
def dml_oneD_batch(
    sub: pd.DataFrame,
    y_col: str,
    D_cols: list,
    X_cols: list,
    block_label_prefix: str,
    seed: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    out = []
    for d in D_cols:
        res, ov, meta = dml_multi_once(
            sub=sub,
            y_col=y_col,
            D_cols=[d],
            X_cols=X_cols,
            block_label=f"{block_label_prefix}_oneD",
            seed=seed,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights,
        )
        if res is None:
            continue
        out.append(res.iloc[0].to_dict())
    return pd.DataFrame(out) if out else pd.DataFrame()


# =========================
# 稳定性
# =========================
def stability_analysis(
    sub, y_col, D_cols, X_cols, block_label,
    base_seed, reps,
    trim_top_frac, do_undersample, maj_keep_ratio, use_wls_weights,
):
    if not D_cols:
        return pd.DataFrame()

    store = {d: [] for d in D_cols}
    store_p = {d: [] for d in D_cols}

    for r in range(reps):
        seed = base_seed + 1000 + r * 13
        res, ov, meta = dml_multi_once(
            sub, y_col, D_cols, X_cols, block_label,
            seed=seed,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights
        )
        if res is None:
            continue
        for _, row in res.iterrows():
            d = row["treatment"]
            store[d].append(float(row["theta_per_1sd"]))
            store_p[d].append(float(row["pval"]))

    rows = []
    for d in D_cols:
        vals = np.array(store[d], dtype=float)
        pvs  = np.array(store_p[d], dtype=float)
        if vals.size == 0:
            continue
        sign_cons = float(np.mean(np.sign(vals) == np.sign(np.nanmedian(vals))))
        pass_p = float(np.mean(pvs < STAB_PVAL_CUT))
        rows.append({
            "block": block_label,
            "treatment": d,
            "n_runs": int(vals.size),
            "median_theta_per_1sd": float(np.nanmedian(vals)),
            "iqr_theta_per_1sd": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            "sign_consistency": sign_cons,
            "pass_rate_p<0.05": pass_p
        })
    return pd.DataFrame(rows).sort_values(["pass_rate_p<0.05", "sign_consistency"], ascending=False)


# =========================
# Placebo：置换 y（cluster 内）
# =========================
def placebo_y_test(
    sub, y_col, D_cols, X_cols, block_label,
    base_seed, reps,
    trim_top_frac, do_undersample, maj_keep_ratio, use_wls_weights,
):
    rng = np.random.RandomState(base_seed + 999)
    if not D_cols:
        return pd.DataFrame()

    real_res, _, meta = dml_multi_once(
        sub, y_col, D_cols, X_cols, block_label,
        seed=base_seed,
        trim_top_frac=trim_top_frac,
        do_undersample=do_undersample,
        maj_keep_ratio=maj_keep_ratio,
        use_wls_weights=use_wls_weights
    )
    if real_res is None:
        return pd.DataFrame([{"block": block_label, "err": "real_run_failed"}])

    real_p = real_res.set_index("treatment")["pval"].to_dict()

    p_counts = {d: 0 for d in D_cols}
    used = 0
    for r in range(reps):
        perm = sub.copy()
        yv = to_num(perm[y_col]).values
        if np.all(~np.isfinite(yv)):
            continue
        perm[y_col] = rng.permutation(perm[y_col].values)

        res, _, _ = dml_multi_once(
            perm, y_col, D_cols, X_cols, block_label + "_placeboY",
            seed=base_seed + 2000 + r,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights
        )
        if res is None:
            continue
        used += 1
        for _, row in res.iterrows():
            d = row["treatment"]
            if float(row["pval"]) <= float(real_p.get(d, 1.0)):
                p_counts[d] += 1

    rows = []
    for d in D_cols:
        if used == 0:
            continue
        rows.append({
            "block": block_label,
            "treatment": d,
            "real_pval": float(real_p.get(d, np.nan)),
            "placebo_reps_used": int(used),
            "placebo_p_le_real_rate": float(p_counts[d] / used)
        })
    return pd.DataFrame(rows).sort_values("placebo_p_le_real_rate", ascending=True)


# =========================
# 主程序
# =========================
def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

    # 检查情境控制列是否存在
    missing_context = [c for c in FORCED_CONTEXT_COLS if c not in df.columns]
    if missing_context:
        raise ValueError(f"输入文件缺少以下情境控制列：{missing_context}")

    df, dup_report = drop_duplicate_by_content(df)

    if OUTCOME_COL not in df.columns or CLUSTER_COL not in df.columns:
        raise ValueError("缺少必要列：success_def 或 cluster_id")

    all_L = [c for c in df.columns if is_L_col(c) and not looks_like_meta(c)]
    all_E = [c for c in df.columns if is_E_col(c) and not looks_like_meta(c)]

    clusters = [v for v in sorted(df[CLUSTER_COL].dropna().unique())]
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLSX)), exist_ok=True)

    runlog = []

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as w:
        dup_report.to_excel(w, index=False, sheet_name="00_dup_by_content")

        for cid in clusters:
            sub = df.loc[df[CLUSTER_COL] == cid].copy()
            n_rows = len(sub)

            if n_rows < MIN_SAMPLES:
                runlog.append({"cluster": cid, "status": "skip", "reason": f"n={n_rows} too small"})
                continue

            def usable(cols):
                keep = []
                for c in cols:
                    s = to_num(sub[c])
                    if s.notna().sum() > 0 and np.nanvar(s.values) > 0:
                        keep.append(c)
                return keep

            L_cols = usable(all_L)
            E_cols = usable(all_E)

            # cluster 内常数/近常数列报告（L/E 分开）
            nearL = find_near_constant_cols(sub, L_cols)
            nearE = find_near_constant_cols(sub, E_cols)
            near_df = pd.DataFrame(
                [{"cluster": cid, "col": c, "reason": r, "block": "L"} for c, r in nearL] +
                [{"cluster": cid, "col": c, "reason": r, "block": "E"} for c, r in nearE]
            )
            if not near_df.empty:
                near_df.to_excel(w, index=False, sheet_name=f"c{cid}_near_const")

            badL = {c for c, _ in nearL}
            badE = {c for c, _ in nearE}
            L_cols = [c for c in L_cols if c not in badL]
            E_cols = [c for c in E_cols if c not in badE]

            # ========= 构造 X：背景变量 =========
            bg_cols = []
            for c in sub.columns:
                if c in [OUTCOME_COL, CLUSTER_COL]:
                    continue
                if is_L_col(c) or is_E_col(c):
                    continue
                if looks_like_meta(c):
                    continue
                bg_cols.append(c)

            # 强制加入情境控制变量
            bg_cols = uniq_keep_order(bg_cols + [c for c in FORCED_CONTEXT_COLS if c in sub.columns])
            bg_cols = usable(bg_cols)

            # ========= L block =========
            if len(L_cols) >= 1:
                # ✅ Att-L 从 D 移除，并加入 X
                L_att = [c for c in L_cols if is_att_kpi(c)]
                L_nonatt = [c for c in L_cols if c not in set(L_att)]

                D_L = L_nonatt
                X_L = usable(uniq_keep_order(bg_cols + L_att))

                resL, ovL, metaL = dml_multi_once(
                    sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}",
                    seed=RANDOM_SEED,
                    trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                    do_undersample=DO_UNDERSAMPLE_DEFAULT,
                    maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                    use_wls_weights=True
                )

                if resL is None:
                    runlog.append({"cluster": cid, "status": "L_fail", "reason": metaL.get("err")})
                else:
                    outL = resL.copy()
                    if DO_FDR:
                        q, passed = fdr_bh(outL["pval"].values, alpha=FDR_ALPHA)
                        outL["qval_bh"] = q
                        outL["fdr_pass"] = passed
                    outL = outL.sort_values(["fdr_pass", "pval", "theta_per_1sd"], ascending=[False, True, False])
                    outL.to_excel(w, index=False, sheet_name=f"L{cid}_main")
                    ovL.to_excel(w, index=False, sheet_name=f"L{cid}_overlap")

                    resL_t0, _, _ = dml_multi_once(
                        sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}_trim0",
                        seed=RANDOM_SEED,
                        trim_top_frac=0.0,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )
                    if resL_t0 is not None:
                        resL_t0.to_excel(w, index=False, sheet_name=f"L{cid}_rob_trim0")

                    resL_nu, _, _ = dml_multi_once(
                        sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}_noUS",
                        seed=RANDOM_SEED,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=False,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )
                    if resL_nu is not None:
                        resL_nu.to_excel(w, index=False, sheet_name=f"L{cid}_rob_noUS")

                    resL_ols, _, _ = dml_multi_once(
                        sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}_OLS",
                        seed=RANDOM_SEED,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=False
                    )
                    if resL_ols is not None:
                        resL_ols.to_excel(w, index=False, sheet_name=f"L{cid}_rob_OLS")

                    oneL = dml_oneD_batch(
                        sub, OUTCOME_COL, D_L, X_L,
                        block_label_prefix=f"L_cluster_{cid}",
                        seed=RANDOM_SEED,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )
                    if not oneL.empty:
                        if DO_FDR:
                            q, passed = fdr_bh(oneL["pval"].values, alpha=FDR_ALPHA)
                            oneL["qval_bh"] = q
                            oneL["fdr_pass"] = passed
                        oneL = oneL.sort_values(["fdr_pass", "pval"], ascending=[False, True])
                        oneL.to_excel(w, index=False, sheet_name=f"L{cid}_oneD")

                    stabL = stability_analysis(
                        sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}",
                        base_seed=RANDOM_SEED, reps=STAB_REPS,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )
                    if not stabL.empty:
                        stabL.to_excel(w, index=False, sheet_name=f"L{cid}_stability")

                    placL = placebo_y_test(
                        sub, OUTCOME_COL, D_L, X_L, f"L_cluster_{cid}",
                        base_seed=RANDOM_SEED, reps=PLACEBO_Y_REPS,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )
                    if not placL.empty:
                        placL.to_excel(w, index=False, sheet_name=f"L{cid}_placeboY")

                    cand = outL.copy()
                    cand["candidate_rule"] = (cand.get("fdr_pass", False) == True) & (cand["overlap_flag"] == False)
                    if not stabL.empty:
                        s = stabL.set_index("treatment")
                        cand["sign_consistency"] = cand["treatment"].map(s["sign_consistency"].to_dict())
                        cand["pass_rate_p<0.05"] = cand["treatment"].map(s["pass_rate_p<0.05"].to_dict())
                        cand["candidate_rule"] = cand["candidate_rule"] & (cand["sign_consistency"].fillna(0) >= 0.8)
                    cand = cand.sort_values(["candidate_rule", "fdr_pass", "pval"], ascending=[False, False, True])
                    cand.to_excel(w, index=False, sheet_name=f"L{cid}_candidates")

                    runlog.append({
                        "cluster": cid,
                        "block": "L",
                        "status": "done",
                        "n_rows": n_rows,
                        "n_D": len(D_L),
                        "n_X": len(X_L),
                        "n_context_in_X": int(sum([1 for c in FORCED_CONTEXT_COLS if c in X_L])),
                        "context_cols_in_X": ", ".join([c for c in FORCED_CONTEXT_COLS if c in X_L])
                    })

            # ========= E block =========
            if len(E_cols) >= 1:
                # ✅ Att-E 从 D 移除，并加入 X（注意：这会让 X_E 包含部分 E 列，仅限 Att-E）
                E_att = [c for c in E_cols if is_att_kpi(c)]
                E_nonatt = [c for c in E_cols if c not in set(E_att)]

                D_E = E_nonatt

                X_E = []
                for c in sub.columns:
                    if c in [OUTCOME_COL, CLUSTER_COL]:
                        continue
                    if is_E_col(c):
                        continue  # 原约束：不把其它 E 放进 X
                    if looks_like_meta(c):
                        continue
                    X_E.append(c)

                # 强制加入情境控制变量
                X_E = uniq_keep_order(X_E + [c for c in FORCED_CONTEXT_COLS if c in sub.columns])
                X_E = usable(uniq_keep_order(X_E + E_att))  # 只把 Att-E 例外加入 X

                if len(D_E) >= 1:
                    resE, ovE, metaE = dml_multi_once(
                        sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}",
                        seed=RANDOM_SEED,
                        trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                        do_undersample=DO_UNDERSAMPLE_DEFAULT,
                        maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                        use_wls_weights=True
                    )

                    if resE is None:
                        runlog.append({"cluster": cid, "status": "E_fail", "reason": metaE.get("err")})
                    else:
                        outE = resE.copy()
                        if DO_FDR:
                            q, passed = fdr_bh(outE["pval"].values, alpha=FDR_ALPHA)
                            outE["qval_bh"] = q
                            outE["fdr_pass"] = passed
                        outE = outE.sort_values(["fdr_pass", "pval", "theta_per_1sd"], ascending=[False, True, False])
                        outE.to_excel(w, index=False, sheet_name=f"E{cid}_main")
                        ovE.to_excel(w, index=False, sheet_name=f"E{cid}_overlap")

                        resE_t0, _, _ = dml_multi_once(
                            sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}_trim0",
                            seed=RANDOM_SEED,
                            trim_top_frac=0.0,
                            do_undersample=DO_UNDERSAMPLE_DEFAULT,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=True
                        )
                        if resE_t0 is not None:
                            resE_t0.to_excel(w, index=False, sheet_name=f"E{cid}_rob_trim0")

                        resE_nu, _, _ = dml_multi_once(
                            sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}_noUS",
                            seed=RANDOM_SEED,
                            trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                            do_undersample=False,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=True
                        )
                        if resE_nu is not None:
                            resE_nu.to_excel(w, index=False, sheet_name=f"E{cid}_rob_noUS")

                        resE_ols, _, _ = dml_multi_once(
                            sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}_OLS",
                            seed=RANDOM_SEED,
                            trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                            do_undersample=DO_UNDERSAMPLE_DEFAULT,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=False
                        )
                        if resE_ols is not None:
                            resE_ols.to_excel(w, index=False, sheet_name=f"E{cid}_rob_OLS")

                        oneE = dml_oneD_batch(
                            sub, OUTCOME_COL, D_E, X_E,
                            block_label_prefix=f"E_cluster_{cid}",
                            seed=RANDOM_SEED,
                            trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                            do_undersample=DO_UNDERSAMPLE_DEFAULT,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=True
                        )
                        if not oneE.empty:
                            if DO_FDR:
                                q, passed = fdr_bh(oneE["pval"].values, alpha=FDR_ALPHA)
                                oneE["qval_bh"] = q
                                oneE["fdr_pass"] = passed
                            oneE = oneE.sort_values(["fdr_pass", "pval"], ascending=[False, True])
                            oneE.to_excel(w, index=False, sheet_name=f"E{cid}_oneD")

                        stabE = stability_analysis(
                            sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}",
                            base_seed=RANDOM_SEED, reps=STAB_REPS,
                            trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                            do_undersample=DO_UNDERSAMPLE_DEFAULT,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=True
                        )
                        if not stabE.empty:
                            stabE.to_excel(w, index=False, sheet_name=f"E{cid}_stability")

                        placE = placebo_y_test(
                            sub, OUTCOME_COL, D_E, X_E, f"E_cluster_{cid}",
                            base_seed=RANDOM_SEED, reps=PLACEBO_Y_REPS,
                            trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                            do_undersample=DO_UNDERSAMPLE_DEFAULT,
                            maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                            use_wls_weights=True
                        )
                        if not placE.empty:
                            placE.to_excel(w, index=False, sheet_name=f"E{cid}_placeboY")

                        cand = outE.copy()
                        cand["candidate_rule"] = (cand.get("fdr_pass", False) == True) & (cand["overlap_flag"] == False)
                        if not stabE.empty:
                            s = stabE.set_index("treatment")
                            cand["sign_consistency"] = cand["treatment"].map(s["sign_consistency"].to_dict())
                            cand["pass_rate_p<0.05"] = cand["treatment"].map(s["pass_rate_p<0.05"].to_dict())
                            cand["candidate_rule"] = cand["candidate_rule"] & (cand["sign_consistency"].fillna(0) >= 0.8)
                        cand = cand.sort_values(["candidate_rule", "fdr_pass", "pval"], ascending=[False, False, True])
                        cand.to_excel(w, index=False, sheet_name=f"E{cid}_candidates")

                        runlog.append({
                            "cluster": cid,
                            "block": "E",
                            "status": "done",
                            "n_rows": n_rows,
                            "n_D": len(D_E),
                            "n_X": len(X_E),
                            "n_context_in_X": int(sum([1 for c in FORCED_CONTEXT_COLS if c in X_E])),
                            "context_cols_in_X": ", ".join([c for c in FORCED_CONTEXT_COLS if c in X_E])
                        })
                else:
                    runlog.append({"cluster": cid, "block": "E", "status": "skip", "reason": "no non-att E treatments left"})

        pd.DataFrame(runlog).to_excel(w, index=False, sheet_name="run_log")

    print(f"[OK] wrote: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()