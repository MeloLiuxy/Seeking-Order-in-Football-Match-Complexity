

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

BASE_SPLITS    = 5
RANDOM_SEED    = 42

# nuisance 训练折下采样（默认开；稳健性会跑 noUS 版本）
DO_UNDERSAMPLE_DEFAULT = True
MAJ_KEEP_RATIO_DEFAULT = 0.5

# 数据门槛
MIN_SAMPLES   = 80
MIN_MINOR     = 12

# trimming（默认 0.02；稳健性会跑 0）
TRIM_TOP_FRAC_DEFAULT = 0.02
MIN_RESID_SD  = 1e-10

# FDR
DO_FDR    = True
FDR_ALPHA = 0.10

# near-constant 判定
NEAR_CONST_VAR_EPS = 1e-12
NEAR_CONST_UNIQUE_MAX = 2

# overlap flag 阈值
FLAG_R2_GE = 0.90
FLAG_RESID_RATIO_LE = 0.30

# gps 分箱数量
GPS_BINS = 5

# 稳定性重复次数
STAB_REPS = 20
STAB_PVAL_CUT = 0.05

# Placebo：y 置换次数（越大越慢）
PLACEBO_Y_REPS = 50


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

    # 白名单：虽然含 minute，但必须进入 X
    if name in FORCED_CONTEXT_COLS:
        return False

    # 原始标签列不进模型
    if name in RAW_CONTEXT_LABEL_COLS:
        return True

    for rgx in EXCLUDE_REGEXES:
        if rgx.search(name):
            return True
    return False

def is_E_col(c):
    return isinstance(c, str) and c.strip().endswith("(E)")

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
# ✅ 识别 Att KPI：不当 treatment，但进 X
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
    # 兜底：含 att 但不含 def（避免把 def_*_att_speed 这类误判成进攻 KPI）
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
        n_estimators=600,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )
    try:
        return CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=3)

def make_d_model(seed):
    return RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

def choose_splits(n):
    if n < 50:
        return max(2, min(3, n - 1))
    return max(2, min(BASE_SPLITS, n // 5, n - 1))

def stratified_undersample(y, tr_idx, do_undersample, maj_keep_ratio, seed):
    """只在训练折：保留多数类的一部分 + 全部少数类；兜底保证两类。"""
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
# 清理：重复列（按内容 hash）/ 常数列
# =========================
def series_digest(s: pd.Series) -> str:
    hv = pd.util.hash_pandas_object(s, index=False).values
    return hashlib.md5(hv.tobytes()).hexdigest()

def drop_duplicate_by_content(df: pd.DataFrame):
    dig2cols = {}
    for c in df.columns:
        dig2cols.setdefault(series_digest(df[c]), []).append(c)

    report = []
    drop_cols = []
    for cols in dig2cols.values():
        if len(cols) >= 2:
            keep = cols[0]
            drops = cols[1:]
            drop_cols.extend(drops)
            report.append({
                "group_size": int(len(cols)),
                "keep": keep,
                "drop": ", ".join(drops),
                "all_cols_in_group": ", ".join(cols)
            })

    df2 = df.drop(columns=list(set(drop_cols)), errors="ignore").copy()
    if report:
        rep_df = pd.DataFrame(report).sort_values(["group_size", "keep"], ascending=[False, True])
    else:
        rep_df = pd.DataFrame([{"group_size": 1, "keep": "", "drop": "", "all_cols_in_group": "No duplicate-by-content columns found."}])
    return df2, rep_df

def find_near_constant_cols(sub_df: pd.DataFrame, cols: list):
    bad = []
    for c in cols:
        s = to_num(sub_df[c]).dropna()
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
# 单特征 DML：一次（支持变体 + overlap 指标）
# =========================
def dml_one_once(
    sub: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: list,
    block_label: str,
    seed: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    cols = uniq_keep_order([y_col, d_col] + x_cols)
    tmp = sub[cols].copy()

    tmp[y_col] = to_num(tmp[y_col])
    tmp[d_col] = to_num(tmp[d_col])
    tmp = tmp.dropna(subset=[y_col, d_col])

    n = len(tmp)
    if n < MIN_SAMPLES:
        return None, None, f"{block_label}:{d_col} n={n} too small"

    # X 数值化 + 中位数填补（剔除全 NaN）
    kept_x = []
    for c in x_cols:
        tmp[c] = to_num(tmp[c])
        med = np.nanmedian(tmp[c].values)
        if np.isnan(med):
            continue
        tmp[c] = tmp[c].fillna(med)
        kept_x.append(c)
    x_cols = kept_x

    y = (tmp[y_col].values > 0.5).astype(int)
    d = tmp[d_col].values.astype(float)
    X = tmp[x_cols].values.astype(float) if x_cols else np.zeros((n, 1))

    pos = int(y.sum()); neg = n - pos
    if min(pos, neg) < MIN_MINOR:
        return None, None, f"{block_label}:{d_col} minority={min(pos,neg)} too small"

    k = choose_splits(n)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    y_hat = np.full(n, np.nan)
    d_hat = np.full(n, np.nan)
    r2y_list, r2d_list = [], []

    for tr, te in skf.split(X, y):
        tr_eff = stratified_undersample(y, tr, do_undersample, maj_keep_ratio, seed)
        if len(np.unique(y[tr_eff])) < 2:
            tr_eff = tr
            if len(np.unique(y[tr_eff])) < 2:
                continue

        Xtr, ytr, dtr = X[tr_eff], y[tr_eff], d[tr_eff]
        Xte, yte, dte = X[te], y[te], d[te]

        my = make_y_model(seed).fit(Xtr, ytr)
        y_hat_te = my.predict_proba(Xte)[:, 1]
        y_hat[te] = y_hat_te
        try:
            r2y_list.append(r2_score(yte, y_hat_te))
        except Exception:
            r2y_list.append(np.nan)

        md = make_d_model(seed).fit(Xtr, dtr)
        d_hat_te = md.predict(Xte)
        d_hat[te] = d_hat_te
        try:
            r2d_list.append(r2_score(dte, d_hat_te))
        except Exception:
            r2d_list.append(np.nan)

    y_res = y - y_hat
    d_res = d - d_hat

    ok = np.isfinite(y_res) & np.isfinite(d_res)
    if not np.any(ok):
        return None, None, f"{block_label}:{d_col} no finite residuals"

    if trim_top_frac and trim_top_frac > 0:
        thr = np.quantile(np.abs(d_res[ok]), 1 - trim_top_frac)
        ok = ok & (np.abs(d_res) <= thr)

    n_used = int(ok.sum())
    if n_used < max(30, int(0.3 * n)):
        return None, None, f"{block_label}:{d_col} usable={n_used} too small"

    sd_d_raw = float(np.nanstd(d, ddof=0))
    sd_d_res = float(np.nanstd(d_res[ok], ddof=1))
    if sd_d_res < MIN_RESID_SD:
        return None, None, f"{block_label}:{d_col} resid var too small"

    r2d = float(np.nanmean(r2d_list))
    resid_ratio = (sd_d_res / sd_d_raw) if (sd_d_raw > 0 and np.isfinite(sd_d_raw)) else np.nan
    overlap_flag = bool((np.isfinite(r2d) and r2d >= FLAG_R2_GE) and (np.isfinite(resid_ratio) and resid_ratio <= FLAG_RESID_RATIO_LE))

    # gps 分箱：按 d_hat 分箱，报告真实 d 的范围
    gps_txt = ""
    try:
        mask = np.isfinite(d_hat) & np.isfinite(d)
        if mask.sum() >= 20:
            q = np.quantile(d_hat[mask], np.linspace(0, 1, GPS_BINS + 1))
            q[0] -= 1e-12
            parts = []
            for b in range(GPS_BINS):
                m2 = mask & (d_hat >= q[b]) & (d_hat <= q[b+1])
                if m2.sum() < 5:
                    parts.append(f"bin{b+1}:n<5")
                else:
                    lo = float(np.nanmin(d[m2])); hi = float(np.nanmax(d[m2]))
                    parts.append(f"bin{b+1}:[{lo:.3g},{hi:.3g}] n={int(m2.sum())}")
            gps_txt = " | ".join(parts)
    except Exception:
        gps_txt = "gps_failed"

    # second stage：WLS(HC3) 或 OLS(HC3)
    Xols = sm.add_constant(d_res[ok], has_constant='add')

    w = None
    if use_wls_weights:
        pr = float(y[ok].mean())
        if 0 < pr < 1:
            w = np.where(y[ok] == 1, 1.0 / max(pr, 1e-6), 1.0 / max(1 - pr, 1e-6))
            w = w / np.mean(w)

    model = sm.WLS(y_res[ok], Xols, weights=w) if use_wls_weights else sm.OLS(y_res[ok], Xols)
    res_ols = model.fit(cov_type="HC3")

    theta = float(res_ols.params[1])
    se    = float(res_ols.bse[1])
    p     = float(res_ols.pvalues[1])
    ci    = res_ols.conf_int(alpha=0.05)
    ci_low, ci_high = float(ci[1, 0]), float(ci[1, 1])

    theta_1sd = theta * sd_d_raw if sd_d_raw > 0 else np.nan

    # 粗敏感性指标（RV-like）
    tval = theta / se if se > 0 else np.nan
    df_approx = max(1, n_used - 2)
    rv_like = (tval**2) / (tval**2 + df_approx) if np.isfinite(tval) else np.nan

    res_row = {
        "block": block_label,
        "treatment": d_col,
        "theta_per_unit": theta,
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
        "r2_d_hat": r2d,
        "sd_d_raw": sd_d_raw,
        "sd_d_resid_used": sd_d_res,
        "resid_sd_ratio": resid_ratio,
        "overlap_flag": overlap_flag,
        "gps_bin_range": gps_txt,
        "trim_top_frac": float(trim_top_frac),
        "undersample": bool(do_undersample),
        "maj_keep_ratio": float(maj_keep_ratio),
        "second_stage": "WLS_HC3" if use_wls_weights else "OLS_HC3",
        "learners": "Y|X: RFc+cal; D|X: RFr"
    }

    overlap_row = {
        "block": block_label,
        "treatment": d_col,
        "r2_d_hat": r2d,
        "sd_d_raw": sd_d_raw,
        "sd_d_resid_used": sd_d_res,
        "resid_sd_ratio": resid_ratio,
        "overlap_flag": overlap_flag,
        "gps_bin_range": gps_txt,
        "n_total": int(n),
        "n_used": int(n_used),
        "pos_rate": float(pos / n),
        "r2_y_hat": float(np.nanmean(r2y_list)),
    }

    return res_row, overlap_row, None


# =========================
# 批量跑一块（只跑 E）
# =========================
def run_block_oneD(
    sub: pd.DataFrame,
    block_label: str,
    D_cols: list,
    X_builder,  # callable(d_col)->x_cols
    seed: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    results = []
    overlaps = []
    dropped = []

    for d in D_cols:
        x_cols = X_builder(d)
        res_row, ov_row, err = dml_one_once(
            sub=sub,
            y_col=OUTCOME_COL,
            d_col=d,
            x_cols=x_cols,
            block_label=block_label,
            seed=seed,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights
        )
        if err is not None:
            dropped.append({"block": block_label, "treatment": d, "reason": err})
        else:
            results.append(res_row)
            overlaps.append(ov_row)

    res_df = pd.DataFrame(results) if results else pd.DataFrame()
    ov_df  = pd.DataFrame(overlaps) if overlaps else pd.DataFrame()
    drop_df = pd.DataFrame(dropped) if dropped else pd.DataFrame()
    return res_df, ov_df, drop_df


# =========================
# 稳定性：重复 seed
# =========================
def stability_block_oneD(
    sub: pd.DataFrame,
    block_label: str,
    D_cols: list,
    X_builder,
    base_seed: int,
    reps: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    store = {d: [] for d in D_cols}
    store_p = {d: [] for d in D_cols}

    for r in range(reps):
        seed = base_seed + 1000 + r * 17
        res_df, _, _ = run_block_oneD(
            sub=sub,
            block_label=block_label,
            D_cols=D_cols,
            X_builder=X_builder,
            seed=seed,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights
        )
        if res_df.empty:
            continue
        for _, row in res_df.iterrows():
            d = row["treatment"]
            store[d].append(float(row["theta_per_1sd"]))
            store_p[d].append(float(row["pval"]))

    rows = []
    for d in D_cols:
        vals = np.asarray(store[d], dtype=float)
        pvs  = np.asarray(store_p[d], dtype=float)
        if vals.size == 0:
            continue
        med = float(np.nanmedian(vals))
        sign_cons = float(np.mean(np.sign(vals) == np.sign(med))) if np.isfinite(med) and med != 0 else float(np.mean(np.sign(vals) != 0))
        pass_p = float(np.mean(pvs < STAB_PVAL_CUT))
        rows.append({
            "block": block_label,
            "treatment": d,
            "n_runs": int(vals.size),
            "median_theta_per_1sd": med,
            "iqr_theta_per_1sd": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            "sign_consistency": sign_cons,
            "pass_rate_p<0.05": pass_p
        })

    return pd.DataFrame(rows).sort_values(["pass_rate_p<0.05", "sign_consistency"], ascending=False) if rows else pd.DataFrame()


# =========================
# Placebo：置换 y（cluster 内）
# =========================
def placebo_y_block_oneD(
    sub: pd.DataFrame,
    block_label: str,
    D_cols: list,
    X_builder,
    base_seed: int,
    reps: int,
    trim_top_frac: float,
    do_undersample: bool,
    maj_keep_ratio: float,
    use_wls_weights: bool,
):
    real_df, _, _ = run_block_oneD(
        sub=sub,
        block_label=block_label,
        D_cols=D_cols,
        X_builder=X_builder,
        seed=base_seed,
        trim_top_frac=trim_top_frac,
        do_undersample=do_undersample,
        maj_keep_ratio=maj_keep_ratio,
        use_wls_weights=use_wls_weights
    )
    if real_df.empty:
        return pd.DataFrame([{"block": block_label, "err": "real_run_failed_or_empty"}])

    real_p = real_df.set_index("treatment")["pval"].to_dict()

    rng = np.random.RandomState(base_seed + 9999)
    counts = {d: 0 for d in D_cols}
    used = 0

    for r in range(reps):
        perm = sub.copy()
        perm[OUTCOME_COL] = rng.permutation(perm[OUTCOME_COL].values)

        perm_df, _, _ = run_block_oneD(
            sub=perm,
            block_label=block_label + "_placeboY",
            D_cols=D_cols,
            X_builder=X_builder,
            seed=base_seed + 20000 + r,
            trim_top_frac=trim_top_frac,
            do_undersample=do_undersample,
            maj_keep_ratio=maj_keep_ratio,
            use_wls_weights=use_wls_weights
        )
        if perm_df.empty:
            continue
        used += 1
        for _, row in perm_df.iterrows():
            d = row["treatment"]
            if float(row["pval"]) <= float(real_p.get(d, 1.0)):
                counts[d] += 1

    rows = []
    for d in D_cols:
        if used == 0:
            continue
        rows.append({
            "block": block_label,
            "treatment": d,
            "real_pval": float(real_p.get(d, np.nan)),
            "placebo_reps_used": int(used),
            "placebo_p_le_real_rate": float(counts[d] / used)
        })

    return pd.DataFrame(rows).sort_values("placebo_p_le_real_rate", ascending=True) if rows else pd.DataFrame()


# =========================
# 主程序（只做 E）
# =========================
def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]
    df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

    if OUTCOME_COL not in df.columns or CLUSTER_COL not in df.columns:
        raise ValueError("缺少必要列：success_def 或 cluster_id")

    # 检查情境控制列是否存在
    missing_context = [c for c in FORCED_CONTEXT_COLS if c not in df.columns]
    if missing_context:
        raise ValueError(f"输入文件缺少以下情境控制列：{missing_context}")

    # 全表：内容重复列去重
    df, dup_report = drop_duplicate_by_content(df)

    # 只识别 E
    all_E = [c for c in df.columns if is_E_col(c) and not looks_like_meta(c)]

    clusters = [v for v in sorted(df[CLUSTER_COL].dropna().unique())]
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_XLSX)), exist_ok=True)

    runlog = []

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as w:
        dup_report.to_excel(w, index=False, sheet_name="00_dup_by_content")

        for cid in clusters:
            sub = df.loc[df[CLUSTER_COL] == cid].copy()
            n_rows = int(len(sub))
            if n_rows < MIN_SAMPLES:
                runlog.append({"cluster": cid, "status": "skip", "reason": f"n={n_rows} too small"})
                continue

            # cluster 内可用列（有值且方差>0）
            def usable(cols):
                keep = []
                for c in cols:
                    s = to_num(sub[c])
                    if s.notna().sum() > 0 and np.nanvar(s.values) > 0:
                        keep.append(c)
                return keep

            E_cols = usable(all_E)

            # cluster 内 near-constant 报告并剔除（只对 E）
            nearE = find_near_constant_cols(sub, E_cols)
            near_df = pd.DataFrame(
                [{"cluster": cid, "col": c, "reason": r, "block": "E"} for c, r in nearE]
            )
            if not near_df.empty:
                near_df.to_excel(w, index=False, sheet_name=f"c{cid}_near_const"[:31])

            badE = {c for c, _ in nearE}
            E_cols = [c for c in E_cols if c not in badE]

            # 背景变量：非 E，非 meta，非 outcome/cluster
            bg_cols = []
            for c in sub.columns:
                if c in [OUTCOME_COL, CLUSTER_COL]:
                    continue
                if is_E_col(c):
                    continue
                if looks_like_meta(c):
                    continue
                bg_cols.append(c)

            # 强制加入情境控制变量
            bg_cols = uniq_keep_order(bg_cols + [c for c in FORCED_CONTEXT_COLS if c in sub.columns])
            bg_cols = usable(bg_cols)

            # ✅ E 中分离 Att / non-Att
            E_att = [c for c in E_cols if is_att_kpi(c)]
            E_nonatt = [c for c in E_cols if c not in set(E_att)]

            if len(E_nonatt) < 1:
                runlog.append({"cluster": cid, "block": "E", "status": "skip", "reason": "no non-att E treatments left"})
                continue

            # X：背景 + E_att（固定，不随 d 变）
            x_fixed = usable(uniq_keep_order(bg_cols + E_att))

            def X_builder_E(_d_col):
                return x_fixed

            mainE, ovE, dropE = run_block_oneD(
                sub=sub,
                block_label=f"E_cluster_{cid}",
                D_cols=E_nonatt,
                X_builder=X_builder_E,
                seed=RANDOM_SEED,
                trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                do_undersample=DO_UNDERSAMPLE_DEFAULT,
                maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=True
            )

            if mainE.empty:
                runlog.append({"cluster": cid, "block": "E", "status": "fail_or_empty", "reason": "mainE empty"})
                continue

            outE = mainE.sort_values(["pval", "theta_per_1sd"], ascending=[True, False]).copy()
            if DO_FDR:
                q, passed = fdr_bh(outE["pval"].values, alpha=FDR_ALPHA)
                outE["qval_bh"] = q
                outE["fdr_pass"] = passed
                outE = outE.sort_values(["fdr_pass", "pval", "theta_per_1sd"], ascending=[False, True, False])

            outE.to_excel(w, index=False, sheet_name=f"E{cid}_main"[:31])
            ovE.to_excel(w, index=False, sheet_name=f"E{cid}_overlap"[:31])
            if not dropE.empty:
                dropE.to_excel(w, index=False, sheet_name=f"E{cid}_drop"[:31])

            # 稳健性：trim=0
            robE_t0, _, _ = run_block_oneD(
                sub=sub, block_label=f"E_cluster_{cid}_trim0",
                D_cols=E_nonatt, X_builder=X_builder_E,
                seed=RANDOM_SEED, trim_top_frac=0.0,
                do_undersample=DO_UNDERSAMPLE_DEFAULT, maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=True
            )
            if not robE_t0.empty:
                robE_t0.to_excel(w, index=False, sheet_name=f"E{cid}_rob_t0"[:31])

            # 稳健性：no undersample
            robE_nu, _, _ = run_block_oneD(
                sub=sub, block_label=f"E_cluster_{cid}_noUS",
                D_cols=E_nonatt, X_builder=X_builder_E,
                seed=RANDOM_SEED, trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                do_undersample=False, maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=True
            )
            if not robE_nu.empty:
                robE_nu.to_excel(w, index=False, sheet_name=f"E{cid}_rob_noUS"[:31])

            # 稳健性：OLS second stage
            robE_ols, _, _ = run_block_oneD(
                sub=sub, block_label=f"E_cluster_{cid}_OLS",
                D_cols=E_nonatt, X_builder=X_builder_E,
                seed=RANDOM_SEED, trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                do_undersample=DO_UNDERSAMPLE_DEFAULT, maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=False
            )
            if not robE_ols.empty:
                robE_ols.to_excel(w, index=False, sheet_name=f"E{cid}_rob_OLS"[:31])

            # 稳定性
            stabE = stability_block_oneD(
                sub=sub,
                block_label=f"E_cluster_{cid}",
                D_cols=E_nonatt,
                X_builder=X_builder_E,
                base_seed=RANDOM_SEED,
                reps=STAB_REPS,
                trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                do_undersample=DO_UNDERSAMPLE_DEFAULT,
                maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=True
            )
            if not stabE.empty:
                stabE.to_excel(w, index=False, sheet_name=f"E{cid}_stability"[:31])

            # Placebo
            placE = placebo_y_block_oneD(
                sub=sub,
                block_label=f"E_cluster_{cid}",
                D_cols=E_nonatt,
                X_builder=X_builder_E,
                base_seed=RANDOM_SEED,
                reps=PLACEBO_Y_REPS,
                trim_top_frac=TRIM_TOP_FRAC_DEFAULT,
                do_undersample=DO_UNDERSAMPLE_DEFAULT,
                maj_keep_ratio=MAJ_KEEP_RATIO_DEFAULT,
                use_wls_weights=True
            )
            if not placE.empty:
                placE.to_excel(w, index=False, sheet_name=f"E{cid}_placeboY"[:31])

            # candidates
            candE = outE.copy()
            candE["candidate_rule"] = (candE.get("fdr_pass", False) == True) & (candE["overlap_flag"] == False)
            if not stabE.empty:
                s = stabE.set_index("treatment")
                candE["sign_consistency"] = candE["treatment"].map(s["sign_consistency"].to_dict())
                candE["pass_rate_p<0.05"] = candE["treatment"].map(s["pass_rate_p<0.05"].to_dict())
                candE["candidate_rule"] = candE["candidate_rule"] & (candE["sign_consistency"].fillna(0) >= 0.8)
            candE = candE.sort_values(["candidate_rule", "fdr_pass", "pval"], ascending=[False, False, True])
            candE.to_excel(w, index=False, sheet_name=f"E{cid}_candidates"[:31])

            runlog.append({
                "cluster": cid,
                "block": "E",
                "status": "done",
                "n_rows": n_rows,
                "n_D_E_nonatt": int(len(E_nonatt)),
                "n_X_bg": int(len(bg_cols)),
                "n_X_E_att": int(len(E_att)),
                "n_X_context": int(sum([1 for c in FORCED_CONTEXT_COLS if c in x_fixed])),
                "n_X_total": int(len(x_fixed)),
                "context_cols_in_X": ", ".join([c for c in FORCED_CONTEXT_COLS if c in x_fixed]),
            })

        pd.DataFrame(runlog).to_excel(w, index=False, sheet_name="run_log")

    print(f"[OK] wrote: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
