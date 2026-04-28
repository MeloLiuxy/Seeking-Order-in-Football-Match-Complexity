
import os
import re
import time
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


# ====================== 配置区 ======================

STAGE = "E"

# 原始特征数据（carry）
CARRY_FEATURES_PATH = r""
CARRY_FEATURES_SHEET = 0

# 输出
OUTPUT_POLICY_XLSX = r""

# 基本列名
OUTCOME_COL = "success_def"
CLUSTER_COL = "cluster_id"

# 稳健性汇总表列名
BLOCK_COL = "block"
TREATMENT_COL = "treatment"
EFFECT_COL = "Effect/1 SD"
EFFECT_PLUS_COL = "Effect/1 SD+"
T_COL = "t"
RV_COL = "RV-like Strength"
N_COL = "N"
POS_RATE_COL = "Pos. Rate"
R2D_COL = "R_(D|X)^2"
QVAL_COL = "BH q-value"
QVAL_PLUS_COL = "BH q-value+"

# 若直接读文件就填路径，否则用 inline
ROBUST_RESULTS_PATH = None

# 硬筛选门槛
QVAL_ALPHA = 0.05
T_ABS_MIN = 4.0
MAX_REL_CHANGE = 0.50

# R2 只做标记
R2D_PRIORITY_MIN = 0.10
R2D_PRIORITY_MAX = 0.70
R2D_BEST_MIN = 0.10
R2D_BEST_MAX = 0.50

# 策略搜索约束
Q_LO = 0.20
Q_HI = 0.80
MAX_L1_STD = 3.0
MIN_KPIS_PER_CLUSTER = 1
MIN_FAIL_SAMPLES = 80

# GA：提速版
GA_POP_SIZE = 30
GA_N_GEN = 40
GA_CX_PROB = 0.8
GA_MUT_PROB = 0.3
GA_MUT_SIGMA = 0.3
GA_ELITISM = 2

# 预测模型
RF_N_ESTIMATORS = 200
RF_MIN_SAMPLES_LEAF = 10

# 目标函数：overall 为主
OBJECTIVE_MODE = "overall"   # overall / weighted
WEIGHT_FAIL = 0.40
WEIGHT_ALL = 0.60

# fail 保护
ENFORCE_FAIL_NONDECREASE = True
FAIL_MIN_GAIN = 0.0

# success 副作用约束
MONITOR_SUCCESS_SIDE_EFFECT = True
ENFORCE_SUCCESS_HARD_CONSTRAINT = True
SUCC_MAX_DROP = 0.01

USE_SUCCESS_SOFT_PENALTY = False
SUCCESS_PENALTY_LAMBDA = 5.0

# 并行
MAX_WORKERS = 3

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 自动排除的背景变量模式（减少无用特征）
EXCLUDE_PATTERNS = [
    r"match_id", r"game_id", r"player_id", r"team_id", r"event_id",
    r"frame", r"second", r"minute", r"period", r"half", r"timestamp",
    r"location_x", r"location_y", r"end_location_x", r"end_location_y",
    r"^x$|^y$|^z$"
]
EXCLUDE_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in EXCLUDE_PATTERNS]


# ====================== 你给的 Carry 稳健性汇总表 ======================

ROBUST_RESULTS_INLINE = """block\ttreatment\tEffect/1 SD\tEffect/1 SD+\tt\tRV-like Strength\tN\tPos. Rate\tR_(D|X)^2\tBH q-value\tBH q-value+
E_cluster_0.0\tAdv_5(E)\t0.038448341\t0.04\t6.440667665\t0.007042242\t5971\t0.339474125\t0.413150443\t1.35942E-10\t2E-10
E_cluster_0.0\tAvg_1_Def(E)\t-0.066239591\t-0.066604382\t-10.40434806\t0.018171212\t5971\t0.339474125\t0.619957809\t3.78995E-25\t7E-25
E_cluster_1.0\tAdv_10(E)\t-0.037363037\t-0.04\t-6.132776441\t0.005440131\t7019\t0.243481977\t0.44528997\t1.38174E-09\t1.49611E-09
E_cluster_1.0\tAdv_5(E)\t0.024251796\t0.03\t4.20904519\t0.002569886\t7019\t0.243481977\t0.353305168\t3.41936E-05\t2.59571E-05
E_cluster_1.0\tAvg_1_Def(E)\t-0.064379822\t-0.066819889\t-9.356232187\t0.012571062\t7019\t0.243481977\t0.529180673\t2.20349E-20\t8.01313E-21
E_cluster_2.0\tAdv_10(E)\t-0.040578156\t-0.04\t-6.689034944\t0.006282618\t7224\t0.253183832\t0.434390019\t4.49294E-11\t3.8633E-11
E_cluster_2.0\tAvg_1_Def(E)\t-0.062992985\t-0.063550049\t-9.525601476\t0.012659098\t7224\t0.253183832\t0.532834143\t4.3757E-21\t4.88426E-21
E_cluster_2.0\tAvg_3_Def(E)\t0.039112294\t0.04\t4.337579591\t0.002651506\t7224\t0.253183832\t0.692454646\t1.6464E-05\t1.74842E-05
"""


# ====================== 工具函数 ======================

def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def looks_like_meta(name: str) -> bool:
    name = str(name)
    for rgx in EXCLUDE_REGEXES:
        if rgx.search(name):
            return True
    return False


def is_att_kpi(name: str) -> bool:
    if name is None:
        return False
    s = str(name).lower()
    return ("_att" in s) or ("disttoattcentroid" in s)


def project_l1_ball(x, l1_budget):
    x = np.asarray(x, dtype=float)
    s = np.sum(np.abs(x))
    if s <= l1_budget:
        return x

    u = np.abs(x)
    if np.all(u == 0):
        return x

    u_sorted = np.sort(u)[::-1]
    cssv = np.cumsum(u_sorted)
    rho = np.nonzero(u_sorted * np.arange(1, len(u) + 1) > (cssv - l1_budget))[0][-1]
    theta = (cssv[rho] - l1_budget) / float(rho + 1)
    return np.sign(x) * np.maximum(u - theta, 0.0)


def rel_change(base, new):
    base = float(base)
    new = float(new)
    if np.isnan(base) or np.isnan(new):
        return np.nan
    if abs(base) < 1e-12:
        return np.inf
    return abs(new - base) / abs(base)


def sign_consistent(base, new):
    if pd.isna(base) or pd.isna(new):
        return False
    if abs(base) < 1e-12 or abs(new) < 1e-12:
        return False
    return np.sign(base) == np.sign(new)


# ====================== 读取数据 ======================

def load_carry_features(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    if OUTCOME_COL not in df.columns or CLUSTER_COL not in df.columns:
        raise ValueError(f"原始特征表缺少 {OUTCOME_COL} 或 {CLUSTER_COL}")
    df[OUTCOME_COL] = to_num(df[OUTCOME_COL])
    return df


def load_robust_results(path=None, inline_text=None):
    if path is None:
        if not inline_text:
            raise ValueError("ROBUST_RESULTS_PATH 为 None 且 inline_text 为空。")
        from io import StringIO
        df = pd.read_csv(StringIO(inline_text), sep="\t")
    else:
        if str(path).lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

    need_cols = [
        BLOCK_COL, TREATMENT_COL, EFFECT_COL, EFFECT_PLUS_COL,
        T_COL, RV_COL, N_COL, POS_RATE_COL, R2D_COL, QVAL_COL, QVAL_PLUS_COL
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"稳健性汇总表缺少列：{c}")

    pattern = re.compile(r"^([LE])_cluster_(\d+(?:\.\d+)?)$", re.IGNORECASE)

    keep_idx, sides, clusters = [], [], []
    for idx, b in enumerate(df[BLOCK_COL].astype(str)):
        m = pattern.match(b.strip())
        if m:
            keep_idx.append(idx)
            sides.append(m.group(1).upper())
            clusters.append(int(round(float(m.group(2)))))

    df = df.iloc[keep_idx].copy()
    df["side"] = sides
    df["cluster"] = clusters

    for c in [EFFECT_COL, EFFECT_PLUS_COL, T_COL, RV_COL, N_COL, POS_RATE_COL, R2D_COL, QVAL_COL, QVAL_PLUS_COL]:
        df[c] = to_num(df[c])

    df["rel_change"] = df.apply(lambda r: rel_change(r[EFFECT_COL], r[EFFECT_PLUS_COL]), axis=1)
    df["sign_consistent"] = df.apply(lambda r: sign_consistent(r[EFFECT_COL], r[EFFECT_PLUS_COL]), axis=1)
    df["r2_priority_flag"] = (df[R2D_COL] >= R2D_PRIORITY_MIN) & (df[R2D_COL] <= R2D_PRIORITY_MAX)
    df["r2_best_range_flag"] = (df[R2D_COL] >= R2D_BEST_MIN) & (df[R2D_COL] <= R2D_BEST_MAX)

    cluster_to_results = {}
    for cid, g in df.groupby("cluster"):
        cluster_to_results[int(cid)] = g.reset_index(drop=True)

    if not cluster_to_results:
        raise RuntimeError("稳健性结果表解析后为空。")
    return cluster_to_results


# ====================== KPI 筛选 ======================

def select_robust_kpis(result_df):
    df = result_df.copy()
    mask = pd.Series(True, index=df.index)
    mask &= (to_num(df[QVAL_COL]) <= QVAL_ALPHA)
    mask &= (to_num(df[QVAL_PLUS_COL]) <= QVAL_ALPHA)
    mask &= (to_num(df[T_COL]).abs() >= T_ABS_MIN)
    mask &= (df["sign_consistent"] == True)
    mask &= (to_num(df["rel_change"]) <= MAX_REL_CHANGE)
    return df[mask].copy()


def dedup_kpi_rows(kpi_df):
    df = kpi_df.copy()
    if df.empty:
        return df

    def score_row(r):
        q = r.get(QVAL_COL, np.nan)
        q2 = r.get(QVAL_PLUS_COL, np.nan)
        ef = r.get(EFFECT_COL, np.nan)
        ts = r.get(T_COL, np.nan)

        q_score = q if not pd.isna(q) else 1.0
        q2_score = q2 if not pd.isna(q2) else 1.0
        ef_score = -abs(ef) if not pd.isna(ef) else 0.0
        ts_score = -abs(ts) if not pd.isna(ts) else 0.0
        return (q_score, q2_score, ef_score, ts_score)

    scores = df.apply(score_row, axis=1, result_type="expand")
    df["_q"] = scores[0]
    df["_q2"] = scores[1]
    df["_ef"] = scores[2]
    df["_ts"] = scores[3]
    df = df.sort_values([TREATMENT_COL, "_q", "_q2", "_ef", "_ts"], ascending=True)
    df = df.drop_duplicates(subset=[TREATMENT_COL], keep="first")
    return df.drop(columns=["_q", "_q2", "_ef", "_ts"], errors="ignore")


def build_theta_and_direction(kpi_df):
    theta_map, dir_map = {}, {}
    for _, row in kpi_df.iterrows():
        name = row[TREATMENT_COL]
        theta = row.get(EFFECT_COL, np.nan)
        if pd.isna(theta):
            continue
        theta = float(theta)
        theta_map[name] = theta
        dir_map[name] = "increase" if theta >= 0 else "decrease"
    return theta_map, dir_map


# ====================== 预测模型 ======================

def choose_model_features(sub_all, def_kpis, att_kpis):
    keep = set(def_kpis) | set(att_kpis)

    bg_cols = []
    for c in sub_all.columns:
        if c in [OUTCOME_COL, CLUSTER_COL]:
            continue
        if c in keep:
            continue
        if looks_like_meta(c):
            continue
        if pd.api.types.is_numeric_dtype(sub_all[c]):
            bg_cols.append(c)

    bg_cols = bg_cols[:12]
    feature_cols = list(def_kpis) + list(att_kpis) + bg_cols
    feature_cols = [c for c in feature_cols if c in sub_all.columns]
    return feature_cols


def train_predict_model(sub_all, feature_cols, random_state=42):
    df = sub_all.copy().dropna(subset=[OUTCOME_COL])

    if not feature_cols:
        raise RuntimeError("没有可用特征列用于训练预测模型。")

    X = df[feature_cols].copy()
    y = df[OUTCOME_COL].copy()

    medians = {}
    for c in feature_cols:
        X[c] = to_num(X[c])
        med = float(X[c].median())
        medians[c] = med
        X[c] = X[c].fillna(med)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=random_state,
    )
    clf = CalibratedClassifierCV(rf, cv=3, method="sigmoid")
    clf.fit(X, y)

    def predict_prob(df_new):
        X_new = df_new.reindex(columns=feature_cols, copy=True)
        for c in feature_cols:
            X_new[c] = to_num(X_new[c])
            X_new[c] = X_new[c].fillna(medians[c])
        return clf.predict_proba(X_new)[:, 1]

    return predict_prob, feature_cols


# ====================== 策略相关 ======================

def compute_kpi_stats(sub_all, kpi_cols):
    stats = {}
    for c in kpi_cols:
        s = to_num(sub_all[c])
        stats[c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "q_lo": float(s.quantile(Q_LO)),
            "q_hi": float(s.quantile(Q_HI)),
            "std": float(s.std(ddof=1)),
        }
    return stats


def apply_shift_with_clip(df_in, x_std, def_kpis, stats, dir_map):
    df_new = df_in.copy()
    for j, kpi in enumerate(def_kpis):
        info = stats[kpi]
        std_val = info["std"]
        if std_val is None or np.isnan(std_val) or std_val <= 1e-8:
            continue

        delta = float(x_std[j]) * float(std_val)
        direction = dir_map.get(kpi, "increase")
        if direction == "increase" and delta < 0:
            delta = 0.0
        if direction == "decrease" and delta > 0:
            delta = 0.0

        old_vals = to_num(df_new[kpi])
        new_vals = np.clip(old_vals + delta, info["q_lo"], info["q_hi"])
        df_new[kpi] = new_vals
    return df_new


def eval_policy_x(
    x_std,
    sub_all,
    sub_fail,
    sub_succ,
    def_kpis,
    stats,
    dir_map,
    predict_prob_func,
    baseline_prob_all,
    baseline_prob_fail,
    baseline_prob_succ,
):
    """
    主目标：overall
    约束：
    - fail 不能变差
    - success 不能显著变差
    """
    df_all_new = apply_shift_with_clip(sub_all, x_std, def_kpis, stats, dir_map)
    v_all = float(np.mean(predict_prob_func(df_all_new)))

    df_fail_new = apply_shift_with_clip(sub_fail, x_std, def_kpis, stats, dir_map)
    v_fail = float(np.mean(predict_prob_func(df_fail_new)))

    if ENFORCE_FAIL_NONDECREASE and (v_fail - baseline_prob_fail) < FAIL_MIN_GAIN:
        return -1e9

    if sub_succ is not None and len(sub_succ) > 0:
        df_succ_new = apply_shift_with_clip(sub_succ, x_std, def_kpis, stats, dir_map)
        v_succ = float(np.mean(predict_prob_func(df_succ_new)))

        if ENFORCE_SUCCESS_HARD_CONSTRAINT:
            succ_drop = baseline_prob_succ - v_succ
            if succ_drop > SUCC_MAX_DROP:
                return -1e9

        if USE_SUCCESS_SOFT_PENALTY:
            succ_drop = max(0.0, baseline_prob_succ - v_succ)
            v_all -= SUCCESS_PENALTY_LAMBDA * succ_drop

    if OBJECTIVE_MODE == "overall":
        return v_all
    elif OBJECTIVE_MODE == "weighted":
        return WEIGHT_FAIL * v_fail + WEIGHT_ALL * v_all
    else:
        raise ValueError(f"未知 OBJECTIVE_MODE: {OBJECTIVE_MODE}")


def ga_search_policy(
    sub_all,
    sub_fail,
    sub_succ,
    def_kpis,
    stats,
    dir_map,
    predict_prob_func,
    x_lower,
    x_upper,
    baseline_prob_all,
    baseline_prob_fail,
    baseline_prob_succ,
    l1_budget=3.0,
    pop_size=30,
    n_generations=40,
    cx_prob=0.8,
    mut_prob=0.3,
    mut_sigma=0.3,
    elitism=2,
    random_state=42,
):
    rng = np.random.RandomState(random_state)
    k = len(def_kpis)
    if k == 0:
        return np.zeros(0, dtype=float), np.nan

    pop = []
    for _ in range(pop_size):
        x = rng.uniform(low=x_lower, high=x_upper, size=k)
        x = project_l1_ball(x, l1_budget)
        x = np.minimum(np.maximum(x, x_lower), x_upper)
        pop.append(x)
    pop = np.vstack(pop)

    def fitness(x):
        return eval_policy_x(
            x_std=x,
            sub_all=sub_all,
            sub_fail=sub_fail,
            sub_succ=sub_succ,
            def_kpis=def_kpis,
            stats=stats,
            dir_map=dir_map,
            predict_prob_func=predict_prob_func,
            baseline_prob_all=baseline_prob_all,
            baseline_prob_fail=baseline_prob_fail,
            baseline_prob_succ=baseline_prob_succ,
        )

    fit = np.array([fitness(ind) for ind in pop])
    best_idx = int(np.argmax(fit))
    best_x = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    for _ in range(n_generations):
        new_pop = []

        elite_idx = fit.argsort()[::-1][:elitism]
        for idx in elite_idx:
            new_pop.append(pop[idx].copy())

        while len(new_pop) < pop_size:
            def tournament():
                i = rng.randint(0, pop_size)
                j = rng.randint(0, pop_size)
                return pop[i] if fit[i] > fit[j] else pop[j]

            p1 = tournament().copy()
            p2 = tournament().copy()

            if rng.rand() < cx_prob:
                alpha = rng.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1, p2

            def mutate(child):
                if rng.rand() < mut_prob:
                    child = child + rng.normal(loc=0.0, scale=mut_sigma, size=k)
                child = project_l1_ball(child, l1_budget)
                child = np.minimum(np.maximum(child, x_lower), x_upper)
                return child

            c1 = mutate(c1)
            c2 = mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.vstack(new_pop)
        fit = np.array([fitness(ind) for ind in pop])

        g_best_idx = int(np.argmax(fit))
        if float(fit[g_best_idx]) > best_fit:
            best_fit = float(fit[g_best_idx])
            best_x = pop[g_best_idx].copy()

    return best_x, best_fit


# ====================== 单个 cluster 任务（并行） ======================

def process_one_cluster(cid, sub_all, robust_df_full):
    stage = "E"
    t_cluster_start = time.time()

    sub_fail = sub_all[sub_all[OUTCOME_COL] == 0].copy()
    sub_succ = sub_all[sub_all[OUTCOME_COL] == 1].copy()

    n_fail = len(sub_fail)
    n_succ = len(sub_succ)

    if n_fail < MIN_FAIL_SAMPLES:
        return {"status": "skip", "reason": f"fail<{MIN_FAIL_SAMPLES}", "cluster": cid, "stage": stage}

    robust_df = robust_df_full[robust_df_full["side"] == "E"].copy()
    if robust_df.empty:
        return {"status": "skip", "reason": "empty robust df", "cluster": cid, "stage": stage}

    kpi_df = select_robust_kpis(robust_df)
    if kpi_df.empty:
        return {"status": "skip", "reason": "no KPI passed", "cluster": cid, "stage": stage}

    kpi_df = dedup_kpi_rows(kpi_df)
    theta_map, dir_map = build_theta_and_direction(kpi_df)
    if not theta_map:
        return {"status": "skip", "reason": "no valid effect", "cluster": cid, "stage": stage}

    kpi_cols_all = sorted([k for k in theta_map.keys() if k in sub_all.columns])
    def_kpis = [c for c in kpi_cols_all if not is_att_kpi(c)]
    att_kpis = [c for c in kpi_cols_all if is_att_kpi(c)]

    if len(def_kpis) < MIN_KPIS_PER_CLUSTER:
        return {"status": "skip", "reason": "too few def kpis", "cluster": cid, "stage": stage}

    stats_def = compute_kpi_stats(sub_all, def_kpis)
    stats_att = compute_kpi_stats(sub_all, att_kpis) if att_kpis else {}

    feature_cols_model = choose_model_features(sub_all, def_kpis, att_kpis)

    t_train_start = time.time()
    predict_prob_func, feature_cols_model = train_predict_model(
        sub_all,
        feature_cols=feature_cols_model,
        random_state=RANDOM_SEED + int(cid),
    )
    train_time_sec = time.time() - t_train_start

    baseline_prob_fail = float(np.mean(predict_prob_func(sub_fail)))
    baseline_prob_all = float(np.mean(predict_prob_func(sub_all)))
    baseline_prob_succ = float(np.mean(predict_prob_func(sub_succ))) if n_succ > 0 else np.nan

    x_lower, x_upper = [], []
    for kpi in def_kpis:
        info = stats_def[kpi]
        std_val = info["std"]
        if std_val is None or np.isnan(std_val) or std_val <= 1e-8:
            l_j, u_j = 0.0, 0.0
        else:
            if dir_map.get(kpi, "increase") == "increase":
                l_j, u_j = 0.0, MAX_L1_STD
            else:
                l_j, u_j = -MAX_L1_STD, 0.0
        x_lower.append(l_j)
        x_upper.append(u_j)

    x_lower = np.array(x_lower, dtype=float)
    x_upper = np.array(x_upper, dtype=float)

    if np.allclose(x_lower, 0.0) and np.allclose(x_upper, 0.0):
        return {"status": "skip", "reason": "no adjustable space", "cluster": cid, "stage": stage}

    t_ga_start = time.time()
    best_x, best_fit = ga_search_policy(
        sub_all=sub_all,
        sub_fail=sub_fail,
        sub_succ=sub_succ,
        def_kpis=def_kpis,
        stats=stats_def,
        dir_map=dir_map,
        predict_prob_func=predict_prob_func,
        x_lower=x_lower,
        x_upper=x_upper,
        baseline_prob_all=baseline_prob_all,
        baseline_prob_fail=baseline_prob_fail,
        baseline_prob_succ=baseline_prob_succ,
        l1_budget=MAX_L1_STD,
        pop_size=GA_POP_SIZE,
        n_generations=GA_N_GEN,
        cx_prob=GA_CX_PROB,
        mut_prob=GA_MUT_PROB,
        mut_sigma=GA_MUT_SIGMA,
        elitism=GA_ELITISM,
        random_state=RANDOM_SEED + 1000 + int(cid),
    )
    ga_time_sec = time.time() - t_ga_start

    fail_shifted = apply_shift_with_clip(sub_fail, best_x, def_kpis, stats_def, dir_map)
    succ_shifted = apply_shift_with_clip(sub_succ, best_x, def_kpis, stats_def, dir_map) if n_succ > 0 else sub_succ
    all_shifted = apply_shift_with_clip(sub_all, best_x, def_kpis, stats_def, dir_map)

    best_prob_fail = float(np.mean(predict_prob_func(fail_shifted)))
    best_prob_all = float(np.mean(predict_prob_func(all_shifted)))
    best_prob_succ = float(np.mean(predict_prob_func(succ_shifted))) if n_succ > 0 else np.nan

    improvement_fail = best_prob_fail - baseline_prob_fail
    improvement_all = best_prob_all - baseline_prob_all
    improvement_succ = best_prob_succ - baseline_prob_succ if n_succ > 0 else np.nan

    best_values_def = {}
    for j, kpi in enumerate(def_kpis):
        info = stats_def[kpi]
        if info["std"] is None or np.isnan(info["std"]):
            approx = info["median"]
        else:
            approx = info["median"] + best_x[j] * info["std"]
        approx = float(np.clip(approx, info["q_lo"], info["q_hi"]))
        best_values_def[kpi] = approx

    rows = []
    def_kpi_index = {k: j for j, k in enumerate(def_kpis)}

    extra_cols_keep = [
        BLOCK_COL, EFFECT_COL, EFFECT_PLUS_COL, T_COL, RV_COL, N_COL,
        POS_RATE_COL, R2D_COL, QVAL_COL, QVAL_PLUS_COL,
        "rel_change", "sign_consistent", "r2_priority_flag", "r2_best_range_flag"
    ]

    for kpi in def_kpis:
        info = stats_def[kpi]
        j = def_kpi_index[kpi]
        theta = theta_map.get(kpi, np.nan)
        direction = dir_map.get(kpi, "increase")

        row = {
            "cluster": cid,
            "stage": stage,
            "kpi": kpi,
            "is_att_kpi": False,
            "direction": direction,
            "effect_per_1sd": theta,
            "base_value_median_all": info["median"],
            "base_value_mean_all": info["mean"],
            "q_lo_all": info["q_lo"],
            "q_hi_all": info["q_hi"],
            "std_all": info["std"],
            "x_ga_std": float(best_x[j]),
            "best_value_ga_approx": float(best_values_def[kpi]),
            "delta_ga_minus_base": float(best_values_def[kpi] - info["median"]),
        }

        kpi_row = kpi_df[kpi_df[TREATMENT_COL] == kpi].iloc[0]
        for c in extra_cols_keep:
            if c in kpi_row.index:
                row[c] = kpi_row[c]
        rows.append(row)

    for kpi in att_kpis:
        info = stats_att[kpi]
        theta = theta_map.get(kpi, np.nan)
        kpi_row = kpi_df[kpi_df[TREATMENT_COL] == kpi].iloc[0]

        row = {
            "cluster": cid,
            "stage": stage,
            "kpi": kpi,
            "is_att_kpi": True,
            "direction": "locked",
            "effect_per_1sd": theta,
            "base_value_median_all": info["median"],
            "base_value_mean_all": info["mean"],
            "q_lo_all": info["q_lo"],
            "q_hi_all": info["q_hi"],
            "std_all": info["std"],
            "x_ga_std": 0.0,
            "best_value_ga_approx": float(info["median"]),
            "delta_ga_minus_base": 0.0,
        }

        for c in extra_cols_keep:
            if c in kpi_row.index:
                row[c] = kpi_row[c]
        rows.append(row)

    df_policy = pd.DataFrame(rows)

    cluster_total_time_sec = time.time() - t_cluster_start

    summary_row = {
        "cluster": cid,
        "stage": stage,
        "n_rows_all": len(sub_all),
        "n_fail": n_fail,
        "n_succ": n_succ,
        "n_def_kpis": len(def_kpis),
        "n_att_kpis": len(att_kpis),
        "baseline_prob_fail_avg": baseline_prob_fail,
        "ga_best_prob_fail_avg": best_prob_fail,
        "ga_improvement_fail": improvement_fail,
        "baseline_prob_succ_avg": baseline_prob_succ,
        "ga_best_prob_succ_avg": best_prob_succ,
        "ga_improvement_succ": improvement_succ,
        "baseline_prob_all_avg": baseline_prob_all,
        "ga_best_prob_all_avg": best_prob_all,
        "ga_improvement_all": improvement_all,
        "ga_best_fitness": float(best_fit),
        "objective_mode": OBJECTIVE_MODE,
        "enforce_fail_nondecrease": ENFORCE_FAIL_NONDECREASE,
        "enforce_success_hard_constraint": ENFORCE_SUCCESS_HARD_CONSTRAINT,
        "succ_max_drop": SUCC_MAX_DROP,
        "Q_LO": Q_LO,
        "Q_HI": Q_HI,
        "MAX_L1_STD": MAX_L1_STD,
        "GA_pop_size": GA_POP_SIZE,
        "GA_n_generations": GA_N_GEN,
        "n_features_model": len(feature_cols_model),
        "train_time_sec": train_time_sec,
        "ga_time_sec": ga_time_sec,
        "cluster_total_time_sec": cluster_total_time_sec,
    }

    return {
        "status": "ok",
        "cluster": cid,
        "stage": stage,
        "policy_df": df_policy,
        "summary_row": summary_row,
    }


# ====================== 并行调度 ======================

def run_for_E_stage_parallel(df_all, cluster_to_results, writer):
    print(f"\n========== 并行计算阶段 E ==========")

    clusters = sorted(df_all[CLUSTER_COL].dropna().unique())
    valid_clusters = [cid for cid in clusters if cid in cluster_to_results]

    futures = []
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for cid in valid_clusters:
            sub_all = df_all[df_all[CLUSTER_COL] == cid].copy()
            robust_df_full = cluster_to_results[cid].copy()

            futures.append(
                ex.submit(
                    process_one_cluster,
                    cid,
                    sub_all,
                    robust_df_full
                )
            )

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res["status"] == "ok":
                s = res["summary_row"]
                print(
                    f"[完成] cluster {res['cluster']} [E] "
                    f"overall Δ={s['ga_improvement_all']:.4f}, "
                    f"fail Δ={s['ga_improvement_fail']:.4f}, "
                    f"succ Δ={s['ga_improvement_succ']:.4f}"
                )
            else:
                print(f"[跳过] cluster {res['cluster']} [E] - {res['reason']}")

    summary_rows = []
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda x: x["cluster"])

    for r in ok_results:
        cid = r["cluster"]
        policy_df = r["policy_df"]
        sheet_name = f"E_cluster_{cid}_policy"[:31]
        policy_df.to_excel(writer, index=False, sheet_name=sheet_name)
        summary_rows.append(r["summary_row"])

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values(["stage", "cluster"])
        df_summary.to_excel(writer, index=False, sheet_name="summary_E")


# ====================== 主流程 ======================

def main():
    t0 = time.time()

    df_all = load_carry_features(CARRY_FEATURES_PATH, CARRY_FEATURES_SHEET)
    cluster_to_results = load_robust_results(
        path=ROBUST_RESULTS_PATH,
        inline_text=ROBUST_RESULTS_INLINE if ROBUST_RESULTS_PATH is None else None,
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_POLICY_XLSX)), exist_ok=True)

    with pd.ExcelWriter(OUTPUT_POLICY_XLSX, engine="xlsxwriter") as writer:
        run_for_E_stage_parallel(df_all, cluster_to_results, writer)

    total_sec = time.time() - t0
    print(f"\n[OK] 已写出结果到：{OUTPUT_POLICY_XLSX}")
    print(f"[总耗时] {total_sec:.2f} 秒，约 {total_sec/60:.2f} 分钟")


if __name__ == "__main__":
    main()
