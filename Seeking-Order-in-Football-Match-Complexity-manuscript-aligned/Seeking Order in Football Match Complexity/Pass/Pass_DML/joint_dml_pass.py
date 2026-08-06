# -*- coding: utf-8 -*-
"""
Pass · Rotating joint and interaction DML for pass-origin and pass-destination-referenced configurations
===========================================================================

This reviewer-facing script reads the completed single-KPI DML workbook, applies
strict prespecified screening, and rotates every KPI that passes the single stage
through a three-mechanism joint interaction model.  For each target KPI, the target
replaces the anchor in its own mechanism family, while the best screened KPI from
the other two families is retained as a companion anchor.  Identical three-KPI
combinations are fitted only once and then mapped back to all target KPIs supplied
by that model.

The script reports average marginal effects, conditional effects at -1/0/+1 SD,
interaction coefficients, a joint Wald test of the three interactions, grouped
cross-fitting, overlap, residual VIF/condition diagnostics, trim sensitivity,
within-match placebo, fold/team stability, alternative nuisance learners and
robustness values for potential omitted-variable bias.  In addition, the RF
outcome nuisance is re-estimated with majority-class undersampling restricted to
every inner/outer TRAINING fold. Calibration-validation and outer-test rows keep
their original class distribution; D|X and the second-stage sample are never
undersampled. External validation is explicitly outside this script because no
external dataset is available.
"""

from __future__ import annotations

import ast
import json
import math
import os

# Prevent BLAS/OpenMP thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import re
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)




class CompleteCaseTransformer(BaseEstimator, TransformerMixin):
    """Fail fast on missing values; never estimate or apply imputation."""

    def __init__(self, strategy=None, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X, y=None):
        self._check(X)
        return self

    def transform(self, X):
        self._check(X)
        return np.asarray(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _check(X):
        values = np.asarray(X, dtype=object)
        if pd.isna(values).any():
            raise ValueError(
                "Missing values detected. The manuscript uses complete-case "
                "screening; remove incomplete rows before model fitting."
            )

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

INPUT_PATH = r""
SHEET_NAME = 0
OUTPUT_XLSX = r""

OUTCOME_COL = "success_def"
CLUSTER_COL = "cluster_id"
MATCH_COL = "__source_file__"       # one source file = one match

# Background categorical variables. Dictionary-like text is supported.
TEAM_COL = "team"
PLAYER_POSITION_COL = "position_name"
PLAY_PATTERN_COL = "play_pattern"
SCORE_STATE_COL = "event_score_state_before"
HOME_AWAY_COL = "pass_team_home_away"

# Categorical controls are one-hot encoded. Constant columns are automatically skipped.
CATEGORICAL_CONTROL_COLS = OrderedDict([
    ("team", TEAM_COL),
    ("position", PLAYER_POSITION_COL),
    ("play_pattern", PLAY_PATTERN_COL),
    ("score_state", SCORE_STATE_COL),
    ("home_away", HOME_AWAY_COL),
])
ADD_CATEGORICAL_DUMMIES = True

RANDOM_SEED = 42
N_SPLITS = 5
ACTIVE_STAGES = ("L", "E'")  # Estimate the two stages separately; never mix their treatments

# Class-imbalance sensitivity. The original-distribution RF joint DML remains the
# primary analysis. Only the RF outcome nuisance training subsets are sampled.
RUN_UNDERSAMPLING_SENSITIVITY = True
UNDERSAMPLING_ACTIVE_STAGES = ACTIVE_STAGES
UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO = 0.50  # 1:2 minority:majority
UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG = 0.30


# Cross-stage attacking-control policy:
# - L target X includes permitted Att(L) and Att(E') KPIs.
# - E' target X includes permitted Att(L) and Att(E') KPIs.
# - The target-matched Att series is removed from BOTH stages.
# - No Def KPI from either stage ever enters X.
INCLUDE_ATT_EPRIME_IN_L_X = False
INCLUDE_ATT_L_IN_EPRIME_X = True
MIN_ROWS = 120
MIN_MATCHES = 20
MIN_CLASS_COUNT = 20


MANUSCRIPT_DEFENSIVE_KPI_BASES = {
    "Adv_5", "Adv_10", "Avg_1_Def", "Avg_3_Def", "Avg_5_Def",
    "DistToDefCentroid", "Area_Def", "Spr_Def",
}
MANUSCRIPT_ATTACKING_KPI_BASES = {
    "Avg_1_Att", "Avg_3_Att", "Avg_5_Att", "DistToAttCentroid",
    "Area_Att", "Spr_Att",
}

# Main inferential choices
MAIN_TRIM_FRAC = 0.02
TRIM_GRID = (0.00, 0.01, 0.02, 0.05)
FDR_ALPHA = 0.05
FOLD_DIRECTION_MIN = 0.80
TEAM_LOO_DIRECTION_MIN = 0.90
PLACEBO_ALPHA = 0.05

# Prespecified primary nuisance learners: Random Forest + Random Forest.
# These are refitted for every target-specific covariate set and every outer fold.
RF_Y_N_TREES = 240
RF_Y_MIN_SAMPLES_LEAF = 8
RF_Y_MAX_FEATURES = "sqrt"
RF_Y_MAX_DEPTH = None
RF_Y_CLASS_WEIGHT = None

RF_D_N_TREES = 240
RF_D_MIN_SAMPLES_LEAF = 5
RF_D_MAX_FEATURES = 0.80
RF_D_MAX_DEPTH = None

# Exact reference copied from the final Pass single-KPI main model.
# The joint script validates these values at startup so the single- and
# multi-treatment estimates remain directly comparable.
SINGLE_MODEL_RF_REFERENCE = {
    "RF_Y_N_TREES": 240,
    "RF_Y_MIN_SAMPLES_LEAF": 8,
    "RF_Y_MAX_FEATURES": "sqrt",
    "RF_Y_MAX_DEPTH": None,
    "RF_Y_CLASS_WEIGHT": None,
    "RF_D_N_TREES": 240,
    "RF_D_MIN_SAMPLES_LEAF": 5,
    "RF_D_MAX_FEATURES": 0.80,
    "RF_D_MAX_DEPTH": None,
}

# Additional nonlinear robustness learners: HGB outcome + ExtraTrees treatment.
HGB_Y_MAX_ITER = 110
HGB_Y_MAX_LEAF_NODES = 15
HGB_Y_MIN_SAMPLES_LEAF = 30
HGB_Y_LEARNING_RATE = 0.08

ET_D_N_TREES = 180
ET_D_MIN_SAMPLES_LEAF = 5
ET_D_MAX_FEATURES = 0.80
N_JOBS = 4

# Optional XGBoost robustness learners. These do not replace the main estimator.
XGB_N_ESTIMATORS = 180
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.85

# Outcome-probability calibration.
# Calibration is nested inside each outer match-grouped cross-fitting training fold.
# The outer test matches are never used to fit the calibrator.
CALIBRATE_MAIN_Y = True
CALIBRATE_HGB_Y = True
CALIBRATE_BASELINE_Y = True
CALIBRATE_XGB_Y = True
CALIBRATION_METHOD = "platt"
CALIBRATION_INNER_SPLITS = 3
CALIBRATION_MIN_ROWS = 120
CALIBRATION_MIN_CLASS_COUNT = 12
PLATT_C = 10000.0
PROB_CLIP_EPS = 1e-5

# Do not use class weighting for XGBoost probability estimation. Class weighting can
# improve ranking while distorting probability calibration. Imbalance is handled by
# the probability loss and the nested calibration step instead.
XGB_SCALE_POS_WEIGHT = 1.0

# Diagnostics
PLACEBO_REPS = 50          # match-preserving OOF placebo repetitions
RUN_HGB_ET_ROBUSTNESS = True  # HGB outcome + ExtraTrees treatment robustness
RUN_BASELINE = True         # Logistic + Ridge nuisance comparison
RUN_XGBOOST_ROBUSTNESS = True  # XGBoost nuisance robustness; safely skipped if unavailable
RUN_TEAM_LOO = True         # second-stage leave-one-team-out check
RUN_JOINT_DML = False
RUN_NONLINEAR_CHECK = False

# Continuous-overlap flags. Diagnostic only; no automatic deletion.
OVERLAP_R2_HIGH = 0.90
OVERLAP_RESID_RATIO_LOW = 0.25
OVERLAP_RESID_RATIO_CAUTION = 0.30  # descriptive caution band, not a conventional cut-off
SUPPORT_BINS = 5

# Sensitivity grid for the reviewer-facing support classification. These are
# descriptive alternatives, not universal cut-offs. The main analysis still uses
# OVERLAP_R2_HIGH and OVERLAP_RESID_RATIO_LOW above.
OVERLAP_R2_SENSITIVITY_GRID = (0.90, 0.95)
OVERLAP_RESID_RATIO_SENSITIVITY_GRID = (0.20, 0.25, 0.30)

# ONLY attacking-side KPIs are eligible KPI covariates. All Def KPIs are excluded.
# Exclude attacking KPIs from the tactical series matching the target treatment.
# Example: when D = Avg_1_Def(E'), Avg_1/3/5_Att(L) and Avg_1/3/5_Att(E')
# are all excluded from X. The same rule applies to L targets.
EXCLUDE_SAME_SERIES_ATT_CONTROLS = True

# True: remove the matched attacking series from BOTH L and E' controls.
# False: remove it only from the target's own stage.
EXCLUDE_SAME_SERIES_ATT_CROSS_STAGE = True

# Reciprocal exclusion between Avg-distance and centroid-distance controls.
#
# When the treatment D is any Avg_*_Def KPI:
#   remove all Avg_*_Att controls and DistToAttCentroid controls
#   from both L and E'.
#
# When the treatment D is DistToDefCentroid:
#   remove DistToAttCentroid controls and all Avg_*_Att controls
#   from both L and E'.
#
# Only attacking-side KPI controls are affected. Every defending KPI is already
# excluded from X by the single-KPI design.
EXCLUDE_AVG_AND_CENTROID_TOGETHER = True

# Families are defined separately for attacking/defending indicators.
# Stage suffixes (L) and (E') are handled automatically.
KPI_FAMILY_PATTERNS = OrderedDict([
    ("distance_centroid_def", [
        r"^Avg_\d+_Def$",
        r"^DistToDefCentroid$",
    ]),
    ("distance_centroid_att", [
        r"^Avg_\d+_Att$",
        r"^DistToAttCentroid$",
    ]),
    ("local_advantage", [r"^Adv_\d+$"]),
    ("structure_shape_def", [
        r"^Area_Def$",
        r"^Spr_Def$",
    ]),
    ("structure_shape_att", [
        r"^Area_Att$",
        r"^Spr_Att$",
    ]),
])

# Stage-specific numeric background controls.
# All score variables use the state BEFORE the current pass.
# Deliberately excluded from X at BOTH stages:
#   location_x, location_y, end_location_x, end_location_y,
#   dx, dy, action_length, action_angle_sin, action_angle_cos,
#   duration, freeze_frame_player_count.
# Therefore the nuisance models do not condition on exact pass location/geometry,
# event duration, or the number of visible players in the freeze frame.
L_BASE_CONTROL_COLS = [
    "period",
    "match_second",
    "event_team_score_before",
    "event_opponent_score_before",
]

# E' uses the same reduced pre-event background as L.
# E' KPIs are still calculated around end_location using the L-time freeze frame,
# but endpoint coordinates, pass geometry, duration, and FF player count are NOT in X.
# Missing or constant columns are skipped safely within each cluster.
E_PRIME_BASE_CONTROL_COLS = [
    "period",
    "match_second",
    "event_team_score_before",
    "event_opponent_score_before",
]

# Explicit audit list: these columns may remain in the source data or be derived during
# preprocessing, but they are never requested as DML covariates in this version.
EXCLUDED_EXACT_SPATIAL_CONTROL_COLS = [
    "location_x",
    "location_y",
    "end_location_x",
    "end_location_y",
    "dx",
    "dy",
    "action_length",
    "action_angle_sin",
    "action_angle_cos",
    "duration",
    "freeze_frame_player_count",
]

# Attacking KPIs never become treatments.
ATT_HINT_PATTERNS = [
    r"DistToAttCentroid",
    r"Area[_\s\-]*Att",
    r"Spr[_\s\-]*Att",
    r"Avg[_\s\-]*\d+[_\s\-]*Att",
    r"Pre[_\s\-]*Att",
    r"Pressure[_\s\-]*Att",
    r"holder.*team0",
    r"attacking",
]

# Pass-aligned rotating joint-model configuration.
# DistToDefCentroid and Avg_*_Def are one mechanism; Area_Def and Spr_Def are another.
JOINT_FAMILY_BASE_KPIS = OrderedDict([
    ("local_advantage", ("Adv_5", "Adv_10")),
    ("distance_centroid", (
        "Avg_1_Def", "Avg_3_Def", "Avg_5_Def", "DistToDefCentroid",
    )),
    ("structure_shape", ("Area_Def", "Spr_Def")),
])

# Cluster- and configuration-specific companion KPIs reported in the manuscript.
# Internal suffix L = pass-origin; E' = pass-destination-referenced.
# Every eligible target replaces the companion from its own mechanism family.
JOINT_ANCHOR_BY_CLUSTER_STAGE = {
    (0, "L"): {
        "local_advantage": "Adv_5(L)",
        "distance_centroid": "Avg_1_Def(L)",
        "structure_shape": "Spr_Def(L)",
    },
    (1, "L"): {
        "local_advantage": "Adv_5(L)",
        "distance_centroid": "Avg_1_Def(L)",
        "structure_shape": "Area_Def(L)",
    },
    (2, "L"): {
        "local_advantage": "Adv_5(L)",
        "distance_centroid": "Avg_1_Def(L)",
        "structure_shape": "Spr_Def(L)",
    },
    (0, "E'"): {
        "local_advantage": "Adv_10(E')",
        "distance_centroid": "Avg_5_Def(E')",
        "structure_shape": "Area_Def(E')",
    },
    (1, "E'"): {
        "local_advantage": "Adv_10(E')",
        "distance_centroid": "Avg_5_Def(E')",
        "structure_shape": "Area_Def(E')",
    },
    (2, "E'"): {
        "local_advantage": "Adv_10(E')",
        "distance_centroid": "Avg_5_Def(E')",
        "structure_shape": "Area_Def(E')",
    },
}

# The public workflow requires candidate eligibility to be established by the
# single-KPI workbook before final interpretation. The rotating analysis still
# evaluates the manuscript KPI inventory so that all joint diagnostics can be
# regenerated; strategy scripts apply the single- and joint-support gates.
ASSUME_ALL_BUILTIN_CANDIDATES_PASSED_SINGLE_STAGE = False

# Raw/object/post-event columns are never auto-added as controls.
RAW_OR_POST_EVENT_PATTERNS = [
    r"^id$", r"^type$", r"^pass$", r"^pass$", r"freeze_frame", r"back_ff",
    r"related_events", r"^location$", r"^end_location$", r"timestamp",
]

# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

@dataclass
class TimerRecord:
    cluster: object
    stage: str
    step: str
    seconds: float
    rows: int = 0
    matches: int = 0
    note: str = ""


class StepTimer:
    def __init__(self, runtime_log: List[TimerRecord], cluster, stage, step,
                 rows: int = 0, matches: int = 0, note: str = ""):
        self.runtime_log = runtime_log
        self.cluster = cluster
        self.stage = stage
        self.step = step
        self.rows = rows
        self.matches = matches
        self.note = note
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[START] cluster={self.cluster} stage={self.stage} step={self.step}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        self.runtime_log.append(TimerRecord(
            cluster=self.cluster,
            stage=self.stage,
            step=self.step,
            seconds=elapsed,
            rows=self.rows,
            matches=self.matches,
            note=self.note if exc_type is None else f"FAILED: {exc_val}",
        ))
        print(f"[DONE ] cluster={self.cluster} stage={self.stage} step={self.step} "
              f"elapsed={format_seconds(elapsed)}")
        return False


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def uniq_keep_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def safe_sheet_name(name: str, used: set) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31]
    base = name
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = (base[:31-len(suffix)] + suffix)
        i += 1
    used.add(name)
    return name


def is_L_col(c: str) -> bool:
    return str(c).strip().endswith("(L)")


def is_Eprime_col(c: str) -> bool:
    return str(c).strip().endswith("(E')")


def is_actual_E_col(c: str) -> bool:
    """Actual terminal-frame E KPI. These columns are ignored completely."""
    s = str(c).strip()
    return s.endswith("(E)") and not s.endswith("(E')")


def remove_stage_suffix(c: str) -> str:
    """Remove the final (L), (E'), or (E) suffix from one KPI name."""
    return re.sub(r"\s*\((?:L|E'|E)\)\s*$", "", str(c).strip())


def get_kpi_family(c: str) -> str:
    """Return the pre-specified tactical family for one KPI."""
    base = remove_stage_suffix(c)
    for family, patterns in KPI_FAMILY_PATTERNS.items():
        if any(re.fullmatch(pattern, base, flags=re.IGNORECASE) for pattern in patterns):
            return family
    # Unlisted KPI: treat it as its own family, so no unrelated KPI is removed.
    return f"singleton::{base}"


def get_kpi_series(c: str) -> str:
    """Return the narrow tactical series used for target-matched Att exclusion.

    The series are deliberately narrower than KPI_FAMILY_PATTERNS:
    Avg, Area, Spread, and Centroid are treated separately. Thus, for example,
    Area_Def removes Area_Att but does not remove Spr_Att or centroid controls.
    """
    base = remove_stage_suffix(c)
    if re.fullmatch(r"Avg_\d+_(?:Def|Att)", base, flags=re.IGNORECASE):
        return "avg"
    if re.fullmatch(r"Area_(?:Def|Att)", base, flags=re.IGNORECASE):
        return "area"
    if re.fullmatch(r"Spr_(?:Def|Att)", base, flags=re.IGNORECASE):
        return "spread"
    if re.fullmatch(r"DistTo(?:Def|Att)Centroid", base, flags=re.IGNORECASE):
        return "centroid"
    if re.fullmatch(r"Adv_\d+", base, flags=re.IGNORECASE):
        return "local_advantage"
    return f"singleton::{base.lower()}"


def attacking_controls_to_exclude_for_targets(
    attacking_cols: Sequence[str],
    selected: Sequence[str],
) -> List[str]:
    """Return attacking KPI controls excluded for the selected treatment(s).

    Base rule:
      remove the attacking-side series matching the defending treatment.

    Additional rule:
      Avg_Def target:
        remove every Avg_Att control and DistToAttCentroid.
      DistToDefCentroid target:
        remove DistToAttCentroid and every Avg_Att control.

    With cross-stage exclusion enabled, the rule is applied to both L and E'.
    """
    if not EXCLUDE_SAME_SERIES_ATT_CONTROLS:
        return []

    target_series = {get_kpi_series(c) for c in selected}
    excluded_series = set(target_series)

    if EXCLUDE_AVG_AND_CENTROID_TOGETHER:
        if "avg" in target_series or "centroid" in target_series:
            excluded_series.update({"avg", "centroid"})

    return [
        c
        for c in attacking_cols
        if get_kpi_series(c) in excluded_series
    ]


def is_att_kpi(c: str) -> bool:
    s = str(c)
    for pat in ATT_HINT_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    low = s.lower()
    return ("att" in low) and ("def" not in low)


def is_raw_or_post_col(c: str) -> bool:
    s = str(c)
    return any(re.search(p, s, flags=re.IGNORECASE) for p in RAW_OR_POST_EVENT_PATTERNS)


def parse_category_name(v) -> str:
    """Parse dictionary-like categorical values and return a stable name."""
    if v is None:
        return "Unknown"
    try:
        if pd.isna(v):
            return "Unknown"
    except Exception:
        pass
    if isinstance(v, dict):
        return str(v.get("name", v.get("id", "Unknown"))).strip()
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null", "<na>"}:
        return "Unknown"
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return str(obj.get("name", obj.get("id", "Unknown"))).strip()
    except Exception:
        pass
    m = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", s)
    return m.group(1).strip() if m else s[:120]


def parse_team_name(v) -> str:
    return parse_category_name(v)


def read_input_table(path: str, sheet_name=0) -> pd.DataFrame:
    """Read CSV or Excel according to the file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported input format: {suffix}")


def add_action_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Create dx/dy/length/angle from start and end coordinates when possible."""
    out = df.copy()

    coordinate_cols = {"location_x", "location_y", "end_location_x", "end_location_y"}
    if coordinate_cols.issubset(out.columns):
        derived_dx = to_num(out["end_location_x"]) - to_num(out["location_x"])
        derived_dy = to_num(out["end_location_y"]) - to_num(out["location_y"])

        if "dx" in out.columns:
            out["dx"] = to_num(out["dx"]).combine_first(derived_dx)
        else:
            out["dx"] = derived_dx

        if "dy" in out.columns:
            out["dy"] = to_num(out["dy"]).combine_first(derived_dy)
        else:
            out["dy"] = derived_dy

    if "dx" in out.columns and "dy" in out.columns:
        dx = to_num(out["dx"])
        dy = to_num(out["dy"])
        out["action_length"] = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)
        out["action_angle_sin"] = np.sin(angle)
        out["action_angle_cos"] = np.cos(angle)

    return out


def numeric_usable_columns(df: pd.DataFrame, cols: Sequence[str],
                           min_nonmissing: int = 20) -> List[str]:
    keep = []
    for c in cols:
        if c not in df.columns:
            continue
        s = to_num(df[c])
        if s.notna().sum() < min_nonmissing:
            continue
        if s.dropna().nunique() <= 1:
            continue
        keep.append(c)
    return keep


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pp = p[ok]
    order = np.argsort(pp)
    ranked = pp[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    back = np.empty(n)
    back[order] = adj
    q[ok] = back
    return q


def apply_fdr_columns(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["q_global"] = bh_fdr(out[p_col].values)
    out["fdr_global_pass"] = out["q_global"] <= FDR_ALPHA
    out["q_within_block"] = np.nan
    for _, idx in out.groupby(["cluster", "stage"], dropna=False).groups.items():
        out.loc[idx, "q_within_block"] = bh_fdr(out.loc[idx, p_col].values)
    out["fdr_block_pass"] = out["q_within_block"] <= FDR_ALPHA
    return out


def check_required_columns(df: pd.DataFrame):
    missing = [c for c in [OUTCOME_COL, CLUSTER_COL, MATCH_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_primary_rf_alignment() -> None:
    """Fail fast if the joint RF settings diverge from the final single-KPI model."""
    current = {
        "RF_Y_N_TREES": RF_Y_N_TREES,
        "RF_Y_MIN_SAMPLES_LEAF": RF_Y_MIN_SAMPLES_LEAF,
        "RF_Y_MAX_FEATURES": RF_Y_MAX_FEATURES,
        "RF_Y_MAX_DEPTH": RF_Y_MAX_DEPTH,
        "RF_Y_CLASS_WEIGHT": RF_Y_CLASS_WEIGHT,
        "RF_D_N_TREES": RF_D_N_TREES,
        "RF_D_MIN_SAMPLES_LEAF": RF_D_MIN_SAMPLES_LEAF,
        "RF_D_MAX_FEATURES": RF_D_MAX_FEATURES,
        "RF_D_MAX_DEPTH": RF_D_MAX_DEPTH,
    }
    mismatches = {
        key: {"joint": current[key], "single_reference": expected}
        for key, expected in SINGLE_MODEL_RF_REFERENCE.items()
        if current.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Primary RF settings no longer match the final Pass single-KPI model: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


# =============================================================================
# 3. GROUPED CROSS-FITTING
# =============================================================================

def make_grouped_splits(y: np.ndarray, groups: np.ndarray,
                        n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = pd.unique(groups)
    k = min(n_splits, len(unique_groups))
    if k < 2:
        raise ValueError("Need at least two matches for cross-fitting.")

    dummy_x = np.zeros((len(y), 1))
    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        splits = list(splitter.split(dummy_x, y, groups=groups))
    else:
        splitter = GroupKFold(n_splits=k)
        splits = list(splitter.split(dummy_x, y, groups=groups))

    # Hard leakage audit.
    for fold, (tr, te) in enumerate(splits):
        tr_groups = set(groups[tr])
        te_groups = set(groups[te])
        if not tr_groups.isdisjoint(te_groups):
            raise RuntimeError(f"Match leakage detected in fold {fold}.")
    return splits


def make_main_y_model(seed: int):
    """Prespecified primary outcome nuisance learner: Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=RF_Y_N_TREES,
        max_depth=RF_Y_MAX_DEPTH,
        min_samples_leaf=RF_Y_MIN_SAMPLES_LEAF,
        max_features=RF_Y_MAX_FEATURES,
        class_weight=RF_Y_CLASS_WEIGHT,
        bootstrap=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


def make_hgb_y_model(seed: int):
    """Additional nonlinear outcome nuisance robustness learner."""
    # Disable the estimator's internal row-wise validation split. Outer and inner
    # validation are both explicitly match-grouped in this script.
    return HistGradientBoostingClassifier(
        learning_rate=HGB_Y_LEARNING_RATE,
        max_iter=HGB_Y_MAX_ITER,
        max_leaf_nodes=HGB_Y_MAX_LEAF_NODES,
        min_samples_leaf=HGB_Y_MIN_SAMPLES_LEAF,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def make_baseline_y_model(seed: int):
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )


def make_main_d_model(seed: int):
    """Prespecified primary treatment nuisance learner: Random Forest regressor."""
    return RandomForestRegressor(
        n_estimators=RF_D_N_TREES,
        max_depth=RF_D_MAX_DEPTH,
        min_samples_leaf=RF_D_MIN_SAMPLES_LEAF,
        max_features=RF_D_MAX_FEATURES,
        bootstrap=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


def make_et_d_model(seed: int):
    """Additional nonlinear treatment nuisance robustness learner."""
    return ExtraTreesRegressor(
        n_estimators=ET_D_N_TREES,
        min_samples_leaf=ET_D_MIN_SAMPLES_LEAF,
        max_features=ET_D_MAX_FEATURES,
        random_state=seed,
        n_jobs=N_JOBS,
        bootstrap=False,
    )

def prepare_fold_X(X: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   scale: bool = False):
    imputer = CompleteCaseTransformer(strategy="median")
    Xtr = imputer.fit_transform(X[tr])
    Xte = imputer.transform(X[te])
    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
    return Xtr, Xte


def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def probability_logit(p: np.ndarray) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1.0 - p))

def undersample_binary_training_indices(
    y_train: np.ndarray,
    target_minority_to_majority_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Randomly reduce only the majority class in a model-training subset.

    The target ratio follows imbalanced-learn semantics:
        n_minority / n_majority = target_minority_to_majority_ratio.
    Every minority row is retained. Validation/test rows never enter this function.
    """
    y_arr = np.asarray(y_train, dtype=int).reshape(-1)
    all_idx = np.arange(len(y_arr), dtype=int)
    classes, counts = np.unique(y_arr, return_counts=True)
    audit: Dict[str, object] = {
        "sampling_method": "random_majority_undersampling_training_only",
        "target_minority_to_majority_ratio": float(target_minority_to_majority_ratio),
        "sampling_applied": False,
        "sampling_reason": "pending",
        "original_n": int(len(y_arr)),
        "sampled_n": int(len(y_arr)),
        "original_class0_n": int(np.sum(y_arr == 0)),
        "original_class1_n": int(np.sum(y_arr == 1)),
        "sampled_class0_n": int(np.sum(y_arr == 0)),
        "sampled_class1_n": int(np.sum(y_arr == 1)),
        "minority_class": np.nan,
        "majority_class": np.nan,
        "original_minority_n": np.nan,
        "original_majority_n": np.nan,
        "sampled_minority_n": np.nan,
        "sampled_majority_n": np.nan,
        "original_minority_to_majority_ratio": np.nan,
        "achieved_minority_to_majority_ratio": np.nan,
    }
    if len(classes) != 2:
        audit["sampling_reason"] = "not_binary_or_single_class"
        return all_idx, audit
    ratio = float(target_minority_to_majority_ratio)
    if not np.isfinite(ratio) or ratio <= 0 or ratio > 1:
        raise ValueError("UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO must be in (0, 1].")

    minority_class = int(classes[int(np.argmin(counts))])
    majority_class = int(classes[int(np.argmax(counts))])
    minority_idx = np.where(y_arr == minority_class)[0]
    majority_idx = np.where(y_arr == majority_class)[0]
    n_minority = len(minority_idx)
    n_majority = len(majority_idx)
    original_ratio = n_minority / n_majority if n_majority else np.nan
    desired_majority = max(n_minority, int(np.floor(n_minority / ratio)))
    audit.update({
        "minority_class": minority_class,
        "majority_class": majority_class,
        "original_minority_n": int(n_minority),
        "original_majority_n": int(n_majority),
        "original_minority_to_majority_ratio": float(original_ratio),
    })
    if n_majority <= desired_majority:
        audit.update({
            "sampling_reason": "already_at_or_above_target_ratio",
            "sampled_minority_n": int(n_minority),
            "sampled_majority_n": int(n_majority),
            "achieved_minority_to_majority_ratio": float(original_ratio),
        })
        return all_idx, audit

    rng = np.random.RandomState(seed)
    kept_majority = rng.choice(majority_idx, size=desired_majority, replace=False)
    sampled_idx = np.concatenate([minority_idx, kept_majority])
    rng.shuffle(sampled_idx)
    sampled_y = y_arr[sampled_idx]
    sampled_minority = int(np.sum(sampled_y == minority_class))
    sampled_majority = int(np.sum(sampled_y == majority_class))
    audit.update({
        "sampling_applied": True,
        "sampling_reason": "majority_rows_randomly_reduced",
        "sampled_n": int(len(sampled_idx)),
        "sampled_class0_n": int(np.sum(sampled_y == 0)),
        "sampled_class1_n": int(np.sum(sampled_y == 1)),
        "sampled_minority_n": sampled_minority,
        "sampled_majority_n": sampled_majority,
        "achieved_minority_to_majority_ratio": (
            sampled_minority / sampled_majority if sampled_majority else np.nan
        ),
    })
    return sampled_idx, audit


def fit_predict_rf_outcome_undersampled(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit the primary RF outcome nuisance after training-only undersampling."""
    imputer = CompleteCaseTransformer(strategy="median")
    Xtr_full = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)
    sample_idx, sample_audit = undersample_binary_training_indices(
        y_train,
        target_minority_to_majority_ratio=UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO,
        seed=seed,
    )
    model = make_main_y_model(seed)
    y_arr = np.asarray(y_train, dtype=int)
    model.fit(Xtr_full[sample_idx], y_arr[sample_idx])
    pred = clip_probability(model.predict_proba(Xte)[:, 1])
    return pred, sample_audit


def nested_group_platt_predict_undersampled(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], pd.DataFrame]:
    """Nested grouped Platt calibration after training-only RF undersampling.

    Inner validation rows used to learn the calibrator and outer test rows retain
    the original class distribution. Each inner model-training subset and the final
    outer-training fit are undersampled independently.
    """
    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train)
    raw_test, outer_sampling = fit_predict_rf_outcome_undersampled(
        X_train, y_train, X_test, seed=seed
    )
    sampling_rows = [dict(
        sampling_level="outer_final_training",
        inner_fold=np.nan,
        **outer_sampling,
    )]
    audit: Dict[str, object] = {
        "learner": "rf_undersampled_training_only",
        "calibration_method": CALIBRATION_METHOD,
        "calibration_requested": bool(CALIBRATE_MAIN_Y),
        "calibration_used": False,
        "calibration_reason": "pending",
        "undersampling_target_ratio": UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO,
        "outer_sampling_applied": outer_sampling["sampling_applied"],
        "outer_original_n": outer_sampling["original_n"],
        "outer_sampled_n": outer_sampling["sampled_n"],
        "outer_original_class0_n": outer_sampling["original_class0_n"],
        "outer_original_class1_n": outer_sampling["original_class1_n"],
        "outer_sampled_class0_n": outer_sampling["sampled_class0_n"],
        "outer_sampled_class1_n": outer_sampling["sampled_class1_n"],
        "outer_achieved_ratio": outer_sampling["achieved_minority_to_majority_ratio"],
        "inner_splits": 0,
        "inner_oof_rows": 0,
        "inner_oof_coverage": 0.0,
        "platt_intercept": np.nan,
        "platt_slope": np.nan,
        "inner_raw_roc_auc": np.nan,
        "inner_raw_pr_auc": np.nan,
        "inner_raw_brier": np.nan,
        "inner_raw_log_loss": np.nan,
        "inner_raw_calibration_intercept": np.nan,
        "inner_raw_calibration_slope": np.nan,
        "inner_raw_ece_10bin": np.nan,
        "inner_calibrated_roc_auc": np.nan,
        "inner_calibrated_pr_auc": np.nan,
        "inner_calibrated_brier": np.nan,
        "inner_calibrated_log_loss": np.nan,
        "inner_calibrated_calibration_intercept": np.nan,
        "inner_calibrated_calibration_slope": np.nan,
        "inner_calibrated_ece_10bin": np.nan,
        "outer_raw_prediction_mean": float(np.mean(raw_test)),
        "outer_calibrated_prediction_mean": float(np.mean(raw_test)),
    }
    if not CALIBRATE_MAIN_Y:
        audit["calibration_reason"] = "not_requested"
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)
    if len(y_train) < CALIBRATION_MIN_ROWS:
        audit["calibration_reason"] = "too_few_training_rows"
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)
    class_counts = np.bincount(y_train, minlength=2)
    if int(class_counts.min()) < CALIBRATION_MIN_CLASS_COUNT:
        audit["calibration_reason"] = "too_few_training_cases_in_one_class"
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)

    inner_splits = make_valid_calibration_splits(
        y_train, groups_train, CALIBRATION_INNER_SPLITS, seed + 100_000
    )
    if not inner_splits:
        audit["calibration_reason"] = "no_valid_inner_grouped_splits"
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)

    inner_raw = np.full(len(y_train), np.nan)
    for inner_fold, (itr, iva) in enumerate(inner_splits):
        pred, sample_audit = fit_predict_rf_outcome_undersampled(
            X_train[itr], y_train[itr], X_train[iva],
            seed=seed + 10_000 + inner_fold,
        )
        inner_raw[iva] = pred
        sampling_rows.append(dict(
            sampling_level="inner_training",
            inner_fold=int(inner_fold),
            **sample_audit,
        ))

    ok = np.isfinite(inner_raw)
    coverage = float(np.mean(ok))
    audit["inner_splits"] = len(inner_splits)
    audit["inner_oof_rows"] = int(ok.sum())
    audit["inner_oof_coverage"] = coverage
    if coverage < 0.95 or len(np.unique(y_train[ok])) < 2:
        audit["calibration_reason"] = "insufficient_inner_oof_coverage"
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)

    calibrator = LogisticRegression(
        C=PLATT_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed + 200_000,
    )
    inner_score = probability_logit(inner_raw[ok]).reshape(-1, 1)
    calibrator.fit(inner_score, y_train[ok])
    calibrated_inner = clip_probability(calibrator.predict_proba(inner_score)[:, 1])
    calibrated_test = clip_probability(
        calibrator.predict_proba(probability_logit(raw_test).reshape(-1, 1))[:, 1]
    )
    for key, value in safe_binary_metrics(y_train[ok], inner_raw[ok]).items():
        audit[f"inner_raw_{key}"] = value
    for key, value in safe_binary_metrics(y_train[ok], calibrated_inner).items():
        audit[f"inner_calibrated_{key}"] = value
    audit.update({
        "calibration_used": True,
        "calibration_reason": "nested_grouped_platt_applied_after_training_only_undersampling",
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "outer_calibrated_prediction_mean": float(np.mean(calibrated_test)),
    })
    return calibrated_test, raw_test, audit, pd.DataFrame(sampling_rows)


def crossfit_undersampled_rf_y(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Dict[str, object]:
    """Grouped OOF outcome predictions for the imbalance-sensitivity branch."""
    n = len(y)
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    calibration_rows = []
    sampling_rows = []
    for fold, (tr, te) in enumerate(splits):
        p, p_raw, audit, sample_df = nested_group_platt_predict_undersampled(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            seed=seed + fold,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": int(fold),
            "outer_train_rows": int(len(tr)),
            "outer_test_rows": int(len(te)),
            "outer_train_matches": int(len(pd.unique(groups[tr]))),
            "outer_test_matches": int(len(pd.unique(groups[te]))),
        })
        for key, value in safe_binary_metrics(y[te], p_raw).items():
            audit[f"outer_raw_{key}"] = value
        for key, value in safe_binary_metrics(y[te], p).items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)
        if not sample_df.empty:
            sample_df = sample_df.copy()
            sample_df.insert(0, "outer_fold", int(fold))
            sampling_rows.append(sample_df)
    return {
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "calibration_audit": pd.DataFrame(calibration_rows),
        "sampling_audit": (
            pd.concat(sampling_rows, ignore_index=True)
            if sampling_rows else pd.DataFrame()
        ),
    }



def fit_predict_raw_outcome_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    learner: str,
    seed: int,
) -> np.ndarray:
    """Fit one raw probability learner with preprocessing learned only from training rows."""
    imputer = CompleteCaseTransformer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)

    learner = str(learner).lower()
    if learner == "logistic":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        model = make_baseline_y_model(seed)
    elif learner == "rf":
        model = make_main_y_model(seed)
    elif learner == "hgb":
        model = make_hgb_y_model(seed)
    elif learner == "xgb":
        model = make_xgb_y_model(seed, y_train)
    else:
        raise ValueError(f"Unknown outcome learner: {learner}")

    model.fit(Xtr, y_train)
    return clip_probability(model.predict_proba(Xte)[:, 1])


def make_valid_calibration_splits(
    y: np.ndarray,
    groups: np.ndarray,
    desired_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Find the largest valid inner grouped split set whose training folds have both classes."""
    max_k = min(int(desired_splits), len(pd.unique(groups)))
    for k in range(max_k, 1, -1):
        try:
            splits = make_grouped_splits(y, groups, k, seed)
        except Exception:
            continue
        valid = True
        for tr, te in splits:
            if len(np.unique(y[tr])) < 2 or len(tr) < 20 or len(te) < 5:
                valid = False
                break
        if valid:
            return splits
    return []


def nested_group_platt_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    learner: str,
    seed: int,
    calibrate: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Fit the raw learner on the outer-training matches and optionally calibrate its
    outer-test probabilities using Platt scaling trained on inner grouped OOF
    predictions from the outer-training matches only.
    """
    y_train = np.asarray(y_train, dtype=int)
    groups_train = np.asarray(groups_train)
    raw_test = fit_predict_raw_outcome_model(
        X_train, y_train, X_test, learner=learner, seed=seed
    )

    audit: Dict[str, object] = {
        "learner": learner,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_requested": bool(calibrate),
        "calibration_used": False,
        "calibration_reason": "not_requested" if not calibrate else "pending",
        "inner_splits": 0,
        "inner_oof_rows": 0,
        "inner_oof_coverage": 0.0,
        "platt_intercept": np.nan,
        "platt_slope": np.nan,
        "inner_raw_roc_auc": np.nan,
        "inner_raw_pr_auc": np.nan,
        "inner_raw_brier": np.nan,
        "inner_raw_log_loss": np.nan,
        "inner_raw_calibration_intercept": np.nan,
        "inner_raw_calibration_slope": np.nan,
        "inner_raw_ece_10bin": np.nan,
        "inner_calibrated_roc_auc": np.nan,
        "inner_calibrated_pr_auc": np.nan,
        "inner_calibrated_brier": np.nan,
        "inner_calibrated_log_loss": np.nan,
        "inner_calibrated_calibration_intercept": np.nan,
        "inner_calibrated_calibration_slope": np.nan,
        "inner_calibrated_ece_10bin": np.nan,
        "outer_raw_prediction_mean": float(np.mean(raw_test)),
        "outer_calibrated_prediction_mean": float(np.mean(raw_test)),
    }

    if not calibrate:
        return raw_test.copy(), raw_test, audit
    if len(y_train) < CALIBRATION_MIN_ROWS:
        audit["calibration_reason"] = "too_few_training_rows"
        return raw_test.copy(), raw_test, audit
    class_counts = np.bincount(y_train, minlength=2)
    if int(class_counts.min()) < CALIBRATION_MIN_CLASS_COUNT:
        audit["calibration_reason"] = "too_few_training_cases_in_one_class"
        return raw_test.copy(), raw_test, audit
    if CALIBRATION_METHOD.lower() != "platt":
        audit["calibration_reason"] = f"unsupported_method::{CALIBRATION_METHOD}"
        return raw_test.copy(), raw_test, audit

    inner_splits = make_valid_calibration_splits(
        y_train, groups_train, CALIBRATION_INNER_SPLITS, seed + 100_000
    )
    if not inner_splits:
        audit["calibration_reason"] = "no_valid_inner_grouped_splits"
        return raw_test.copy(), raw_test, audit

    inner_raw = np.full(len(y_train), np.nan)
    for inner_fold, (itr, iva) in enumerate(inner_splits):
        inner_raw[iva] = fit_predict_raw_outcome_model(
            X_train[itr], y_train[itr], X_train[iva], learner=learner,
            seed=seed + 10_000 + inner_fold,
        )

    ok = np.isfinite(inner_raw)
    coverage = float(np.mean(ok))
    audit["inner_splits"] = len(inner_splits)
    audit["inner_oof_rows"] = int(ok.sum())
    audit["inner_oof_coverage"] = coverage
    if coverage < 0.95 or len(np.unique(y_train[ok])) < 2:
        audit["calibration_reason"] = "insufficient_inner_oof_coverage"
        return raw_test.copy(), raw_test, audit

    calibrator = LogisticRegression(
        C=PLATT_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=seed + 200_000,
    )
    inner_score = probability_logit(inner_raw[ok]).reshape(-1, 1)
    calibrator.fit(inner_score, y_train[ok])

    calibrated_inner = clip_probability(
        calibrator.predict_proba(inner_score)[:, 1]
    )
    calibrated_test = clip_probability(
        calibrator.predict_proba(probability_logit(raw_test).reshape(-1, 1))[:, 1]
    )

    raw_metrics = safe_binary_metrics(y_train[ok], inner_raw[ok])
    cal_metrics = safe_binary_metrics(y_train[ok], calibrated_inner)
    for key, value in raw_metrics.items():
        audit[f"inner_raw_{key}"] = value
    for key, value in cal_metrics.items():
        audit[f"inner_calibrated_{key}"] = value

    audit.update({
        "calibration_used": True,
        "calibration_reason": "nested_grouped_platt_applied",
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "outer_calibrated_prediction_mean": float(np.mean(calibrated_test)),
    })
    return calibrated_test, raw_test, audit


def crossfit_main_and_baseline(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    run_baseline: bool,
):
    n, k_d = D.shape
    yhat_main = np.full(n, np.nan)
    yhat_main_raw = np.full(n, np.nan)
    dhat_main = np.full((n, k_d), np.nan)
    fold_id = np.full(n, -1, dtype=int)

    yhat_base = np.full(n, np.nan) if run_baseline else None
    yhat_base_raw = np.full(n, np.nan) if run_baseline else None
    dhat_base = np.full((n, k_d), np.nan) if run_baseline else None

    fold_rows = []
    calibration_rows = []

    for fold, (tr, te) in enumerate(splits):
        fold_id[te] = fold

        p_main, p_main_raw, main_audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="rf", seed=seed + fold, calibrate=CALIBRATE_MAIN_Y,
        )
        yhat_main[te] = p_main
        yhat_main_raw[te] = p_main_raw
        main_audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        raw_outer = safe_binary_metrics(y[te], p_main_raw)
        cal_outer = safe_binary_metrics(y[te], p_main)
        for key, value in raw_outer.items():
            main_audit[f"outer_raw_{key}"] = value
        for key, value in cal_outer.items():
            main_audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(main_audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        d_model = make_main_d_model(seed + 100 + fold)
        d_model.fit(Xtr, D[tr])
        pred_d = d_model.predict(Xte)
        if pred_d.ndim == 1:
            pred_d = pred_d[:, None]
        dhat_main[te, :] = pred_d

        if run_baseline:
            p_base, p_base_raw, base_audit = nested_group_platt_predict(
                X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
                learner="logistic", seed=seed + 200 + fold,
                calibrate=CALIBRATE_BASELINE_Y,
            )
            yhat_base[te] = p_base
            yhat_base_raw[te] = p_base_raw
            base_audit.update({
                "outer_fold": fold,
                "outer_train_rows": len(tr),
                "outer_test_rows": len(te),
                "outer_train_matches": len(pd.unique(groups[tr])),
                "outer_test_matches": len(pd.unique(groups[te])),
            })
            raw_outer_b = safe_binary_metrics(y[te], p_base_raw)
            cal_outer_b = safe_binary_metrics(y[te], p_base)
            for key, value in raw_outer_b.items():
                base_audit[f"outer_raw_{key}"] = value
            for key, value in cal_outer_b.items():
                base_audit[f"outer_calibrated_{key}"] = value
            calibration_rows.append(base_audit)

            Xtr_s, Xte_s = prepare_fold_X(X, tr, te, scale=True)
            d_base = Ridge(alpha=1.0)
            d_base.fit(Xtr_s, D[tr])
            pred_db = d_base.predict(Xte_s)
            if pred_db.ndim == 1:
                pred_db = pred_db[:, None]
            dhat_base[te, :] = pred_db

        fold_rows.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "train_matches": len(pd.unique(groups[tr])),
            "test_matches": len(pd.unique(groups[te])),
            "train_positive_rate": float(np.mean(y[tr])),
            "test_positive_rate": float(np.mean(y[te])),
            "match_leakage": len(set(groups[tr]).intersection(set(groups[te]))),
            "main_calibration_used": bool(main_audit.get("calibration_used", False)),
            "baseline_calibration_used": (
                bool(base_audit.get("calibration_used", False)) if run_baseline else np.nan
            ),
        })

    return {
        "yhat_main": yhat_main,
        "yhat_main_raw": yhat_main_raw,
        "dhat_main": dhat_main,
        "yhat_base": yhat_base,
        "yhat_base_raw": yhat_base_raw,
        "dhat_base": dhat_base,
        "fold_id": fold_id,
        "fold_audit": pd.DataFrame(fold_rows),
        "calibration_audit": pd.DataFrame(calibration_rows),
    }




def crossfit_hgb_et(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
):
    """Target-specific grouped OOF predictions from HGB + ExtraTrees learners."""
    n, k_d = D.shape
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    dhat = np.full((n, k_d), np.nan)
    calibration_rows = []
    for fold, (tr, te) in enumerate(splits):
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="hgb", seed=seed + fold, calibrate=CALIBRATE_HGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        raw_outer = safe_binary_metrics(y[te], p_raw)
        cal_outer = safe_binary_metrics(y[te], p)
        for key, value in raw_outer.items():
            audit[f"outer_raw_{key}"] = value
        for key, value in cal_outer.items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        dm = make_et_d_model(seed + 100 + fold)
        dm.fit(Xtr, D[tr, 0])
        dhat[te, 0] = np.asarray(dm.predict(Xte)).reshape(-1)
    return {
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "dhat": dhat,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }


def make_xgb_y_model(seed: int, y_train: np.ndarray):
    """Optional XGBoost classifier used only for learner-robustness analysis."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not available")
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=XGB_SCALE_POS_WEIGHT,
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=0,
    )


def make_xgb_d_model(seed: int):
    """Optional XGBoost regressor used only for learner-robustness analysis."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not available")
    return XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=0,
    )


def crossfit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
):
    """Target-specific grouped OOF predictions from calibrated XGBoost learners."""
    n, k_d = D.shape
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    dhat = np.full((n, k_d), np.nan)
    calibration_rows = []
    for fold, (tr, te) in enumerate(splits):
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="xgb", seed=seed + fold, calibrate=CALIBRATE_XGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        raw_outer = safe_binary_metrics(y[te], p_raw)
        cal_outer = safe_binary_metrics(y[te], p)
        for key, value in raw_outer.items():
            audit[f"outer_raw_{key}"] = value
        for key, value in cal_outer.items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        dm = make_xgb_d_model(seed + 100 + fold)
        dm.fit(Xtr, D[tr, 0])
        pred = dm.predict(Xte)
        dhat[te, 0] = np.asarray(pred).reshape(-1)
    return {
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "dhat": dhat,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }



def crossfit_d_only(
    X: np.ndarray,
    D: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    n, k_d = D.shape
    dhat = np.full((n, k_d), np.nan)
    for fold, (tr, te) in enumerate(splits):
        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        model = make_main_d_model(seed + 500 + fold)
        model.fit(Xtr, D[tr])
        pred = model.predict(Xte)
        if pred.ndim == 1:
            pred = pred[:, None]
        dhat[te] = pred
    return dhat


# =============================================================================
# 4. PERFORMANCE AND OVERLAP DIAGNOSTICS
# =============================================================================

def safe_binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(y) & np.isfinite(p)
    y2 = y[ok].astype(float)
    p2 = np.clip(p[ok].astype(float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(ok.sum()),
        "positive_rate": float(np.mean(y2)) if len(y2) else np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
        "log_loss": np.nan,
        "calibration_intercept": np.nan,
        "calibration_slope": np.nan,
        "ece_10bin": np.nan,
    }
    if len(y2) == 0:
        return out
    if len(np.unique(y2)) == 2:
        out["roc_auc"] = float(roc_auc_score(y2, p2))
        out["pr_auc"] = float(average_precision_score(y2, p2))
    out["brier"] = float(brier_score_loss(y2, p2))
    out["log_loss"] = float(log_loss(y2, p2, labels=[0, 1]))

    # Calibration intercept/slope from y ~ 1 + logit(p), reported descriptively.
    try:
        lp = np.log(p2 / (1 - p2))
        cal = sm.GLM(y2, sm.add_constant(lp), family=sm.families.Binomial()).fit(disp=0)
        out["calibration_intercept"] = float(cal.params[0])
        out["calibration_slope"] = float(cal.params[1])
    except Exception:
        pass

    # Expected calibration error using ten fixed probability bins.
    try:
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p2, edges[1:-1], right=False), 0, 9)
        ece = 0.0
        for b in range(10):
            m = idx == b
            if np.any(m):
                ece += float(np.mean(m)) * abs(float(np.mean(y2[m])) - float(np.mean(p2[m])))
        out["ece_10bin"] = float(ece)
    except Exception:
        pass
    return out


def outcome_performance_rows(cluster, stage, y, yhat, model_name):
    m = safe_binary_metrics(y, yhat)
    m.update({"cluster": cluster, "stage": stage, "model": model_name})
    return m


def treatment_performance_rows(cluster, stage, D, Dhat, names, model_name):
    rows = []
    for j, name in enumerate(names):
        ok = np.isfinite(D[:, j]) & np.isfinite(Dhat[:, j])
        if ok.sum() < 5:
            continue
        raw_sd = float(np.std(D[ok, j], ddof=1))
        rmse = float(math.sqrt(mean_squared_error(D[ok, j], Dhat[ok, j])))
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "model": model_name,
            "treatment": name,
            "n": int(ok.sum()),
            "r2": float(r2_score(D[ok, j], Dhat[ok, j])),
            "rmse": rmse,
            "mae": float(mean_absolute_error(D[ok, j], Dhat[ok, j])),
            "raw_sd": raw_sd,
            "nrmse_over_sd": rmse / raw_sd if raw_sd > 0 else np.nan,
        })
    return rows


def support_bin_text(d: np.ndarray, dhat: np.ndarray, bins: int = 5) -> str:
    ok = np.isfinite(d) & np.isfinite(dhat)
    if ok.sum() < max(30, bins * 5):
        return "insufficient"
    d2 = d[ok]
    h2 = dhat[ok]
    try:
        edges = np.unique(np.quantile(h2, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return "predicted treatment nearly constant"
        labels = []
        for b in range(len(edges) - 1):
            if b == len(edges) - 2:
                mask = (h2 >= edges[b]) & (h2 <= edges[b + 1])
            else:
                mask = (h2 >= edges[b]) & (h2 < edges[b + 1])
            if mask.sum() < 5:
                labels.append(f"b{b+1}:n={int(mask.sum())}")
            else:
                p5, p95 = np.quantile(d2[mask], [0.05, 0.95])
                labels.append(f"b{b+1}:[{p5:.3g},{p95:.3g}] n={int(mask.sum())}")
        return " | ".join(labels)
    except Exception as e:
        return f"failed:{e}"



def support_bin_rows(cluster, stage, d, dhat, d_res, treatment, bins: int = 5):
    """Numeric common-support table across predicted-treatment quantile strata."""
    ok = np.isfinite(d) & np.isfinite(dhat) & np.isfinite(d_res)
    if ok.sum() < max(30, bins * 5):
        return []
    raw = np.asarray(d)[ok]
    pred = np.asarray(dhat)[ok]
    resid = np.asarray(d_res)[ok]
    try:
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return []
        rows = []
        for b in range(len(edges) - 1):
            if b == len(edges) - 2:
                m = (pred >= edges[b]) & (pred <= edges[b + 1])
            else:
                m = (pred >= edges[b]) & (pred < edges[b + 1])
            if not np.any(m):
                continue
            rq = np.quantile(raw[m], [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            pq = np.quantile(pred[m], [0.05, 0.50, 0.95])
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "treatment": treatment,
                "predicted_bin": b + 1,
                "n": int(m.sum()),
                "pred_p05": float(pq[0]),
                "pred_p50": float(pq[1]),
                "pred_p95": float(pq[2]),
                "raw_mean": float(np.mean(raw[m])),
                "raw_sd": float(np.std(raw[m], ddof=1)) if m.sum() > 1 else np.nan,
                "raw_p01": float(rq[0]),
                "raw_p05": float(rq[1]),
                "raw_p25": float(rq[2]),
                "raw_p50": float(rq[3]),
                "raw_p75": float(rq[4]),
                "raw_p95": float(rq[5]),
                "raw_p99": float(rq[6]),
                "residual_mean": float(np.mean(resid[m])),
                "residual_sd": float(np.std(resid[m], ddof=1)) if m.sum() > 1 else np.nan,
            })
        return rows
    except Exception:
        return []


def overlap_status_label(r2: float, ratio: float) -> str:
    """Descriptive status. Only FAIL is the prespecified severe-support flag."""
    if np.isfinite(r2) and np.isfinite(ratio):
        if r2 >= OVERLAP_R2_HIGH and ratio <= OVERLAP_RESID_RATIO_LOW:
            return "FAIL_severe_support"
        if r2 >= OVERLAP_R2_HIGH and ratio <= OVERLAP_RESID_RATIO_CAUTION:
            return "CAUTION_near_threshold"
    return "PASS_no_severe_flag"


def overlap_rows(cluster, stage, D, Dhat, D_res, names):
    rows = []
    for j, name in enumerate(names):
        raw = D[:, j]
        pred = Dhat[:, j]
        resid = D_res[:, j]
        ok = np.isfinite(raw) & np.isfinite(pred) & np.isfinite(resid)
        if ok.sum() < 10:
            continue
        sd_raw = float(np.std(raw[ok], ddof=1))
        sd_res = float(np.std(resid[ok], ddof=1))
        ratio = sd_res / sd_raw if sd_raw > 0 else np.nan
        r2 = float(r2_score(raw[ok], pred[ok]))
        q = np.quantile(resid[ok], [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "treatment": name,
            "n": int(ok.sum()),
            "raw_sd": sd_raw,
            "residual_sd": sd_res,
            "residual_sd_ratio": ratio,
            "treatment_oof_r2": r2,
            "resid_p01": q[0],
            "resid_p05": q[1],
            "resid_p25": q[2],
            "resid_p50": q[3],
            "resid_p75": q[4],
            "resid_p95": q[5],
            "resid_p99": q[6],
            "support_by_predicted_bin": support_bin_text(raw, pred, SUPPORT_BINS),
            "overlap_flag": bool(
                np.isfinite(r2) and np.isfinite(ratio) and
                r2 >= OVERLAP_R2_HIGH and ratio <= OVERLAP_RESID_RATIO_LOW
            ),
            "overlap_status": overlap_status_label(r2, ratio),
        })
    return rows


# =============================================================================
# 5. SECOND-STAGE DML AND SENSITIVITY STATISTICS
# =============================================================================

def robustness_value_from_t(t_stat: float, dof: int,
                            q: float = 1.0, alpha: float = 1.0) -> float:
    """Formula used by sensemakr/PySensemakr for a t-statistic and dof."""
    if not np.isfinite(t_stat) or dof <= 1:
        return np.nan
    fq = q * abs(t_stat / np.sqrt(dof))
    if alpha >= 1.0:
        f_crit = 0.0
    else:
        f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
    fqa = fq - f_crit
    if fqa < 0:
        return 0.0
    rv = 0.5 * (np.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    if f_crit > 0 and fq > 1 / f_crit:
        rv = (fq ** 2 - f_crit ** 2) / (1 + fq ** 2)
    return float(np.clip(rv, 0, 1))


def partial_r2_from_t(t_stat: float, dof: int) -> float:
    if not np.isfinite(t_stat) or dof <= 0:
        return np.nan
    return float((t_stat ** 2) / (t_stat ** 2 + dof))


def make_trim_mask(D_res: np.ndarray, trim_frac: float) -> np.ndarray:
    ok = np.all(np.isfinite(D_res), axis=1)
    if trim_frac <= 0 or ok.sum() == 0:
        return ok
    z = D_res[ok].copy()
    sd = np.std(z, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    score = np.max(np.abs(z / sd), axis=1)
    threshold = np.quantile(score, 1 - trim_frac)
    keep_local = score <= threshold
    mask = np.zeros(len(D_res), dtype=bool)
    mask[np.where(ok)[0]] = keep_local
    return mask


def fit_clustered_second_stage(
    y_res: np.ndarray,
    D_res: np.ndarray,
    names: Sequence[str],
    groups: np.ndarray,
    raw_sd_map: Dict[str, float],
    cluster,
    stage: str,
    model_type: str,
    trim_frac: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    mask = np.isfinite(y_res) & np.all(np.isfinite(D_res), axis=1)
    trim_mask = make_trim_mask(D_res, trim_frac)
    mask &= trim_mask

    n = int(mask.sum())
    n_groups = int(len(pd.unique(groups[mask]))) if n else 0
    if n < max(40, len(names) + 10) or n_groups < 5:
        return pd.DataFrame(), {
            "n_used": n,
            "n_matches": n_groups,
            "error": "insufficient second-stage sample",
        }

    X2 = sm.add_constant(D_res[mask], has_constant="add")
    fit = sm.WLS(
        y_res[mask], X2, weights=np.ones(n, dtype=float)
    ).fit(cov_type="HC3", use_t=True)

    params = np.asarray(fit.params)[1:]
    ses = np.asarray(fit.bse)[1:]
    dof_cluster = max(1, int(fit.df_resid))
    tcrit = stats.t.ppf(0.975, dof_cluster)

    rows = []
    for j, name in enumerate(names):
        beta = float(params[j])
        se = float(ses[j])
        tval = beta / se if se > 0 else np.nan
        pval = float(2 * stats.t.sf(abs(tval), dof_cluster)) if np.isfinite(tval) else np.nan
        ci_low = beta - tcrit * se
        ci_high = beta + tcrit * se
        sd_raw = float(raw_sd_map.get(name, 1.0))
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "model_type": model_type,
            "treatment": name,
            "theta_per_unit": beta,
            "se_per_unit": se,
            "ci_low_per_unit": ci_low,
            "ci_high_per_unit": ci_high,
            "theta_per_1sd": beta * sd_raw,
            "se_per_1sd": se * sd_raw,
            "ci_low_per_1sd": ci_low * sd_raw,
            "ci_high_per_1sd": ci_high * sd_raw,
            "hc3_t_stat": tval,
            "t_stat_cluster": tval,  # deprecated compatibility alias
            "p_value": pval,
            "partial_r2_cluster_t": partial_r2_from_t(tval, dof_cluster),
            "robustness_value_point": robustness_value_from_t(tval, dof_cluster, q=1, alpha=1),
            "robustness_value_alpha05": robustness_value_from_t(tval, dof_cluster, q=1, alpha=0.05),
            "n_total": int(len(y_res)),
            "n_used": n,
            "trimmed_n": int(len(y_res) - n),
            "trimmed_proportion": float(1 - n / len(y_res)),
            "trim_fraction_rule": float(trim_frac),
            "n_matches": n_groups,
            "hc3_residual_df": dof_cluster,
            "cluster_se_df": dof_cluster,  # deprecated compatibility alias
            "second_stage": "unit-weight residual WLS + HC3 SE",
            "design_condition_number": float(np.linalg.cond(X2)),
        })

    info = {
        "n_used": n,
        "n_matches": n_groups,
        "r2_second_stage": float(fit.rsquared),
        "condition_number": float(np.linalg.cond(X2)),
        "error": "",
    }
    return pd.DataFrame(rows), info


def run_oneD_all_trims(y_res, D_res, names, groups, raw_sd_map,
                       cluster, stage, model_type):
    out = []
    for j, name in enumerate(names):
        for trim in TRIM_GRID:
            res, _ = fit_clustered_second_stage(
                y_res=y_res,
                D_res=D_res[:, [j]],
                names=[name],
                groups=groups,
                raw_sd_map=raw_sd_map,
                cluster=cluster,
                stage=stage,
                model_type=model_type,
                trim_frac=trim,
            )
            if not res.empty:
                out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# =============================================================================
# 6. FAST PLACEBO AND STABILITY CHECKS USING OOF RESIDUALS
# =============================================================================

def vector_slopes(y: np.ndarray, D: np.ndarray) -> np.ndarray:
    y_c = y - np.mean(y)
    D_c = D - np.mean(D, axis=0, keepdims=True)
    den = np.sum(D_c ** 2, axis=0)
    num = D_c.T @ y_c
    out = np.full(D.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def placebo_within_match(
    y_res: np.ndarray,
    D_res: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    raw_sd_map: Dict[str, float],
    cluster,
    stage: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    ok = np.isfinite(y_res) & np.all(np.isfinite(D_res), axis=1)
    y = y_res[ok]
    D = D_res[ok]
    g = groups[ok]
    if len(y) < 40:
        return pd.DataFrame()

    observed = vector_slopes(y, D)
    rng = np.random.RandomState(seed)
    group_indices = [np.where(g == gv)[0] for gv in pd.unique(g)]
    null = np.full((reps, D.shape[1]), np.nan)

    for r in range(reps):
        Dp = np.empty_like(D)
        for idx in group_indices:
            if len(idx) <= 1:
                Dp[idx] = D[idx]
            else:
                Dp[idx] = D[idx][rng.permutation(len(idx))]
        null[r] = vector_slopes(y, Dp)

    rows = []
    for j, name in enumerate(names):
        vals = null[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        empirical_p = (1 + np.sum(np.abs(vals) >= abs(observed[j]))) / (len(vals) + 1)
        sd_raw = raw_sd_map.get(name, 1.0)
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "treatment": name,
            "placebo_reps": int(len(vals)),
            "observed_theta_per_1sd_untrimmed": float(observed[j] * sd_raw),
            "null_mean_per_1sd": float(np.mean(vals) * sd_raw),
            "null_sd_per_1sd": float(np.std(vals, ddof=1) * sd_raw),
            "null_p025_per_1sd": float(np.quantile(vals, 0.025) * sd_raw),
            "null_p975_per_1sd": float(np.quantile(vals, 0.975) * sd_raw),
            "empirical_placebo_p": float(empirical_p),
            "permutation_scope": "treatment residual permuted within match",
        })
    return pd.DataFrame(rows)


def fold_stability(y_res, D_res, fold_id, names, raw_sd_map, cluster, stage):
    long_rows = []
    for f in sorted(pd.unique(fold_id)):
        if f < 0:
            continue
        mask = (fold_id == f) & np.isfinite(y_res) & np.all(np.isfinite(D_res), axis=1)
        if mask.sum() < 20:
            continue
        slopes = vector_slopes(y_res[mask], D_res[mask])
        for j, name in enumerate(names):
            long_rows.append({
                "cluster": cluster,
                "stage": stage,
                "fold": int(f),
                "treatment": name,
                "theta_per_1sd": float(slopes[j] * raw_sd_map.get(name, 1.0)),
                "n": int(mask.sum()),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summaries = []
    for name, g in long_df.groupby("treatment"):
        vals = g["theta_per_1sd"].values
        med = float(np.nanmedian(vals))
        summaries.append({
            "cluster": cluster,
            "stage": stage,
            "treatment": name,
            "n_folds": len(vals),
            "median_theta_per_1sd": med,
            "min_theta_per_1sd": float(np.nanmin(vals)),
            "max_theta_per_1sd": float(np.nanmax(vals)),
            "iqr_theta_per_1sd": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            "sign_consistency": float(np.mean(np.sign(vals) == np.sign(med))) if med != 0 else np.nan,
        })
    return long_df, pd.DataFrame(summaries)


def team_loo_stability(y_res, D_res, teams, names, raw_sd_map, cluster, stage):
    unique_teams = [t for t in pd.unique(teams) if pd.notna(t)]
    long_rows = []
    for t in unique_teams:
        mask = (teams != t) & np.isfinite(y_res) & np.all(np.isfinite(D_res), axis=1)
        if mask.sum() < 40:
            continue
        slopes = vector_slopes(y_res[mask], D_res[mask])
        for j, name in enumerate(names):
            long_rows.append({
                "cluster": cluster,
                "stage": stage,
                "left_out_team": str(t),
                "treatment": name,
                "theta_per_1sd": float(slopes[j] * raw_sd_map.get(name, 1.0)),
                "n": int(mask.sum()),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summaries = []
    for name, g in long_df.groupby("treatment"):
        vals = g["theta_per_1sd"].values
        med = float(np.nanmedian(vals))
        summaries.append({
            "cluster": cluster,
            "stage": stage,
            "treatment": name,
            "n_leave_one_team_runs": len(vals),
            "median_theta_per_1sd": med,
            "min_theta_per_1sd": float(np.nanmin(vals)),
            "max_theta_per_1sd": float(np.nanmax(vals)),
            "iqr_theta_per_1sd": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
            "sign_consistency": float(np.mean(np.sign(vals) == np.sign(med))) if med != 0 else np.nan,
        })
    return long_df, pd.DataFrame(summaries)


# =============================================================================
# 7. CORRELATION, VIF, JOINT, AND NONLINEAR MODELS
# =============================================================================

def correlation_long(D: np.ndarray, names: Sequence[str], cluster, stage):
    corr = pd.DataFrame(D, columns=names).corr(method="pearson")
    rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "kpi_1": a,
                "kpi_2": b,
                "pearson_r": corr.iloc[i, j],
            })
    return pd.DataFrame(rows), corr


def vif_from_corr(corr: pd.DataFrame, cluster, stage):
    if corr.empty:
        return pd.DataFrame()
    arr = corr.values.astype(float)
    arr = np.nan_to_num(arr, nan=0.0)
    np.fill_diagonal(arr, 1.0)
    inv = np.linalg.pinv(arr)
    return pd.DataFrame({
        "cluster": cluster,
        "stage": stage,
        "treatment": corr.columns,
        "vif_from_inverse_correlation": np.diag(inv),
    })


def _stage_kpi(base: str, stage: str) -> str:
    return f"{base}({stage})"


def build_rotating_joint_specs(cluster, stage: str, D_cols: Sequence[str]):
    """Build Pass-style target rotations and deduplicate identical joint sets.

    Returns
    -------
    model_specs:
        One record per distinct three-treatment joint model.
    target_map:
        One record per target KPI, including the model supplying its coefficient.
    """
    available = set(D_cols)
    key = (int(cluster), str(stage))
    if key not in JOINT_ANCHOR_BY_CLUSTER_STAGE:
        raise KeyError(f"Missing joint anchor configuration for {key}")
    anchors = JOINT_ANCHOR_BY_CLUSTER_STAGE[key]

    required_families = list(JOINT_FAMILY_BASE_KPIS)
    if set(anchors) != set(required_families):
        raise ValueError(f"Incomplete anchor families for {key}: {anchors}")

    model_by_treatments = OrderedDict()
    target_map = []

    for family, bases in JOINT_FAMILY_BASE_KPIS.items():
        for base in bases:
            target = _stage_kpi(base, stage)
            selected_by_family = OrderedDict()
            for other_family in required_families:
                selected_by_family[other_family] = (
                    target if other_family == family else anchors[other_family]
                )
            selected = tuple(selected_by_family.values())
            missing = [name for name in selected if name not in available]

            if missing:
                target_map.append({
                    "cluster": cluster,
                    "stage": stage,
                    "target": target,
                    "target_family": family,
                    "joint_spec": "",
                    "joint_kpis": " | ".join(selected),
                    "configured_family_anchor": anchors[family],
                    "missing_kpis": " | ".join(missing),
                    "single_model_pass_assumed": False,
                    "selection_basis": (
                        "manuscript KPI inventory rotated for diagnostic evaluation; "
                        "single- and joint-support gates are applied before strategy selection"
                    ),
                    "status": "SKIP_missing_KPI",
                })
                continue

            if selected not in model_by_treatments:
                spec_name = "Rotate__" + "__".join(
                    re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
                    for name in selected
                )
                model_by_treatments[selected] = {
                    "joint_spec": spec_name,
                    "selected": list(selected),
                    "target_members": [],
                }
            record = model_by_treatments[selected]
            record["target_members"].append(target)
            target_map.append({
                "cluster": cluster,
                "stage": stage,
                "target": target,
                "target_family": family,
                "joint_spec": record["joint_spec"],
                "joint_kpis": " | ".join(selected),
                "configured_family_anchor": anchors[family],
                "missing_kpis": "",
                "single_model_pass_assumed": False,
                "selection_basis": (
                    "all built-in candidates admitted to joint stage; "
                    "target replaces its own family anchor"
                ),
                "status": "RUN",
            })

    model_specs = list(model_by_treatments.values())
    return model_specs, target_map


def scale_residuals_by_raw_sd(D_res: np.ndarray, names: Sequence[str],
                              raw_sd_map: Dict[str, float]):
    """Scale each treatment residual by the raw KPI SD.

    Coefficients from the resulting joint model are directly interpretable as
    outcome-probability changes per one raw-data SD of the KPI, and the design
    matrix is not dominated by mixed physical units.
    """
    out = np.asarray(D_res, dtype=float).copy()
    for j, name in enumerate(names):
        sd = float(raw_sd_map.get(name, np.nan))
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        out[:, j] /= sd
    return out


def make_interaction_treatment(D_df: pd.DataFrame, pair: Tuple[str, str]):
    """Create one pre-specified interaction from standardized raw KPI values."""
    a, b = pair
    if a not in D_df.columns or b not in D_df.columns:
        return pd.DataFrame(index=D_df.index), {}
    za = (D_df[a] - D_df[a].mean()) / D_df[a].std(ddof=1)
    zb = (D_df[b] - D_df[b].mean()) / D_df[b].std(ddof=1)
    name = f"INT::{a}*{b}"
    out = pd.DataFrame({name: za * zb}, index=D_df.index)
    return out, {name: 1.0}



# =============================================================================
# 8. TARGET-SPECIFIC DATA PREPARATION
# =============================================================================

def prepare_context_dummies(
    sub: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]]]:
    """One-hot encode team, player position, play pattern, score state, and home/away."""
    if TEAM_COL in sub.columns:
        team_names = sub[TEAM_COL].map(parse_team_name).astype(str)
    else:
        team_names = pd.Series("Unknown", index=sub.index, dtype="object")

    if not ADD_CATEGORICAL_DUMMIES:
        return (
            pd.DataFrame(index=sub.index),
            team_names.values,
            {prefix: [] for prefix in CATEGORICAL_CONTROL_COLS},
        )

    dummy_frames = []
    dummy_groups: Dict[str, List[str]] = {}

    for prefix, source_col in CATEGORICAL_CONTROL_COLS.items():
        if source_col not in sub.columns:
            dummy_groups[prefix] = []
            continue

        values = sub[source_col].map(parse_category_name).astype(str)

        # A constant categorical variable contributes no information.
        if values.nunique(dropna=False) <= 1:
            dummy_groups[prefix] = []
            continue

        dummies = pd.get_dummies(
            values,
            prefix=prefix,
            dtype=float,
            drop_first=True,
        )
        dummies = dummies.loc[:, dummies.nunique(dropna=False) > 1]
        dummy_groups[prefix] = list(dummies.columns)

        if not dummies.empty:
            dummy_frames.append(dummies)

    if dummy_frames:
        all_dummies = pd.concat(dummy_frames, axis=1)
        all_dummies = all_dummies.loc[:, ~all_dummies.columns.duplicated()].copy()
    else:
        all_dummies = pd.DataFrame(index=sub.index)

    return all_dummies, team_names.values, dummy_groups


def cluster_seed(cid, stage: str, offset: int = 0) -> int:
    try:
        base = int(float(cid)) * 1000
    except Exception:
        base = abs(hash(str(cid))) % 100000
    stage_offset = 0 if stage == "L" else 500000
    return int(RANDOM_SEED + base + stage_offset + offset)


def build_inventory(sub: pd.DataFrame) -> Dict[str, object]:
    """Identify L/E' KPI families and audit ignored actual-E columns."""
    all_L = numeric_usable_columns(sub, [c for c in sub.columns if is_L_col(c)])
    all_E = numeric_usable_columns(sub, [c for c in sub.columns if is_Eprime_col(c)])
    ignored_actual_E = numeric_usable_columns(
        sub,
        [c for c in sub.columns if is_actual_E_col(c)],
    )
    L_att = [c for c in all_L if is_att_kpi(c)]
    E_att = [c for c in all_E if is_att_kpi(c)]
    L_att = [c for c in L_att if remove_stage_suffix(c) in MANUSCRIPT_ATTACKING_KPI_BASES]
    E_att = [c for c in E_att if remove_stage_suffix(c) in MANUSCRIPT_ATTACKING_KPI_BASES]
    D_L = [c for c in all_L if c not in set(L_att) and remove_stage_suffix(c) in MANUSCRIPT_DEFENSIVE_KPI_BASES]
    D_E = [c for c in all_E if c not in set(E_att) and remove_stage_suffix(c) in MANUSCRIPT_DEFENSIVE_KPI_BASES]
    L_base = numeric_usable_columns(sub, L_BASE_CONTROL_COLS)
    E_base = numeric_usable_columns(sub, E_PRIME_BASE_CONTROL_COLS)
    categorical_dummies, teams, categorical_groups = prepare_context_dummies(sub)
    return {
        "all_L": all_L,
        "all_E": all_E,
        "ignored_actual_E": ignored_actual_E,
        "L_att": L_att,
        "E_att": E_att,
        "D_by_stage": {"L": D_L, "E'": D_E},
        "base_by_stage": {"L": L_base, "E'": E_base},
        "categorical_dummies": categorical_dummies,
        "categorical_groups": categorical_groups,
        "teams": teams,
    }


def build_target_dataset(
    sub: pd.DataFrame,
    inventory: Dict[str, object],
    treatment_cols: Sequence[str],
    stage: str,
) -> Dict[str, object]:
    """Build a single- or multi-treatment DML dataset with matched Att-series exclusion.

    For every target KPI, X contains only the reduced background variables and
    attacking-side Att(L)/Att(E') KPI controls. The complete attacking KPI series
    matching the defending treatment is removed from BOTH stages, and every Def KPI
    remains excluded from X.

    Examples
    --------
    D = Avg_1_Def(L) or Avg_1_Def(E'):
        remove Avg_1/3/5_Att(L) and Avg_1/3/5_Att(E') from X.

    D = Area_Def(L) or Area_Def(E'):
        remove Area_Att(L) and Area_Att(E') from X only; Spr_Att and centroid
        controls remain available.

    D = Adv_5(...):
        no Att-only counterpart exists, so no Att KPI is removed by this rule.
    """
    stage = str(stage).strip().upper()
    if stage not in {"L", "E'"}:
        raise ValueError(f"Unknown stage: {stage}")

    selected = uniq_keep_order(list(treatment_cols))
    if len(selected) < 1:
        raise ValueError(
            "At least one treatment must be supplied to build_target_dataset()."
        )

    missing = [c for c in selected if c not in sub.columns]
    if missing:
        raise ValueError(f"Treatment columns missing: {missing}")

    wrong_stage = [
        c for c in selected
        if (stage == "L" and not is_L_col(c))
        or (stage == "E'" and not is_Eprime_col(c))
    ]
    if wrong_stage:
        raise ValueError(f"Treatments do not match stage {stage}: {wrong_stage}")

    attacking_targets = [c for c in selected if is_att_kpi(c)]
    if attacking_targets:
        raise ValueError(f"Attacking KPIs cannot be treatments: {attacking_targets}")

    y_ser = to_num(sub[OUTCOME_COL])
    group_ser = sub[MATCH_COL]
    D_df_full = sub[selected].apply(to_num)

    complete = y_ser.notna() & group_ser.notna() & D_df_full.notna().all(axis=1)
    work = sub.loc[complete].copy()
    D_df = D_df_full.loc[complete].copy()

    all_L = list(inventory["all_L"])
    all_E = list(inventory["all_E"])
    ignored_actual_E = list(inventory.get("ignored_actual_E", []))
    L_att_all = list(inventory["L_att"])
    E_att_all = list(inventory["E_att"])
    base_controls = list(inventory["base_by_stage"][stage])

    L_def = [c for c in all_L if c not in set(L_att_all)]
    E_def = [c for c in all_E if c not in set(E_att_all)]

    # Always exclude the matched Att series in the target's own stage. With the
    # cross-stage switch enabled (default), exclude the same series in the other
    # stage as well because both Att(L) and Att(E') are otherwise included in X.
    exclude_from_L_att = (
        attacking_controls_to_exclude_for_targets(L_att_all, selected)
        if stage == "L" or EXCLUDE_SAME_SERIES_ATT_CROSS_STAGE
        else []
    )
    exclude_from_E_att = (
        attacking_controls_to_exclude_for_targets(E_att_all, selected)
        if stage == "E'" or EXCLUDE_SAME_SERIES_ATT_CROSS_STAGE
        else []
    )

    excluded_att_set = set(exclude_from_L_att) | set(exclude_from_E_att)
    L_att_controls = [c for c in L_att_all if c not in excluded_att_set]
    E_att_controls = [c for c in E_att_all if c not in excluded_att_set]

    if stage == "L":
        earlier_stage_attacking_controls = []
        same_stage_attacking_controls = list(L_att_controls)
        cross_stage_attacking_controls = (
            list(E_att_controls) if INCLUDE_ATT_EPRIME_IN_L_X else []
        )
        requested_numeric = uniq_keep_order(
            base_controls
            + same_stage_attacking_controls
            + cross_stage_attacking_controls
        )
        excluded_defensive_controls = uniq_keep_order(
            [c for c in L_def if c not in set(selected)] + list(E_def)
        )
        estimand = (
            "joint conditional effects of the selected defending L KPI vector given reduced pre-event "
            "background and permitted Att(L)/Att(E') controls; the target-matched "
            "attacking series is removed from both stages; when D is Avg_Def, "
            "all Avg_Att and DistToAttCentroid controls are removed; when D is "
            "DistToDefCentroid, DistToAttCentroid and all Avg_Att controls are "
            "removed; no defending KPI, actual-E KPI, exact location, geometry, "
            "duration, or freeze-frame-count variable enters X"
        )
    else:
        earlier_stage_attacking_controls = (
            list(L_att_controls) if INCLUDE_ATT_L_IN_EPRIME_X else []
        )
        same_stage_attacking_controls = list(E_att_controls)
        cross_stage_attacking_controls = list(earlier_stage_attacking_controls)
        requested_numeric = uniq_keep_order(
            base_controls
            + earlier_stage_attacking_controls
            + same_stage_attacking_controls
        )
        excluded_defensive_controls = uniq_keep_order(
            list(L_def) + [c for c in E_def if c not in set(selected)]
        )
        estimand = (
            "joint conditional effects of the selected defending E' KPI vector measured around the "
            "pass endpoint in the L-time freeze frame, given reduced pre-event "
            "background and permitted Att(L)/Att(E') controls; the target-matched "
            "attacking series is removed from both stages; when D is Avg_Def, "
            "all Avg_Att and DistToAttCentroid controls are removed; when D is "
            "DistToDefCentroid, DistToAttCentroid and all Avg_Att controls are "
            "removed; no defending KPI, actual-E KPI, exact location, geometry, "
            "duration, or freeze-frame-count variable enters X"
        )

    usable_numeric = numeric_usable_columns(work, requested_numeric)
    X_num_df = (
        work[usable_numeric].apply(to_num)
        if usable_numeric
        else pd.DataFrame(index=work.index)
    )

    categorical_dummies_all = inventory["categorical_dummies"]
    categorical_dummies = categorical_dummies_all.loc[complete].copy()
    X_df = pd.concat([X_num_df, categorical_dummies], axis=1)
    X_df = X_df.loc[:, ~X_df.columns.duplicated()].copy()

    # Hard audits: the target, every defending KPI, and all excluded matched Att
    # controls must be absent from X.
    all_defensive_kpis = set(L_def) | set(E_def)
    X_df = X_df.drop(
        columns=[c for c in X_df.columns if c in all_defensive_kpis],
        errors="ignore",
    )
    X_df = X_df.drop(columns=selected, errors="ignore")
    X_df = X_df.drop(columns=list(excluded_att_set), errors="ignore")
    X_df = X_df.drop(columns=ignored_actual_E, errors="ignore")

    leaked_actual_E = [c for c in X_df.columns if is_actual_E_col(c)]
    if leaked_actual_E:
        raise RuntimeError(
            f"Actual E-stage KPI leaked into X: {leaked_actual_E}"
        )

    leaked_def = [c for c in X_df.columns if c in all_defensive_kpis]
    if leaked_def:
        raise RuntimeError(f"Defending KPI leaked into X: {leaked_def}")

    leaked_matched_att = [c for c in X_df.columns if c in excluded_att_set]
    if leaked_matched_att:
        raise RuntimeError(
            f"Target-matched attacking KPI series leaked into X: {leaked_matched_att}"
        )

    teams_all = np.asarray(inventory["teams"], dtype=object)
    teams = teams_all[np.where(complete.values)[0]]
    groups = group_ser.loc[complete].astype(str).values
    y = (y_ser.loc[complete].values > 0.5).astype(int)

    return {
        "stage": stage,
        "treatment_cols": selected,
        "D_df": D_df,
        "X_df": X_df,
        "y": y,
        "groups": groups,
        "teams": teams,
        "complete_mask": complete,
        "requested_numeric_controls": requested_numeric,
        "usable_numeric_controls": usable_numeric,
        "base_controls": base_controls,
        "earlier_stage_attacking_controls": earlier_stage_attacking_controls,
        "same_stage_attacking_controls": same_stage_attacking_controls,
        "cross_stage_attacking_controls": cross_stage_attacking_controls,
        "target_series": sorted({get_kpi_series(c) for c in selected}),
        "treatment_families": sorted({get_kpi_family(c) for c in selected}),
        "excluded_same_series_att_L": uniq_keep_order(exclude_from_L_att),
        "excluded_same_series_att_Eprime": uniq_keep_order(exclude_from_E_att),
        "excluded_same_series_attacking_controls": uniq_keep_order(
            list(exclude_from_L_att) + list(exclude_from_E_att)
        ),
        "excluded_defensive_controls": excluded_defensive_controls,
        "ignored_actual_E_controls": ignored_actual_E,
        "categorical_control_cols": list(categorical_dummies.columns),
        "team_control_cols": list(inventory["categorical_groups"].get("team", [])),
        "position_control_cols": list(inventory["categorical_groups"].get("position", [])),
        "play_pattern_control_cols": list(inventory["categorical_groups"].get("play_pattern", [])),
        "score_state_control_cols": list(inventory["categorical_groups"].get("score_state", [])),
        "home_away_control_cols": list(inventory["categorical_groups"].get("home_away", [])),
        "estimand": estimand,
    }


def nonlinear_extra_features(D_df: pd.DataFrame, selected: Sequence[str], stage: str):
    """One pre-specified quadratic term and one theory-driven interaction."""
    extras = pd.DataFrame(index=D_df.index)
    if not selected:
        return extras

    z = {}
    for c in selected:
        if c not in D_df.columns:
            continue
        sd = float(D_df[c].std(ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z[c] = (D_df[c] - float(D_df[c].mean())) / sd

    first = next(iter(z), None)
    if first is not None:
        extras[f"SQ::{first}"] = z[first] ** 2

    a, b = INTERACTION_PAIRS.get(stage, ("", ""))
    if a in z and b in z:
        extras[f"INT::{a}*{b}"] = z[a] * z[b]
    elif len(z) >= 2:
        aa, bb = list(z.keys())[:2]
        extras[f"INT::{aa}*{bb}"] = z[aa] * z[bb]
    return extras



# =============================================================================
# 9. REVIEWER-READY CONSOLIDATED REPORTS
# =============================================================================

def _select_model_metrics(df: pd.DataFrame, model: str, prefix: str,
                          treatment_col: str = "treatment") -> pd.DataFrame:
    if df is None or df.empty or "model" not in df.columns:
        return pd.DataFrame()
    z = df.loc[df["model"] == model].copy()
    if z.empty:
        return z
    if "treatment_specific_X" in z.columns and treatment_col not in z.columns:
        z = z.rename(columns={"treatment_specific_X": treatment_col})
    keys = [c for c in ["cluster", "stage", treatment_col] if c in z.columns]
    metric_cols = [c for c in z.columns if c not in keys + ["model"]]
    z = z[keys + metric_cols]
    z = z.rename(columns={c: f"{prefix}{c}" for c in metric_cols})
    return z


def _main_trim_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.loc[np.isclose(df["trim_fraction_rule"], MAIN_TRIM_FRAC)].copy()


def _trim_summary(oneD_df: pd.DataFrame) -> pd.DataFrame:
    if oneD_df is None or oneD_df.empty:
        return pd.DataFrame()
    keys = ["cluster", "stage", "treatment"]
    rows = []
    for key, g in oneD_df.groupby(keys, dropna=False):
        g = g.sort_values("trim_fraction_rule")
        rec = dict(zip(keys, key if isinstance(key, tuple) else [key]))
        values = []
        for _, r in g.iterrows():
            frac = float(r["trim_fraction_rule"])
            label = f"trim_theta_{int(round(frac * 100)):02d}pct"
            rec[label] = r["theta_per_1sd"]
            values.append(float(r["theta_per_1sd"]))
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            nz = arr[np.abs(arr) > 1e-12]
            rec["trim_sign_consistency"] = (
                float(max(np.mean(arr >= 0), np.mean(arr <= 0))) if len(arr) else np.nan
            )
            main_rows = g.loc[np.isclose(g["trim_fraction_rule"], MAIN_TRIM_FRAC)]
            main_theta = float(main_rows.iloc[0]["theta_per_1sd"]) if not main_rows.empty else float(np.median(arr))
            rec["trim_max_abs_change_from_main"] = float(np.max(np.abs(arr - main_theta)))
            rec["trim_max_relative_change_from_main"] = (
                rec["trim_max_abs_change_from_main"] / abs(main_theta)
                if abs(main_theta) > 1e-12 else np.nan
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def build_reviewer_model_report(
    main_df: pd.DataFrame,
    all_trims_df: pd.DataFrame,
    hgb_et_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    y_perf_df: pd.DataFrame,
    d_perf_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
    fold_sum_df: pd.DataFrame,
    team_sum_df: pd.DataFrame,
    data_audit_df: pd.DataFrame,
) -> pd.DataFrame:
    if main_df is None or main_df.empty:
        return pd.DataFrame()
    keys = ["cluster", "stage", "treatment"]
    out = main_df.copy()

    # Outcome nuisance performance.
    for model, prefix in [
        ("RandomForestClassifier_NestedPlatt", "y_main_"),
        ("HistGradientBoosting_NestedPlatt", "y_hgb_"),
        ("LogisticRegression_NestedPlatt", "y_linear_"),
        ("XGBoostClassifier_NestedPlatt", "y_xgb_"),
    ]:
        m = _select_model_metrics(y_perf_df, model, prefix)
        if not m.empty:
            out = out.merge(m, on=keys, how="left")

    # Treatment nuisance performance.
    for model, prefix in [
        ("RandomForestRegressor_singleoutput", "d_main_"),
        ("ExtraTrees_singleoutput", "d_hgb_"),
        ("Ridge_singleoutput", "d_linear_"),
        ("XGBoostRegressor_singleoutput", "d_xgb_"),
    ]:
        m = _select_model_metrics(d_perf_df, model, prefix)
        if not m.empty:
            out = out.merge(m, on=keys, how="left")

    # Full overlap table (main already contains a subset).
    if overlap_df is not None and not overlap_df.empty:
        add_cols = [c for c in overlap_df.columns if c not in keys and c not in out.columns]
        out = out.merge(overlap_df[keys + add_cols], on=keys, how="left")

    # Alternative learner effect estimates at the main trim rule.
    for df_alt, prefix in [(hgb_et_df, "hgb_"), (baseline_df, "linear_"), (xgb_df, "xgb_")]:
        z = _main_trim_rows(df_alt)
        if not z.empty:
            keep = keys + [c for c in [
                "theta_per_1sd", "se_per_1sd", "ci_low_per_1sd", "ci_high_per_1sd",
                "p_value", "q_global", "fdr_global_pass"
            ] if c in z.columns]
            z = z[keep].rename(columns={c: f"{prefix}{c}" for c in keep if c not in keys})
            out = out.merge(z, on=keys, how="left")

    trim_sum = _trim_summary(all_trims_df)
    if not trim_sum.empty:
        out = out.merge(trim_sum, on=keys, how="left")

    # Audit/control-set details.
    if data_audit_df is not None and not data_audit_df.empty:
        audit_keep = keys + [c for c in [
            "rows_cluster", "rows_used", "rows_lost", "rows_lost_proportion",
            "matches", "n_positive", "n_negative", "positive_rate", "n_controls",
            "base_controls", "same_stage_attacking_controls", "cross_stage_attacking_controls",
            "excluded_same_series_attacking_controls", "excluded_defensive_controls",
            "ignored_actual_E_controls", "categorical_controls", "estimand"
        ] if c in data_audit_df.columns]
        audit = data_audit_df[audit_keep].drop_duplicates(keys)
        duplicate_cols = [c for c in audit.columns if c in out.columns and c not in keys]
        audit = audit.drop(columns=duplicate_cols)
        out = out.merge(audit, on=keys, how="left")

    # Directional robustness.
    if "hgb_theta_per_1sd" in out.columns:
        out["hgb_direction_agreement"] = np.where(
            np.isfinite(out["hgb_theta_per_1sd"]),
            np.sign(out["theta_per_1sd"]) == np.sign(out["hgb_theta_per_1sd"]),
            np.nan,
        )
        out["hgb_abs_effect_difference"] = np.abs(out["theta_per_1sd"] - out["hgb_theta_per_1sd"])
    else:
        out["hgb_direction_agreement"] = np.nan

    if "linear_theta_per_1sd" in out.columns:
        out["linear_direction_agreement"] = np.where(
            np.isfinite(out["linear_theta_per_1sd"]),
            np.sign(out["theta_per_1sd"]) == np.sign(out["linear_theta_per_1sd"]),
            np.nan,
        )
        out["linear_abs_effect_difference"] = np.abs(out["theta_per_1sd"] - out["linear_theta_per_1sd"])
    else:
        out["linear_direction_agreement"] = np.nan

    if "xgb_theta_per_1sd" in out.columns:
        out["xgb_direction_agreement"] = np.where(
            np.isfinite(out["xgb_theta_per_1sd"]),
            np.sign(out["theta_per_1sd"]) == np.sign(out["xgb_theta_per_1sd"]),
            np.nan,
        )
        out["xgb_abs_effect_difference"] = np.abs(out["theta_per_1sd"] - out["xgb_theta_per_1sd"])
    else:
        out["xgb_direction_agreement"] = np.nan

    # Explicit descriptive support status.
    if "overlap_status" not in out.columns:
        out["overlap_status"] = [
            overlap_status_label(r, q) for r, q in zip(out["treatment_oof_r2"], out["residual_sd_ratio"])
        ]

    # Reviewer-oriented evidence label. This does not replace the continuous diagnostics.
    def evidence_label(r):
        if bool(r.get("overlap_flag", False)):
            return "NOT_PRIMARY_overlap_failure"
        if not bool(r.get("fdr_global_pass", False)):
            return "NOT_PRIMARY_not_FDR_significant"
        ci_low = pd.to_numeric(pd.Series([r.get("ci_low_per_1sd")]), errors="coerce").iloc[0]
        ci_high = pd.to_numeric(pd.Series([r.get("ci_high_per_1sd")]), errors="coerce").iloc[0]
        if not (np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)):
            return "NOT_PRIMARY_CI_includes_zero"
        if float(r.get("fold_sign_consistency", 0) or 0) < FOLD_DIRECTION_MIN:
            return "CAUTION_fold_instability"
        if float(r.get("team_loo_sign_consistency", 0) or 0) < TEAM_LOO_DIRECTION_MIN:
            return "CAUTION_team_LOO_instability"
        if float(r.get("empirical_placebo_p", 1) or 1) > PLACEBO_ALPHA:
            return "CAUTION_placebo"
        if r.get("overlap_status") == "CAUTION_near_threshold":
            return "PRIMARY_with_support_caution"
        if pd.notna(r.get("hgb_direction_agreement")) and not bool(r.get("hgb_direction_agreement")):
            return "CAUTION_hgb_direction"
        if pd.notna(r.get("linear_direction_agreement")) and not bool(r.get("linear_direction_agreement")):
            return "CAUTION_linear_direction"
        if pd.notna(r.get("xgb_direction_agreement")) and not bool(r.get("xgb_direction_agreement")):
            return "CAUTION_xgb_direction"
        return "PRIMARY"

    out["reviewer_evidence_status"] = out.apply(evidence_label, axis=1)
    return out


def build_reviewer_overview(report: pd.DataFrame, fold_audit_df: pd.DataFrame) -> pd.DataFrame:
    if report is None or report.empty:
        return pd.DataFrame()
    groups = [("ALL", "ALL", report)]
    groups += [(c, s, g) for (c, s), g in report.groupby(["cluster", "stage"], dropna=False)]
    rows = []
    for c, s, g in groups:
        fdr_s = g["fdr_global_pass"].fillna(False) if "fdr_global_pass" in g.columns else pd.Series(False, index=g.index)
        overlap_s = g["overlap_flag"].fillna(False) if "overlap_flag" in g.columns else pd.Series(False, index=g.index)
        overlap_status_s = g["overlap_status"].astype(str) if "overlap_status" in g.columns else pd.Series("", index=g.index)
        candidate_s = g["candidate_for_discussion"].fillna(False) if "candidate_for_discussion" in g.columns else pd.Series(False, index=g.index)
        evidence_s = g["reviewer_evidence_status"].astype(str) if "reviewer_evidence_status" in g.columns else pd.Series("", index=g.index)
        rec = {
            "cluster": c,
            "stage": s,
            "n_treatments": len(g),
            "n_fdr_global_pass": int(fdr_s.sum()),
            "n_overlap_fail": int(overlap_s.sum()),
            "n_overlap_caution": int((overlap_status_s == "CAUTION_near_threshold").sum()),
            "n_candidate_for_discussion": int(candidate_s.sum()),
            "n_reviewer_primary": int(evidence_s.str.startswith("PRIMARY").sum()),
        }
        for col in [
            "y_main_roc_auc", "y_main_pr_auc", "y_main_brier", "y_main_log_loss",
            "y_main_calibration_intercept", "y_main_calibration_slope", "y_main_ece_10bin",
            "d_main_r2", "d_main_nrmse_over_sd", "residual_sd_ratio",
            "fold_sign_consistency", "team_loo_sign_consistency", "empirical_placebo_p",
        ]:
            if col in g.columns:
                rec[f"median_{col}"] = float(pd.to_numeric(g[col], errors="coerce").median())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if fold_audit_df is not None and not fold_audit_df.empty:
        out["max_match_leakage"] = int(pd.to_numeric(fold_audit_df["match_leakage"], errors="coerce").max())
    return out



def build_overlap_threshold_sensitivity(overlap_df: pd.DataFrame) -> pd.DataFrame:
    """Report how support classifications change across plausible diagnostic grids."""
    if overlap_df is None or overlap_df.empty:
        return pd.DataFrame()
    rows = []
    grouping = [("ALL", "ALL", overlap_df)]
    grouping += [(c, st, g) for (c, st), g in overlap_df.groupby(["cluster", "stage"], dropna=False)]
    for r2_cut in OVERLAP_R2_SENSITIVITY_GRID:
        for resid_cut in OVERLAP_RESID_RATIO_SENSITIVITY_GRID:
            for cluster, stage, g in grouping:
                r2 = pd.to_numeric(g["treatment_oof_r2"], errors="coerce")
                rr = pd.to_numeric(g["residual_sd_ratio"], errors="coerce")
                flag = (r2 >= float(r2_cut)) & (rr <= float(resid_cut))
                names = g.loc[flag, "treatment"].astype(str).tolist()
                rows.append({
                    "cluster": cluster,
                    "stage": stage,
                    "r2_threshold": float(r2_cut),
                    "residual_sd_ratio_threshold": float(resid_cut),
                    "n_models": int(len(g)),
                    "n_flagged": int(flag.sum()),
                    "flagged_proportion": float(flag.mean()) if len(g) else np.nan,
                    "flagged_treatments": " | ".join(names),
                    "is_prespecified_main_rule": bool(
                        np.isclose(r2_cut, OVERLAP_R2_HIGH) and
                        np.isclose(resid_cut, OVERLAP_RESID_RATIO_LOW)
                    ),
                })
    return pd.DataFrame(rows)


def build_reviewer_checklist(
    report: pd.DataFrame,
    y_perf_df: pd.DataFrame,
    d_perf_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    support_df: pd.DataFrame,
    all_trims_df: pd.DataFrame,
    hgb_et_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    fold_sum_df: pd.DataFrame,
    team_sum_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
    fold_audit_df: pd.DataFrame,
    data_audit_df: pd.DataFrame,
    calibration_audit_df: pd.DataFrame,
    overlap_threshold_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    no_leak = (
        fold_audit_df is not None and not fold_audit_df.empty and
        pd.to_numeric(fold_audit_df["match_leakage"], errors="coerce").fillna(1).max() == 0
    )
    items = [
        ("Grouped cross-fitting by match", "Train/test matches are disjoint in every fold", "21_fold_audit", "PASS" if no_leak else "CHECK"),
        ("Outcome nuisance discrimination/calibration", "Raw and nested-calibrated OOF ROC-AUC, PR-AUC, Brier, log loss, calibration intercept/slope, ECE", "11_y_performance and 02_reviewer_full_report", "REPORTED" if y_perf_df is not None and not y_perf_df.empty else "MISSING"),
        ("Leakage-safe probability calibration", "Inner match-grouped OOF Platt scaling fitted only inside each outer training fold", "27_y_calibration_audit", "REPORTED" if calibration_audit_df is not None and not calibration_audit_df.empty else "MISSING"),
        ("Treatment nuisance prediction", "OOF R2, RMSE, MAE, normalized RMSE", "12_D_performance and 02_reviewer_full_report", "REPORTED" if d_perf_df is not None and not d_perf_df.empty else "MISSING"),
        ("Continuous-treatment residual support", "OOF treatment R2, residual SD ratio, residual quantiles, severe-support flag", "10_overlap", "REPORTED" if overlap_df is not None and not overlap_df.empty else "MISSING"),
        ("Common support across predicted-treatment strata", "Raw treatment distribution in predicted-treatment quantile bins", "03_reviewer_support_bins", "REPORTED" if support_df is not None and not support_df.empty else "MISSING"),
        ("Support-threshold sensitivity", "Counts and identities of flagged treatments across R2 and residual-SD threshold grids", "28_overlap_threshold_sens", "REPORTED" if overlap_threshold_sensitivity_df is not None and not overlap_threshold_sensitivity_df.empty else "MISSING"),
        ("Trimming sensitivity", "0%, 1%, 2%, and 5% residual-support trimming", "06_oneD_all_trims and 02_reviewer_full_report", "REPORTED" if all_trims_df is not None and not all_trims_df.empty else "MISSING"),
        ("Primary RF+RF learner", "Nested-Platt RandomForestClassifier outcome nuisance + RandomForestRegressor treatment nuisance", "05_main_oneD, 11_y_performance, 12_D_performance", "REPORTED" if report is not None and not report.empty else "MISSING"),
        ("HGB + ExtraTrees learner robustness", "Nested-Platt HistGradientBoosting outcome nuisance + ExtraTrees treatment nuisance", "08_hgb_et_oneD", "REPORTED" if hgb_et_df is not None and not hgb_et_df.empty else "NOT_RUN"),
        ("Linear learner robustness", "Nested-Platt Logistic outcome nuisance + Ridge treatment nuisance", "07_linear_oneD", "REPORTED" if baseline_df is not None and not baseline_df.empty else "NOT_RUN"),
        ("XGBoost learner robustness", "Unweighted nested-Platt XGBoost outcome nuisance + XGBoost treatment nuisance", "09_xgboost_oneD", "REPORTED" if xgb_df is not None and not xgb_df.empty else ("UNAVAILABLE" if not HAS_XGBOOST else "NOT_RUN")),
        ("Cross-fold stability", "Fold-specific effect direction and IQR", "14_fold_stab_sum / 15_fold_stab_long", "REPORTED" if fold_sum_df is not None and not fold_sum_df.empty else "MISSING"),
        ("Leave-one-team-out stability", "Effect direction and IQR after omitting each team", "16_teamLOO_sum / 17_teamLOO_long", "REPORTED" if team_sum_df is not None and not team_sum_df.empty else "NOT_RUN"),
        ("Placebo test", "Within-match permutation of treatment residuals", "13_placebo", "REPORTED" if placebo_df is not None and not placebo_df.empty else "MISSING"),
        ("Multiple testing", "BH-FDR globally and within cluster-stage", "05_main_oneD / 02_reviewer_full_report", "REPORTED" if report is not None and not report.empty and "q_global" in report.columns else "MISSING"),
        ("Second-stage uncertainty", "unit-weight residual WLS with HC3 heteroskedasticity-robust standard errors", "05_main_oneD", "REPORTED" if report is not None and not report.empty else "MISSING"),
        ("Covariate and leakage audit", "Exact controls, same-series exclusions, all defensive KPI exclusions", "20_data_audit and 04_manifest", "REPORTED" if data_audit_df is not None and not data_audit_df.empty else "MISSING"),
        ("Cluster-number sensitivity K=2-5", "Silhouette, CH, DB, inertia, minimum cluster size, ARI stability", "Separate clustering-sensitivity outputs", "SEPARATE_ANALYSIS_REQUIRED"),
    ]
    return pd.DataFrame(items, columns=["reviewer_requirement", "reported_evidence", "workbook_location", "status"])


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================

def main():
    total_start = time.perf_counter()
    runtime_log: List[TimerRecord] = []
    run_log = []
    data_audit = []
    fold_audits = []
    y_perf = []
    d_perf = []
    overlap_all = []
    oneD_all = []
    hgb_et_oneD_all = []
    baseline_oneD_all = []
    xgb_oneD_all = []
    support_bin_all = []
    placebo_all = []
    fold_stab_long_all = []
    fold_stab_summary_all = []
    team_loo_long_all = []
    team_loo_summary_all = []
    corr_all = []
    vif_all = []
    joint_all = []
    nonlinear_all = []
    joint_selection_log = []
    calibration_audits = []

    print("=" * 100)
    print("PASS · SINGLE-KPI DML · L AND E' SEPARATELY · PASS-ALIGNED RF+RF PRIMARY")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_XLSX}")
    print("=" * 100)

    with StepTimer(runtime_log, "ALL", "ALL", "read_and_prepare"):
        df = read_input_table(INPUT_PATH, sheet_name=SHEET_NAME)
        df.columns = [str(c).strip() for c in df.columns]
        check_required_columns(df)

        df[OUTCOME_COL] = to_num(df[OUTCOME_COL])
        df = add_action_geometry(df)

        eprime_def_columns = [
            c for c in df.columns
            if is_Eprime_col(c) and not is_att_kpi(c)
        ]
        if not eprime_def_columns:
            actual_e_columns = [
                c for c in df.columns
                if is_actual_E_col(c)
            ]
            raise ValueError(
                "输入数据没有可用的Def(E') KPI。"
                "本代码不会把实际(E) KPI改名或代替(E')。"
                f"检测到的实际E列数量={len(actual_e_columns)}。"
                "请先在L时刻freeze_frame上围绕pass end_location计算E' KPI。"
            )

    clusters = sorted(df[CLUSTER_COL].dropna().unique())

    for cid in clusters:
        cluster_start = time.perf_counter()
        sub = df.loc[df[CLUSTER_COL] == cid].copy()
        n_rows_cluster = len(sub)
        n_matches_cluster = sub[MATCH_COL].nunique(dropna=True)

        print("\n" + "#" * 100)
        print(f"CLUSTER {cid}: rows={n_rows_cluster:,}, matches={n_matches_cluster}")
        print("#" * 100)

        with StepTimer(runtime_log, cid, "ALL", "identify_KPIs", n_rows_cluster, n_matches_cluster):
            inventory = build_inventory(sub)

        for stage in ACTIVE_STAGES:
            D_candidates = list(inventory["D_by_stage"].get(stage, []))
            if not D_candidates:
                run_log.append({"cluster": cid, "stage": stage, "status": "skip", "reason": f"no {stage} non-Att KPI treatments"})
                continue

            corr_complete = sub[D_candidates].apply(to_num).dropna()
            if len(corr_complete) >= 20:
                with StepTimer(runtime_log, cid, stage, "correlation_vif", len(corr_complete), n_matches_cluster):
                    corr_long, corr_matrix = correlation_long(
                        corr_complete.values.astype(float), D_candidates, cid, stage
                    )
                    corr_all.append(corr_long)
                    vif_all.append(vif_from_corr(corr_matrix, cid, stage))

            # --------------------------------------------------------------
            # One target KPI at a time. X is rebuilt and nuisances are refit.
            # --------------------------------------------------------------
            for d_index, d_col in enumerate(D_candidates):
                target_start = time.perf_counter()
                print("\n" + "-" * 100)
                print(f"[TARGET] cluster={cid} stage={stage} D={d_col}")
                print("-" * 100)
                try:
                    with StepTimer(runtime_log, cid, stage, f"prepare::{d_col}", n_rows_cluster, n_matches_cluster):
                        block = build_target_dataset(sub, inventory, [d_col], stage)

                    X_df = block["X_df"]
                    D_df = block["D_df"]
                    y = block["y"]
                    groups = block["groups"]
                    teams = block["teams"]
                    n = len(y)
                    n_matches = len(pd.unique(groups))
                    n_pos = int(y.sum())
                    n_neg = int(n - n_pos)

                    data_audit.append({
                        "cluster": cid,
                        "stage": stage,
                        "model_scope": "one_KPI_overall_conditional_effect",
                        "treatment": d_col,
                        "rows_cluster": n_rows_cluster,
                        "rows_used": n,
                        "rows_lost": n_rows_cluster - n,
                        "rows_lost_proportion": 1 - n / n_rows_cluster if n_rows_cluster else np.nan,
                        "matches": n_matches,
                        "n_positive": n_pos,
                        "n_negative": n_neg,
                        "positive_rate": n_pos / n if n else np.nan,
                        "n_controls": X_df.shape[1],
                        "target_in_controls": d_col in X_df.columns,
                        "base_controls": " | ".join(block["base_controls"]),
                        "earlier_stage_attacking_controls": " | ".join(block["earlier_stage_attacking_controls"]),
                        "same_stage_attacking_controls": " | ".join(block["same_stage_attacking_controls"]),
                        "cross_stage_attacking_controls": " | ".join(block["cross_stage_attacking_controls"]),
                        "target_series": " | ".join(block["target_series"]),
                        "treatment_families": " | ".join(block["treatment_families"]),
                        "excluded_same_series_att_L": " | ".join(block["excluded_same_series_att_L"]),
                        "excluded_same_series_att_Eprime": " | ".join(block["excluded_same_series_att_Eprime"]),
                        "excluded_same_series_attacking_controls": " | ".join(block["excluded_same_series_attacking_controls"]),
                        "excluded_defensive_controls": " | ".join(block["excluded_defensive_controls"]),
                        "ignored_actual_E_controls": " | ".join(block["ignored_actual_E_controls"]),
                        "usable_numeric_controls": " | ".join(block["usable_numeric_controls"]),
                        "categorical_controls": " | ".join(block["categorical_control_cols"]),
                        "team_controls": " | ".join(block["team_control_cols"]),
                        "position_controls": " | ".join(block["position_control_cols"]),
                        "play_pattern_controls": " | ".join(block["play_pattern_control_cols"]),
                        "score_state_controls": " | ".join(block["score_state_control_cols"]),
                        "home_away_controls": " | ".join(block["home_away_control_cols"]),
                        "estimand": block["estimand"],
                    })

                    if (
                        n < MIN_ROWS or n_matches < MIN_MATCHES or
                        min(n_pos, n_neg) < MIN_CLASS_COUNT or X_df.shape[1] == 0
                    ):
                        reason = f"skip n={n}, matches={n_matches}, class_min={min(n_pos,n_neg)}, X={X_df.shape[1]}"
                        print(f"[SKIP] {reason}")
                        run_log.append({"cluster": cid, "stage": stage, "treatment": d_col, "status": "skip", "reason": reason})
                        continue

                    X = X_df.values.astype(float)
                    D = D_df[[d_col]].values.astype(float)
                    raw_sd_map = {d_col: float(np.std(D[:, 0], ddof=1))}

                    with StepTimer(runtime_log, cid, stage, f"folds::{d_col}", n, n_matches):
                        splits = make_grouped_splits(
                            y, groups, N_SPLITS, cluster_seed(cid, stage, d_index)
                        )

                    with StepTimer(runtime_log, cid, stage, f"crossfit::{d_col}", n, n_matches,
                                   note=f"target-specific X={X.shape[1]}"):
                        cf = crossfit_main_and_baseline(
                            X=X, y=y, D=D, groups=groups, splits=splits,
                            seed=cluster_seed(cid, stage, 100 + d_index * 13),
                            run_baseline=RUN_BASELINE,
                        )

                    fa = cf["fold_audit"].copy()
                    fa.insert(0, "treatment", d_col)
                    fa.insert(0, "stage", stage)
                    fa.insert(0, "cluster", cid)
                    fold_audits.append(fa)

                    cal_audit = cf.get("calibration_audit", pd.DataFrame()).copy()
                    if not cal_audit.empty:
                        cal_audit.insert(0, "treatment", d_col)
                        cal_audit.insert(0, "stage", stage)
                        cal_audit.insert(0, "cluster", cid)
                        calibration_audits.append(cal_audit)

                    yhat = cf["yhat_main"]
                    Dhat = cf["dhat_main"]
                    y_res = y - yhat
                    D_res = D - Dhat

                    yp_raw = outcome_performance_rows(
                        cid, stage, y, cf["yhat_main_raw"], "RandomForestClassifier_raw"
                    )
                    yp_raw["treatment_specific_X"] = d_col
                    yp_raw["n_controls"] = X.shape[1]
                    y_perf.append(yp_raw)

                    yp = outcome_performance_rows(
                        cid, stage, y, yhat, "RandomForestClassifier_NestedPlatt"
                    )
                    yp["treatment_specific_X"] = d_col
                    yp["n_controls"] = X.shape[1]
                    y_perf.append(yp)

                    dp = treatment_performance_rows(cid, stage, D, Dhat, [d_col], "RandomForestRegressor_singleoutput")
                    for row in dp:
                        row["n_controls"] = X.shape[1]
                    d_perf.extend(dp)
                    overlap_all.extend(overlap_rows(cid, stage, D, Dhat, D_res, [d_col]))
                    support_bin_all.extend(
                        support_bin_rows(cid, stage, D[:, 0], Dhat[:, 0], D_res[:, 0], d_col, SUPPORT_BINS)
                    )

                    with StepTimer(runtime_log, cid, stage, f"oneD_trims::{d_col}", n, n_matches):
                        one = run_oneD_all_trims(
                            y_res, D_res, [d_col], groups, raw_sd_map,
                            cid, stage, "random_forest_primary_target_specific_X",
                        )
                        if not one.empty:
                            one["n_controls"] = X.shape[1]
                            oneD_all.append(one)

                    if RUN_BASELINE:
                        yhat_b = cf["yhat_base"]
                        Dhat_b = cf["dhat_base"]
                        y_res_b = y - yhat_b
                        D_res_b = D - Dhat_b
                        ypb_raw = outcome_performance_rows(
                            cid, stage, y, cf["yhat_base_raw"], "LogisticRegression_raw"
                        )
                        ypb_raw["treatment_specific_X"] = d_col
                        ypb_raw["n_controls"] = X.shape[1]
                        y_perf.append(ypb_raw)

                        ypb = outcome_performance_rows(
                            cid, stage, y, yhat_b, "LogisticRegression_NestedPlatt"
                        )
                        ypb["treatment_specific_X"] = d_col
                        ypb["n_controls"] = X.shape[1]
                        y_perf.append(ypb)
                        dpb = treatment_performance_rows(cid, stage, D, Dhat_b, [d_col], "Ridge_singleoutput")
                        for row in dpb:
                            row["n_controls"] = X.shape[1]
                        d_perf.extend(dpb)
                        with StepTimer(runtime_log, cid, stage, f"baseline::{d_col}", n, n_matches):
                            base_one = run_oneD_all_trims(
                                y_res_b, D_res_b, [d_col], groups, raw_sd_map,
                                cid, stage, "linear_baseline_target_specific_X",
                            )
                            if not base_one.empty:
                                base_one["n_controls"] = X.shape[1]
                                baseline_oneD_all.append(base_one)

                    if RUN_HGB_ET_ROBUSTNESS:
                        with StepTimer(runtime_log, cid, stage, f"hgb_et::{d_col}", n, n_matches):
                            hcf = crossfit_hgb_et(
                                X=X, y=y, D=D, groups=groups, splits=splits,
                                seed=cluster_seed(cid, stage, 3500 + d_index * 19),
                            )
                            yhat_h = hcf["yhat"]
                            Dhat_h = hcf["dhat"]
                            y_res_h = y - yhat_h
                            D_res_h = D - Dhat_h

                            hcal = hcf.get("calibration_audit", pd.DataFrame()).copy()
                            if not hcal.empty:
                                hcal.insert(0, "treatment", d_col)
                                hcal.insert(0, "stage", stage)
                                hcal.insert(0, "cluster", cid)
                                calibration_audits.append(hcal)

                            yph_raw = outcome_performance_rows(
                                cid, stage, y, hcf["yhat_raw"], "HistGradientBoosting_raw"
                            )
                            yph_raw["treatment_specific_X"] = d_col
                            yph_raw["n_controls"] = X.shape[1]
                            y_perf.append(yph_raw)

                            yph = outcome_performance_rows(
                                cid, stage, y, yhat_h, "HistGradientBoosting_NestedPlatt"
                            )
                            yph["treatment_specific_X"] = d_col
                            yph["n_controls"] = X.shape[1]
                            y_perf.append(yph)
                            dph = treatment_performance_rows(
                                cid, stage, D, Dhat_h, [d_col], "ExtraTrees_singleoutput"
                            )
                            for row in dph:
                                row["n_controls"] = X.shape[1]
                            d_perf.extend(dph)
                            hgb_one = run_oneD_all_trims(
                                y_res_h, D_res_h, [d_col], groups, raw_sd_map,
                                cid, stage, "hgb_et_robustness_target_specific_X",
                            )
                            if not hgb_one.empty:
                                hgb_one["n_controls"] = X.shape[1]
                                hgb_et_oneD_all.append(hgb_one)

                    if RUN_XGBOOST_ROBUSTNESS and HAS_XGBOOST:
                        with StepTimer(runtime_log, cid, stage, f"xgboost::{d_col}", n, n_matches):
                            xcf = crossfit_xgboost(
                                X=X, y=y, D=D, groups=groups, splits=splits,
                                seed=cluster_seed(cid, stage, 5000 + d_index * 17),
                            )
                            yhat_x = xcf["yhat"]
                            Dhat_x = xcf["dhat"]
                            y_res_x = y - yhat_x
                            D_res_x = D - Dhat_x

                            xcal = xcf.get("calibration_audit", pd.DataFrame()).copy()
                            if not xcal.empty:
                                xcal.insert(0, "treatment", d_col)
                                xcal.insert(0, "stage", stage)
                                xcal.insert(0, "cluster", cid)
                                calibration_audits.append(xcal)

                            ypx_raw = outcome_performance_rows(
                                cid, stage, y, xcf["yhat_raw"], "XGBoostClassifier_raw"
                            )
                            ypx_raw["treatment_specific_X"] = d_col
                            ypx_raw["n_controls"] = X.shape[1]
                            y_perf.append(ypx_raw)

                            ypx = outcome_performance_rows(
                                cid, stage, y, yhat_x, "XGBoostClassifier_NestedPlatt"
                            )
                            ypx["treatment_specific_X"] = d_col
                            ypx["n_controls"] = X.shape[1]
                            y_perf.append(ypx)
                            dpx = treatment_performance_rows(
                                cid, stage, D, Dhat_x, [d_col], "XGBoostRegressor_singleoutput"
                            )
                            for row in dpx:
                                row["n_controls"] = X.shape[1]
                            d_perf.extend(dpx)
                            xgb_one = run_oneD_all_trims(
                                y_res_x, D_res_x, [d_col], groups, raw_sd_map,
                                cid, stage, "xgboost_robustness_target_specific_X",
                            )
                            if not xgb_one.empty:
                                xgb_one["n_controls"] = X.shape[1]
                                xgb_oneD_all.append(xgb_one)

                    with StepTimer(runtime_log, cid, stage, f"placebo::{d_col}", n, n_matches,
                                   note=f"reps={PLACEBO_REPS}; OOF residual permutation within match"):
                        placebo_df = placebo_within_match(
                            y_res, D_res, groups, [d_col], raw_sd_map,
                            cid, stage, PLACEBO_REPS,
                            cluster_seed(cid, stage, 9000 + d_index),
                        )
                        if not placebo_df.empty:
                            placebo_all.append(placebo_df)

                    with StepTimer(runtime_log, cid, stage, f"fold_stability::{d_col}", n, n_matches):
                        fold_long, fold_sum = fold_stability(
                            y_res, D_res, cf["fold_id"], [d_col], raw_sd_map, cid, stage
                        )
                        if not fold_long.empty:
                            fold_stab_long_all.append(fold_long)
                            fold_stab_summary_all.append(fold_sum)

                    if RUN_TEAM_LOO:
                        with StepTimer(runtime_log, cid, stage, f"team_LOO::{d_col}", n, n_matches):
                            tlong, tsum = team_loo_stability(
                                y_res, D_res, teams, [d_col], raw_sd_map, cid, stage
                            )
                            if not tlong.empty:
                                team_loo_long_all.append(tlong)
                                team_loo_summary_all.append(tsum)

                    elapsed_target = time.perf_counter() - target_start
                    run_log.append({
                        "cluster": cid, "stage": stage, "treatment": d_col,
                        "status": "done", "rows": n, "matches": n_matches,
                        "n_X": X.shape[1], "elapsed_seconds": elapsed_target,
                        "elapsed_hms": format_seconds(elapsed_target),
                    })

                except Exception as e:
                    print(f"[ERROR] cluster={cid} stage={stage} treatment={d_col}: {repr(e)}")
                    run_log.append({
                        "cluster": cid, "stage": stage, "treatment": d_col,
                        "status": "error", "reason": repr(e),
                    })

            # --------------------------------------------------------------
            # Pre-specified joint DML for this stage.
            # --------------------------------------------------------------
            joint_specs = []
            main_joint_cache = None
            if RUN_JOINT_DML:
                joint_specs, selection_log = available_joint_specs(stage, D_candidates)
                for row in selection_log:
                    row.update({"cluster": cid})
                    joint_selection_log.append(row)
                for spec_i, (spec_name, selected) in enumerate(joint_specs):
                    try:
                        with StepTimer(runtime_log, cid, stage, f"joint_prepare::{spec_name}", n_rows_cluster, n_matches_cluster,
                                       note=" | ".join(selected)):
                            jb = build_target_dataset(sub, inventory, selected, stage)
                        Xj_df = jb["X_df"]
                        Dj_df = jb["D_df"]
                        yj = jb["y"]
                        gj = jb["groups"]
                        nj = len(yj)
                        nmj = len(pd.unique(gj))
                        if nj < MIN_ROWS or nmj < MIN_MATCHES or min(int(yj.sum()), int(nj-yj.sum())) < MIN_CLASS_COUNT:
                            raise ValueError(f"joint sample insufficient n={nj}, matches={nmj}")

                        Xj = Xj_df.values.astype(float)
                        Dj = Dj_df[selected].values.astype(float)
                        raw_sd_j = {c: float(np.std(Dj[:, k], ddof=1)) for k, c in enumerate(selected)}
                        splits_j = make_grouped_splits(
                            yj, gj, N_SPLITS, cluster_seed(cid, stage, 20000 + spec_i)
                        )

                        with StepTimer(runtime_log, cid, stage, f"joint_crossfit::{spec_name}", nj, nmj,
                                       note=f"joint D={len(selected)}, X={Xj.shape[1]}"):
                            cfj = crossfit_main_and_baseline(
                                X=Xj, y=yj, D=Dj, groups=gj, splits=splits_j,
                                seed=cluster_seed(cid, stage, 21000 + spec_i * 17),
                                run_baseline=False,
                            )
                        yres_j = yj - cfj["yhat_main"]
                        Dres_j = Dj - cfj["dhat_main"]
                        Dres_j_z = scale_residuals_by_raw_sd(Dres_j, selected, raw_sd_j)
                        joint_res, _ = fit_clustered_second_stage(
                            y_res=yres_j, D_res=Dres_j_z, names=selected, groups=gj,
                            raw_sd_map={c: 1.0 for c in selected}, cluster=cid, stage=stage,
                            model_type=f"joint::{spec_name}::stage_specific_full_controls",
                            trim_frac=MAIN_TRIM_FRAC,
                        )
                        if not joint_res.empty:
                            joint_res["joint_spec"] = spec_name
                            joint_res["n_controls"] = Xj.shape[1]
                            joint_res["controls_design"] = (
                                "other L only" if stage == "L" else "all L + remaining E'"
                            ) + " + stage context + team + position + play pattern + score + home/away"
                            joint_res["joint_scale"] = "treatment residual / raw KPI SD"
                            joint_all.append(joint_res)

                        data_audit.append({
                            "cluster": cid, "stage": stage, "model_scope": f"joint::{spec_name}",
                            "treatment": " | ".join(selected), "rows_cluster": n_rows_cluster,
                            "rows_used": nj, "rows_lost": n_rows_cluster - nj,
                            "rows_lost_proportion": 1 - nj/n_rows_cluster if n_rows_cluster else np.nan,
                            "matches": nmj, "n_positive": int(yj.sum()), "n_negative": int(nj-yj.sum()),
                            "positive_rate": float(yj.mean()), "n_controls": Xj.shape[1],
                            "target_in_controls": any(c in Xj_df.columns for c in selected),
                            "base_controls": " | ".join(jb["base_controls"]),
                            "earlier_stage_attacking_controls": " | ".join(jb["earlier_stage_attacking_controls"]),
                            "same_stage_attacking_controls": " | ".join(jb["same_stage_attacking_controls"]),
                            "target_series": " | ".join(jb["target_series"]),
                            "treatment_families": " | ".join(jb["treatment_families"]),
                            "excluded_same_series_att_L": " | ".join(jb["excluded_same_series_att_L"]),
                            "excluded_same_series_att_Eprime": " | ".join(jb["excluded_same_series_att_Eprime"]),
                            "excluded_same_series_attacking_controls": " | ".join(jb["excluded_same_series_attacking_controls"]),
                            "excluded_defensive_controls": " | ".join(jb["excluded_defensive_controls"]),
                            "ignored_actual_E_controls": " | ".join(jb["ignored_actual_E_controls"]),
                            "usable_numeric_controls": " | ".join(jb["usable_numeric_controls"]),
                            "categorical_controls": " | ".join(jb["categorical_control_cols"]),
                            "team_controls": " | ".join(jb["team_control_cols"]),
                            "position_controls": " | ".join(jb["position_control_cols"]),
                            "play_pattern_controls": " | ".join(jb["play_pattern_control_cols"]),
                            "score_state_controls": " | ".join(jb["score_state_control_cols"]),
                            "home_away_controls": " | ".join(jb["home_away_control_cols"]),
                            "estimand": jb["estimand"],
                        })

                        if spec_name == "main_direct_local_structure":
                            main_joint_cache = {
                                "selected": selected, "block": jb, "X": Xj, "D": Dj,
                                "D_df": Dj_df[selected].copy(), "y": yj, "groups": gj,
                                "splits": splits_j, "y_res": yres_j, "D_res": Dres_j,
                                "raw_sd": raw_sd_j,
                            }
                    except Exception as e:
                        run_log.append({
                            "cluster": cid, "stage": stage, "treatment": f"joint::{spec_name}",
                            "status": "error", "reason": repr(e),
                        })
                        print(f"[ERROR] joint cluster={cid} stage={stage} spec={spec_name}: {repr(e)}")

            # --------------------------------------------------------------
            # One quadratic and one interaction for the main joint model.
            # --------------------------------------------------------------
            if RUN_NONLINEAR_CHECK and main_joint_cache is not None:
                try:
                    cache = main_joint_cache
                    extras_df = nonlinear_extra_features(cache["D_df"], cache["selected"], stage)
                    if not extras_df.empty and extras_df.notna().all(axis=1).all():
                        with StepTimer(runtime_log, cid, stage, "nonlinear_crossfit", len(cache["y"]), len(pd.unique(cache["groups"])),
                                       note=" | ".join(extras_df.columns)):
                            extras = extras_df.values.astype(float)
                            extras_hat = crossfit_d_only(
                                X=cache["X"], D=extras, splits=cache["splits"],
                                seed=cluster_seed(cid, stage, 30000),
                            )
                        extras_res = extras - extras_hat
                        names_nl = list(cache["selected"]) + list(extras_df.columns)
                        raw_sd_nl = dict(cache["raw_sd"])
                        for j, c in enumerate(extras_df.columns):
                            raw_sd_nl[c] = float(np.std(extras[:, j], ddof=1))
                        design_res = np.column_stack([cache["D_res"], extras_res])
                        design_res_z = scale_residuals_by_raw_sd(design_res, names_nl, raw_sd_nl)
                        nl_res, _ = fit_clustered_second_stage(
                            y_res=cache["y_res"], D_res=design_res_z, names=names_nl,
                            groups=cache["groups"], raw_sd_map={c: 1.0 for c in names_nl},
                            cluster=cid, stage=stage,
                            model_type="joint_main_plus_quadratic_plus_interaction::stage_specific_full_controls",
                            trim_frac=MAIN_TRIM_FRAC,
                        )
                        if not nl_res.empty:
                            nl_res["joint_spec"] = "main_direct_local_structure"
                            nl_res["n_controls"] = cache["X"].shape[1]
                            nl_res["joint_scale"] = "all residualized terms divided by raw SD"
                            nonlinear_all.append(nl_res)
                except Exception as e:
                    print(f"[ERROR] nonlinear cluster={cid} stage={stage}: {repr(e)}")
                    run_log.append({
                        "cluster": cid, "stage": stage, "treatment": "nonlinear",
                        "status": "error", "reason": repr(e),
                    })

        cluster_elapsed = time.perf_counter() - cluster_start
        print(f"[CLUSTER {cid} FINISHED] elapsed={format_seconds(cluster_elapsed)}")

    # ----------------------------------------------------------------------
    # Assemble and FDR-correct all results. Nothing is silently discarded.
    # ----------------------------------------------------------------------
    oneD_df = pd.concat(oneD_all, ignore_index=True) if oneD_all else pd.DataFrame()
    hgb_et_df = pd.concat(hgb_et_oneD_all, ignore_index=True) if hgb_et_oneD_all else pd.DataFrame()
    baseline_df = pd.concat(baseline_oneD_all, ignore_index=True) if baseline_oneD_all else pd.DataFrame()
    xgb_df = pd.concat(xgb_oneD_all, ignore_index=True) if xgb_oneD_all else pd.DataFrame()
    support_bin_df = pd.DataFrame(support_bin_all)
    joint_df = pd.concat(joint_all, ignore_index=True) if joint_all else pd.DataFrame()
    nonlinear_df = pd.concat(nonlinear_all, ignore_index=True) if nonlinear_all else pd.DataFrame()

    if not oneD_df.empty:
        main_mask = np.isclose(oneD_df["trim_fraction_rule"], MAIN_TRIM_FRAC)
        main_oneD_df = apply_fdr_columns(oneD_df.loc[main_mask].copy())
        oneD_df = pd.concat([main_oneD_df, oneD_df.loc[~main_mask].copy()], ignore_index=True)
    else:
        main_oneD_df = pd.DataFrame()

    if not hgb_et_df.empty:
        main_mask_h = np.isclose(hgb_et_df["trim_fraction_rule"], MAIN_TRIM_FRAC)
        hgb_main = apply_fdr_columns(hgb_et_df.loc[main_mask_h].copy())
        hgb_et_df = pd.concat([hgb_main, hgb_et_df.loc[~main_mask_h].copy()], ignore_index=True)

    if not baseline_df.empty:
        main_mask_b = np.isclose(baseline_df["trim_fraction_rule"], MAIN_TRIM_FRAC)
        base_main = apply_fdr_columns(baseline_df.loc[main_mask_b].copy())
        baseline_df = pd.concat([base_main, baseline_df.loc[~main_mask_b].copy()], ignore_index=True)

    if not xgb_df.empty:
        main_mask_x = np.isclose(xgb_df["trim_fraction_rule"], MAIN_TRIM_FRAC)
        xgb_main = apply_fdr_columns(xgb_df.loc[main_mask_x].copy())
        xgb_df = pd.concat([xgb_main, xgb_df.loc[~main_mask_x].copy()], ignore_index=True)

    if not joint_df.empty:
        joint_df = apply_fdr_columns(joint_df)
    if not nonlinear_df.empty:
        nonlinear_df = apply_fdr_columns(nonlinear_df)

    overlap_df = pd.DataFrame(overlap_all)
    fold_stab_sum_df = pd.concat(fold_stab_summary_all, ignore_index=True) if fold_stab_summary_all else pd.DataFrame()
    team_loo_sum_df = pd.concat(team_loo_summary_all, ignore_index=True) if team_loo_summary_all else pd.DataFrame()
    placebo_df = pd.concat(placebo_all, ignore_index=True) if placebo_all else pd.DataFrame()

    if not main_oneD_df.empty:
        key = ["cluster", "stage", "treatment"]
        if not overlap_df.empty:
            main_oneD_df = main_oneD_df.merge(
                overlap_df[key + ["treatment_oof_r2", "residual_sd_ratio", "overlap_flag"]],
                on=key, how="left",
            )
        if not fold_stab_sum_df.empty:
            main_oneD_df = main_oneD_df.merge(
                fold_stab_sum_df[key + ["sign_consistency", "iqr_theta_per_1sd"]].rename(columns={
                    "sign_consistency": "fold_sign_consistency",
                    "iqr_theta_per_1sd": "fold_effect_iqr",
                }), on=key, how="left",
            )
        if not team_loo_sum_df.empty:
            main_oneD_df = main_oneD_df.merge(
                team_loo_sum_df[key + ["sign_consistency", "iqr_theta_per_1sd"]].rename(columns={
                    "sign_consistency": "team_loo_sign_consistency",
                    "iqr_theta_per_1sd": "team_loo_effect_iqr",
                }), on=key, how="left",
            )
        if not placebo_df.empty:
            main_oneD_df = main_oneD_df.merge(
                placebo_df[key + ["empirical_placebo_p"]], on=key, how="left",
            )
        overlap_flag_series = main_oneD_df.get("overlap_flag", pd.Series(False, index=main_oneD_df.index)).fillna(False)
        fold_sign_series = main_oneD_df.get("fold_sign_consistency", pd.Series(0.0, index=main_oneD_df.index)).fillna(0)
        team_sign_series = main_oneD_df.get("team_loo_sign_consistency", pd.Series(0.0, index=main_oneD_df.index)).fillna(0)
        placebo_p_series = main_oneD_df.get("empirical_placebo_p", pd.Series(1.0, index=main_oneD_df.index)).fillna(1)
        ci_excludes_zero = (
            (pd.to_numeric(main_oneD_df["ci_low_per_1sd"], errors="coerce") > 0) |
            (pd.to_numeric(main_oneD_df["ci_high_per_1sd"], errors="coerce") < 0)
        )
        main_oneD_df["candidate_for_discussion"] = (
            ci_excludes_zero &
            (main_oneD_df["q_global"] <= FDR_ALPHA) &
            (~overlap_flag_series) &
            (fold_sign_series >= FOLD_DIRECTION_MIN) &
            (team_sign_series >= TEAM_LOO_DIRECTION_MIN) &
            (placebo_p_series <= PLACEBO_ALPHA)
        )
        # This column records the manuscript's core statistical/support gates.
        # Trimming, undersampling, and alternative-learner direction checks are
        # reported separately and are applied before final representative selection.
        main_oneD_df["primary_retention_core"] = main_oneD_df["candidate_for_discussion"]


    y_perf_df = pd.DataFrame(y_perf)
    d_perf_df = pd.DataFrame(d_perf)
    data_audit_df = pd.DataFrame(data_audit)
    fold_audit_df = pd.concat(fold_audits, ignore_index=True) if fold_audits else pd.DataFrame()
    calibration_audit_df = (
        pd.concat(calibration_audits, ignore_index=True) if calibration_audits else pd.DataFrame()
    )

    reviewer_full_df = build_reviewer_model_report(
        main_df=main_oneD_df,
        all_trims_df=oneD_df,
        hgb_et_df=hgb_et_df,
        baseline_df=baseline_df,
        xgb_df=xgb_df,
        y_perf_df=y_perf_df,
        d_perf_df=d_perf_df,
        overlap_df=overlap_df,
        placebo_df=placebo_df,
        fold_sum_df=fold_stab_sum_df,
        team_sum_df=team_loo_sum_df,
        data_audit_df=data_audit_df,
    )
    reviewer_overview_df = build_reviewer_overview(reviewer_full_df, fold_audit_df)
    overlap_threshold_sensitivity_df = build_overlap_threshold_sensitivity(overlap_df)

    reviewer_checklist_df = build_reviewer_checklist(
        report=reviewer_full_df,
        y_perf_df=y_perf_df,
        d_perf_df=d_perf_df,
        overlap_df=overlap_df,
        support_df=support_bin_df,
        all_trims_df=oneD_df,
        hgb_et_df=hgb_et_df,
        baseline_df=baseline_df,
        xgb_df=xgb_df,
        fold_sum_df=fold_stab_sum_df,
        team_sum_df=team_loo_sum_df,
        placebo_df=placebo_df,
        fold_audit_df=fold_audit_df,
        data_audit_df=data_audit_df,
        calibration_audit_df=calibration_audit_df,
        overlap_threshold_sensitivity_df=overlap_threshold_sensitivity_df,
    )

    analysis_elapsed = time.perf_counter() - total_start
    runtime_log.append(TimerRecord(
        cluster="ALL", stage="ALL", step="ANALYSIS_BEFORE_WRITE",
        seconds=analysis_elapsed, rows=len(df), matches=df[MATCH_COL].nunique(),
        note="all analysis steps before final Excel writing",
    ))

    manifest = pd.DataFrame([{
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_XLSX,
        "rows_input": len(df),
        "matches_input": df[MATCH_COL].nunique(),
        "clusters": len(clusters),
        "analysis_stages": "L and E'",
        "actual_E_policy": "all columns ending in (E) ignored as treatments and controls",
        "actual_E_columns_detected": " | ".join(
            [c for c in df.columns if is_actual_E_col(c)]
        ),
        "Eprime_columns_required": True,
        "L_estimand": "one defending Pass L KPI conditional on reduced background plus permitted Att(L)/Att(E') KPIs; target-matched Att controls are removed from both stages; Avg and centroid attacking controls are reciprocally excluded; every Def KPI and every actual-E KPI is excluded from X",
        "Eprime_estimand": "one defending Pass E' KPI conditional on reduced background plus permitted Att(L)/Att(E') KPIs; target-matched Att controls are removed from both stages; Avg and centroid attacking controls are reciprocally excluded; every Def KPI and every actual-E KPI is excluded from X",
        "temporal_rule": "only period, match time, pre-pass score/context, categorical controls, and permitted Att KPIs enter X; Att(E') is computed from the L-time freeze frame around the realized endpoint; actual E-frame variables are excluded",
        "numeric_background_controls_L": " | ".join(L_BASE_CONTROL_COLS),
        "numeric_background_controls_Eprime": " | ".join(E_PRIME_BASE_CONTROL_COLS),
        "explicitly_excluded_controls": " | ".join(EXCLUDED_EXACT_SPATIAL_CONTROL_COLS),
        "categorical_background_sources": " | ".join(f"{k}:{v}" for k, v in CATEGORICAL_CONTROL_COLS.items()),
        "single_kpi_control_policy": "L and E': reduced background + permitted Att(L) + permitted Att(E') only; target-matched Att controls are removed from both stages; Avg_Def removes all Avg_Att plus DistToAttCentroid; DistToDefCentroid removes DistToAttCentroid plus all Avg_Att; every Def KPI, every actual-E KPI, and all exact position/geometry/duration/FF-count variables are excluded from X",
        "same_series_att_exclusion": EXCLUDE_SAME_SERIES_ATT_CONTROLS,
        "same_series_att_cross_stage_exclusion": EXCLUDE_SAME_SERIES_ATT_CROSS_STAGE,
        "avg_centroid_reciprocal_exclusion": EXCLUDE_AVG_AND_CENTROID_TOGETHER,
        "defending_kpis_in_X": "none",
        "kpi_family_patterns_for_reporting_only": json.dumps(KPI_FAMILY_PATTERNS, ensure_ascii=False),
        "same_series_rule": "Avg_Def -> all Avg_Att plus DistToAttCentroid; DistToDefCentroid -> DistToAttCentroid plus all Avg_Att; Area_Def -> Area_Att; Spr_Def -> Spr_Att; Adv has no Att-only counterpart",
        "match_group_column": MATCH_COL,
        "crossfit_splits": N_SPLITS,
        "main_trim_fraction": MAIN_TRIM_FRAC,
        "trim_grid": json.dumps(TRIM_GRID),
        "fdr_alpha": FDR_ALPHA,
        "placebo_reps": PLACEBO_REPS,
        "main_y_learner": "RandomForestClassifier with nested match-grouped Platt calibration, refitted for each target-specific X",
        "main_d_learner": "RandomForestRegressor single-output refitted for each target KPI",
        "hgb_et_robustness": "HistGradientBoostingClassifier with nested match-grouped Platt calibration + ExtraTreesRegressor target-specific X" if RUN_HGB_ET_ROBUSTNESS else "not run",
        "baseline_y_learner": "LogisticRegression with nested match-grouped Platt calibration, target-specific X" if RUN_BASELINE else "not run",
        "baseline_d_learner": "Ridge single-output target-specific X" if RUN_BASELINE else "not run",
        "xgboost_robustness": (
            "unweighted XGBClassifier with nested match-grouped Platt calibration + XGBRegressor target-specific X"
            if RUN_XGBOOST_ROBUSTNESS and HAS_XGBOOST else
            ("requested but xgboost unavailable" if RUN_XGBOOST_ROBUSTNESS else "not run")
        ),
        "outcome_nuisance_metrics": "raw and nested-calibrated OOF ROC-AUC | PR-AUC | Brier | log loss | calibration intercept | calibration slope | ECE-10",
        "outcome_calibration_design": "inner match-grouped OOF Platt scaling within each outer training fold; outer test matches never used for calibration",
        "calibrate_main_y": CALIBRATE_MAIN_Y,
        "calibrate_hgb_y": CALIBRATE_HGB_Y,
        "calibrate_baseline_y": CALIBRATE_BASELINE_Y,
        "calibrate_xgb_y": CALIBRATE_XGB_Y,
        "calibration_inner_splits": CALIBRATION_INNER_SPLITS,
        "xgb_scale_pos_weight": XGB_SCALE_POS_WEIGHT,
        "treatment_nuisance_metrics": "OOF R2 | RMSE | MAE | normalized RMSE/SD",
        "overlap_diagnostic_note": "no universal continuous-treatment cut-off; FAIL is a prespecified severe residual-support flag; CAUTION is descriptive only",
        "overlap_fail_rule": f"OOF D R2 >= {OVERLAP_R2_HIGH} AND residual SD ratio <= {OVERLAP_RESID_RATIO_LOW}",
        "overlap_caution_rule": f"OOF D R2 >= {OVERLAP_R2_HIGH} AND {OVERLAP_RESID_RATIO_LOW} < residual SD ratio <= {OVERLAP_RESID_RATIO_CAUTION}",
        "common_support_report": f"raw treatment distribution across {SUPPORT_BINS} predicted-treatment quantile bins",
        "support_threshold_sensitivity_grid": f"R2={OVERLAP_R2_SENSITIVITY_GRID}; residual_SD_ratio={OVERLAP_RESID_RATIO_SENSITIVITY_GRID}",
        "second_stage": "unit-weight residual WLS with HC3 heteroskedasticity-robust standard errors",
        "joint_DML": "not run in this single-KPI script",
        "nonlinear_joint_check": "not run in this single-KPI script",
        "interpretation_note": "Pass E' KPIs are measured around the corrected pass endpoint using the L-time freeze frame; every actual E-stage KPI is ignored.",
        "analysis_seconds_before_excel_write": analysis_elapsed,
        "analysis_hms_before_excel_write": format_seconds(analysis_elapsed),
    }])

    out_path = Path(OUTPUT_XLSX)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    used_sheets = set()
    write_start = time.perf_counter()
    print("[START] final Excel writing")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        tables = OrderedDict([
            ("00_reviewer_checklist", reviewer_checklist_df),
            ("01_reviewer_overview", reviewer_overview_df),
            ("02_reviewer_full_report", reviewer_full_df),
            ("03_reviewer_support_bins", support_bin_df),
            ("04_manifest", manifest),
            ("05_main_oneD", main_oneD_df),
            ("06_oneD_all_trims", oneD_df),
            ("07_linear_oneD", baseline_df),
            ("08_hgb_et_oneD", hgb_et_df),
            ("09_xgboost_oneD", xgb_df),
            ("10_overlap", overlap_df),
            ("11_y_performance", y_perf_df),
            ("12_D_performance", d_perf_df),
            ("13_placebo", placebo_df),
            ("14_fold_stab_sum", fold_stab_sum_df),
            ("15_fold_stab_long", pd.concat(fold_stab_long_all, ignore_index=True) if fold_stab_long_all else pd.DataFrame()),
            ("16_teamLOO_sum", team_loo_sum_df),
            ("17_teamLOO_long", pd.concat(team_loo_long_all, ignore_index=True) if team_loo_long_all else pd.DataFrame()),
            ("18_correlations", pd.concat(corr_all, ignore_index=True) if corr_all else pd.DataFrame()),
            ("19_VIF", pd.concat(vif_all, ignore_index=True) if vif_all else pd.DataFrame()),
            ("20_data_audit", data_audit_df),
            ("21_fold_audit", fold_audit_df),
            ("22_run_log", pd.DataFrame(run_log)),
            ("23_runtime", pd.DataFrame([r.__dict__ for r in runtime_log])),
            ("24_joint_DML", joint_df),
            ("25_nonlinear", nonlinear_df),
            ("26_joint_selection", pd.DataFrame(joint_selection_log)),
            ("27_y_calibration_audit", calibration_audit_df),
            ("28_overlap_threshold_sens", overlap_threshold_sensitivity_df),
        ])
        for name, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame([{"message": "No results produced for this table."}])
            sheet = safe_sheet_name(name, used_sheets)
            table.to_excel(writer, index=False, sheet_name=sheet)

        workbook = writer.book
        fmt_header = workbook.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78",
            "text_wrap": True, "valign": "top", "border": 1,
        })
        fmt_pct = workbook.add_format({"num_format": "0.000"})
        fmt_wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, 0, max(0, tables[sheet_name].shape[1] - 1) if sheet_name in tables else 0)
            worksheet.set_row(0, 34, fmt_header)
            worksheet.set_column(0, 4, 17)
            worksheet.set_column(5, 20, 15, fmt_pct)
            worksheet.set_column(21, 80, 22, fmt_wrap)

            # Reviewer-facing conditional formatting.
            if sheet_name == "02_reviewer_full_report" and not reviewer_full_df.empty:
                cols = {c: i for i, c in enumerate(reviewer_full_df.columns)}
                nrows = len(reviewer_full_df)
                if "overlap_status" in cols:
                    ci = cols["overlap_status"]
                    worksheet.conditional_format(1, ci, nrows, ci, {
                        "type": "text", "criteria": "containing", "value": "FAIL",
                        "format": workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#9C0006"}),
                    })
                    worksheet.conditional_format(1, ci, nrows, ci, {
                        "type": "text", "criteria": "containing", "value": "CAUTION",
                        "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"}),
                    })
                if "reviewer_evidence_status" in cols:
                    ci = cols["reviewer_evidence_status"]
                    worksheet.conditional_format(1, ci, nrows, ci, {
                        "type": "text", "criteria": "containing", "value": "PRIMARY",
                        "format": workbook.add_format({"bg_color": "#D9EAD3", "font_color": "#274E13"}),
                    })

    write_elapsed = time.perf_counter() - write_start
    runtime_log.append(TimerRecord(
        cluster="ALL", stage="ALL", step="WRITE_EXCEL",
        seconds=write_elapsed, rows=len(df), matches=df[MATCH_COL].nunique(),
        note="final workbook writing",
    ))
    wall_elapsed = time.perf_counter() - total_start
    runtime_log.append(TimerRecord(
        cluster="ALL", stage="ALL", step="TOTAL_WALL_CLOCK",
        seconds=wall_elapsed, rows=len(df), matches=df[MATCH_COL].nunique(),
        note="including final workbook writing",
    ))
    runtime_csv = out_path.with_name(out_path.stem + "_runtime.csv")
    pd.DataFrame([r.__dict__ for r in runtime_log]).to_csv(runtime_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE ] final Excel writing elapsed={format_seconds(write_elapsed)}")
    print("\n" + "=" * 100)
    print(f"[OK] Wrote: {OUTPUT_XLSX}")
    print(f"[TOTAL RUNTIME] {format_seconds(wall_elapsed)} ({wall_elapsed:.2f} seconds)")
    print(f"[RUNTIME CSV] {runtime_csv}")
    print("=" * 100)


# 11. JOINT-ONLY REVIEWER PIPELINE
# =============================================================================

JOINT_MAIN_TRIM_FRAC = MAIN_TRIM_FRAC
JOINT_TRIM_GRID = TRIM_GRID
JOINT_FOLD_SIGN_MIN = 0.80
JOINT_TEAM_SIGN_MIN = 0.90
JOINT_PLACEBO_P_MAX = 0.10
JOINT_RESIDUAL_VIF_CAUTION = 5.0
JOINT_RESIDUAL_VIF_SEVERE = 10.0
JOINT_CONDITION_CAUTION = 30.0
JOINT_CONDITION_SEVERE = 100.0


def joint_crossfit_hgb_et_multi(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
):
    """Grouped OOF HGB outcome predictions and one ExtraTrees model per D."""
    n, k_d = D.shape
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    dhat = np.full((n, k_d), np.nan)
    calibration_rows = []

    for fold, (tr, te) in enumerate(splits):
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr],
            X_test=X[te], learner="hgb", seed=seed + fold,
            calibrate=CALIBRATE_HGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        raw_outer = safe_binary_metrics(y[te], p_raw)
        cal_outer = safe_binary_metrics(y[te], p)
        for key, value in raw_outer.items():
            audit[f"outer_raw_{key}"] = value
        for key, value in cal_outer.items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        for j in range(k_d):
            model = make_et_d_model(seed + 1000 + fold * 101 + j)
            model.fit(Xtr, D[tr, j])
            dhat[te, j] = np.asarray(model.predict(Xte)).reshape(-1)

    return {
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "dhat": dhat,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }


def joint_crossfit_xgb_multi(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
):
    """Grouped OOF calibrated XGBoost outcome and one XGB regressor per D."""
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is unavailable")

    n, k_d = D.shape
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    dhat = np.full((n, k_d), np.nan)
    calibration_rows = []

    for fold, (tr, te) in enumerate(splits):
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr],
            X_test=X[te], learner="xgb", seed=seed + fold,
            calibrate=CALIBRATE_XGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        raw_outer = safe_binary_metrics(y[te], p_raw)
        cal_outer = safe_binary_metrics(y[te], p)
        for key, value in raw_outer.items():
            audit[f"outer_raw_{key}"] = value
        for key, value in cal_outer.items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        for j in range(k_d):
            model = make_xgb_d_model(seed + 1000 + fold * 101 + j)
            model.fit(Xtr, D[tr, j])
            dhat[te, j] = np.asarray(model.predict(Xte)).reshape(-1)

    return {
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "dhat": dhat,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }


def joint_all_trim_results(
    y_res: np.ndarray,
    D_res_z: np.ndarray,
    names: Sequence[str],
    groups: np.ndarray,
    cluster,
    stage: str,
    joint_spec: str,
    model_type: str,
) -> pd.DataFrame:
    frames = []
    for trim_frac in JOINT_TRIM_GRID:
        result, info = fit_clustered_second_stage(
            y_res=y_res,
            D_res=D_res_z,
            names=names,
            groups=groups,
            raw_sd_map={name: 1.0 for name in names},
            cluster=cluster,
            stage=stage,
            model_type=model_type,
            trim_frac=trim_frac,
        )
        if result.empty:
            continue
        result["joint_spec"] = joint_spec
        result["joint_kpis"] = " | ".join(names)
        result["joint_second_stage_r2"] = info.get("r2_second_stage", np.nan)
        result["joint_condition_number"] = info.get("condition_number", np.nan)
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def joint_point_estimates(y_res: np.ndarray, D_res_z: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y_res) & np.all(np.isfinite(D_res_z), axis=1)
    if mask.sum() < D_res_z.shape[1] + 10:
        return np.full(D_res_z.shape[1], np.nan)
    design = np.column_stack([np.ones(mask.sum()), D_res_z[mask]])
    coef = np.linalg.lstsq(design, y_res[mask], rcond=None)[0]
    return np.asarray(coef[1:], dtype=float)


def joint_placebo_within_match(
    y_res: np.ndarray,
    D_res_z: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Permute the full residual-treatment vector jointly within each match."""
    mask = np.isfinite(y_res) & np.all(np.isfinite(D_res_z), axis=1)
    y = y_res[mask]
    D = D_res_z[mask]
    g = groups[mask]
    if len(y) < 40:
        return pd.DataFrame()

    observed = joint_point_estimates(y, D)
    rng = np.random.RandomState(seed)
    group_indices = [np.where(g == value)[0] for value in pd.unique(g)]
    null = np.full((reps, D.shape[1]), np.nan)

    for r in range(reps):
        Dp = np.empty_like(D)
        for idx in group_indices:
            if len(idx) <= 1:
                Dp[idx] = D[idx]
            else:
                # The same row permutation is applied to every D column, preserving
                # the treatment correlation structure inside each match.
                perm = rng.permutation(len(idx))
                Dp[idx] = D[idx][perm]
        null[r] = joint_point_estimates(y, Dp)

    rows = []
    for j, name in enumerate(names):
        values = null[:, j]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        p_emp = (1 + np.sum(np.abs(values) >= abs(observed[j]))) / (len(values) + 1)
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "treatment": name,
            "placebo_reps": int(len(values)),
            "observed_joint_theta_per_1sd_untrimmed": float(observed[j]),
            "null_mean": float(np.mean(values)),
            "null_sd": float(np.std(values, ddof=1)),
            "null_p025": float(np.quantile(values, 0.025)),
            "null_p975": float(np.quantile(values, 0.975)),
            "empirical_placebo_p": float(p_emp),
            "permutation_scope": "joint residual-treatment vector permuted within match",
        })
    return pd.DataFrame(rows)


def joint_fold_stability(
    y_res: np.ndarray,
    D_res_z: np.ndarray,
    fold_id: np.ndarray,
    names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
):
    rows = []
    for fold in sorted(pd.unique(fold_id)):
        if fold < 0:
            continue
        mask = (
            (fold_id == fold)
            & np.isfinite(y_res)
            & np.all(np.isfinite(D_res_z), axis=1)
        )
        if mask.sum() < max(30, len(names) + 10):
            continue
        coef = joint_point_estimates(y_res[mask], D_res_z[mask])
        for j, name in enumerate(names):
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "fold": int(fold),
                "treatment": name,
                "theta_per_1sd": float(coef[j]),
                "n": int(mask.sum()),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summaries = []
    for name, group in long_df.groupby("treatment"):
        values = group["theta_per_1sd"].to_numpy(dtype=float)
        median = float(np.nanmedian(values))
        summaries.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "treatment": name,
            "n_folds": len(values),
            "median_theta_per_1sd": median,
            "min_theta_per_1sd": float(np.nanmin(values)),
            "max_theta_per_1sd": float(np.nanmax(values)),
            "iqr_theta_per_1sd": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
            "fold_sign_consistency": (
                float(np.mean(np.sign(values) == np.sign(median))) if median != 0 else np.nan
            ),
        })
    return long_df, pd.DataFrame(summaries)


def joint_team_loo_stability(
    y_res: np.ndarray,
    D_res_z: np.ndarray,
    teams: np.ndarray,
    names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
):
    rows = []
    valid_teams = [value for value in pd.unique(teams) if pd.notna(value)]
    full_coef = joint_point_estimates(y_res, D_res_z)

    for team in valid_teams:
        mask = (
            (teams != team)
            & np.isfinite(y_res)
            & np.all(np.isfinite(D_res_z), axis=1)
        )
        if mask.sum() < max(40, len(names) + 10):
            continue
        coef = joint_point_estimates(y_res[mask], D_res_z[mask])
        for j, name in enumerate(names):
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "left_out_team": str(team),
                "treatment": name,
                "theta_per_1sd": float(coef[j]),
                "n": int(mask.sum()),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summaries = []
    for j, name in enumerate(names):
        values = long_df.loc[long_df["treatment"] == name, "theta_per_1sd"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        summaries.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "treatment": name,
            "full_untrimmed_theta_per_1sd": float(full_coef[j]),
            "n_teams_left_out": len(values),
            "team_loo_median_theta_per_1sd": float(np.nanmedian(values)),
            "team_loo_min_theta_per_1sd": float(np.nanmin(values)),
            "team_loo_max_theta_per_1sd": float(np.nanmax(values)),
            "team_loo_iqr_theta_per_1sd": float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)),
            "team_loo_sign_consistency": (
                float(np.mean(np.sign(values) == np.sign(full_coef[j])))
                if np.isfinite(full_coef[j]) and full_coef[j] != 0 else np.nan
            ),
        })
    return long_df, pd.DataFrame(summaries)


def joint_correlation_and_vif(
    matrix: np.ndarray,
    names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
    matrix_type: str,
):
    frame = pd.DataFrame(matrix, columns=names)
    corr = frame.corr(method="pearson")
    corr_rows = []
    for i, first in enumerate(names):
        for j, second in enumerate(names):
            if j < i:
                continue
            corr_rows.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "matrix_type": matrix_type,
                "kpi_1": first,
                "kpi_2": second,
                "pearson_r": float(corr.iloc[i, j]),
            })

    array = np.nan_to_num(corr.to_numpy(dtype=float), nan=0.0)
    np.fill_diagonal(array, 1.0)
    inverse = np.linalg.pinv(array)
    vif_rows = []
    for j, name in enumerate(names):
        vif = float(inverse[j, j])
        vif_rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "matrix_type": matrix_type,
            "treatment": name,
            "VIF": vif,
            "vif_status": (
                "SEVERE" if vif >= JOINT_RESIDUAL_VIF_SEVERE
                else "CAUTION" if vif >= JOINT_RESIDUAL_VIF_CAUTION
                else "PASS"
            ),
        })
    return pd.DataFrame(corr_rows), pd.DataFrame(vif_rows)


def joint_wald_test(
    y_res: np.ndarray,
    D_res_z: np.ndarray,
    groups: np.ndarray,
    names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
    trim_frac: float,
    model_type: str,
) -> pd.DataFrame:
    mask = np.isfinite(y_res) & np.all(np.isfinite(D_res_z), axis=1)
    mask &= make_trim_mask(D_res_z, trim_frac)
    n = int(mask.sum())
    n_matches = int(len(pd.unique(groups[mask]))) if n else 0
    if n < max(40, len(names) + 10) or n_matches < 5:
        return pd.DataFrame()

    design = sm.add_constant(D_res_z[mask], has_constant="add")
    fit = sm.WLS(
        y_res[mask], design, weights=np.ones(n, dtype=float)
    ).fit(cov_type="HC3", use_t=True)
    restriction = np.zeros((len(names), len(names) + 1))
    restriction[:, 1:] = np.eye(len(names))

    statistic = np.nan
    p_value = np.nan
    df_num = len(names)
    df_denom = max(1, n_matches - 1)
    test_type = "HC3-robust Wald"
    try:
        test = fit.wald_test(restriction, use_f=True, scalar=True)
        statistic = float(np.asarray(test.statistic).reshape(-1)[0])
        p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
        df_num = float(getattr(test, "df_num", len(names)))
        df_denom = float(getattr(test, "df_denom", max(1, n_matches - 1)))
        test_type = "HC3-robust F/Wald"
    except Exception:
        try:
            test = fit.wald_test(restriction, scalar=True)
            statistic = float(np.asarray(test.statistic).reshape(-1)[0])
            p_value = float(np.asarray(test.pvalue).reshape(-1)[0])
        except Exception:
            pass

    condition = float(np.linalg.cond(design))
    return pd.DataFrame([{
        "cluster": cluster,
        "stage": stage,
        "joint_spec": joint_spec,
        "model_type": model_type,
        "joint_kpis": " | ".join(names),
        "trim_fraction_rule": float(trim_frac),
        "n_used": n,
        "n_matches": n_matches,
        "wald_statistic": statistic,
        "wald_p_value": p_value,
        "df_num": df_num,
        "df_denom": df_denom,
        "test_type": test_type,
        "joint_condition_number": condition,
        "condition_status": (
            "SEVERE" if condition >= JOINT_CONDITION_SEVERE
            else "CAUTION" if condition >= JOINT_CONDITION_CAUTION
            else "PASS"
        ),
    }])


def add_joint_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame
    out = frame.copy()
    out["q_global"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
    out["fdr_global_pass"] = out["q_global"] <= FDR_ALPHA
    out["q_within_stage"] = np.nan
    out["q_within_joint_spec"] = np.nan

    for _, idx in out.groupby(["cluster", "stage"], dropna=False).groups.items():
        out.loc[idx, "q_within_stage"] = bh_fdr(out.loc[idx, "p_value"].to_numpy(dtype=float))

    for _, idx in out.groupby(["cluster", "stage", "joint_spec"], dropna=False).groups.items():
        out.loc[idx, "q_within_joint_spec"] = bh_fdr(out.loc[idx, "p_value"].to_numpy(dtype=float))

    out["fdr_stage_pass"] = out["q_within_stage"] <= FDR_ALPHA
    out["fdr_joint_spec_pass"] = out["q_within_joint_spec"] <= FDR_ALPHA
    return out


def main_trim_only(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    values = pd.to_numeric(frame["trim_fraction_rule"], errors="coerce")
    return frame.loc[np.isclose(values, JOINT_MAIN_TRIM_FRAC)].copy()


def build_joint_main_report(
    main_results: pd.DataFrame,
    alternative_results: Dict[str, pd.DataFrame],
    overlap: pd.DataFrame,
    placebo: pd.DataFrame,
    fold_summary: pd.DataFrame,
    team_summary: pd.DataFrame,
    residual_vif: pd.DataFrame,
    wald: pd.DataFrame,
    data_audit: pd.DataFrame,
) -> pd.DataFrame:
    if main_results.empty:
        return pd.DataFrame()

    keys = ["cluster", "stage", "joint_spec", "treatment"]
    out = main_results.copy()

    for diagnostic in [overlap, placebo, fold_summary, team_summary, residual_vif]:
        if diagnostic is None or diagnostic.empty:
            continue
        add_cols = [c for c in diagnostic.columns if c not in keys and c not in out.columns]
        out = out.merge(diagnostic[keys + add_cols], on=keys, how="left")

    if wald is not None and not wald.empty:
        wald_keys = ["cluster", "stage", "joint_spec"]
        wald_cols = [c for c in wald.columns if c not in wald_keys and c not in out.columns]
        out = out.merge(wald[wald_keys + wald_cols], on=wald_keys, how="left")

    if data_audit is not None and not data_audit.empty:
        audit_cols = keys + [
            c for c in [
                "rows_used", "matches", "positive_rate", "n_controls",
                "base_controls", "same_stage_attacking_controls",
                "cross_stage_attacking_controls", "excluded_same_series_attacking_controls",
                "excluded_defensive_controls", "categorical_controls", "estimand",
            ] if c in data_audit.columns
        ]
        audit = data_audit[audit_cols].drop_duplicates(keys)
        out = out.merge(audit, on=keys, how="left")

    for label, frame in alternative_results.items():
        alt = main_trim_only(frame)
        if alt.empty:
            continue
        keep = keys + [
            c for c in [
                "theta_per_1sd", "se_per_1sd", "ci_low_per_1sd",
                "ci_high_per_1sd", "p_value", "q_global", "fdr_global_pass",
            ] if c in alt.columns
        ]
        alt = alt[keep].rename(columns={
            c: c if c in keys else f"{label}_{c}" for c in keep
        })
        out = out.merge(alt, on=keys, how="left")
        theta_col = f"{label}_theta_per_1sd"
        if theta_col in out.columns:
            out[f"{label}_direction_agreement"] = np.where(
                np.isfinite(pd.to_numeric(out[theta_col], errors="coerce")),
                np.sign(out["theta_per_1sd"]) == np.sign(out[theta_col]),
                np.nan,
            )

    def classify(row: pd.Series) -> str:
        if bool(row.get("overlap_flag", False)):
            return "NOT_PRIMARY_overlap_failure"
        if float(row.get("VIF", 0) or 0) >= JOINT_RESIDUAL_VIF_SEVERE:
            return "NOT_PRIMARY_residual_collinearity"
        if not bool(row.get("fdr_global_pass", False)):
            return "SECONDARY_not_global_FDR"
        if float(row.get("fold_sign_consistency", 0) or 0) < JOINT_FOLD_SIGN_MIN:
            return "CAUTION_fold_instability"
        if float(row.get("team_loo_sign_consistency", 0) or 0) < JOINT_TEAM_SIGN_MIN:
            return "CAUTION_team_instability"
        if float(row.get("empirical_placebo_p", 1) or 1) > JOINT_PLACEBO_P_MAX:
            return "CAUTION_placebo"
        for label in alternative_results:
            agree = row.get(f"{label}_direction_agreement")
            if pd.notna(agree) and not bool(agree):
                return f"CAUTION_{label}_direction_reversal"
        if float(row.get("VIF", 0) or 0) >= JOINT_RESIDUAL_VIF_CAUTION:
            return "PRIMARY_with_VIF_caution"
        return "PRIMARY_independent_effect"

    out["reviewer_evidence_status"] = out.apply(classify, axis=1)
    out["candidate_for_core_discussion"] = out["reviewer_evidence_status"].astype(str).str.startswith("PRIMARY")
    return out


def build_rotating_target_selection(
    full_report: pd.DataFrame,
    target_map: pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly one joint result row for each configured target KPI.

    Multiplicity is recalculated over the target coefficients only, avoiding
    repeated companion coefficients being counted as separate screening tests.
    """
    if target_map is None or target_map.empty:
        return pd.DataFrame()

    out = target_map.copy()
    if full_report is None or full_report.empty:
        out["target_result_status"] = "missing_all_joint_results"
        return out

    report = full_report.copy().rename(columns={"treatment": "target"})
    keys = ["cluster", "stage", "joint_spec", "target"]
    keep = keys + [
        c for c in [
            "theta_per_1sd", "se_per_1sd", "ci_low_per_1sd",
            "ci_high_per_1sd", "t_stat_cluster", "p_value",
            "partial_r2_cluster_t", "robustness_value_point",
            "overlap_flag", "residual_sd_ratio", "VIF",
            "design_condition_number", "fold_sign_consistency",
            "team_loo_sign_consistency", "empirical_placebo_p",
            "ridge_direction_agreement", "hgb_direction_agreement",
            "xgb_direction_agreement", "reviewer_evidence_status",
            "candidate_for_core_discussion", "n_used", "n_matches",
            "same_stage_attacking_controls",
            "cross_stage_attacking_controls",
            "excluded_same_series_attacking_controls",
        ] if c in report.columns
    ]
    report = report[keep].drop_duplicates(keys)
    out = out.merge(report, on=keys, how="left")

    valid = out["p_value"].notna() & out["status"].eq("RUN")
    out["target_q_global"] = np.nan
    out.loc[valid, "target_q_global"] = bh_fdr(
        pd.to_numeric(out.loc[valid, "p_value"], errors="coerce").to_numpy(float)
    )
    out["target_q_within_cluster_stage"] = np.nan
    for _, idx in out.loc[valid].groupby(["cluster", "stage"]).groups.items():
        out.loc[idx, "target_q_within_cluster_stage"] = bh_fdr(
            pd.to_numeric(out.loc[idx, "p_value"], errors="coerce").to_numpy(float)
        )

    def robust_pass(row: pd.Series) -> bool:
        if pd.isna(row.get("p_value")):
            return False
        if float(row.get("target_q_global", np.inf)) > FDR_ALPHA:
            return False
        if bool(row.get("overlap_flag", False)):
            return False
        vif = pd.to_numeric(pd.Series([row.get("VIF", np.nan)]), errors="coerce").iloc[0]
        if pd.notna(vif) and float(vif) >= JOINT_RESIDUAL_VIF_SEVERE:
            return False
        fold = pd.to_numeric(pd.Series([row.get("fold_sign_consistency", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(fold) or float(fold) < JOINT_FOLD_SIGN_MIN:
            return False
        team = pd.to_numeric(pd.Series([row.get("team_loo_sign_consistency", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(team) or float(team) < JOINT_TEAM_SIGN_MIN:
            return False
        placebo = pd.to_numeric(pd.Series([row.get("empirical_placebo_p", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(placebo) or float(placebo) > JOINT_PLACEBO_P_MAX:
            return False
        for col in [
            "ridge_direction_agreement",
            "hgb_direction_agreement",
            "xgb_direction_agreement",
        ]:
            if col in row.index and pd.notna(row.get(col)) and not bool(row.get(col)):
                return False
        return True

    out["joint_target_fdr_pass"] = out["target_q_global"] <= FDR_ALPHA
    out["joint_statistical_and_robustness_pass"] = out.apply(robust_pass, axis=1)
    out["selected_for_multi_kpi_optimization"] = (
        out["single_model_pass_assumed"].fillna(False)
        & out["joint_statistical_and_robustness_pass"].fillna(False)
    )
    out["single_joint_direction_check"] = (
        "not enforced in this no-single-workbook script; compare with the "
        "separate Pass single-KPI report before optimization"
    )

    out["family_winner_recommended"] = False
    passed = out.loc[out["selected_for_multi_kpi_optimization"]].copy()
    if not passed.empty:
        passed["abs_theta"] = pd.to_numeric(
            passed["theta_per_1sd"], errors="coerce"
        ).abs()
        winners = (
            passed.sort_values(
                ["target_q_global", "target_q_within_cluster_stage", "abs_theta", "target"],
                ascending=[True, True, False, True],
            )
            .groupby(["cluster", "stage", "target_family"], as_index=False)
            .head(1)
            .index
        )
        out.loc[winners, "family_winner_recommended"] = True

    out["target_result_status"] = np.where(
        out["p_value"].notna(), "done", "missing_joint_result"
    )
    return out


def build_joint_overview(full_report: pd.DataFrame, fold_audit: pd.DataFrame) -> pd.DataFrame:
    if full_report.empty:
        return pd.DataFrame()
    groups = [("ALL", "ALL", "ALL", full_report)]
    groups += [
        (cluster, stage, spec, group)
        for (cluster, stage, spec), group in full_report.groupby(
            ["cluster", "stage", "joint_spec"], dropna=False
        )
    ]
    rows = []
    for cluster, stage, spec, group in groups:
        rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": spec,
            "n_coefficients": len(group),
            "n_global_FDR_pass": int(group["fdr_global_pass"].fillna(False).sum()),
            "n_core_discussion": int(group["candidate_for_core_discussion"].fillna(False).sum()),
            "n_overlap_fail": int(group.get("overlap_flag", pd.Series(False, index=group.index)).fillna(False).sum()),
            "max_residual_VIF": float(pd.to_numeric(group.get("VIF", np.nan), errors="coerce").max()),
            "median_fold_sign_consistency": float(pd.to_numeric(group.get("fold_sign_consistency", np.nan), errors="coerce").median()),
            "median_team_LOO_sign_consistency": float(pd.to_numeric(group.get("team_loo_sign_consistency", np.nan), errors="coerce").median()),
            "median_placebo_p": float(pd.to_numeric(group.get("empirical_placebo_p", np.nan), errors="coerce").median()),
            "joint_wald_p": float(pd.to_numeric(group.get("wald_p_value", np.nan), errors="coerce").dropna().iloc[0]) if pd.to_numeric(group.get("wald_p_value", np.nan), errors="coerce").notna().any() else np.nan,
        })
    if fold_audit is not None and not fold_audit.empty:
        max_leak = int(pd.to_numeric(fold_audit["match_leakage"], errors="coerce").max())
        for row in rows:
            row["max_match_leakage"] = max_leak
    return pd.DataFrame(rows)


def build_paper_wide_table(main_report: pd.DataFrame) -> pd.DataFrame:
    if main_report.empty:
        return pd.DataFrame()
    frame = main_report.copy()
    frame["effect_95CI_pp"] = frame.apply(
        lambda row: (
            f"{100 * row['theta_per_1sd']:.2f} "
            f"[{100 * row['ci_low_per_1sd']:.2f}, {100 * row['ci_high_per_1sd']:.2f}]"
        ),
        axis=1,
    )
    frame["q_global_text"] = frame["q_global"].map(
        lambda value: "" if pd.isna(value) else f"{value:.4g}"
    )
    frame["cluster_label"] = frame["cluster"].map(lambda value: f"Cluster {value}")
    rows = []
    for (stage, spec, treatment), group in frame.groupby(
        ["stage", "joint_spec", "treatment"], dropna=False
    ):
        row = {
            "stage": stage,
            "joint_spec": spec,
            "treatment": treatment,
        }
        for _, item in group.iterrows():
            label = item["cluster_label"]
            row[f"{label} effect [95% CI], pp"] = item["effect_95CI_pp"]
            row[f"{label} global q"] = item["q_global_text"]
            row[f"{label} status"] = item.get("reviewer_evidence_status", "")
        rows.append(row)
    return pd.DataFrame(rows)


def format_joint_workbook(writer: pd.ExcelWriter, tables: OrderedDict) -> None:
    workbook = writer.book
    header = workbook.add_format({
        "bold": True,
        "font_color": "#FFFFFF",
        "bg_color": "#1F4E78",
        "text_wrap": True,
        "valign": "top",
        "border": 1,
    })
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    number = workbook.add_format({"num_format": "0.0000"})

    for sheet, table in tables.items():
        worksheet = writer.sheets[sheet]
        worksheet.freeze_panes(1, 0)
        if len(table.columns):
            worksheet.autofilter(0, 0, max(1, len(table)), len(table.columns) - 1)
        worksheet.set_row(0, 34, header)
        for index, column in enumerate(table.columns):
            worksheet.write(0, index, str(column), header)
            lower = str(column).lower()
            if any(token in lower for token in [
                "controls", "estimand", "kpis", "status", "note", "reason",
                "input_path", "output_path", "selection", "scope", "rule",
            ]):
                worksheet.set_column(index, index, 42, wrap)
            elif any(token in lower for token in [
                "theta", "se_", "ci_", "p_value", "q_", "r2", "rmse",
                "vif", "correlation", "ratio", "auc", "brier", "ece",
            ]):
                worksheet.set_column(index, index, 18, number)
            else:
                worksheet.set_column(index, index, min(max(len(str(column)) + 2, 13), 24))

        if sheet == "02_main_joint_report" and not table.empty:
            cols = {column: i for i, column in enumerate(table.columns)}
            nrows = len(table)
            if "reviewer_evidence_status" in cols:
                ci = cols["reviewer_evidence_status"]
                worksheet.conditional_format(1, ci, nrows, ci, {
                    "type": "text", "criteria": "containing", "value": "PRIMARY",
                    "format": workbook.add_format({"bg_color": "#D9EAD3", "font_color": "#274E13"}),
                })
                worksheet.conditional_format(1, ci, nrows, ci, {
                    "type": "text", "criteria": "containing", "value": "CAUTION",
                    "format": workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"}),
                })
                worksheet.conditional_format(1, ci, nrows, ci, {
                    "type": "text", "criteria": "containing", "value": "NOT_PRIMARY",
                    "format": workbook.add_format({"bg_color": "#F4CCCC", "font_color": "#9C0006"}),
                })


def joint_main():
    validate_primary_rf_alignment()
    total_start = time.perf_counter()
    runtime_log: List[TimerRecord] = []
    run_log = []
    selection_rows = []
    data_audit_rows = []
    fold_audit_frames = []
    calibration_frames = []
    y_performance_rows = []
    d_performance_rows = []
    overlap_rows_all = []
    support_bin_rows_all = []
    raw_corr_frames = []
    raw_vif_frames = []
    residual_corr_frames = []
    residual_vif_frames = []
    main_trim_frames = []
    baseline_trim_frames = []
    hgb_trim_frames = []
    xgb_trim_frames = []
    wald_frames = []
    placebo_frames = []
    fold_long_frames = []
    fold_summary_frames = []
    team_long_frames = []
    team_summary_frames = []

    print("=" * 110)
    print("PASS · ROTATING JOINT DML · PASS-ORIGIN AND PASS-DESTINATION-REFERENCED CONFIGURATIONS")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_XLSX}")
    print("=" * 110)

    with StepTimer(runtime_log, "ALL", "ALL", "read_and_prepare"):
        df = read_input_table(INPUT_PATH, sheet_name=SHEET_NAME)
        df.columns = [str(column).strip() for column in df.columns]

        # Resolve the Pass home/away field to the canonical Pass-equivalent name.
        if HOME_AWAY_COL not in df.columns:
            for alternative in ["move_team_home_away", "team_home_away", "home_away"]:
                if alternative in df.columns:
                    df[HOME_AWAY_COL] = df[alternative]
                    break

        check_required_columns(df)

        df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

        eprime_def_columns = [
            column for column in df.columns
            if is_Eprime_col(column) and not is_att_kpi(column)
        ]
        if not eprime_def_columns:
            actual_e_columns = [
                column for column in df.columns
                if is_actual_E_col(column)
            ]
            raise ValueError(
                "输入数据没有可用的Def(E') KPI。"
                "本代码不会把实际(E) KPI改名或代替(E')。"
                f"检测到的实际E列数量={len(actual_e_columns)}。"
                "请先在L时刻freeze_frame上围绕pass end_location计算E' KPI。"
            )

    clusters = sorted(df[CLUSTER_COL].dropna().unique())

    for cluster in clusters:
        sub = df.loc[df[CLUSTER_COL] == cluster].copy()
        rows_cluster = len(sub)
        matches_cluster = sub[MATCH_COL].nunique(dropna=True)
        print("\n" + "#" * 110)
        print(f"CLUSTER {cluster}: rows={rows_cluster:,}, matches={matches_cluster:,}")
        print("#" * 110)

        with StepTimer(runtime_log, cluster, "ALL", "identify_KPIs", rows_cluster, matches_cluster):
            inventory = build_inventory(sub)

        for stage in ACTIVE_STAGES:
            available = set(inventory["D_by_stage"].get(stage, []))
            stage_specs, stage_target_map = build_rotating_joint_specs(
                cluster, stage, available
            )
            selection_rows.extend(stage_target_map)

            for spec_index, spec_record in enumerate(stage_specs):
                joint_spec = spec_record["joint_spec"]
                selected = list(spec_record["selected"])
                target_members = list(spec_record["target_members"])
                missing = [name for name in selected if name not in available]
                if missing:
                    run_log.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "target_members": " | ".join(target_members),
                        "status": "skip",
                        "reason": f"missing KPI: {missing}",
                    })
                    continue

                spec_start = time.perf_counter()
                try:
                    with StepTimer(
                        runtime_log, cluster, stage, f"prepare::{joint_spec}",
                        rows_cluster, matches_cluster, note=" | ".join(selected),
                    ):
                        block = build_target_dataset(sub, inventory, selected, stage)

                    X_df = block["X_df"]
                    D_df = block["D_df"]
                    y = block["y"]
                    groups = block["groups"]
                    teams = block["teams"]
                    n = len(y)
                    n_matches = len(pd.unique(groups))
                    n_positive = int(y.sum())
                    n_negative = int(n - n_positive)

                    for treatment in selected:
                        data_audit_rows.append({
                            "cluster": cluster,
                            "stage": stage,
                            "joint_spec": joint_spec,
                            "treatment": treatment,
                            "joint_kpis": " | ".join(selected),
                            "rows_cluster": rows_cluster,
                            "rows_used": n,
                            "rows_lost": rows_cluster - n,
                            "rows_lost_proportion": 1 - n / rows_cluster if rows_cluster else np.nan,
                            "matches": n_matches,
                            "n_positive": n_positive,
                            "n_negative": n_negative,
                            "positive_rate": n_positive / n if n else np.nan,
                            "n_controls": X_df.shape[1],
                            "target_in_controls": any(name in X_df.columns for name in selected),
                            "base_controls": " | ".join(block["base_controls"]),
                            "earlier_stage_attacking_controls": " | ".join(block["earlier_stage_attacking_controls"]),
                            "same_stage_attacking_controls": " | ".join(block["same_stage_attacking_controls"]),
                            "cross_stage_attacking_controls": " | ".join(block["cross_stage_attacking_controls"]),
                            "excluded_same_series_attacking_controls": " | ".join(block["excluded_same_series_attacking_controls"]),
                            "excluded_defensive_controls": " | ".join(block["excluded_defensive_controls"]),
                            "ignored_actual_E_controls": " | ".join(block.get("ignored_actual_E_controls", [])),
                            "usable_numeric_controls": " | ".join(block["usable_numeric_controls"]),
                            "categorical_controls": " | ".join(block["categorical_control_cols"]),
                            "estimand": block["estimand"],
                        })

                    if (
                        n < MIN_ROWS
                        or n_matches < MIN_MATCHES
                        or min(n_positive, n_negative) < MIN_CLASS_COUNT
                        or X_df.shape[1] == 0
                    ):
                        raise ValueError(
                            f"insufficient sample n={n}, matches={n_matches}, "
                            f"class_min={min(n_positive, n_negative)}, X={X_df.shape[1]}"
                        )

                    X = X_df.to_numpy(dtype=float)
                    D = D_df[selected].to_numpy(dtype=float)
                    raw_sd_map = {
                        name: float(np.std(D[:, j], ddof=1))
                        for j, name in enumerate(selected)
                    }
                    if any((not np.isfinite(value) or value <= 1e-12) for value in raw_sd_map.values()):
                        raise ValueError(f"constant treatment in {joint_spec}: {raw_sd_map}")

                    raw_corr, raw_vif = joint_correlation_and_vif(
                        D, selected, cluster, stage, joint_spec, "raw_treatments"
                    )
                    raw_corr_frames.append(raw_corr)
                    raw_vif_frames.append(raw_vif)

                    splits = make_grouped_splits(
                        y, groups, N_SPLITS,
                        cluster_seed(cluster, stage, 30000 + spec_index * 31),
                    )

                    with StepTimer(
                        runtime_log, cluster, stage, f"crossfit_RF_Ridge::{joint_spec}",
                        n, n_matches, note=f"joint D={len(selected)}, X={X.shape[1]}",
                    ):
                        main_cf = crossfit_main_and_baseline(
                            X=X, y=y, D=D, groups=groups, splits=splits,
                            seed=cluster_seed(cluster, stage, 31000 + spec_index * 41),
                            run_baseline=RUN_BASELINE,
                        )

                    fold_audit = main_cf["fold_audit"].copy()
                    fold_audit.insert(0, "joint_spec", joint_spec)
                    fold_audit.insert(0, "stage", stage)
                    fold_audit.insert(0, "cluster", cluster)
                    fold_audit_frames.append(fold_audit)

                    calibration = main_cf.get("calibration_audit", pd.DataFrame()).copy()
                    if not calibration.empty:
                        calibration.insert(0, "joint_spec", joint_spec)
                        calibration.insert(0, "stage", stage)
                        calibration.insert(0, "cluster", cluster)
                        calibration_frames.append(calibration)

                    yhat_main = main_cf["yhat_main"]
                    Dhat_main = main_cf["dhat_main"]
                    y_res_main = y - yhat_main
                    D_res_main = D - Dhat_main
                    D_res_main_z = scale_residuals_by_raw_sd(D_res_main, selected, raw_sd_map)

                    yrow_raw = outcome_performance_rows(
                        cluster, stage, y, main_cf["yhat_main_raw"], "RF_raw"
                    )
                    yrow_raw.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    y_performance_rows.append(yrow_raw)
                    yrow = outcome_performance_rows(
                        cluster, stage, y, yhat_main, "RF_NestedPlatt"
                    )
                    yrow.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    y_performance_rows.append(yrow)

                    drows = treatment_performance_rows(
                        cluster, stage, D, Dhat_main, selected, "RF_multioutput_D"
                    )
                    for row in drows:
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    d_performance_rows.extend(drows)

                    for row in overlap_rows(cluster, stage, D, Dhat_main, D_res_main, selected):
                        row["joint_spec"] = joint_spec
                        overlap_rows_all.append(row)
                    for j, name in enumerate(selected):
                        rows = support_bin_rows(
                            cluster, stage, D[:, j], Dhat_main[:, j], D_res_main[:, j],
                            name, SUPPORT_BINS,
                        )
                        for row in rows:
                            row["joint_spec"] = joint_spec
                        support_bin_rows_all.extend(rows)

                    residual_corr, residual_vif = joint_correlation_and_vif(
                        D_res_main_z, selected, cluster, stage, joint_spec,
                        "OOF_residuals_scaled_by_raw_SD",
                    )
                    residual_corr_frames.append(residual_corr)
                    residual_vif_frames.append(residual_vif)

                    main_trim = joint_all_trim_results(
                        y_res_main, D_res_main_z, selected, groups, cluster, stage,
                        joint_spec, "RF_Y + RF_multioutput_D",
                    )
                    if not main_trim.empty:
                        main_trim_frames.append(main_trim)

                    wald_main = joint_wald_test(
                        y_res_main, D_res_main_z, groups, selected, cluster, stage,
                        joint_spec, JOINT_MAIN_TRIM_FRAC, "RF_Y + RF_multioutput_D",
                    )
                    if not wald_main.empty:
                        wald_frames.append(wald_main)

                    placebo = joint_placebo_within_match(
                        y_res_main, D_res_main_z, groups, selected, cluster, stage,
                        joint_spec, PLACEBO_REPS,
                        cluster_seed(cluster, stage, 39000 + spec_index * 43),
                    )
                    if not placebo.empty:
                        placebo_frames.append(placebo)

                    fold_long, fold_summary = joint_fold_stability(
                        y_res_main, D_res_main_z, main_cf["fold_id"], selected,
                        cluster, stage, joint_spec,
                    )
                    if not fold_long.empty:
                        fold_long_frames.append(fold_long)
                        fold_summary_frames.append(fold_summary)

                    if RUN_TEAM_LOO:
                        team_long, team_summary = joint_team_loo_stability(
                            y_res_main, D_res_main_z, teams, selected,
                            cluster, stage, joint_spec,
                        )
                        if not team_long.empty:
                            team_long_frames.append(team_long)
                            team_summary_frames.append(team_summary)

                    if RUN_BASELINE:
                        yhat_base = main_cf["yhat_base"]
                        Dhat_base = main_cf["dhat_base"]
                        yrow_base = outcome_performance_rows(
                            cluster, stage, y, yhat_base, "Logistic_NestedPlatt"
                        )
                        yrow_base.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(yrow_base)
                        drows_base = treatment_performance_rows(
                            cluster, stage, D, Dhat_base, selected, "Ridge_multioutput_D"
                        )
                        for row in drows_base:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows_base)
                        base_res = joint_all_trim_results(
                            y - yhat_base,
                            scale_residuals_by_raw_sd(D - Dhat_base, selected, raw_sd_map),
                            selected, groups, cluster, stage, joint_spec,
                            "Logistic_Y + Ridge_multioutput_D",
                        )
                        if not base_res.empty:
                            baseline_trim_frames.append(base_res)

                    if RUN_HGB_ET_ROBUSTNESS:
                        with StepTimer(
                            runtime_log, cluster, stage, f"crossfit_HGB_ET::{joint_spec}",
                            n, n_matches,
                        ):
                            hgb_cf = joint_crossfit_hgb_et_multi(
                                X, y, D, groups, splits,
                                cluster_seed(cluster, stage, 41000 + spec_index * 47),
                            )
                        hcal = hgb_cf.get("calibration_audit", pd.DataFrame()).copy()
                        if not hcal.empty:
                            hcal.insert(0, "joint_spec", joint_spec)
                            hcal.insert(0, "stage", stage)
                            hcal.insert(0, "cluster", cluster)
                            calibration_frames.append(hcal)
                        yrow_h = outcome_performance_rows(
                            cluster, stage, y, hgb_cf["yhat"], "HGB_NestedPlatt"
                        )
                        yrow_h.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(yrow_h)
                        drows_h = treatment_performance_rows(
                            cluster, stage, D, hgb_cf["dhat"], selected, "ExtraTrees_separate_D"
                        )
                        for row in drows_h:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows_h)
                        hgb_res = joint_all_trim_results(
                            y - hgb_cf["yhat"],
                            scale_residuals_by_raw_sd(D - hgb_cf["dhat"], selected, raw_sd_map),
                            selected, groups, cluster, stage, joint_spec,
                            "HGB_Y + ExtraTrees_separate_D",
                        )
                        if not hgb_res.empty:
                            hgb_trim_frames.append(hgb_res)

                    if RUN_XGBOOST_ROBUSTNESS and HAS_XGBOOST:
                        with StepTimer(
                            runtime_log, cluster, stage, f"crossfit_XGB::{joint_spec}",
                            n, n_matches,
                        ):
                            xgb_cf = joint_crossfit_xgb_multi(
                                X, y, D, groups, splits,
                                cluster_seed(cluster, stage, 51000 + spec_index * 53),
                            )
                        xcal = xgb_cf.get("calibration_audit", pd.DataFrame()).copy()
                        if not xcal.empty:
                            xcal.insert(0, "joint_spec", joint_spec)
                            xcal.insert(0, "stage", stage)
                            xcal.insert(0, "cluster", cluster)
                            calibration_frames.append(xcal)
                        yrow_x = outcome_performance_rows(
                            cluster, stage, y, xgb_cf["yhat"], "XGB_NestedPlatt"
                        )
                        yrow_x.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(yrow_x)
                        drows_x = treatment_performance_rows(
                            cluster, stage, D, xgb_cf["dhat"], selected, "XGB_separate_D"
                        )
                        for row in drows_x:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows_x)
                        xgb_res = joint_all_trim_results(
                            y - xgb_cf["yhat"],
                            scale_residuals_by_raw_sd(D - xgb_cf["dhat"], selected, raw_sd_map),
                            selected, groups, cluster, stage, joint_spec,
                            "XGB_Y + XGB_separate_D",
                        )
                        if not xgb_res.empty:
                            xgb_trim_frames.append(xgb_res)

                    run_log.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "joint_kpis": " | ".join(selected),
                        "target_members": " | ".join(target_members),
                        "status": "done",
                        "rows": n,
                        "matches": n_matches,
                        "n_controls": X.shape[1],
                        "elapsed_seconds": time.perf_counter() - spec_start,
                    })

                except Exception as error:
                    print(f"[ERROR] cluster={cluster} stage={stage} spec={joint_spec}: {error!r}")
                    run_log.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "status": "error",
                        "reason": repr(error),
                    })

    concat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    main_all_trims = add_joint_fdr(concat(main_trim_frames))
    baseline_all_trims = add_joint_fdr(concat(baseline_trim_frames))
    hgb_all_trims = add_joint_fdr(concat(hgb_trim_frames))
    xgb_all_trims = add_joint_fdr(concat(xgb_trim_frames))
    main_results = main_trim_only(main_all_trims)

    y_performance = pd.DataFrame(y_performance_rows)
    d_performance = pd.DataFrame(d_performance_rows)
    overlap = pd.DataFrame(overlap_rows_all)
    support_bins = pd.DataFrame(support_bin_rows_all)
    raw_corr = concat(raw_corr_frames)
    raw_vif = concat(raw_vif_frames)
    residual_corr = concat(residual_corr_frames)
    residual_vif = concat(residual_vif_frames)
    wald = concat(wald_frames)
    placebo = concat(placebo_frames)
    fold_long = concat(fold_long_frames)
    fold_summary = concat(fold_summary_frames)
    team_long = concat(team_long_frames)
    team_summary = concat(team_summary_frames)
    fold_audit = concat(fold_audit_frames)
    calibration_audit = concat(calibration_frames)
    data_audit = pd.DataFrame(data_audit_rows)

    alternatives = {
        "ridge": baseline_all_trims,
        "hgb": hgb_all_trims,
        "xgb": xgb_all_trims,
    }
    main_report = build_joint_main_report(
        main_results, alternatives, overlap, placebo, fold_summary,
        team_summary, residual_vif, wald, data_audit,
    )
    target_model_map = pd.DataFrame(selection_rows)
    target_selection = build_rotating_target_selection(main_report, target_model_map)
    overview = build_joint_overview(main_report, fold_audit)
    paper_wide = build_paper_wide_table(main_report)

    robustness_frames = []
    for label, frame in [
        ("RF_main", main_all_trims),
        ("Ridge", baseline_all_trims),
        ("HGB_ET", hgb_all_trims),
        ("XGBoost", xgb_all_trims),
    ]:
        subset = main_trim_only(frame)
        if subset.empty:
            continue
        subset = subset.copy()
        subset.insert(0, "learner_set", label)
        robustness_frames.append(subset)
    robustness_comparison = concat(robustness_frames)

    checklist = pd.DataFrame([
        {"requirement": "Pass-style rotating joint models", "evidence": "Eight target tests per cluster-stage, deduplicated to six distinct three-mechanism models", "sheet": "23_target_selection / 24_joint_model_map", "status": "REPORTED"},
        {"requirement": "Joint conditional coefficients", "evidence": "All D residuals enter the second stage simultaneously", "sheet": "02_main_joint_report", "status": "REPORTED" if not main_report.empty else "MISSING"},
        {"requirement": "Match-grouped cross-fitting", "evidence": "Five folds with disjoint matches", "sheet": "20_fold_audit", "status": "PASS" if not fold_audit.empty and pd.to_numeric(fold_audit["match_leakage"], errors="coerce").max() == 0 else "CHECK"},
        {"requirement": "Joint multicollinearity", "evidence": "Raw and OOF-residual correlations, VIF, condition number", "sheet": "09-12", "status": "REPORTED"},
        {"requirement": "Continuous-treatment support", "evidence": "OOF D R2, residual SD ratio, support bins", "sheet": "08_overlap / 13_support_bins", "status": "REPORTED"},
        {"requirement": "Trim sensitivity", "evidence": "0%, 1%, 2%, 5% multivariate residual trimming", "sheet": "04_all_trims_RF", "status": "REPORTED"},
        {"requirement": "Learner robustness", "evidence": "RF, Ridge, HGB/ET, optional XGB", "sheet": "05_robustness", "status": "REPORTED"},
        {"requirement": "Joint Wald test", "evidence": "All joint coefficients tested simultaneously", "sheet": "14_joint_Wald", "status": "REPORTED"},
        {"requirement": "Placebo", "evidence": "Joint D-residual vector permuted within match", "sheet": "15_placebo", "status": "REPORTED"},
        {"requirement": "Fold and team stability", "evidence": "Joint coefficients re-estimated by fold and leave-one-team-out", "sheet": "16-19", "status": "REPORTED"},
        {"requirement": "Actual E exclusion", "evidence": "Only L and E'; actual E never used", "sheet": "25_manifest", "status": "PASS"},
        {"requirement": "Pass sample audit", "evidence": "The dedicated Pass K3 input is used without unrelated event-type exclusions", "sheet": "25_manifest", "status": "PASS"},
    ])

    manifest = pd.DataFrame([{
        "input_path": INPUT_PATH,
        "output_path": OUTPUT_XLSX,
        "rows_input_after_exclusion": len(df),
        "matches_input": df[MATCH_COL].nunique(dropna=True),
        "clusters": len(clusters),
        "outcome": OUTCOME_COL,
        "analysis_stages": "L and E' separately; actual E ignored",
        "actual_E_policy": "all columns ending in (E) ignored as treatments and controls",
        "actual_E_columns_detected": " | ".join(
            [column for column in df.columns if is_actual_E_col(column)]
        ),
        "Eprime_columns_required": True,
        "joint_families": json.dumps(JOINT_FAMILY_BASE_KPIS, ensure_ascii=False),
        "joint_anchor_rule": json.dumps({f"C{key[0]}_{key[1]}": value for key, value in JOINT_ANCHOR_BY_CLUSTER_STAGE.items()}, ensure_ascii=False),
        "default_anchor_combination": "Adv_5 + Avg_1_Def + Spr_Def",
        "target_tests_expected": 48,
        "distinct_joint_models_expected": 36,
        "single_result_workbook_input": "none",
        "selection_rule": "each target replaces its own family anchor; companion families retain prespecified anchors",
        "main_y_learner": "RandomForestClassifier + nested grouped Platt",
        "main_y_rf_params": json.dumps({
            "n_estimators": RF_Y_N_TREES,
            "min_samples_leaf": RF_Y_MIN_SAMPLES_LEAF,
            "max_features": RF_Y_MAX_FEATURES,
            "max_depth": RF_Y_MAX_DEPTH,
            "class_weight": RF_Y_CLASS_WEIGHT,
        }, ensure_ascii=False),
        "main_D_learner": "multi-output RandomForestRegressor",
        "main_D_rf_params": json.dumps({
            "n_estimators": RF_D_N_TREES,
            "min_samples_leaf": RF_D_MIN_SAMPLES_LEAF,
            "max_features": RF_D_MAX_FEATURES,
            "max_depth": RF_D_MAX_DEPTH,
        }, ensure_ascii=False),
        "rf_alignment_with_single_model": "PASS (startup-validated)",
        "baseline": "LogisticRegression + multi-output Ridge",
        "nonlinear_robustness": "HGB + separate ExtraTrees; optional XGB + separate XGB regressors",
        "crossfit": f"{N_SPLITS}-fold grouped by {MATCH_COL}",
        "main_trim": JOINT_MAIN_TRIM_FRAC,
        "trim_grid": json.dumps(JOINT_TRIM_GRID),
        "placebo_reps": PLACEBO_REPS,
        "second_stage": "joint unit-weight WLS on OOF residuals + HC3 SE",
        "scale": "each D residual divided by raw KPI SD; theta is probability change per 1 raw SD",
        "attacking_control_rule": "target-matched Att series removed from both L and E'; Avg/Centroid reciprocal exclusion",
        "defensive_control_rule": "no Def KPI outside the joint D vector enters X",
        "forbidden_controls": "exact locations, endpoint coordinates, pass geometry, duration, freeze-frame count, and all actual-E KPIs",
        "Eprime_definition": "endpoint-centered KPI from pass-instant freeze frame; not arrival-time player positions",
        "FDR": "BH global, within cluster-stage, and within joint spec",
        "joint_VIF_caution": JOINT_RESIDUAL_VIF_CAUTION,
        "joint_VIF_severe": JOINT_RESIDUAL_VIF_SEVERE,
        "condition_caution": JOINT_CONDITION_CAUTION,
        "condition_severe": JOINT_CONDITION_SEVERE,
        "xgboost_available": HAS_XGBOOST,
    }])

    runtime_frame = pd.DataFrame([record.__dict__ for record in runtime_log])
    run_log_frame = pd.DataFrame(run_log)
    selection_frame = target_selection
    model_map_frame = target_model_map

    tables = OrderedDict([
        ("00_reviewer_checklist", checklist),
        ("01_overview", overview),
        ("02_main_joint_report", main_report),
        ("03_paper_wide_table", paper_wide),
        ("04_all_trims_RF", main_all_trims),
        ("05_robustness", robustness_comparison),
        ("06_Y_performance", y_performance),
        ("07_D_performance", d_performance),
        ("08_overlap", overlap),
        ("09_raw_correlations", raw_corr),
        ("10_raw_VIF", raw_vif),
        ("11_residual_correlations", residual_corr),
        ("12_residual_VIF", residual_vif),
        ("13_support_bins", support_bins),
        ("14_joint_Wald", wald),
        ("15_placebo", placebo),
        ("16_fold_stability_sum", fold_summary),
        ("17_fold_stability_long", fold_long),
        ("18_teamLOO_sum", team_summary),
        ("19_teamLOO_long", team_long),
        ("20_fold_audit", fold_audit),
        ("21_calibration_audit", calibration_audit),
        ("22_data_audit", data_audit),
        ("23_target_selection", selection_frame),
        ("24_joint_model_map", model_map_frame),
        ("25_manifest", manifest),
        ("26_run_log", run_log_frame),
        ("27_runtime", runtime_frame),
    ])

    output = Path(OUTPUT_XLSX)
    output.parent.mkdir(parents=True, exist_ok=True)
    used = set()
    final_tables = OrderedDict()
    with pd.ExcelWriter(
        output, engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        for requested, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame([{"message": "No results produced for this table."}])
            sheet = safe_sheet_name(requested, used)
            table.to_excel(writer, index=False, sheet_name=sheet)
            final_tables[sheet] = table
        format_joint_workbook(writer, final_tables)

    elapsed = time.perf_counter() - total_start
    runtime_csv = output.with_name(output.stem + "_runtime.csv")
    pd.DataFrame([record.__dict__ for record in runtime_log]).to_csv(
        runtime_csv, index=False, encoding="utf-8-sig"
    )
    print("=" * 110)
    print(f"[DONE] {OUTPUT_XLSX}")
    print(f"[RUNTIME] {format_seconds(elapsed)}")
    print(f"[RUNTIME CSV] {runtime_csv}")
    print("=" * 110)

# =============================================================================
# 12. FIXED REPRESENTATIVE-KPI JOINT INTERACTION DML
# =============================================================================
# This is the active reviewer-facing Pass multivariable validation pipeline.
# The legacy rotating-joint functions above are retained only for traceability and
# are not called by __main__.

INTERACTION_OUTPUT_XLSX = r""

# One model per cluster-stage.  Selection was locked before running the interaction
# models.  Distance follows the requested deterministic fallback:
# Avg_1_Def -> Avg_3_Def -> Avg_5_Def.  The current single-KPI diagnostics only
# rejected Avg_1_Def(E') in Cluster 2, so that block uses Avg_3_Def(E').
INTERACTION_ANCHOR_BY_CLUSTER_STAGE = {
    (0, "L"):  ("Adv_5(L)",  "Avg_1_Def(L)",  "Area_Def(L)"),
    (0, "E'"): ("Adv_5(E')", "Avg_1_Def(E')", "Area_Def(E')"),
    (1, "L"):  ("Adv_5(L)",  "Avg_1_Def(L)",  "Area_Def(L)"),
    (1, "E'"): ("Adv_5(E')", "Avg_1_Def(E')", "Area_Def(E')"),
    (2, "L"):  ("Adv_5(L)",  "Avg_1_Def(L)",  "Area_Def(L)"),
    (2, "E'"): ("Adv_5(E')", "Avg_3_Def(E')", "Area_Def(E')"),
}

DISTANCE_FALLBACK_ORDER = ("Avg_1_Def", "Avg_3_Def", "Avg_5_Def")
INTERACTION_LEVELS = (-1.0, 0.0, 1.0)
INTERACTION_TERM_FDR_ALPHA = FDR_ALPHA
INTERACTION_AME_FDR_ALPHA = FDR_ALPHA
INTERACTION_WALD_FDR_ALPHA = FDR_ALPHA

# Stability thresholds used only to classify the reviewer-facing summary.  They do
# not delete estimates from the detailed output.
INTERACTION_FOLD_SIGN_MIN = 0.80
INTERACTION_TEAM_SIGN_MIN = 0.90
INTERACTION_PLACEBO_P_MAX = 0.10
INTERACTION_ALT_DIRECTION_MIN = 3  # require all available alternatives (up to three)


def interaction_term_names(main_names: Sequence[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    if len(main_names) != 3:
        raise ValueError("The fixed interaction model requires exactly three main KPIs.")
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = list(main_names) + [
        f"INT::{main_names[i]}*{main_names[j]}" for i, j in pairs
    ]
    return names, pairs


def build_interaction_terms_with_training_scale(
    D_train_raw: np.ndarray,
    D_test_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create three z-scored main terms and three pairwise products.

    Means and standard deviations are learned from the outer-training matches only
    and then applied to both the outer-training and outer-test rows.  Thus neither
    main-term scaling nor interaction construction leaks information from the outer
    test fold.
    """
    means = np.mean(D_train_raw, axis=0)
    sds = np.std(D_train_raw, axis=0, ddof=1)
    if np.any(~np.isfinite(sds)) or np.any(sds <= 1e-12):
        raise ValueError(f"Non-usable treatment SD in training fold: {sds}")
    z_train = (D_train_raw - means) / sds
    z_test = (D_test_raw - means) / sds
    pairs = [(0, 1), (0, 2), (1, 2)]
    int_train = np.column_stack([z_train[:, i] * z_train[:, j] for i, j in pairs])
    int_test = np.column_stack([z_test[:, i] * z_test[:, j] for i, j in pairs])
    return (
        np.column_stack([z_train, int_train]),
        np.column_stack([z_test, int_test]),
        means,
        sds,
    )


def crossfit_interaction_main_and_baseline(
    X: np.ndarray,
    y: np.ndarray,
    D_raw: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    main_names: Sequence[str],
    run_baseline: bool,
):
    n = len(y)
    term_names, _ = interaction_term_names(main_names)
    k_t = len(term_names)
    T_obs = np.full((n, k_t), np.nan)
    yhat_main = np.full(n, np.nan)
    yhat_main_raw = np.full(n, np.nan)
    that_main = np.full((n, k_t), np.nan)
    fold_id = np.full(n, -1, dtype=int)

    yhat_base = np.full(n, np.nan) if run_baseline else None
    yhat_base_raw = np.full(n, np.nan) if run_baseline else None
    that_base = np.full((n, k_t), np.nan) if run_baseline else None

    fold_rows = []
    calibration_rows = []
    standardization_rows = []

    for fold, (tr, te) in enumerate(splits):
        fold_id[te] = fold
        Ttr, Tte, means, sds = build_interaction_terms_with_training_scale(
            D_raw[tr], D_raw[te]
        )
        T_obs[te] = Tte

        for j, name in enumerate(main_names):
            standardization_rows.append({
                "outer_fold": fold,
                "main_kpi": name,
                "training_mean": float(means[j]),
                "training_sd": float(sds[j]),
                "train_rows": len(tr),
                "test_rows": len(te),
                "train_matches": len(pd.unique(groups[tr])),
                "test_matches": len(pd.unique(groups[te])),
            })

        p_main, p_main_raw, main_audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="rf", seed=seed + fold, calibrate=CALIBRATE_MAIN_Y,
        )
        yhat_main[te] = p_main
        yhat_main_raw[te] = p_main_raw
        main_audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        for key, value in safe_binary_metrics(y[te], p_main_raw).items():
            main_audit[f"outer_raw_{key}"] = value
        for key, value in safe_binary_metrics(y[te], p_main).items():
            main_audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(main_audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        d_model = make_main_d_model(seed + 1000 + fold)
        d_model.fit(Xtr, Ttr)
        pred = np.asarray(d_model.predict(Xte))
        if pred.ndim == 1:
            pred = pred[:, None]
        that_main[te] = pred

        base_audit = None
        if run_baseline:
            p_base, p_base_raw, base_audit = nested_group_platt_predict(
                X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
                learner="logistic", seed=seed + 2000 + fold,
                calibrate=CALIBRATE_BASELINE_Y,
            )
            yhat_base[te] = p_base
            yhat_base_raw[te] = p_base_raw
            base_audit.update({
                "outer_fold": fold,
                "outer_train_rows": len(tr),
                "outer_test_rows": len(te),
                "outer_train_matches": len(pd.unique(groups[tr])),
                "outer_test_matches": len(pd.unique(groups[te])),
            })
            for key, value in safe_binary_metrics(y[te], p_base_raw).items():
                base_audit[f"outer_raw_{key}"] = value
            for key, value in safe_binary_metrics(y[te], p_base).items():
                base_audit[f"outer_calibrated_{key}"] = value
            calibration_rows.append(base_audit)

            Xtr_s, Xte_s = prepare_fold_X(X, tr, te, scale=True)
            d_base = Ridge(alpha=1.0)
            d_base.fit(Xtr_s, Ttr)
            pred_base = np.asarray(d_base.predict(Xte_s))
            if pred_base.ndim == 1:
                pred_base = pred_base[:, None]
            that_base[te] = pred_base

        fold_rows.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "train_matches": len(pd.unique(groups[tr])),
            "test_matches": len(pd.unique(groups[te])),
            "train_positive_rate": float(np.mean(y[tr])),
            "test_positive_rate": float(np.mean(y[te])),
            "match_leakage": len(set(groups[tr]).intersection(set(groups[te]))),
            "main_calibration_used": bool(main_audit.get("calibration_used", False)),
            "baseline_calibration_used": (
                bool(base_audit.get("calibration_used", False)) if base_audit else np.nan
            ),
        })

    return {
        "term_names": term_names,
        "T_obs": T_obs,
        "yhat_main": yhat_main,
        "yhat_main_raw": yhat_main_raw,
        "that_main": that_main,
        "yhat_base": yhat_base,
        "yhat_base_raw": yhat_base_raw,
        "that_base": that_base,
        "fold_id": fold_id,
        "fold_audit": pd.DataFrame(fold_rows),
        "calibration_audit": pd.DataFrame(calibration_rows),
        "standardization_audit": pd.DataFrame(standardization_rows),
    }


def crossfit_interaction_hgb_et(
    X: np.ndarray,
    y: np.ndarray,
    D_raw: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    main_names: Sequence[str],
):
    n = len(y)
    term_names, _ = interaction_term_names(main_names)
    k_t = len(term_names)
    T_obs = np.full((n, k_t), np.nan)
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    that = np.full((n, k_t), np.nan)
    calibration_rows = []

    for fold, (tr, te) in enumerate(splits):
        Ttr, Tte, _, _ = build_interaction_terms_with_training_scale(D_raw[tr], D_raw[te])
        T_obs[te] = Tte
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="hgb", seed=seed + fold, calibrate=CALIBRATE_HGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        for key, value in safe_binary_metrics(y[te], p_raw).items():
            audit[f"outer_raw_{key}"] = value
        for key, value in safe_binary_metrics(y[te], p).items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        for j in range(k_t):
            model = make_et_d_model(seed + 1000 + fold * 101 + j)
            model.fit(Xtr, Ttr[:, j])
            that[te, j] = np.asarray(model.predict(Xte)).reshape(-1)

    return {
        "term_names": term_names,
        "T_obs": T_obs,
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "that": that,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }


def crossfit_interaction_xgb(
    X: np.ndarray,
    y: np.ndarray,
    D_raw: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    main_names: Sequence[str],
):
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is unavailable")
    n = len(y)
    term_names, _ = interaction_term_names(main_names)
    k_t = len(term_names)
    T_obs = np.full((n, k_t), np.nan)
    yhat = np.full(n, np.nan)
    yhat_raw = np.full(n, np.nan)
    that = np.full((n, k_t), np.nan)
    calibration_rows = []

    for fold, (tr, te) in enumerate(splits):
        Ttr, Tte, _, _ = build_interaction_terms_with_training_scale(D_raw[tr], D_raw[te])
        T_obs[te] = Tte
        p, p_raw, audit = nested_group_platt_predict(
            X_train=X[tr], y_train=y[tr], groups_train=groups[tr], X_test=X[te],
            learner="xgb", seed=seed + fold, calibrate=CALIBRATE_XGB_Y,
        )
        yhat[te] = p
        yhat_raw[te] = p_raw
        audit.update({
            "outer_fold": fold,
            "outer_train_rows": len(tr),
            "outer_test_rows": len(te),
            "outer_train_matches": len(pd.unique(groups[tr])),
            "outer_test_matches": len(pd.unique(groups[te])),
        })
        for key, value in safe_binary_metrics(y[te], p_raw).items():
            audit[f"outer_raw_{key}"] = value
        for key, value in safe_binary_metrics(y[te], p).items():
            audit[f"outer_calibrated_{key}"] = value
        calibration_rows.append(audit)

        Xtr, Xte = prepare_fold_X(X, tr, te, scale=False)
        for j in range(k_t):
            model = make_xgb_d_model(seed + 1000 + fold * 101 + j)
            model.fit(Xtr, Ttr[:, j])
            that[te, j] = np.asarray(model.predict(Xte)).reshape(-1)

    return {
        "term_names": term_names,
        "T_obs": T_obs,
        "yhat": yhat,
        "yhat_raw": yhat_raw,
        "that": that,
        "calibration_audit": pd.DataFrame(calibration_rows),
    }


def _effect_row_from_gradient(
    params: np.ndarray,
    cov: np.ndarray,
    gradient: np.ndarray,
    dof_cluster: int,
) -> Dict[str, float]:
    estimate = float(gradient @ params)
    variance = float(gradient @ cov @ gradient)
    se = math.sqrt(max(variance, 0.0))
    tval = estimate / se if se > 0 else np.nan
    pval = float(2 * stats.t.sf(abs(tval), dof_cluster)) if np.isfinite(tval) else np.nan
    tcrit = stats.t.ppf(0.975, dof_cluster)
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - tcrit * se,
        "ci_high": estimate + tcrit * se,
        "hc3_t_stat": tval,
            "t_stat_cluster": tval,  # deprecated compatibility alias
        "p_value": pval,
        "partial_r2_cluster_t": partial_r2_from_t(tval, dof_cluster),
        "robustness_value_point": robustness_value_from_t(tval, dof_cluster, q=1, alpha=1),
        "robustness_value_alpha05": robustness_value_from_t(tval, dof_cluster, q=1, alpha=0.05),
    }


def fit_fixed_interaction_second_stage(
    y_res: np.ndarray,
    T_res: np.ndarray,
    T_obs: np.ndarray,
    term_names: Sequence[str],
    main_names: Sequence[str],
    groups: np.ndarray,
    cluster,
    stage: str,
    joint_spec: str,
    learner_set: str,
    trim_frac: float,
    include_conditional: bool = True,
):
    mask = np.isfinite(y_res) & np.all(np.isfinite(T_res), axis=1) & np.all(np.isfinite(T_obs), axis=1)
    mask &= make_trim_mask(T_res, trim_frac)
    n = int(mask.sum())
    n_groups = int(len(pd.unique(groups[mask]))) if n else 0
    if n < 50 or n_groups < 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    design = sm.add_constant(T_res[mask], has_constant="add")
    fit = sm.WLS(
        y_res[mask], design, weights=np.ones(n, dtype=float)
    ).fit(cov_type="HC3", use_t=True)
    params = np.asarray(fit.params, dtype=float)[1:]
    cov_full = np.asarray(fit.cov_params(), dtype=float)
    cov = cov_full[1:, 1:]
    dof_cluster = max(1, int(fit.df_resid))
    tcrit = stats.t.ppf(0.975, dof_cluster)
    term_rows = []

    for j, name in enumerate(term_names):
        beta = float(params[j])
        se = float(math.sqrt(max(cov[j, j], 0.0)))
        tval = beta / se if se > 0 else np.nan
        pval = float(2 * stats.t.sf(abs(tval), dof_cluster)) if np.isfinite(tval) else np.nan
        term_rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "learner_set": learner_set,
            "term": name,
            "effect_kind": "main_coefficient_at_mean" if j < 3 else "pairwise_interaction",
            "estimate": beta,
            "se": se,
            "ci_low": beta - tcrit * se,
            "ci_high": beta + tcrit * se,
            "hc3_t_stat": tval,
            "t_stat_cluster": tval,  # deprecated compatibility alias
            "p_value": pval,
            "partial_r2_cluster_t": partial_r2_from_t(tval, dof_cluster),
            "robustness_value_point": robustness_value_from_t(tval, dof_cluster, q=1, alpha=1),
            "robustness_value_alpha05": robustness_value_from_t(tval, dof_cluster, q=1, alpha=0.05),
            "trim_fraction_rule": float(trim_frac),
            "n_used": n,
            "n_matches": n_groups,
            "hc3_residual_df": dof_cluster,
            "cluster_se_df": dof_cluster,  # deprecated compatibility alias
            "second_stage_r2": float(fit.rsquared),
            "design_condition_number": float(np.linalg.cond(design)),
            "scale": (
                "probability change per 1-SD main KPI when the other two standardized KPIs equal zero"
                if j < 3 else
                "probability change per one-unit product of two training-fold-standardized KPIs"
            ),
        })

    pair_to_index = {(0, 1): 3, (0, 2): 4, (1, 2): 5}
    ame_rows = []
    conditional_rows = []
    mean_z = np.mean(T_obs[mask, :3], axis=0)

    for target_idx, target in enumerate(main_names):
        moderators = [idx for idx in range(3) if idx != target_idx]
        gradient = np.zeros(len(term_names), dtype=float)
        gradient[target_idx] = 1.0
        for moderator_idx in moderators:
            pair = tuple(sorted((target_idx, moderator_idx)))
            gradient[pair_to_index[pair]] = mean_z[moderator_idx]
        values = _effect_row_from_gradient(params, cov, gradient, dof_cluster)
        ame_rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "learner_set": learner_set,
            "main_kpi": target,
            "effect_type": "average_marginal_effect",
            "moderator_1": main_names[moderators[0]],
            "moderator_1_mean_z": float(mean_z[moderators[0]]),
            "moderator_2": main_names[moderators[1]],
            "moderator_2_mean_z": float(mean_z[moderators[1]]),
            **values,
            "trim_fraction_rule": float(trim_frac),
            "n_used": n,
            "n_matches": n_groups,
            "hc3_residual_df": dof_cluster,
            "cluster_se_df": dof_cluster,  # deprecated compatibility alias
            "second_stage_r2": float(fit.rsquared),
            "design_condition_number": float(np.linalg.cond(design)),
            "scale": "average probability change per 1-SD increase in the target KPI over observed tactical configurations",
        })

        if include_conditional:
            for level_a in INTERACTION_LEVELS:
                for level_b in INTERACTION_LEVELS:
                    conditional_gradient = np.zeros(len(term_names), dtype=float)
                    conditional_gradient[target_idx] = 1.0
                    pair_a = tuple(sorted((target_idx, moderators[0])))
                    pair_b = tuple(sorted((target_idx, moderators[1])))
                    conditional_gradient[pair_to_index[pair_a]] = level_a
                    conditional_gradient[pair_to_index[pair_b]] = level_b
                    cond = _effect_row_from_gradient(params, cov, conditional_gradient, dof_cluster)
                    conditional_rows.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "learner_set": learner_set,
                        "main_kpi": target,
                        "moderator_1": main_names[moderators[0]],
                        "moderator_1_level_z": float(level_a),
                        "moderator_2": main_names[moderators[1]],
                        "moderator_2_level_z": float(level_b),
                        **cond,
                        "trim_fraction_rule": float(trim_frac),
                        "n_used": n,
                        "n_matches": n_groups,
                        "scale": "conditional probability change per 1-SD increase in the target KPI",
                    })

    def run_wald(indices: Sequence[int], label: str) -> Dict[str, object]:
        restriction = np.zeros((len(indices), 1 + len(term_names)), dtype=float)
        for row_i, term_i in enumerate(indices):
            restriction[row_i, 1 + term_i] = 1.0
        try:
            test = fit.wald_test(restriction, use_f=True, scalar=True)
            stat = float(np.asarray(test.statistic).reshape(-1)[0])
            pval = float(np.asarray(test.pvalue).reshape(-1)[0])
        except Exception:
            stat, pval = np.nan, np.nan
        return {
            "cluster": cluster,
            "stage": stage,
            "joint_spec": joint_spec,
            "learner_set": learner_set,
            "wald_block": label,
            "terms_tested": " | ".join(term_names[i] for i in indices),
            "n_terms": len(indices),
            "wald_F": stat,
            "p_value": pval,
            "trim_fraction_rule": float(trim_frac),
            "n_used": n,
            "n_matches": n_groups,
            "full_second_stage_r2": float(fit.rsquared),
            "design_condition_number": float(np.linalg.cond(design)),
        }

    wald_rows = [
        run_wald([0, 1, 2], "three_main_terms"),
        run_wald([3, 4, 5], "three_interactions_jointly"),
        run_wald([0, 1, 2, 3, 4, 5], "all_six_terms"),
    ]

    # Nested additive second stage for a transparent test of whether interactions
    # materially improve the residual outcome equation.
    add_design = sm.add_constant(T_res[mask, :3], has_constant="add")
    add_fit = sm.WLS(
        y_res[mask], add_design, weights=np.ones(n, dtype=float)
    ).fit(cov_type="HC3", use_t=True)
    for row in wald_rows:
        row["additive_second_stage_r2"] = float(add_fit.rsquared)
        row["interaction_increment_r2"] = float(fit.rsquared - add_fit.rsquared)

    return (
        pd.DataFrame(term_rows),
        pd.DataFrame(ame_rows),
        pd.DataFrame(conditional_rows),
        pd.DataFrame(wald_rows),
    )


def add_fixed_interaction_fdr(
    term_df: pd.DataFrame,
    ame_df: pd.DataFrame,
    conditional_df: pd.DataFrame,
    wald_df: pd.DataFrame,
):
    terms = term_df.copy()
    ames = ame_df.copy()
    cond = conditional_df.copy()
    wald = wald_df.copy()

    if not terms.empty:
        terms["q_global_by_effect_kind"] = np.nan
        for _, idx in terms.groupby(
            ["learner_set", "trim_fraction_rule", "effect_kind"], dropna=False
        ).groups.items():
            terms.loc[idx, "q_global_by_effect_kind"] = bh_fdr(
                pd.to_numeric(terms.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        terms["q_within_cluster_stage"] = np.nan
        for _, idx in terms.groupby(
            ["learner_set", "trim_fraction_rule", "cluster", "stage"], dropna=False
        ).groups.items():
            terms.loc[idx, "q_within_cluster_stage"] = bh_fdr(
                pd.to_numeric(terms.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        terms["fdr_global_pass"] = terms["q_global_by_effect_kind"] <= FDR_ALPHA

    if not ames.empty:
        ames["q_global"] = np.nan
        for _, idx in ames.groupby(
            ["learner_set", "trim_fraction_rule"], dropna=False
        ).groups.items():
            ames.loc[idx, "q_global"] = bh_fdr(
                pd.to_numeric(ames.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        ames["q_within_cluster_stage"] = np.nan
        for _, idx in ames.groupby(
            ["learner_set", "trim_fraction_rule", "cluster", "stage"], dropna=False
        ).groups.items():
            ames.loc[idx, "q_within_cluster_stage"] = bh_fdr(
                pd.to_numeric(ames.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        ames["fdr_global_pass"] = ames["q_global"] <= FDR_ALPHA

    if not cond.empty:
        cond["q_within_target_grid"] = np.nan
        for _, idx in cond.groupby(
            ["learner_set", "cluster", "stage", "joint_spec", "main_kpi"], dropna=False
        ).groups.items():
            cond.loc[idx, "q_within_target_grid"] = bh_fdr(
                pd.to_numeric(cond.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        cond["fdr_within_target_grid_pass"] = cond["q_within_target_grid"] <= FDR_ALPHA

    if not wald.empty:
        wald["q_global_by_wald_block"] = np.nan
        for _, idx in wald.groupby(
            ["learner_set", "trim_fraction_rule", "wald_block"], dropna=False
        ).groups.items():
            wald.loc[idx, "q_global_by_wald_block"] = bh_fdr(
                pd.to_numeric(wald.loc[idx, "p_value"], errors="coerce").to_numpy()
            )
        wald["fdr_global_pass"] = wald["q_global_by_wald_block"] <= FDR_ALPHA

    return terms, ames, cond, wald


def interaction_ame_point_estimates(
    y_res: np.ndarray,
    T_res: np.ndarray,
    T_obs: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(y_res) & np.all(np.isfinite(T_res), axis=1) & np.all(np.isfinite(T_obs), axis=1)
    if mask.sum() < 30:
        return np.full(3, np.nan)
    design = np.column_stack([np.ones(mask.sum()), T_res[mask]])
    params = np.linalg.lstsq(design, y_res[mask], rcond=None)[0][1:]
    mean_z = np.mean(T_obs[mask, :3], axis=0)
    pair_to_index = {(0, 1): 3, (0, 2): 4, (1, 2): 5}
    out = []
    for j in range(3):
        value = params[j]
        for k in range(3):
            if k == j:
                continue
            value += params[pair_to_index[tuple(sorted((j, k)))]] * mean_z[k]
        out.append(float(value))
    return np.asarray(out)


def interaction_ame_fold_stability(
    y_res: np.ndarray,
    T_res: np.ndarray,
    T_obs: np.ndarray,
    fold_id: np.ndarray,
    main_names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
):
    rows = []
    for fold in sorted(pd.unique(fold_id)):
        if fold < 0:
            continue
        mask = fold_id == fold
        values = interaction_ame_point_estimates(y_res[mask], T_res[mask], T_obs[mask])
        for j, name in enumerate(main_names):
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "fold": int(fold),
                "main_kpi": name,
                "ame": float(values[j]),
                "n": int(mask.sum()),
            })
    long = pd.DataFrame(rows)
    summary = []
    if not long.empty:
        for name, group in long.groupby("main_kpi"):
            values = pd.to_numeric(group["ame"], errors="coerce").dropna().to_numpy()
            median = float(np.median(values)) if len(values) else np.nan
            summary.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "main_kpi": name,
                "n_folds": len(values),
                "median_ame": median,
                "min_ame": float(np.min(values)) if len(values) else np.nan,
                "max_ame": float(np.max(values)) if len(values) else np.nan,
                "sign_consistency": (
                    float(np.mean(np.sign(values) == np.sign(median)))
                    if len(values) and median != 0 else np.nan
                ),
            })
    return long, pd.DataFrame(summary)


def interaction_ame_team_loo_stability(
    y_res: np.ndarray,
    T_res: np.ndarray,
    T_obs: np.ndarray,
    teams: np.ndarray,
    main_names: Sequence[str],
    cluster,
    stage: str,
    joint_spec: str,
):
    rows = []
    for team in [value for value in pd.unique(teams) if pd.notna(value)]:
        mask = teams != team
        if mask.sum() < 40:
            continue
        values = interaction_ame_point_estimates(y_res[mask], T_res[mask], T_obs[mask])
        for j, name in enumerate(main_names):
            rows.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "left_out_team": str(team),
                "main_kpi": name,
                "ame": float(values[j]),
                "n": int(mask.sum()),
            })
    long = pd.DataFrame(rows)
    summary = []
    if not long.empty:
        for name, group in long.groupby("main_kpi"):
            values = pd.to_numeric(group["ame"], errors="coerce").dropna().to_numpy()
            median = float(np.median(values)) if len(values) else np.nan
            summary.append({
                "cluster": cluster,
                "stage": stage,
                "joint_spec": joint_spec,
                "main_kpi": name,
                "n_leave_one_team_runs": len(values),
                "median_ame": median,
                "min_ame": float(np.min(values)) if len(values) else np.nan,
                "max_ame": float(np.max(values)) if len(values) else np.nan,
                "sign_consistency": (
                    float(np.mean(np.sign(values) == np.sign(median)))
                    if len(values) and median != 0 else np.nan
                ),
            })
    return long, pd.DataFrame(summary)


def build_interaction_learner_summary(ame_df: pd.DataFrame, term_df: pd.DataFrame):
    if ame_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    main = ame_df.loc[np.isclose(pd.to_numeric(ame_df["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)].copy()
    terms = term_df.loc[np.isclose(pd.to_numeric(term_df["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)].copy()

    def summarize(frame, id_col, estimate_col="estimate"):
        if frame.empty:
            return pd.DataFrame()
        key_cols = ["cluster", "stage", "joint_spec", id_col]
        pivot = frame.pivot_table(index=key_cols, columns="learner_set", values=estimate_col, aggfunc="first").reset_index()
        if "RF_main" not in pivot.columns:
            return pivot
        alt_cols = [name for name in ["Ridge", "HGB_ET", "XGBoost"] if name in pivot.columns]
        for alt in alt_cols:
            pivot[f"direction_same_{alt}"] = (
                np.sign(pd.to_numeric(pivot[alt], errors="coerce"))
                == np.sign(pd.to_numeric(pivot["RF_main"], errors="coerce"))
            )
        direction_cols = [f"direction_same_{alt}" for alt in alt_cols]
        pivot["n_alternative_learners"] = len(alt_cols)
        pivot["n_alternative_same_direction"] = (
            pivot[direction_cols].sum(axis=1) if direction_cols else 0
        )
        pivot["all_available_alternatives_same_direction"] = (
            pivot["n_alternative_same_direction"] == pivot["n_alternative_learners"]
        )
        return pivot

    return summarize(main, "main_kpi"), summarize(terms, "term")


def build_ame_trim_stability(ame_df: pd.DataFrame) -> pd.DataFrame:
    if ame_df is None or ame_df.empty:
        return pd.DataFrame()
    rf = ame_df.loc[ame_df["learner_set"] == "RF_main"].copy()
    rows = []
    for keys, group in rf.groupby(["cluster", "stage", "joint_spec", "main_kpi"], dropna=False):
        values = pd.to_numeric(group["estimate"], errors="coerce").to_numpy()
        pvals = pd.to_numeric(group["p_value"], errors="coerce").to_numpy()
        trims = pd.to_numeric(group["trim_fraction_rule"], errors="coerce").to_numpy()
        ok = np.isfinite(values) & np.isfinite(trims)
        values = values[ok]
        pvals = pvals[ok]
        trims = trims[ok]
        if len(values) == 0:
            continue
        main_mask = np.isclose(trims, MAIN_TRIM_FRAC)
        reference = float(values[np.where(main_mask)[0][0]]) if np.any(main_mask) else float(np.median(values))
        rows.append({
            "cluster": keys[0],
            "stage": keys[1],
            "joint_spec": keys[2],
            "main_kpi": keys[3],
            "n_trim_settings": len(values),
            "trim_min_ame": float(np.min(values)),
            "trim_max_ame": float(np.max(values)),
            "trim_sign_consistency_with_main": float(np.mean(np.sign(values) == np.sign(reference))) if reference != 0 else np.nan,
            "n_trim_p_lt_05": int(np.sum(pvals < 0.05)),
            "all_trim_direction_same": bool(np.all(np.sign(values) == np.sign(reference))) if reference != 0 else False,
        })
    return pd.DataFrame(rows)


def build_interaction_main_summary(
    ame_main: pd.DataFrame,
    term_main: pd.DataFrame,
    overlap: pd.DataFrame,
    residual_vif: pd.DataFrame,
    placebo: pd.DataFrame,
    fold_ame_summary: pd.DataFrame,
    team_ame_summary: pd.DataFrame,
    learner_ame_summary: pd.DataFrame,
    trim_ame_summary: pd.DataFrame,
    wald_main: pd.DataFrame,
):
    if ame_main.empty:
        return pd.DataFrame()
    out = ame_main.copy()
    keys = ["cluster", "stage", "joint_spec", "main_kpi"]

    if not overlap.empty:
        ov = overlap.rename(columns={"treatment": "main_kpi"})
        cols = keys + [c for c in [
            "treatment_oof_r2", "residual_sd_ratio", "overlap_status", "overlap_flag"
        ] if c in ov.columns]
        out = out.merge(ov[cols], on=keys, how="left")

    if not residual_vif.empty:
        rv = residual_vif.rename(columns={"treatment": "main_kpi"})
        vif_col = "VIF" if "VIF" in rv.columns else "vif_from_inverse_correlation"
        if vif_col in rv.columns:
            out = out.merge(rv[keys + [vif_col]].rename(columns={vif_col: "residual_VIF"}), on=keys, how="left")

    if not placebo.empty:
        pl = placebo.rename(columns={"treatment": "main_kpi"})
        pcol = "empirical_placebo_p"
        if pcol in pl.columns:
            out = out.merge(pl[keys + [pcol]], on=keys, how="left")

    if not fold_ame_summary.empty:
        cols = keys + [c for c in ["n_folds", "median_ame", "min_ame", "max_ame", "sign_consistency"] if c in fold_ame_summary.columns]
        out = out.merge(
            fold_ame_summary[cols].rename(columns={"sign_consistency": "fold_ame_sign_consistency"}),
            on=keys, how="left"
        )

    if not team_ame_summary.empty:
        cols = keys + [c for c in ["n_leave_one_team_runs", "median_ame", "min_ame", "max_ame", "sign_consistency"] if c in team_ame_summary.columns]
        rename = {
            "median_ame": "teamLOO_median_ame",
            "min_ame": "teamLOO_min_ame",
            "max_ame": "teamLOO_max_ame",
            "sign_consistency": "teamLOO_ame_sign_consistency",
        }
        out = out.merge(team_ame_summary[cols].rename(columns=rename), on=keys, how="left")

    if not learner_ame_summary.empty:
        learner_cols = keys + [c for c in learner_ame_summary.columns if c not in keys]
        out = out.merge(learner_ame_summary[learner_cols], on=keys, how="left")

    if trim_ame_summary is not None and not trim_ame_summary.empty:
        trim_cols = keys + [c for c in trim_ame_summary.columns if c not in keys]
        out = out.merge(trim_ame_summary[trim_cols], on=keys, how="left")

    interaction_wald = wald_main.loc[wald_main["wald_block"] == "three_interactions_jointly"].copy() if not wald_main.empty else pd.DataFrame()
    if not interaction_wald.empty:
        wcols = ["cluster", "stage", "joint_spec", "wald_F", "p_value", "q_global_by_wald_block", "interaction_increment_r2"]
        wcols = [c for c in wcols if c in interaction_wald.columns]
        out = out.merge(
            interaction_wald[wcols].rename(columns={
                "wald_F": "interaction_joint_Wald_F",
                "p_value": "interaction_joint_p",
                "q_global_by_wald_block": "interaction_joint_q",
            }),
            on=["cluster", "stage", "joint_spec"], how="left"
        )

    def classify(row):
        q_ok = pd.notna(row.get("q_global")) and float(row.get("q_global")) <= FDR_ALPHA
        overlap_ok = str(row.get("overlap_status", "PASS")).startswith("PASS")
        vif = pd.to_numeric(pd.Series([row.get("residual_VIF", np.nan)]), errors="coerce").iloc[0]
        vif_ok = pd.isna(vif) or float(vif) < JOINT_RESIDUAL_VIF_SEVERE
        fold = pd.to_numeric(pd.Series([row.get("fold_ame_sign_consistency", np.nan)]), errors="coerce").iloc[0]
        team = pd.to_numeric(pd.Series([row.get("teamLOO_ame_sign_consistency", np.nan)]), errors="coerce").iloc[0]
        fold_ok = pd.isna(fold) or float(fold) >= INTERACTION_FOLD_SIGN_MIN
        team_ok = pd.isna(team) or float(team) >= INTERACTION_TEAM_SIGN_MIN
        placebo_p = pd.to_numeric(pd.Series([row.get("empirical_placebo_p", np.nan)]), errors="coerce").iloc[0]
        placebo_ok = pd.isna(placebo_p) or float(placebo_p) <= INTERACTION_PLACEBO_P_MAX
        trim_sign = pd.to_numeric(pd.Series([row.get("trim_sign_consistency_with_main", np.nan)]), errors="coerce").iloc[0]
        n_trim = pd.to_numeric(pd.Series([row.get("n_trim_settings", 0)]), errors="coerce").fillna(0).iloc[0]
        n_trim_sig = pd.to_numeric(pd.Series([row.get("n_trim_p_lt_05", 0)]), errors="coerce").fillna(0).iloc[0]
        trim_ok = (pd.isna(trim_sign) or float(trim_sign) >= 1.0) and (n_trim == 0 or n_trim_sig >= max(1, int(n_trim) - 1))
        alt_same = pd.to_numeric(pd.Series([row.get("n_alternative_same_direction", 0)]), errors="coerce").fillna(0).iloc[0]
        alt_total = pd.to_numeric(pd.Series([row.get("n_alternative_learners", 0)]), errors="coerce").fillna(0).iloc[0]
        alt_ok = alt_total == 0 or alt_same >= min(INTERACTION_ALT_DIRECTION_MIN, alt_total)
        int_q = pd.to_numeric(pd.Series([row.get("interaction_joint_q", np.nan)]), errors="coerce").iloc[0]
        if q_ok and overlap_ok and vif_ok and fold_ok and team_ok and placebo_ok and trim_ok and alt_ok:
            return "CONFIRMED_average_multivariable_effect"
        if (not q_ok) and pd.notna(int_q) and float(int_q) <= FDR_ALPHA:
            return "CONTEXT_DEPENDENT_interaction_evidence"
        return "UNCERTAIN_multivariable_evidence"

    out["evidence_classification"] = out.apply(classify, axis=1)
    return out



# =============================================================================
# 12A. ALL-PASSED-KPI ROTATING INTERACTION CONFIGURATION
# =============================================================================
# The original fixed-representative pipeline is retained below for traceability.
# The active __main__ entry point at the end of this file calls
# all_passed_interaction_main().

SINGLE_RESULT_XLSX = r""
SINGLE_MAIN_SHEET = "05_main_oneD"
SINGLE_TRIM_SHEET = "06_oneD_all_trims"
SINGLE_ALTERNATIVE_SHEETS = OrderedDict([
    ("Ridge", "07_linear_oneD"),
    ("HGB_ET", "08_hgb_et_oneD"),
    ("XGBoost", "09_xgboost_oneD"),
])

ALL_PASSED_INTERACTION_OUTPUT_XLSX = r""

# Strict single-stage admission rule.  These settings reproduce the intended
# screening logic: a KPI must pass the RF main model and remain robust across all
# available trim and nuisance-learner specifications before entering the joint stage.
SINGLE_SCREEN_FOLD_SIGN_MIN = 0.80
SINGLE_SCREEN_TEAM_SIGN_MIN = 0.90
SINGLE_SCREEN_PLACEBO_P_MAX = 0.05
SINGLE_SCREEN_REQUIRE_NO_SEVERE_OVERLAP = True
SINGLE_SCREEN_REQUIRE_ALL_TRIM_DIRECTIONS = True
SINGLE_SCREEN_MIN_TRIM_P_LT_05 = len(TRIM_GRID)  # default: 4/4 trim settings
SINGLE_SCREEN_REQUIRE_ALL_AVAILABLE_ALT_DIRECTIONS = True
SINGLE_SCREEN_REQUIRE_ALL_AVAILABLE_ALT_FDR = True

# Companion anchors are selected separately in every cluster-stage-family from the
# KPIs that pass the strict single-stage screen.  Smaller global q is preferred;
# ties are broken by larger OVB robustness value and then larger absolute effect.
ANCHOR_SELECTION_RULE = (
    "largest absolute standardized single-KPI AME; then smallest global q; "
    "then largest RV_alpha05"
)


def _joint_family_name(treatment: str) -> Optional[str]:
    """Map one defending KPI to the three rotating joint-model mechanisms."""
    base = remove_stage_suffix(treatment)
    if re.fullmatch(r"Adv_\d+", base, flags=re.IGNORECASE):
        return "local_advantage"
    if re.fullmatch(
        r"(?:Avg_\d+_Def|DistToDefCentroid)", base, flags=re.IGNORECASE
    ):
        return "distance_centroid"
    if re.fullmatch(r"(?:Area_Def|Spr_Def)", base, flags=re.IGNORECASE):
        return "structure_shape"
    return None


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _safe_read_result_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        return pd.DataFrame()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_single_kpi_screening() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the single-KPI workbook and determine joint-stage admission.

    Returns
    -------
    audit:
        One row per built-in KPI with every screening component and a failure reason.
    passed:
        Strictly screened rows admitted to the rotating joint-interaction stage.
    """
    path = Path(SINGLE_RESULT_XLSX)
    if not path.exists():
        raise FileNotFoundError(
            f"Single-KPI result workbook not found: {SINGLE_RESULT_XLSX}"
        )

    main = _safe_read_result_sheet(SINGLE_RESULT_XLSX, SINGLE_MAIN_SHEET)
    if main.empty:
        raise ValueError(f"No rows found in {SINGLE_MAIN_SHEET}")
    required = {
        "cluster", "stage", "treatment", "theta_per_1sd", "p_value", "q_global"
    }
    missing = sorted(required.difference(main.columns))
    if missing:
        raise ValueError(f"Single-KPI main sheet missing columns: {missing}")

    main = main.loc[main["stage"].astype(str).isin(ACTIVE_STAGES)].copy()
    main["joint_family"] = main["treatment"].map(_joint_family_name)
    allowed_bases = {
        base
        for bases in JOINT_FAMILY_BASE_KPIS.values()
        for base in bases
    }
    main["builtin_joint_candidate"] = main["treatment"].map(
        lambda value: remove_stage_suffix(value) in allowed_bases
    )
    main = main.loc[main["builtin_joint_candidate"] & main["joint_family"].notna()].copy()

    numeric_cols = [
        "theta_per_1sd", "p_value", "q_global", "fold_sign_consistency",
        "team_loo_sign_consistency", "empirical_placebo_p",
        "robustness_value_alpha05",
    ]
    for column in numeric_cols:
        if column in main.columns:
            main[column] = pd.to_numeric(main[column], errors="coerce")

    keys = ["cluster", "stage", "treatment"]

    # Trim stability.
    trims = _safe_read_result_sheet(SINGLE_RESULT_XLSX, SINGLE_TRIM_SHEET)
    trim_rows = []
    if not trims.empty:
        trims = trims.loc[trims["stage"].astype(str).isin(ACTIVE_STAGES)].copy()
        for column in ["theta_per_1sd", "p_value", "trim_fraction_rule"]:
            if column in trims.columns:
                trims[column] = pd.to_numeric(trims[column], errors="coerce")
        main_theta = main.set_index(keys)["theta_per_1sd"].to_dict()
        for group_key, group in trims.groupby(keys, dropna=False):
            reference = main_theta.get(tuple(group_key), np.nan)
            values = pd.to_numeric(group["theta_per_1sd"], errors="coerce")
            pvals = pd.to_numeric(group["p_value"], errors="coerce")
            valid = values.notna()
            same = bool(
                valid.any()
                and np.isfinite(reference)
                and np.all(np.sign(values.loc[valid]) == np.sign(reference))
            )
            trim_rows.append({
                "cluster": group_key[0],
                "stage": group_key[1],
                "treatment": group_key[2],
                "n_trim_settings": int(valid.sum()),
                "n_trim_p_lt_05": int((pvals.loc[valid] < 0.05).sum()),
                "all_trim_directions_same": same,
                "trim_min_effect": float(values.loc[valid].min()) if valid.any() else np.nan,
                "trim_max_effect": float(values.loc[valid].max()) if valid.any() else np.nan,
            })
    trim_summary = pd.DataFrame(trim_rows)
    if not trim_summary.empty:
        main = main.merge(trim_summary, on=keys, how="left")
    else:
        main["n_trim_settings"] = 0
        main["n_trim_p_lt_05"] = 0
        main["all_trim_directions_same"] = False
        main["trim_min_effect"] = np.nan
        main["trim_max_effect"] = np.nan

    # Alternative nuisance learners at the prespecified main trimming rule.
    available_alt_labels = []
    for label, sheet in SINGLE_ALTERNATIVE_SHEETS.items():
        alt = _safe_read_result_sheet(SINGLE_RESULT_XLSX, sheet)
        if alt.empty:
            continue
        required_alt_columns = set(keys + ["theta_per_1sd"])
        if not required_alt_columns.issubset(alt.columns):
            continue
        alt = alt.loc[alt["stage"].astype(str).isin(ACTIVE_STAGES)].copy()
        if alt.empty:
            continue
        available_alt_labels.append(label)
        if "trim_fraction_rule" in alt.columns:
            trim_value = pd.to_numeric(alt["trim_fraction_rule"], errors="coerce")
            alt = alt.loc[np.isclose(trim_value, MAIN_TRIM_FRAC)].copy()
        keep = keys + [
            column for column in ["theta_per_1sd", "q_global", "fdr_global_pass"]
            if column in alt.columns
        ]
        alt = alt[keep].drop_duplicates(keys)
        alt = alt.rename(columns={
            "theta_per_1sd": f"{label}_theta_per_1sd",
            "q_global": f"{label}_q_global",
            "fdr_global_pass": f"{label}_fdr_global_pass",
        })
        main = main.merge(alt, on=keys, how="left")
        theta_col = f"{label}_theta_per_1sd"
        main[f"{label}_direction_same"] = (
            np.sign(pd.to_numeric(main["theta_per_1sd"], errors="coerce"))
            == np.sign(pd.to_numeric(main[theta_col], errors="coerce"))
        )

    main["main_global_fdr_pass"] = pd.to_numeric(
        main["q_global"], errors="coerce"
    ) <= FDR_ALPHA
    overlap_flag = main.get("overlap_flag", pd.Series(False, index=main.index))
    main["single_overlap_pass"] = ~overlap_flag.map(_safe_bool)
    main["single_fold_pass"] = pd.to_numeric(
        main.get("fold_sign_consistency", np.nan), errors="coerce"
    ) >= SINGLE_SCREEN_FOLD_SIGN_MIN
    main["single_team_pass"] = pd.to_numeric(
        main.get("team_loo_sign_consistency", np.nan), errors="coerce"
    ) >= SINGLE_SCREEN_TEAM_SIGN_MIN
    main["single_placebo_pass"] = pd.to_numeric(
        main.get("empirical_placebo_p", np.nan), errors="coerce"
    ) <= SINGLE_SCREEN_PLACEBO_P_MAX
    main["single_trim_pass"] = (
        pd.to_numeric(main["n_trim_settings"], errors="coerce") >= len(TRIM_GRID)
    ) & (
        pd.to_numeric(main["n_trim_p_lt_05"], errors="coerce")
        >= SINGLE_SCREEN_MIN_TRIM_P_LT_05
    )
    if SINGLE_SCREEN_REQUIRE_ALL_TRIM_DIRECTIONS:
        main["single_trim_pass"] &= main["all_trim_directions_same"].map(_safe_bool)

    alt_direction_cols = [f"{label}_direction_same" for label in available_alt_labels]
    alt_fdr_cols = [f"{label}_fdr_global_pass" for label in available_alt_labels]
    main["n_alternative_learners_available"] = len(available_alt_labels)
    main["n_alternative_directions_same"] = (
        main[alt_direction_cols].fillna(False).astype(bool).sum(axis=1)
        if alt_direction_cols else 0
    )
    main["n_alternative_fdr_pass"] = (
        main[alt_fdr_cols].apply(lambda column: column.map(_safe_bool)).sum(axis=1)
        if alt_fdr_cols else 0
    )
    main["single_alternative_direction_pass"] = (
        True if not alt_direction_cols else
        main["n_alternative_directions_same"].eq(len(alt_direction_cols))
    )
    main["single_alternative_fdr_pass"] = (
        True if not alt_fdr_cols else
        main["n_alternative_fdr_pass"].eq(len(alt_fdr_cols))
    )

    components = [
        "main_global_fdr_pass", "single_fold_pass", "single_team_pass",
        "single_placebo_pass", "single_trim_pass",
    ]
    if SINGLE_SCREEN_REQUIRE_NO_SEVERE_OVERLAP:
        components.append("single_overlap_pass")
    if SINGLE_SCREEN_REQUIRE_ALL_AVAILABLE_ALT_DIRECTIONS:
        components.append("single_alternative_direction_pass")
    if SINGLE_SCREEN_REQUIRE_ALL_AVAILABLE_ALT_FDR:
        components.append("single_alternative_fdr_pass")

    main["single_stage_pass"] = True
    for column in components:
        main["single_stage_pass"] &= main[column].fillna(False).astype(bool)

    reason_labels = OrderedDict([
        ("main_global_fdr_pass", "main_global_FDR"),
        ("single_overlap_pass", "overlap"),
        ("single_fold_pass", "fold_stability"),
        ("single_team_pass", "team_LOO_stability"),
        ("single_placebo_pass", "placebo"),
        ("single_trim_pass", "trim_robustness"),
        ("single_alternative_direction_pass", "alternative_direction"),
        ("single_alternative_fdr_pass", "alternative_FDR"),
    ])

    def failure_reason(row: pd.Series) -> str:
        failed = [
            label for column, label in reason_labels.items()
            if column in row.index and not _safe_bool(row.get(column), False)
        ]
        return "PASS" if not failed else "FAIL::" + " | ".join(failed)

    main["single_selection_reason"] = main.apply(failure_reason, axis=1)
    main["abs_single_theta"] = pd.to_numeric(
        main["theta_per_1sd"], errors="coerce"
    ).abs()
    passed = main.loc[main["single_stage_pass"]].copy()
    return main.sort_values(keys).reset_index(drop=True), passed.sort_values(keys).reset_index(drop=True)


def _choose_family_anchor(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["q_rank"] = pd.to_numeric(ranked["q_global"], errors="coerce").fillna(np.inf)
    ranked["rv_rank"] = pd.to_numeric(
        ranked.get("robustness_value_alpha05", np.nan), errors="coerce"
    ).fillna(-np.inf)
    ranked["effect_rank"] = pd.to_numeric(
        ranked["theta_per_1sd"], errors="coerce"
    ).abs().fillna(-np.inf)
    ranked = ranked.sort_values(
        ["effect_rank", "q_rank", "rv_rank", "treatment"],
        ascending=[False, True, False, True],
    )
    return ranked.iloc[0]


def build_screened_rotating_interaction_specs(
    cluster,
    stage: str,
    available_kpis: Sequence[str],
    passed_single: pd.DataFrame,
):
    """Create one target mapping for every strictly screened single-KPI result.

    The best screened KPI in each of the three families is the companion anchor.
    A target replaces the anchor in its own family.  Duplicate three-KPI sets are
    fitted once and mapped to all target KPIs represented by the model.
    """
    available = set(available_kpis)
    candidates = passed_single.loc[
        (passed_single["cluster"] == cluster)
        & (passed_single["stage"].astype(str) == str(stage))
    ].copy()
    candidates = candidates.loc[candidates["treatment"].isin(available)].copy()

    required_families = ["local_advantage", "distance_centroid", "structure_shape"]
    anchors = OrderedDict()
    anchor_audit = []
    missing_families = []
    for family in required_families:
        family_rows = candidates.loc[candidates["joint_family"] == family].copy()
        if family_rows.empty:
            missing_families.append(family)
            continue
        winner = _choose_family_anchor(family_rows)
        anchors[family] = winner["treatment"]
        anchor_audit.append({
            "cluster": cluster,
            "stage": stage,
            "family": family,
            "selected_anchor": winner["treatment"],
            "single_theta_per_1sd": winner.get("theta_per_1sd", np.nan),
            "single_q_global": winner.get("q_global", np.nan),
            "single_RV_alpha05": winner.get("robustness_value_alpha05", np.nan),
            "selection_rule": ANCHOR_SELECTION_RULE,
            "n_screened_candidates_in_family": len(family_rows),
            "locked_from_single_results_before_joint_run": True,
        })

    target_map = []
    if missing_families:
        for _, row in candidates.iterrows():
            target_map.append({
                "cluster": cluster,
                "stage": stage,
                "target": row["treatment"],
                "target_family": row["joint_family"],
                "joint_spec": "",
                "joint_kpis": "",
                "target_members_in_model": "",
                "status": "SKIP_missing_family_anchor",
                "missing_family_anchors": " | ".join(missing_families),
                "single_stage_pass": True,
                "single_theta_per_1sd": row.get("theta_per_1sd", np.nan),
                "single_q_global": row.get("q_global", np.nan),
                "single_RV_alpha05": row.get("robustness_value_alpha05", np.nan),
            })
        return [], target_map, anchor_audit

    model_by_selected = OrderedDict()
    for _, row in candidates.sort_values(["joint_family", "treatment"]).iterrows():
        target = row["treatment"]
        family = row["joint_family"]
        selected_by_family = OrderedDict((fam, anchors[fam]) for fam in required_families)
        selected_by_family[family] = target
        selected = tuple(selected_by_family[fam] for fam in required_families)
        missing = [name for name in selected if name not in available]
        if missing:
            target_map.append({
                "cluster": cluster,
                "stage": stage,
                "target": target,
                "target_family": family,
                "joint_spec": "",
                "joint_kpis": " | ".join(selected),
                "target_members_in_model": "",
                "status": "SKIP_missing_KPI",
                "missing_kpis": " | ".join(missing),
                "single_stage_pass": True,
                "single_theta_per_1sd": row.get("theta_per_1sd", np.nan),
                "single_q_global": row.get("q_global", np.nan),
                "single_RV_alpha05": row.get("robustness_value_alpha05", np.nan),
            })
            continue

        if selected not in model_by_selected:
            stage_tag = "Eprime" if stage == "E'" else "L"
            token = "__".join(
                re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
                for name in selected
            )
            model_by_selected[selected] = {
                "joint_spec": f"RotateInteraction_C{int(cluster)}_{stage_tag}__{token}",
                "selected": list(selected),
                "target_members": [],
            }
        model = model_by_selected[selected]
        model["target_members"].append(target)
        target_map.append({
            "cluster": cluster,
            "stage": stage,
            "target": target,
            "target_family": family,
            "joint_spec": model["joint_spec"],
            "joint_kpis": " | ".join(selected),
            "local_anchor": anchors["local_advantage"],
            "distance_anchor": anchors["distance_centroid"],
            "structure_anchor": anchors["structure_shape"],
            "configured_family_anchor": anchors[family],
            "status": "RUN",
            "missing_kpis": "",
            "missing_family_anchors": "",
            "single_stage_pass": True,
            "single_theta_per_1sd": row.get("theta_per_1sd", np.nan),
            "single_q_global": row.get("q_global", np.nan),
            "single_RV_alpha05": row.get("robustness_value_alpha05", np.nan),
            "single_selection_reason": row.get("single_selection_reason", "PASS"),
        })

    for model in model_by_selected.values():
        members = " | ".join(model["target_members"])
        for row in target_map:
            if row.get("joint_spec") == model["joint_spec"]:
                row["target_members_in_model"] = members
    return list(model_by_selected.values()), target_map, anchor_audit


def _conditional_target_summary(conditional: pd.DataFrame) -> pd.DataFrame:
    if conditional is None or conditional.empty:
        return pd.DataFrame()
    frame = conditional.copy()
    frame = frame.loc[frame["learner_set"] == "RF_main"].copy()
    rows = []
    for keys, group in frame.groupby(
        ["cluster", "stage", "joint_spec", "main_kpi"], dropna=False
    ):
        effects = pd.to_numeric(group["estimate"], errors="coerce")
        qvals = pd.to_numeric(group.get("q_within_target_grid", np.nan), errors="coerce")
        pos_sig = bool(((effects > 0) & (qvals <= FDR_ALPHA)).any())
        neg_sig = bool(((effects < 0) & (qvals <= FDR_ALPHA)).any())
        rows.append({
            "cluster": keys[0],
            "stage": keys[1],
            "joint_spec": keys[2],
            "target": keys[3],
            "conditional_min_effect": float(effects.min()),
            "conditional_max_effect": float(effects.max()),
            "conditional_sign_change": bool((effects.min() < 0) and (effects.max() > 0)),
            "significant_positive_condition": pos_sig,
            "significant_negative_condition": neg_sig,
            "robust_conditional_sign_change": bool(pos_sig and neg_sig),
            "n_conditional_grid_points": int(effects.notna().sum()),
            "n_conditional_FDR_pass": int((qvals <= FDR_ALPHA).sum()),
        })
    return pd.DataFrame(rows)


def build_rotating_target_ame_report(
    target_map: pd.DataFrame,
    all_model_summary: pd.DataFrame,
    conditional: pd.DataFrame,
) -> pd.DataFrame:
    """Extract exactly one joint AME result for every screened target KPI."""
    if target_map is None or target_map.empty:
        return pd.DataFrame()
    out = target_map.copy()
    if all_model_summary is None or all_model_summary.empty:
        out["target_result_status"] = "missing_all_joint_results"
        return out

    summary = all_model_summary.rename(columns={"main_kpi": "target"}).copy()
    keys = ["cluster", "stage", "joint_spec", "target"]
    keep = keys + [column for column in summary.columns if column not in keys]
    summary = summary[keep].drop_duplicates(keys)
    out = out.merge(summary, on=keys, how="left", suffixes=("_single_map", ""))

    cond_summary = _conditional_target_summary(conditional)
    if not cond_summary.empty:
        out = out.merge(cond_summary, on=keys, how="left")

    valid = out["status"].eq("RUN") & pd.to_numeric(
        out.get("p_value", np.nan), errors="coerce"
    ).notna()
    out["target_q_global"] = np.nan
    if valid.any():
        out.loc[valid, "target_q_global"] = bh_fdr(
            pd.to_numeric(out.loc[valid, "p_value"], errors="coerce").to_numpy()
        )
    out["target_q_within_cluster_stage"] = np.nan
    for _, index in out.loc[valid].groupby(["cluster", "stage"]).groups.items():
        out.loc[index, "target_q_within_cluster_stage"] = bh_fdr(
            pd.to_numeric(out.loc[index, "p_value"], errors="coerce").to_numpy()
        )

    single_theta = pd.to_numeric(out.get("single_theta_per_1sd", np.nan), errors="coerce")
    joint_theta = pd.to_numeric(out.get("estimate", np.nan), errors="coerce")
    out["single_to_joint_direction_same"] = np.where(
        single_theta.notna() & joint_theta.notna(),
        np.sign(single_theta) == np.sign(joint_theta),
        np.nan,
    )
    out["single_to_joint_abs_change"] = (joint_theta - single_theta).abs()

    confirmed = out.get("evidence_classification", "").astype(str).eq(
        "CONFIRMED_average_multivariable_effect"
    )
    contextual = out.get("evidence_classification", "").astype(str).eq(
        "CONTEXT_DEPENDENT_interaction_evidence"
    )
    target_fdr = pd.to_numeric(out["target_q_global"], errors="coerce") <= FDR_ALPHA
    out["eligible_as_average_policy_lever"] = confirmed & target_fdr
    out["eligible_for_interaction_aware_optimizer"] = (
        out["single_stage_pass"].map(_safe_bool)
        & (confirmed | contextual)
    )

    def policy_class(row: pd.Series) -> str:
        if not _safe_bool(row.get("eligible_for_interaction_aware_optimizer"), False):
            return "DO_NOT_USE_FOR_POLICY_OPTIMIZATION"
        if _safe_bool(row.get("robust_conditional_sign_change"), False):
            return "CONDITIONAL_DIRECTION_ONLY"
        direction_same = row.get("single_to_joint_direction_same")
        if pd.notna(direction_same) and not _safe_bool(direction_same, False):
            return "CONDITIONAL_DIRECTION_ONLY"
        estimate = pd.to_numeric(pd.Series([row.get("estimate", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(estimate):
            return "DO_NOT_USE_FOR_POLICY_OPTIMIZATION"
        return "FIXED_POSITIVE_DIRECTION" if estimate > 0 else "FIXED_NEGATIVE_DIRECTION"

    out["policy_constraint_class"] = out.apply(policy_class, axis=1)
    out["target_result_status"] = np.where(
        pd.to_numeric(out.get("estimate", np.nan), errors="coerce").notna(),
        "done", "missing_joint_target_result"
    )
    return out.sort_values(["cluster", "stage", "target_family", "target"]).reset_index(drop=True)

def fixed_interaction_main():
    validate_primary_rf_alignment()
    total_start = time.perf_counter()
    runtime_log: List[TimerRecord] = []
    run_log = []
    anchor_rows = []
    data_audit_rows = []
    fold_audit_frames = []
    calibration_frames = []
    standardization_frames = []
    y_performance_rows = []
    d_performance_rows = []
    overlap_rows_all = []
    support_bin_rows_all = []
    raw_main_corr_frames = []
    raw_main_vif_frames = []
    term_corr_frames = []
    term_vif_frames = []
    term_frames = []
    ame_frames = []
    conditional_frames = []
    wald_frames = []
    placebo_frames = []
    fold_term_long_frames = []
    fold_term_summary_frames = []
    team_term_long_frames = []
    team_term_summary_frames = []
    fold_ame_long_frames = []
    fold_ame_summary_frames = []
    team_ame_long_frames = []
    team_ame_summary_frames = []

    print("=" * 110)
    print("PASS · FIXED REPRESENTATIVE-KPI JOINT INTERACTION DML")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {INTERACTION_OUTPUT_XLSX}")
    print("=" * 110)

    with StepTimer(runtime_log, "ALL", "ALL", "read_and_prepare"):
        df = read_input_table(INPUT_PATH, sheet_name=SHEET_NAME)
        df.columns = [str(column).strip() for column in df.columns]
        if HOME_AWAY_COL not in df.columns:
            for alternative in ["move_team_home_away", "team_home_away", "home_away"]:
                if alternative in df.columns:
                    df[HOME_AWAY_COL] = df[alternative]
                    break
        check_required_columns(df)
        df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

        stage_detectors = {"L": is_L_col, "E'": is_Eprime_col}
        for active_stage in ACTIVE_STAGES:
            detector = stage_detectors[active_stage]
            if not [
                column for column in df.columns
                if detector(column) and not is_att_kpi(column)
            ]:
                raise ValueError(
                    f"No usable Def({active_stage}) KPI columns were detected."
                )

    clusters = sorted(df[CLUSTER_COL].dropna().unique())
    for cluster in clusters:
        sub = df.loc[df[CLUSTER_COL] == cluster].copy()
        rows_cluster = len(sub)
        matches_cluster = sub[MATCH_COL].nunique(dropna=True)
        with StepTimer(runtime_log, cluster, "ALL", "identify_KPIs", rows_cluster, matches_cluster):
            inventory = build_inventory(sub)

        for stage in ACTIVE_STAGES:
            key = (int(cluster), str(stage))
            if key not in INTERACTION_ANCHOR_BY_CLUSTER_STAGE:
                run_log.append({"cluster": cluster, "stage": stage, "status": "skip", "reason": "anchor configuration missing"})
                continue
            selected = list(INTERACTION_ANCHOR_BY_CLUSTER_STAGE[key])
            joint_spec = f"FixedInteraction_C{int(cluster)}_{'Eprime' if stage == "E'" else 'L'}"
            available = set(inventory["D_by_stage"].get(stage, []))
            missing = [name for name in selected if name not in available]

            for family, name in zip(["local_advantage", "distance_centroid", "structure_shape"], selected):
                anchor_rows.append({
                    "cluster": cluster,
                    "stage": stage,
                    "family": family,
                    "selected_anchor": name,
                    "selection_rule": (
                        "Adv_5 fixed for local tactical meaning; "
                        "distance uses Avg_1 -> Avg_3 -> Avg_5 stability fallback; "
                        "Area_Def fixed for interpretable structural coverage"
                    ),
                    "distance_fallback_order": " -> ".join(DISTANCE_FALLBACK_ORDER),
                    "specific_reason": (
                        "Avg_1_Def(E') failed the prespecified trim-stability screen; Avg_3_Def(E') was the next stable candidate"
                        if key == (2, "E'") and family == "distance_centroid"
                        else "first prespecified representative passed the single-KPI stability screen"
                    ),
                    "locked_before_interaction_run": True,
                })

            if missing:
                run_log.append({
                    "cluster": cluster, "stage": stage, "joint_spec": joint_spec,
                    "status": "skip", "reason": f"missing KPI: {missing}",
                })
                continue

            spec_start = time.perf_counter()
            try:
                with StepTimer(runtime_log, cluster, stage, f"prepare::{joint_spec}", rows_cluster, matches_cluster):
                    block = build_target_dataset(sub, inventory, selected, stage)
                X_df = block["X_df"]
                D_df = block["D_df"]
                y = block["y"]
                groups = block["groups"]
                teams = block["teams"]
                n = len(y)
                n_matches = len(pd.unique(groups))
                n_positive = int(y.sum())
                n_negative = int(n - n_positive)
                if (
                    n < MIN_ROWS or n_matches < MIN_MATCHES
                    or min(n_positive, n_negative) < MIN_CLASS_COUNT
                    or X_df.shape[1] == 0
                ):
                    raise ValueError(
                        f"insufficient sample n={n}, matches={n_matches}, "
                        f"class_min={min(n_positive, n_negative)}, X={X_df.shape[1]}"
                    )

                X = X_df.to_numpy(dtype=float)
                D_raw = D_df[selected].to_numpy(dtype=float)
                term_names, _ = interaction_term_names(selected)

                for treatment in selected:
                    data_audit_rows.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "main_kpi": treatment,
                        "joint_main_kpis": " | ".join(selected),
                        "joint_terms": " | ".join(term_names),
                        "rows_cluster": rows_cluster,
                        "rows_used": n,
                        "rows_lost": rows_cluster - n,
                        "rows_lost_proportion": 1 - n / rows_cluster if rows_cluster else np.nan,
                        "matches": n_matches,
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "positive_rate": n_positive / n if n else np.nan,
                        "n_controls": X_df.shape[1],
                        "main_kpi_in_controls": treatment in X_df.columns,
                        "all_selected_in_controls": any(name in X_df.columns for name in selected),
                        "base_controls": " | ".join(block["base_controls"]),
                        "earlier_stage_attacking_controls": " | ".join(block["earlier_stage_attacking_controls"]),
                        "same_stage_attacking_controls": " | ".join(block["same_stage_attacking_controls"]),
                        "cross_stage_attacking_controls": " | ".join(block["cross_stage_attacking_controls"]),
                        "excluded_same_series_attacking_controls": " | ".join(block["excluded_same_series_attacking_controls"]),
                        "excluded_defensive_controls": " | ".join(block["excluded_defensive_controls"]),
                        "ignored_actual_E_controls": " | ".join(block.get("ignored_actual_E_controls", [])),
                        "usable_numeric_controls": " | ".join(block["usable_numeric_controls"]),
                        "categorical_controls": " | ".join(block["categorical_control_cols"]),
                        "estimand": (
                            "average and conditional effects of three representative defending mechanisms, "
                            "including all pairwise interactions, within observed tactical support"
                        ),
                    })

                raw_corr, raw_vif = joint_correlation_and_vif(
                    D_raw, selected, cluster, stage, joint_spec, "raw_main_KPIs"
                )
                raw_main_corr_frames.append(raw_corr)
                raw_main_vif_frames.append(raw_vif)

                splits = make_grouped_splits(
                    y, groups, N_SPLITS, cluster_seed(cluster, stage, 70000)
                )
                with StepTimer(runtime_log, cluster, stage, f"crossfit_RF_Ridge::{joint_spec}", n, n_matches, note="3 main + 3 interactions"):
                    main_cf = crossfit_interaction_main_and_baseline(
                        X, y, D_raw, groups, splits,
                        cluster_seed(cluster, stage, 71000), selected, RUN_BASELINE,
                    )

                fold_audit = main_cf["fold_audit"].copy()
                fold_audit.insert(0, "joint_spec", joint_spec)
                fold_audit.insert(0, "stage", stage)
                fold_audit.insert(0, "cluster", cluster)
                fold_audit_frames.append(fold_audit)
                standardization = main_cf["standardization_audit"].copy()
                standardization.insert(0, "joint_spec", joint_spec)
                standardization.insert(0, "stage", stage)
                standardization.insert(0, "cluster", cluster)
                standardization_frames.append(standardization)
                calibration = main_cf["calibration_audit"].copy()
                if not calibration.empty:
                    calibration.insert(0, "joint_spec", joint_spec)
                    calibration.insert(0, "stage", stage)
                    calibration.insert(0, "cluster", cluster)
                    calibration_frames.append(calibration)

                T_obs = main_cf["T_obs"]
                y_res = y - main_cf["yhat_main"]
                T_res = T_obs - main_cf["that_main"]

                for model_name, yhat in [
                    ("RF_raw", main_cf["yhat_main_raw"]),
                    ("RF_NestedPlatt", main_cf["yhat_main"]),
                ]:
                    row = outcome_performance_rows(cluster, stage, y, yhat, model_name)
                    row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    y_performance_rows.append(row)
                drows = treatment_performance_rows(
                    cluster, stage, T_obs, main_cf["that_main"], term_names,
                    "RF_multioutput_interaction_terms",
                )
                for row in drows:
                    row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                d_performance_rows.extend(drows)

                for row in overlap_rows(cluster, stage, T_obs, main_cf["that_main"], T_res, term_names):
                    row["joint_spec"] = joint_spec
                    overlap_rows_all.append(row)
                for j, term in enumerate(term_names):
                    rows = support_bin_rows(cluster, stage, T_obs[:, j], main_cf["that_main"][:, j], T_res[:, j], term, SUPPORT_BINS)
                    for row in rows:
                        row["joint_spec"] = joint_spec
                    support_bin_rows_all.extend(rows)

                term_corr, term_vif = joint_correlation_and_vif(
                    T_res, term_names, cluster, stage, joint_spec,
                    "OOF_residual_main_and_interaction_terms",
                )
                term_corr_frames.append(term_corr)
                term_vif_frames.append(term_vif)

                for trim_frac in TRIM_GRID:
                    terms, ames, cond, wald = fit_fixed_interaction_second_stage(
                        y_res, T_res, T_obs, term_names, selected, groups,
                        cluster, stage, joint_spec, "RF_main", trim_frac,
                        include_conditional=np.isclose(trim_frac, MAIN_TRIM_FRAC),
                    )
                    term_frames.append(terms)
                    ame_frames.append(ames)
                    if not cond.empty:
                        conditional_frames.append(cond)
                    wald_frames.append(wald)

                placebo = joint_placebo_within_match(
                    y_res, T_res, groups, term_names, cluster, stage, joint_spec,
                    PLACEBO_REPS, cluster_seed(cluster, stage, 79000),
                )
                if not placebo.empty:
                    placebo_frames.append(placebo)

                fold_term_long, fold_term_summary = joint_fold_stability(
                    y_res, T_res, main_cf["fold_id"], term_names,
                    cluster, stage, joint_spec,
                )
                if not fold_term_long.empty:
                    fold_term_long_frames.append(fold_term_long)
                    fold_term_summary_frames.append(fold_term_summary)
                fold_ame_long, fold_ame_summary = interaction_ame_fold_stability(
                    y_res, T_res, T_obs, main_cf["fold_id"], selected,
                    cluster, stage, joint_spec,
                )
                fold_ame_long_frames.append(fold_ame_long)
                fold_ame_summary_frames.append(fold_ame_summary)

                if RUN_TEAM_LOO:
                    team_term_long, team_term_summary = joint_team_loo_stability(
                        y_res, T_res, teams, term_names, cluster, stage, joint_spec,
                    )
                    if not team_term_long.empty:
                        team_term_long_frames.append(team_term_long)
                        team_term_summary_frames.append(team_term_summary)
                    team_ame_long, team_ame_summary = interaction_ame_team_loo_stability(
                        y_res, T_res, T_obs, teams, selected, cluster, stage, joint_spec,
                    )
                    team_ame_long_frames.append(team_ame_long)
                    team_ame_summary_frames.append(team_ame_summary)

                if RUN_BASELINE:
                    yhat = main_cf["yhat_base"]
                    that = main_cf["that_base"]
                    for model_name, pred in [
                        ("Logistic_raw", main_cf["yhat_base_raw"]),
                        ("Logistic_NestedPlatt", yhat),
                    ]:
                        row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(row)
                    drows = treatment_performance_rows(cluster, stage, T_obs, that, term_names, "Ridge_multioutput_interaction_terms")
                    for row in drows:
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    d_performance_rows.extend(drows)
                    terms, ames, _, wald = fit_fixed_interaction_second_stage(
                        y - yhat, T_obs - that, T_obs, term_names, selected, groups,
                        cluster, stage, joint_spec, "Ridge", MAIN_TRIM_FRAC,
                        include_conditional=False,
                    )
                    term_frames.append(terms)
                    ame_frames.append(ames)
                    wald_frames.append(wald)

                if RUN_HGB_ET_ROBUSTNESS:
                    with StepTimer(runtime_log, cluster, stage, f"crossfit_HGB_ET::{joint_spec}", n, n_matches):
                        hgb = crossfit_interaction_hgb_et(
                            X, y, D_raw, groups, splits,
                            cluster_seed(cluster, stage, 81000), selected,
                        )
                    hcal = hgb["calibration_audit"].copy()
                    if not hcal.empty:
                        hcal.insert(0, "joint_spec", joint_spec)
                        hcal.insert(0, "stage", stage)
                        hcal.insert(0, "cluster", cluster)
                        calibration_frames.append(hcal)
                    for model_name, pred in [("HGB_raw", hgb["yhat_raw"]), ("HGB_NestedPlatt", hgb["yhat"])]:
                        row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(row)
                    drows = treatment_performance_rows(cluster, stage, hgb["T_obs"], hgb["that"], term_names, "ExtraTrees_separate_interaction_terms")
                    for row in drows:
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    d_performance_rows.extend(drows)
                    terms, ames, _, wald = fit_fixed_interaction_second_stage(
                        y - hgb["yhat"], hgb["T_obs"] - hgb["that"], hgb["T_obs"],
                        term_names, selected, groups, cluster, stage, joint_spec,
                        "HGB_ET", MAIN_TRIM_FRAC, include_conditional=False,
                    )
                    term_frames.append(terms)
                    ame_frames.append(ames)
                    wald_frames.append(wald)

                if RUN_XGBOOST_ROBUSTNESS and HAS_XGBOOST:
                    with StepTimer(runtime_log, cluster, stage, f"crossfit_XGB::{joint_spec}", n, n_matches):
                        xgb = crossfit_interaction_xgb(
                            X, y, D_raw, groups, splits,
                            cluster_seed(cluster, stage, 91000), selected,
                        )
                    xcal = xgb["calibration_audit"].copy()
                    if not xcal.empty:
                        xcal.insert(0, "joint_spec", joint_spec)
                        xcal.insert(0, "stage", stage)
                        xcal.insert(0, "cluster", cluster)
                        calibration_frames.append(xcal)
                    for model_name, pred in [("XGB_raw", xgb["yhat_raw"]), ("XGB_NestedPlatt", xgb["yhat"])]:
                        row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(row)
                    drows = treatment_performance_rows(cluster, stage, xgb["T_obs"], xgb["that"], term_names, "XGB_separate_interaction_terms")
                    for row in drows:
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    d_performance_rows.extend(drows)
                    terms, ames, _, wald = fit_fixed_interaction_second_stage(
                        y - xgb["yhat"], xgb["T_obs"] - xgb["that"], xgb["T_obs"],
                        term_names, selected, groups, cluster, stage, joint_spec,
                        "XGBoost", MAIN_TRIM_FRAC, include_conditional=False,
                    )
                    term_frames.append(terms)
                    ame_frames.append(ames)
                    wald_frames.append(wald)

                run_log.append({
                    "cluster": cluster,
                    "stage": stage,
                    "joint_spec": joint_spec,
                    "joint_main_kpis": " | ".join(selected),
                    "status": "done",
                    "rows": n,
                    "matches": n_matches,
                    "n_controls": X.shape[1],
                    "elapsed_seconds": time.perf_counter() - spec_start,
                })

            except Exception as error:
                print(f"[ERROR] cluster={cluster} stage={stage} spec={joint_spec}: {error!r}")
                run_log.append({
                    "cluster": cluster, "stage": stage, "joint_spec": joint_spec,
                    "status": "error", "reason": repr(error),
                })

    concat = lambda frames: pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any(f is not None and not f.empty for f in frames) else pd.DataFrame()
    terms, ames, conditional, wald = add_fixed_interaction_fdr(
        concat(term_frames), concat(ame_frames), concat(conditional_frames), concat(wald_frames)
    )
    main_terms = terms.loc[
        (terms["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(terms["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not terms.empty else pd.DataFrame()
    main_ames = ames.loc[
        (ames["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(ames["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not ames.empty else pd.DataFrame()
    main_wald = wald.loc[
        (wald["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(wald["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not wald.empty else pd.DataFrame()

    overlap = pd.DataFrame(overlap_rows_all)
    support_bins = pd.DataFrame(support_bin_rows_all)
    residual_vif = concat(term_vif_frames)
    placebo = concat(placebo_frames)
    fold_ame_summary = concat(fold_ame_summary_frames)
    team_ame_summary = concat(team_ame_summary_frames)
    learner_ame_summary, learner_term_summary = build_interaction_learner_summary(ames, terms)
    trim_ame_summary = build_ame_trim_stability(ames)
    main_summary = build_interaction_main_summary(
        main_ames, main_terms, overlap, residual_vif, placebo,
        fold_ame_summary, team_ame_summary, learner_ame_summary, trim_ame_summary, main_wald,
    )

    overview_rows = []
    for (cluster, stage), group in main_summary.groupby(["cluster", "stage"], dropna=False) if not main_summary.empty else []:
        overview_rows.append({
            "cluster": cluster,
            "stage": stage,
            "joint_spec": group["joint_spec"].iloc[0],
            "representative_KPIs": " | ".join(group["main_kpi"].astype(str)),
            "n_confirmed_average_effects": int((group["evidence_classification"] == "CONFIRMED_average_multivariable_effect").sum()),
            "n_context_dependent": int((group["evidence_classification"] == "CONTEXT_DEPENDENT_interaction_evidence").sum()),
            "interaction_joint_q": pd.to_numeric(group.get("interaction_joint_q", np.nan), errors="coerce").dropna().iloc[0] if pd.to_numeric(group.get("interaction_joint_q", np.nan), errors="coerce").notna().any() else np.nan,
            "interaction_increment_r2": pd.to_numeric(group.get("interaction_increment_r2", np.nan), errors="coerce").dropna().iloc[0] if pd.to_numeric(group.get("interaction_increment_r2", np.nan), errors="coerce").notna().any() else np.nan,
        })
    overview = pd.DataFrame(overview_rows)

    checklist = pd.DataFrame([
        {"requirement": "Multivariable causal estimation", "evidence": "Three representative mechanisms estimated simultaneously", "sheet": "02_main_AME_report / 03_term_report", "status": "REPORTED" if not main_summary.empty else "MISSING"},
        {"requirement": "Non-additive KPI interactions", "evidence": "All three pairwise products included; joint interaction Wald test reported", "sheet": "03_term_report / 05_nested_Wald", "status": "REPORTED"},
        {"requirement": "Average and conditional effects", "evidence": "AME plus -1/0/+1 SD moderator grids", "sheet": "02_main_AME_report / 04_conditional_effects", "status": "REPORTED"},
        {"requirement": "No scaling leakage", "evidence": "KPI means/SDs learned only from outer-training matches", "sheet": "25_standardization_audit", "status": "PASS"},
        {"requirement": "Match-grouped cross-fitting", "evidence": "Five folds with disjoint matches", "sheet": "24_fold_audit", "status": "PASS"},
        {"requirement": "KPI multicollinearity", "evidence": "Raw main-KPI and OOF residual-term correlations/VIF plus condition number", "sheet": "12-15", "status": "REPORTED"},
        {"requirement": "Overlap/common support", "evidence": "OOF treatment R2, residual SD ratio and support bins for all six terms", "sheet": "10_overlap / 16_support_bins", "status": "REPORTED"},
        {"requirement": "Trim sensitivity", "evidence": "0%, 1%, 2%, 5% RF estimates", "sheet": "06_AME_all_trims / 07_terms_all_trims", "status": "REPORTED"},
        {"requirement": "Alternative nuisance learners", "evidence": "RF, Logistic/Ridge, HGB/ET, optional XGB", "sheet": "08-09", "status": "REPORTED"},
        {"requirement": "Unobserved-confounding sensitivity", "evidence": "Partial R2 and robustness values for AMEs and coefficients", "sheet": "02-03", "status": "REPORTED"},
        {"requirement": "Placebo and stability", "evidence": "Within-match placebo, fold and leave-one-team-out checks", "sheet": "17-23", "status": "REPORTED"},
        {"requirement": "External validation", "evidence": "Not available in the current data environment; explicitly separated from this script", "sheet": "28_manifest", "status": "NOT_RUN_no_external_data"},
    ])

    manifest = pd.DataFrame([{
        "input_path": INPUT_PATH,
        "output_path": INTERACTION_OUTPUT_XLSX,
        "rows_input_after_exclusion": len(df),
        "matches_input": df[MATCH_COL].nunique(dropna=True),
        "outcome": OUTCOME_COL,
        "analysis": "six fixed representative-KPI joint interaction DML models (3 clusters x L/E')",
        "main_analysis_relationship": "single-KPI DML remains the primary effect analysis; this script is the principal multivariable interaction validation",
        "anchor_plan": json.dumps({f"C{k[0]}_{k[1]}": v for k, v in INTERACTION_ANCHOR_BY_CLUSTER_STAGE.items()}, ensure_ascii=False),
        "distance_fallback_rule": "Avg_1_Def -> Avg_3_Def -> Avg_5_Def",
        "model_terms": "three fold-standardized main KPIs + all three pairwise products",
        "interaction_construction": "main KPIs standardized using outer-training matches only; products constructed before treatment nuisance residualization",
        "main_effect_interpretation": "coefficient at other standardized KPIs=0; AME is the primary multivariable KPI effect",
        "conditional_effect_grid": "both other mechanisms at -1, 0, +1 SD",
        "main_y_learner": "RandomForestClassifier + nested match-grouped Platt calibration",
        "main_D_learner": "multi-output RandomForestRegressor for six treatment basis terms",
        "alternative_learners": "Logistic/Ridge, HGB/ExtraTrees, optional XGBoost",
        "undersampling_sensitivity": (
            "RF outcome nuisance only; random majority undersampling inside each inner/outer training subset; "
            "inner validation and outer test rows preserve the original distribution; primary RF D|X residuals reused"
            if RUN_UNDERSAMPLING_SENSITIVITY else "not run"
        ),
        "undersampling_active_stages": " | ".join(UNDERSAMPLING_ACTIVE_STAGES),
        "undersampling_target_minority_to_majority_ratio": UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO,
        "undersampling_magnitude_change_flag": UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG,
        "undersampling_screening_policy": "not an admission gate; strict single-KPI primary-model screening remains unchanged",
        "crossfit": f"{N_SPLITS}-fold grouped by {MATCH_COL}",
        "main_trim": MAIN_TRIM_FRAC,
        "trim_grid": json.dumps(TRIM_GRID),
        "second_stage": "unit-weight WLS on six OOF residualized basis terms + HC3 SE",
        "FDR": "BH separately for 18 AMEs, 18 interactions/main coefficients by type, and six interaction-block Wald tests",
        "OVB": "partial R2 and robustness values reported; detailed observed-covariate benchmarking remains in the single-KPI workbook",
        "actual_E_policy": "actual (E) KPIs ignored; only L and E' analyzed",
        "external_validation": "not run because no external dataset is available",
        "xgboost_available": HAS_XGBOOST,
    }])

    runtime_frame = pd.DataFrame([record.__dict__ for record in runtime_log])
    run_log_frame = pd.DataFrame(run_log)
    tables = OrderedDict([
        ("00_reviewer_checklist", checklist),
        ("01_overview", overview),
        ("02_main_AME_report", main_summary),
        ("03_term_report", main_terms),
        ("04_conditional_effects", conditional),
        ("05_nested_Wald", main_wald),
        ("06_AME_all_trims", ames.loc[ames["learner_set"] == "RF_main"].copy() if not ames.empty else ames),
        ("07_terms_all_trims", terms.loc[terms["learner_set"] == "RF_main"].copy() if not terms.empty else terms),
        ("08_AME_learner_robust", learner_ame_summary),
        ("09_term_learner_robust", learner_term_summary),
        ("10_overlap", overlap),
        ("11_Y_performance", pd.DataFrame(y_performance_rows)),
        ("12_D_performance", pd.DataFrame(d_performance_rows)),
        ("13_raw_main_corr", concat(raw_main_corr_frames)),
        ("14_raw_main_VIF", concat(raw_main_vif_frames)),
        ("15_residual_term_corr", concat(term_corr_frames)),
        ("16_residual_term_VIF", residual_vif),
        ("17_support_bins", support_bins),
        ("18_placebo", placebo),
        ("19_fold_AME_summary", fold_ame_summary),
        ("20_fold_AME_long", concat(fold_ame_long_frames)),
        ("21_team_AME_summary", team_ame_summary),
        ("22_team_AME_long", concat(team_ame_long_frames)),
        ("23_fold_term_summary", concat(fold_term_summary_frames)),
        ("24_team_term_summary", concat(team_term_summary_frames)),
        ("25_fold_audit", concat(fold_audit_frames)),
        ("26_calibration_audit", concat(calibration_frames)),
        ("27_standardization_audit", concat(standardization_frames)),
        ("28_data_audit", pd.DataFrame(data_audit_rows)),
        ("29_anchor_plan", pd.DataFrame(anchor_rows)),
        ("30_manifest", manifest),
        ("31_run_log", run_log_frame),
        ("32_runtime", runtime_frame),
    ])

    output = Path(INTERACTION_OUTPUT_XLSX)
    output.parent.mkdir(parents=True, exist_ok=True)
    used = set()
    final_tables = OrderedDict()
    with pd.ExcelWriter(
        output, engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        for requested, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame([{"message": "No results produced for this table."}])
            sheet = safe_sheet_name(requested, used)
            table.to_excel(writer, index=False, sheet_name=sheet)
            final_tables[sheet] = table
        format_joint_workbook(writer, final_tables)

    elapsed = time.perf_counter() - total_start
    runtime_csv = output.with_name(output.stem + "_runtime.csv")
    pd.DataFrame([record.__dict__ for record in runtime_log]).to_csv(
        runtime_csv, index=False, encoding="utf-8-sig"
    )
    print("=" * 110)
    print(f"[DONE] {INTERACTION_OUTPUT_XLSX}")
    print(f"[RUNTIME] {format_seconds(elapsed)}")
    print(f"[RUNTIME CSV] {runtime_csv}")
    print("=" * 110)



def _conditional_summary_for_learner(
    conditional: pd.DataFrame,
    learner_set: str,
) -> pd.DataFrame:
    """Summarize conditional sign patterns for one joint-model learner set."""
    if conditional is None or conditional.empty:
        return pd.DataFrame()
    frame = conditional.loc[
        conditional["learner_set"].astype(str).eq(str(learner_set))
        & np.isclose(
            pd.to_numeric(conditional["trim_fraction_rule"], errors="coerce"),
            MAIN_TRIM_FRAC,
        )
    ].copy()
    rows = []
    for keys, group in frame.groupby(
        ["cluster", "stage", "joint_spec", "main_kpi"], dropna=False
    ):
        est = pd.to_numeric(group["estimate"], errors="coerce")
        q = pd.to_numeric(group.get("q_within_target_grid", np.nan), errors="coerce")
        significant = q <= FDR_ALPHA
        significant_positive = bool(((est > 0) & significant).any())
        significant_negative = bool(((est < 0) & significant).any())
        rows.append({
            "cluster": keys[0],
            "stage": keys[1],
            "joint_spec": keys[2],
            "target": keys[3],
            "conditional_min_effect": float(est.min()) if est.notna().any() else np.nan,
            "conditional_max_effect": float(est.max()) if est.notna().any() else np.nan,
            "n_conditional_grid_points": int(len(group)),
            "n_conditional_FDR_pass": int(significant.fillna(False).sum()),
            "robust_conditional_sign_change": (
                significant_positive and significant_negative
            ),
        })
    return pd.DataFrame(rows)


def build_undersampling_target_compare(
    target_map: pd.DataFrame,
    primary_target_summary: pd.DataFrame,
    undersampled_ames: pd.DataFrame,
    undersampled_conditional: pd.DataFrame,
    undersampled_wald: pd.DataFrame,
) -> pd.DataFrame:
    """Compare target-specific primary AMEs with training-fold undersampling AMEs."""
    if target_map is None or target_map.empty or undersampled_ames is None or undersampled_ames.empty:
        return pd.DataFrame()
    us = undersampled_ames.loc[
        undersampled_ames["learner_set"].eq("RF_undersampled")
        & np.isclose(
            pd.to_numeric(undersampled_ames["trim_fraction_rule"], errors="coerce"),
            MAIN_TRIM_FRAC,
        )
    ].copy().rename(columns={"main_kpi": "target"})
    keys = ["cluster", "stage", "joint_spec", "target"]
    keep = keys + [c for c in us.columns if c not in keys]
    out = target_map.merge(us[keep].drop_duplicates(keys), on=keys, how="left")

    valid = out["status"].eq("RUN") & pd.to_numeric(out.get("p_value"), errors="coerce").notna()
    out["undersampled_target_q_global"] = np.nan
    if valid.any():
        out.loc[valid, "undersampled_target_q_global"] = bh_fdr(
            pd.to_numeric(out.loc[valid, "p_value"], errors="coerce").to_numpy()
        )
    out["undersampled_target_q_within_cluster_stage"] = np.nan
    for _, idx in out.loc[valid].groupby(["cluster", "stage"]).groups.items():
        out.loc[idx, "undersampled_target_q_within_cluster_stage"] = bh_fdr(
            pd.to_numeric(out.loc[idx, "p_value"], errors="coerce").to_numpy()
        )

    cond = _conditional_summary_for_learner(
        undersampled_conditional, "RF_undersampled"
    )
    if not cond.empty:
        cond = cond.rename(columns={
            "conditional_min_effect": "undersampled_conditional_min_effect",
            "conditional_max_effect": "undersampled_conditional_max_effect",
            "n_conditional_grid_points": "undersampled_n_conditional_grid_points",
            "n_conditional_FDR_pass": "undersampled_n_conditional_FDR_pass",
            "robust_conditional_sign_change": "undersampled_robust_conditional_sign_change",
        })
        out = out.merge(cond, on=keys, how="left")

    if undersampled_wald is not None and not undersampled_wald.empty:
        uw = undersampled_wald.loc[
            undersampled_wald["learner_set"].eq("RF_undersampled")
            & undersampled_wald["wald_block"].eq("three_interactions_jointly")
            & np.isclose(
                pd.to_numeric(undersampled_wald["trim_fraction_rule"], errors="coerce"),
                MAIN_TRIM_FRAC,
            )
        ].copy()
        cols = ["cluster", "stage", "joint_spec", "wald_F", "p_value",
                "q_global_by_wald_block", "interaction_increment_r2"]
        out = out.merge(
            uw[cols].drop_duplicates(["cluster", "stage", "joint_spec"]).rename(columns={
                "wald_F": "undersampled_interaction_Wald_F",
                "p_value": "undersampled_interaction_p",
                "q_global_by_wald_block": "undersampled_interaction_q",
                "interaction_increment_r2": "undersampled_interaction_increment_r2",
            }),
            on=["cluster", "stage", "joint_spec"], how="left",
        )

    if primary_target_summary is not None and not primary_target_summary.empty:
        pcols = keys + [c for c in [
            "estimate", "se", "ci_low", "ci_high", "target_q_global",
            "robust_conditional_sign_change", "interaction_joint_q",
            "interaction_increment_r2", "policy_constraint_class",
            "eligible_for_interaction_aware_optimizer",
        ] if c in primary_target_summary.columns]
        primary = primary_target_summary[pcols].drop_duplicates(keys).rename(columns={
            "estimate": "primary_AME",
            "se": "primary_AME_se",
            "ci_low": "primary_AME_ci_low",
            "ci_high": "primary_AME_ci_high",
            "target_q_global": "primary_target_q_global",
            "robust_conditional_sign_change": "primary_robust_conditional_sign_change",
            "interaction_joint_q": "primary_interaction_q",
            "interaction_increment_r2": "primary_interaction_increment_r2",
            "policy_constraint_class": "primary_policy_constraint_class",
            "eligible_for_interaction_aware_optimizer": "primary_optimizer_eligible",
        })
        out = out.merge(primary, on=keys, how="left")

    out = out.rename(columns={
        "estimate": "undersampled_AME",
        "se": "undersampled_AME_se",
        "ci_low": "undersampled_AME_ci_low",
        "ci_high": "undersampled_AME_ci_high",
        "p_value": "undersampled_AME_p",
        "q_global": "undersampled_model_level_q_global",
    })
    primary = pd.to_numeric(out.get("primary_AME"), errors="coerce")
    sampled = pd.to_numeric(out.get("undersampled_AME"), errors="coerce")
    out["AME_direction_same_undersampling"] = np.where(
        primary.notna() & sampled.notna(), np.sign(primary) == np.sign(sampled), np.nan
    )
    out["AME_absolute_change_undersampling"] = (sampled - primary).abs()
    out["AME_relative_absolute_change_undersampling"] = (
        out["AME_absolute_change_undersampling"] / primary.abs().replace(0, np.nan)
    )
    primary_cond = out.get("primary_robust_conditional_sign_change", pd.Series(False, index=out.index)).map(_safe_bool)
    sampled_cond = out.get("undersampled_robust_conditional_sign_change", pd.Series(False, index=out.index)).map(_safe_bool)
    out["conditional_sign_change_same_undersampling"] = primary_cond == sampled_cond

    def classify(row: pd.Series) -> str:
        if not np.isfinite(pd.to_numeric(pd.Series([row.get("undersampled_AME")]), errors="coerce").iloc[0]):
            return "NOT_RUN_OR_MISSING"
        if not _safe_bool(row.get("AME_direction_same_undersampling"), False):
            return "DIRECTION_SENSITIVE"
        relative = pd.to_numeric(pd.Series([row.get("AME_relative_absolute_change_undersampling")]), errors="coerce").iloc[0]
        if np.isfinite(relative) and relative > UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG:
            return "MAGNITUDE_SENSITIVE"
        uq = pd.to_numeric(pd.Series([row.get("undersampled_target_q_global")]), errors="coerce").iloc[0]
        if not np.isfinite(uq) or uq > FDR_ALPHA:
            return "SIGNIFICANCE_SENSITIVE"
        if not _safe_bool(row.get("conditional_sign_change_same_undersampling"), True):
            return "CONDITIONAL_PATTERN_SENSITIVE"
        pq = pd.to_numeric(pd.Series([row.get("primary_interaction_q")]), errors="coerce").iloc[0]
        sq = pd.to_numeric(pd.Series([row.get("undersampled_interaction_q")]), errors="coerce").iloc[0]
        if np.isfinite(pq) and pq <= FDR_ALPHA and (not np.isfinite(sq) or sq > FDR_ALPHA):
            return "INTERACTION_BLOCK_SENSITIVE"
        return "ROBUST"

    out["imbalance_sensitivity_class"] = out.apply(classify, axis=1)
    out["imbalance_sensitivity_flag"] = out["imbalance_sensitivity_class"].ne("ROBUST")
    out["undersampling_target_ratio"] = UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
    out["optimizer_eligible_after_undersampling"] = (
        out.get("primary_optimizer_eligible", False).map(_safe_bool)
        & out["AME_direction_same_undersampling"].map(_safe_bool)
        & (pd.to_numeric(out["undersampled_target_q_global"], errors="coerce") <= FDR_ALPHA)
    )
    return out.sort_values(["cluster", "stage", "target_family", "target"]).reset_index(drop=True)


def build_undersampling_term_compare(
    primary_terms: pd.DataFrame,
    undersampled_terms: pd.DataFrame,
) -> pd.DataFrame:
    if primary_terms is None or primary_terms.empty or undersampled_terms is None or undersampled_terms.empty:
        return pd.DataFrame()
    keys = ["cluster", "stage", "joint_spec", "term", "effect_kind"]
    p = primary_terms[keys + ["estimate", "se", "ci_low", "ci_high", "p_value", "q_global_by_effect_kind"]].copy().rename(columns={
        "estimate": "primary_estimate", "se": "primary_se", "ci_low": "primary_ci_low",
        "ci_high": "primary_ci_high", "p_value": "primary_p",
        "q_global_by_effect_kind": "primary_q_global_by_effect_kind",
    })
    u = undersampled_terms.loc[
        undersampled_terms["learner_set"].eq("RF_undersampled")
        & np.isclose(pd.to_numeric(undersampled_terms["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ][keys + ["estimate", "se", "ci_low", "ci_high", "p_value", "q_global_by_effect_kind"]].copy().rename(columns={
        "estimate": "undersampled_estimate", "se": "undersampled_se",
        "ci_low": "undersampled_ci_low", "ci_high": "undersampled_ci_high",
        "p_value": "undersampled_p", "q_global_by_effect_kind": "undersampled_q_global_by_effect_kind",
    })
    out = p.merge(u, on=keys, how="outer")
    pe = pd.to_numeric(out["primary_estimate"], errors="coerce")
    ue = pd.to_numeric(out["undersampled_estimate"], errors="coerce")
    out["direction_same"] = np.where(pe.notna() & ue.notna(), np.sign(pe) == np.sign(ue), np.nan)
    out["absolute_change"] = (ue - pe).abs()
    out["relative_absolute_change"] = out["absolute_change"] / pe.abs().replace(0, np.nan)
    out["undersampling_target_ratio"] = UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
    return out.sort_values(keys).reset_index(drop=True)


def build_undersampling_wald_compare(
    primary_wald: pd.DataFrame,
    undersampled_wald: pd.DataFrame,
) -> pd.DataFrame:
    if primary_wald is None or primary_wald.empty or undersampled_wald is None or undersampled_wald.empty:
        return pd.DataFrame()
    keys = ["cluster", "stage", "joint_spec", "wald_block"]
    cols = keys + ["wald_F", "p_value", "q_global_by_wald_block", "full_second_stage_r2", "additive_second_stage_r2", "interaction_increment_r2"]
    p = primary_wald[cols].copy().rename(columns={c: f"primary_{c}" for c in cols if c not in keys})
    u = undersampled_wald.loc[
        undersampled_wald["learner_set"].eq("RF_undersampled")
        & np.isclose(pd.to_numeric(undersampled_wald["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ][cols].copy().rename(columns={c: f"undersampled_{c}" for c in cols if c not in keys})
    out = p.merge(u, on=keys, how="outer")
    out["undersampling_target_ratio"] = UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
    return out.sort_values(keys).reset_index(drop=True)


def all_passed_interaction_main():
    validate_primary_rf_alignment()
    total_start = time.perf_counter()
    runtime_log: List[TimerRecord] = []
    run_log = []
    anchor_rows = []
    selection_rows = []
    single_screening_audit = pd.DataFrame()
    passed_single = pd.DataFrame()
    data_audit_rows = []
    fold_audit_frames = []
    calibration_frames = []
    standardization_frames = []
    y_performance_rows = []
    d_performance_rows = []
    overlap_rows_all = []
    support_bin_rows_all = []
    raw_main_corr_frames = []
    raw_main_vif_frames = []
    term_corr_frames = []
    term_vif_frames = []
    term_frames = []
    ame_frames = []
    conditional_frames = []
    wald_frames = []
    placebo_frames = []
    fold_term_long_frames = []
    fold_term_summary_frames = []
    team_term_long_frames = []
    team_term_summary_frames = []
    fold_ame_long_frames = []
    fold_ame_summary_frames = []
    team_ame_long_frames = []
    team_ame_summary_frames = []
    undersample_term_frames = []
    undersample_ame_frames = []
    undersample_conditional_frames = []
    undersample_wald_frames = []
    undersample_y_performance_rows = []
    undersample_sampling_frames = []
    undersample_calibration_frames = []

    print("=" * 110)
    print("PASS · ALL SINGLE-STAGE-PASSED KPI ROTATING JOINT INTERACTION DML")
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {ALL_PASSED_INTERACTION_OUTPUT_XLSX}")
    print("=" * 110)

    with StepTimer(runtime_log, "ALL", "ALL", "read_and_prepare"):
        df = read_input_table(INPUT_PATH, sheet_name=SHEET_NAME)
        df.columns = [str(column).strip() for column in df.columns]
        if HOME_AWAY_COL not in df.columns:
            for alternative in ["move_team_home_away", "team_home_away", "home_away"]:
                if alternative in df.columns:
                    df[HOME_AWAY_COL] = df[alternative]
                    break
        check_required_columns(df)
        df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

        stage_detectors = {"L": is_L_col, "E'": is_Eprime_col}
        for active_stage in ACTIVE_STAGES:
            detector = stage_detectors[active_stage]
            if not [
                column for column in df.columns
                if detector(column) and not is_att_kpi(column)
            ]:
                raise ValueError(
                    f"No usable Def({active_stage}) KPI columns were detected."
                )

    with StepTimer(runtime_log, "ALL", "ALL", "load_single_KPI_screening"):
        single_screening_audit, passed_single = load_single_kpi_screening()
    if passed_single.empty:
        raise ValueError("No KPI passed the strict single-stage screen.")

    clusters = sorted(df[CLUSTER_COL].dropna().unique())
    for cluster in clusters:
        sub = df.loc[df[CLUSTER_COL] == cluster].copy()
        rows_cluster = len(sub)
        matches_cluster = sub[MATCH_COL].nunique(dropna=True)
        with StepTimer(runtime_log, cluster, "ALL", "identify_KPIs", rows_cluster, matches_cluster):
            inventory = build_inventory(sub)

        for stage in ACTIVE_STAGES:
            available = set(inventory["D_by_stage"].get(stage, []))
            stage_specs, stage_target_map, stage_anchor_rows = (
                build_screened_rotating_interaction_specs(
                    cluster, stage, available, passed_single
                )
            )
            selection_rows.extend(stage_target_map)
            anchor_rows.extend(stage_anchor_rows)

            if not stage_specs:
                run_log.append({
                    "cluster": cluster, "stage": stage, "status": "skip",
                    "reason": "no runnable rotating interaction specification",
                })
                continue

            for spec_index, spec_record in enumerate(stage_specs):
                joint_spec = spec_record["joint_spec"]
                selected = list(spec_record["selected"])
                target_members = list(spec_record["target_members"])
                missing = [name for name in selected if name not in available]
                if missing:
                    run_log.append({
                        "cluster": cluster, "stage": stage,
                        "joint_spec": joint_spec,
                        "target_members": " | ".join(target_members),
                        "status": "skip", "reason": f"missing KPI: {missing}",
                    })
                    continue

                spec_start = time.perf_counter()
                try:
                    with StepTimer(runtime_log, cluster, stage, f"prepare::{joint_spec}", rows_cluster, matches_cluster):
                        block = build_target_dataset(sub, inventory, selected, stage)
                    X_df = block["X_df"]
                    D_df = block["D_df"]
                    y = block["y"]
                    groups = block["groups"]
                    teams = block["teams"]
                    n = len(y)
                    n_matches = len(pd.unique(groups))
                    n_positive = int(y.sum())
                    n_negative = int(n - n_positive)
                    if (
                        n < MIN_ROWS or n_matches < MIN_MATCHES
                        or min(n_positive, n_negative) < MIN_CLASS_COUNT
                        or X_df.shape[1] == 0
                    ):
                        raise ValueError(
                            f"insufficient sample n={n}, matches={n_matches}, "
                            f"class_min={min(n_positive, n_negative)}, X={X_df.shape[1]}"
                        )

                    X = X_df.to_numpy(dtype=float)
                    D_raw = D_df[selected].to_numpy(dtype=float)
                    term_names, _ = interaction_term_names(selected)

                    for treatment in selected:
                        data_audit_rows.append({
                            "cluster": cluster,
                            "stage": stage,
                            "joint_spec": joint_spec,
                            "main_kpi": treatment,
                            "joint_main_kpis": " | ".join(selected),
                            "joint_terms": " | ".join(term_names),
                            "target_members": " | ".join(target_members),
                            "rows_cluster": rows_cluster,
                            "rows_used": n,
                            "rows_lost": rows_cluster - n,
                            "rows_lost_proportion": 1 - n / rows_cluster if rows_cluster else np.nan,
                            "matches": n_matches,
                            "n_positive": n_positive,
                            "n_negative": n_negative,
                            "positive_rate": n_positive / n if n else np.nan,
                            "n_controls": X_df.shape[1],
                            "main_kpi_in_controls": treatment in X_df.columns,
                            "all_selected_in_controls": any(name in X_df.columns for name in selected),
                            "base_controls": " | ".join(block["base_controls"]),
                            "earlier_stage_attacking_controls": " | ".join(block["earlier_stage_attacking_controls"]),
                            "same_stage_attacking_controls": " | ".join(block["same_stage_attacking_controls"]),
                            "cross_stage_attacking_controls": " | ".join(block["cross_stage_attacking_controls"]),
                            "excluded_same_series_attacking_controls": " | ".join(block["excluded_same_series_attacking_controls"]),
                            "excluded_defensive_controls": " | ".join(block["excluded_defensive_controls"]),
                            "ignored_actual_E_controls": " | ".join(block.get("ignored_actual_E_controls", [])),
                            "usable_numeric_controls": " | ".join(block["usable_numeric_controls"]),
                            "categorical_controls": " | ".join(block["categorical_control_cols"]),
                            "estimand": (
                                "target-rotating average and conditional effects of three defending mechanisms, "
                                "including all pairwise interactions, within observed tactical support"
                            ),
                        })

                    raw_corr, raw_vif = joint_correlation_and_vif(
                        D_raw, selected, cluster, stage, joint_spec, "raw_main_KPIs"
                    )
                    raw_main_corr_frames.append(raw_corr)
                    raw_main_vif_frames.append(raw_vif)

                    splits = make_grouped_splits(
                        y, groups, N_SPLITS, cluster_seed(cluster, stage, 70000 + spec_index * 101)
                    )
                    with StepTimer(runtime_log, cluster, stage, f"crossfit_RF_Ridge::{joint_spec}", n, n_matches, note="3 main + 3 interactions"):
                        main_cf = crossfit_interaction_main_and_baseline(
                            X, y, D_raw, groups, splits,
                            cluster_seed(cluster, stage, 71000 + spec_index * 103), selected, RUN_BASELINE,
                        )

                    fold_audit = main_cf["fold_audit"].copy()
                    fold_audit.insert(0, "joint_spec", joint_spec)
                    fold_audit.insert(0, "stage", stage)
                    fold_audit.insert(0, "cluster", cluster)
                    fold_audit_frames.append(fold_audit)
                    standardization = main_cf["standardization_audit"].copy()
                    standardization.insert(0, "joint_spec", joint_spec)
                    standardization.insert(0, "stage", stage)
                    standardization.insert(0, "cluster", cluster)
                    standardization_frames.append(standardization)
                    calibration = main_cf["calibration_audit"].copy()
                    if not calibration.empty:
                        calibration.insert(0, "joint_spec", joint_spec)
                        calibration.insert(0, "stage", stage)
                        calibration.insert(0, "cluster", cluster)
                        calibration_frames.append(calibration)

                    T_obs = main_cf["T_obs"]
                    y_res = y - main_cf["yhat_main"]
                    T_res = T_obs - main_cf["that_main"]

                    for model_name, yhat in [
                        ("RF_raw", main_cf["yhat_main_raw"]),
                        ("RF_NestedPlatt", main_cf["yhat_main"]),
                    ]:
                        row = outcome_performance_rows(cluster, stage, y, yhat, model_name)
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        y_performance_rows.append(row)
                    drows = treatment_performance_rows(
                        cluster, stage, T_obs, main_cf["that_main"], term_names,
                        "RF_multioutput_interaction_terms",
                    )
                    for row in drows:
                        row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                    d_performance_rows.extend(drows)

                    for row in overlap_rows(cluster, stage, T_obs, main_cf["that_main"], T_res, term_names):
                        row["joint_spec"] = joint_spec
                        overlap_rows_all.append(row)
                    for j, term in enumerate(term_names):
                        rows = support_bin_rows(cluster, stage, T_obs[:, j], main_cf["that_main"][:, j], T_res[:, j], term, SUPPORT_BINS)
                        for row in rows:
                            row["joint_spec"] = joint_spec
                        support_bin_rows_all.extend(rows)

                    term_corr, term_vif = joint_correlation_and_vif(
                        T_res, term_names, cluster, stage, joint_spec,
                        "OOF_residual_main_and_interaction_terms",
                    )
                    term_corr_frames.append(term_corr)
                    term_vif_frames.append(term_vif)

                    for trim_frac in TRIM_GRID:
                        terms, ames, cond, wald = fit_fixed_interaction_second_stage(
                            y_res, T_res, T_obs, term_names, selected, groups,
                            cluster, stage, joint_spec, "RF_main", trim_frac,
                            include_conditional=np.isclose(trim_frac, MAIN_TRIM_FRAC),
                        )
                        term_frames.append(terms)
                        ame_frames.append(ames)
                        if not cond.empty:
                            conditional_frames.append(cond)
                        wald_frames.append(wald)

                    if (
                        RUN_UNDERSAMPLING_SENSITIVITY
                        and stage in UNDERSAMPLING_ACTIVE_STAGES
                    ):
                        with StepTimer(
                            runtime_log, cluster, stage,
                            f"crossfit_RF_undersampled::{joint_spec}", n, n_matches,
                            note=(
                                "RF Y|X majority undersampling inside inner/outer training folds; "
                                f"target minority:majority={UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO:.3f}; "
                                "primary RF D|X residuals reused"
                            ),
                        ):
                            us_cf = crossfit_undersampled_rf_y(
                                X, y, groups, splits,
                                cluster_seed(cluster, stage, 76000 + spec_index * 127),
                            )
                        us_cal = us_cf["calibration_audit"].copy()
                        if not us_cal.empty:
                            us_cal.insert(0, "joint_spec", joint_spec)
                            us_cal.insert(0, "stage", stage)
                            us_cal.insert(0, "cluster", cluster)
                            undersample_calibration_frames.append(us_cal)
                        us_sampling = us_cf["sampling_audit"].copy()
                        if not us_sampling.empty:
                            us_sampling.insert(0, "joint_spec", joint_spec)
                            us_sampling.insert(0, "stage", stage)
                            us_sampling.insert(0, "cluster", cluster)
                            undersample_sampling_frames.append(us_sampling)

                        for model_name, pred in [
                            ("RF_training_fold_undersampled_raw", us_cf["yhat_raw"]),
                            ("RF_training_fold_undersampled_NestedPlatt", us_cf["yhat"]),
                        ]:
                            row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                            row.update({
                                "joint_spec": joint_spec,
                                "n_controls": X.shape[1],
                                "undersampling_target_ratio": UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO,
                            })
                            undersample_y_performance_rows.append(row)

                        # Reuse the primary T_obs and D|X residuals so the sensitivity
                        # isolates only the treatment of outcome-class imbalance.
                        us_y_res = y - us_cf["yhat"]
                        us_terms, us_ames, us_cond, us_wald = fit_fixed_interaction_second_stage(
                            us_y_res, T_res, T_obs, term_names, selected, groups,
                            cluster, stage, joint_spec, "RF_undersampled", MAIN_TRIM_FRAC,
                            include_conditional=True,
                        )
                        if not us_terms.empty:
                            undersample_term_frames.append(us_terms)
                        if not us_ames.empty:
                            undersample_ame_frames.append(us_ames)
                        if not us_cond.empty:
                            undersample_conditional_frames.append(us_cond)
                        if not us_wald.empty:
                            undersample_wald_frames.append(us_wald)

                    placebo = joint_placebo_within_match(
                        y_res, T_res, groups, term_names, cluster, stage, joint_spec,
                        PLACEBO_REPS, cluster_seed(cluster, stage, 79000 + spec_index * 107),
                    )
                    if not placebo.empty:
                        placebo_frames.append(placebo)

                    fold_term_long, fold_term_summary = joint_fold_stability(
                        y_res, T_res, main_cf["fold_id"], term_names,
                        cluster, stage, joint_spec,
                    )
                    if not fold_term_long.empty:
                        fold_term_long_frames.append(fold_term_long)
                        fold_term_summary_frames.append(fold_term_summary)
                    fold_ame_long, fold_ame_summary = interaction_ame_fold_stability(
                        y_res, T_res, T_obs, main_cf["fold_id"], selected,
                        cluster, stage, joint_spec,
                    )
                    fold_ame_long_frames.append(fold_ame_long)
                    fold_ame_summary_frames.append(fold_ame_summary)

                    if RUN_TEAM_LOO:
                        team_term_long, team_term_summary = joint_team_loo_stability(
                            y_res, T_res, teams, term_names, cluster, stage, joint_spec,
                        )
                        if not team_term_long.empty:
                            team_term_long_frames.append(team_term_long)
                            team_term_summary_frames.append(team_term_summary)
                        team_ame_long, team_ame_summary = interaction_ame_team_loo_stability(
                            y_res, T_res, T_obs, teams, selected, cluster, stage, joint_spec,
                        )
                        team_ame_long_frames.append(team_ame_long)
                        team_ame_summary_frames.append(team_ame_summary)

                    if RUN_BASELINE:
                        yhat = main_cf["yhat_base"]
                        that = main_cf["that_base"]
                        for model_name, pred in [
                            ("Logistic_raw", main_cf["yhat_base_raw"]),
                            ("Logistic_NestedPlatt", yhat),
                        ]:
                            row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                            y_performance_rows.append(row)
                        drows = treatment_performance_rows(cluster, stage, T_obs, that, term_names, "Ridge_multioutput_interaction_terms")
                        for row in drows:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows)
                        terms, ames, _, wald = fit_fixed_interaction_second_stage(
                            y - yhat, T_obs - that, T_obs, term_names, selected, groups,
                            cluster, stage, joint_spec, "Ridge", MAIN_TRIM_FRAC,
                            include_conditional=False,
                        )
                        term_frames.append(terms)
                        ame_frames.append(ames)
                        wald_frames.append(wald)

                    if RUN_HGB_ET_ROBUSTNESS:
                        with StepTimer(runtime_log, cluster, stage, f"crossfit_HGB_ET::{joint_spec}", n, n_matches):
                            hgb = crossfit_interaction_hgb_et(
                                X, y, D_raw, groups, splits,
                                cluster_seed(cluster, stage, 81000 + spec_index * 109), selected,
                            )
                        hcal = hgb["calibration_audit"].copy()
                        if not hcal.empty:
                            hcal.insert(0, "joint_spec", joint_spec)
                            hcal.insert(0, "stage", stage)
                            hcal.insert(0, "cluster", cluster)
                            calibration_frames.append(hcal)
                        for model_name, pred in [("HGB_raw", hgb["yhat_raw"]), ("HGB_NestedPlatt", hgb["yhat"])]:
                            row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                            y_performance_rows.append(row)
                        drows = treatment_performance_rows(cluster, stage, hgb["T_obs"], hgb["that"], term_names, "ExtraTrees_separate_interaction_terms")
                        for row in drows:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows)
                        terms, ames, _, wald = fit_fixed_interaction_second_stage(
                            y - hgb["yhat"], hgb["T_obs"] - hgb["that"], hgb["T_obs"],
                            term_names, selected, groups, cluster, stage, joint_spec,
                            "HGB_ET", MAIN_TRIM_FRAC, include_conditional=False,
                        )
                        term_frames.append(terms)
                        ame_frames.append(ames)
                        wald_frames.append(wald)

                    if RUN_XGBOOST_ROBUSTNESS and HAS_XGBOOST:
                        with StepTimer(runtime_log, cluster, stage, f"crossfit_XGB::{joint_spec}", n, n_matches):
                            xgb = crossfit_interaction_xgb(
                                X, y, D_raw, groups, splits,
                                cluster_seed(cluster, stage, 91000 + spec_index * 113), selected,
                            )
                        xcal = xgb["calibration_audit"].copy()
                        if not xcal.empty:
                            xcal.insert(0, "joint_spec", joint_spec)
                            xcal.insert(0, "stage", stage)
                            xcal.insert(0, "cluster", cluster)
                            calibration_frames.append(xcal)
                        for model_name, pred in [("XGB_raw", xgb["yhat_raw"]), ("XGB_NestedPlatt", xgb["yhat"])]:
                            row = outcome_performance_rows(cluster, stage, y, pred, model_name)
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                            y_performance_rows.append(row)
                        drows = treatment_performance_rows(cluster, stage, xgb["T_obs"], xgb["that"], term_names, "XGB_separate_interaction_terms")
                        for row in drows:
                            row.update({"joint_spec": joint_spec, "n_controls": X.shape[1]})
                        d_performance_rows.extend(drows)
                        terms, ames, _, wald = fit_fixed_interaction_second_stage(
                            y - xgb["yhat"], xgb["T_obs"] - xgb["that"], xgb["T_obs"],
                            term_names, selected, groups, cluster, stage, joint_spec,
                            "XGBoost", MAIN_TRIM_FRAC, include_conditional=False,
                        )
                        term_frames.append(terms)
                        ame_frames.append(ames)
                        wald_frames.append(wald)

                    run_log.append({
                        "cluster": cluster,
                        "stage": stage,
                        "joint_spec": joint_spec,
                        "joint_main_kpis": " | ".join(selected),
                        "target_members": " | ".join(target_members),
                        "status": "done",
                        "rows": n,
                        "matches": n_matches,
                        "n_controls": X.shape[1],
                        "elapsed_seconds": time.perf_counter() - spec_start,
                    })

                except Exception as error:
                    print(f"[ERROR] cluster={cluster} stage={stage} spec={joint_spec}: {error!r}")
                    run_log.append({
                        "cluster": cluster, "stage": stage, "joint_spec": joint_spec,
                        "status": "error", "reason": repr(error),
                    })

    concat = lambda frames: pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if any(f is not None and not f.empty for f in frames) else pd.DataFrame()
    terms, ames, conditional, wald = add_fixed_interaction_fdr(
        concat(term_frames), concat(ame_frames), concat(conditional_frames), concat(wald_frames)
    )
    undersample_terms, undersample_ames, undersample_conditional, undersample_wald = (
        add_fixed_interaction_fdr(
            concat(undersample_term_frames),
            concat(undersample_ame_frames),
            concat(undersample_conditional_frames),
            concat(undersample_wald_frames),
        )
    )
    main_terms = terms.loc[
        (terms["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(terms["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not terms.empty else pd.DataFrame()
    main_ames = ames.loc[
        (ames["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(ames["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not ames.empty else pd.DataFrame()
    main_wald = wald.loc[
        (wald["learner_set"] == "RF_main")
        & np.isclose(pd.to_numeric(wald["trim_fraction_rule"], errors="coerce"), MAIN_TRIM_FRAC)
    ].copy() if not wald.empty else pd.DataFrame()

    overlap = pd.DataFrame(overlap_rows_all)
    support_bins = pd.DataFrame(support_bin_rows_all)
    residual_vif = concat(term_vif_frames)
    placebo = concat(placebo_frames)
    fold_ame_summary = concat(fold_ame_summary_frames)
    team_ame_summary = concat(team_ame_summary_frames)
    learner_ame_summary, learner_term_summary = build_interaction_learner_summary(ames, terms)
    trim_ame_summary = build_ame_trim_stability(ames)
    main_summary = build_interaction_main_summary(
        main_ames, main_terms, overlap, residual_vif, placebo,
        fold_ame_summary, team_ame_summary, learner_ame_summary, trim_ame_summary, main_wald,
    )

    target_map = pd.DataFrame(selection_rows)
    target_summary = build_rotating_target_ame_report(
        target_map, main_summary, conditional
    )
    undersample_target_compare = build_undersampling_target_compare(
        target_map, target_summary, undersample_ames,
        undersample_conditional, undersample_wald,
    )
    undersample_term_compare = build_undersampling_term_compare(
        main_terms, undersample_terms
    )
    undersample_wald_compare = build_undersampling_wald_compare(
        main_wald, undersample_wald
    )
    if not target_summary.empty and not undersample_target_compare.empty:
        compare_keys = ["cluster", "stage", "joint_spec", "target"]
        compare_cols = compare_keys + [c for c in [
            "undersampled_AME", "undersampled_AME_se",
            "undersampled_AME_ci_low", "undersampled_AME_ci_high",
            "undersampled_AME_p", "undersampled_target_q_global",
            "AME_direction_same_undersampling",
            "AME_relative_absolute_change_undersampling",
            "undersampled_robust_conditional_sign_change",
            "conditional_sign_change_same_undersampling",
            "undersampled_interaction_q",
            "undersampled_interaction_increment_r2",
            "imbalance_sensitivity_class", "imbalance_sensitivity_flag",
            "optimizer_eligible_after_undersampling",
            "undersampling_target_ratio",
        ] if c in undersample_target_compare.columns]
        target_summary = target_summary.merge(
            undersample_target_compare[compare_cols].drop_duplicates(compare_keys),
            on=compare_keys, how="left",
        )

    overview_rows = []
    if not target_summary.empty:
        for (cluster, stage), group in target_summary.groupby(["cluster", "stage"], dropna=False):
            overview_rows.append({
                "cluster": cluster,
                "stage": stage,
                "n_single_stage_passed_targets": int(group["single_stage_pass"].map(_safe_bool).sum()),
                "n_distinct_joint_models": int(group["joint_spec"].replace("", np.nan).nunique(dropna=True)),
                "n_target_results_done": int(group["target_result_status"].eq("done").sum()),
                "n_target_global_FDR_pass": int((pd.to_numeric(group["target_q_global"], errors="coerce") <= FDR_ALPHA).sum()),
                "n_average_policy_levers": int(group["eligible_as_average_policy_lever"].map(_safe_bool).sum()),
                "n_interaction_optimizer_eligible": int(group["eligible_for_interaction_aware_optimizer"].map(_safe_bool).sum()),
                "n_fixed_positive": int(group["policy_constraint_class"].eq("FIXED_POSITIVE_DIRECTION").sum()),
                "n_fixed_negative": int(group["policy_constraint_class"].eq("FIXED_NEGATIVE_DIRECTION").sum()),
                "n_conditional_direction_only": int(group["policy_constraint_class"].eq("CONDITIONAL_DIRECTION_ONLY").sum()),
                "n_single_to_joint_direction_reversals": int(group["single_to_joint_direction_same"].eq(False).sum()),
                "n_undersampling_robust": int(group.get("imbalance_sensitivity_class", pd.Series(index=group.index, dtype=object)).eq("ROBUST").sum()),
                "n_undersampling_sensitive": int(group.get("imbalance_sensitivity_flag", pd.Series(False, index=group.index)).map(_safe_bool).sum()),
                "n_optimizer_eligible_after_undersampling": int(group.get("optimizer_eligible_after_undersampling", pd.Series(False, index=group.index)).map(_safe_bool).sum()),
            })
    overview = pd.DataFrame(overview_rows)

    checklist = pd.DataFrame([
        {"requirement": "Multivariable causal estimation", "evidence": "Every strictly screened single-KPI target is rotated through a three-mechanism interaction model", "sheet": "02_target_AME_report / 03_all_model_AME / 04_term_report", "status": "REPORTED" if not main_summary.empty else "MISSING"},
        {"requirement": "Non-additive KPI interactions", "evidence": "All three pairwise products included; joint interaction Wald test reported", "sheet": "04_term_report / 06_nested_Wald", "status": "REPORTED"},
        {"requirement": "Average and conditional effects", "evidence": "AME plus -1/0/+1 SD moderator grids", "sheet": "02_target_AME_report / 05_conditional_effects", "status": "REPORTED"},
        {"requirement": "No scaling leakage", "evidence": "KPI means/SDs learned only from outer-training matches", "sheet": "28_standardization_audit", "status": "PASS"},
        {"requirement": "Match-grouped cross-fitting", "evidence": "Five folds with disjoint matches", "sheet": "26_fold_audit", "status": "PASS"},
        {"requirement": "KPI multicollinearity", "evidence": "Raw main-KPI and OOF residual-term correlations/VIF plus condition number", "sheet": "14-17", "status": "REPORTED"},
        {"requirement": "Overlap/common support", "evidence": "OOF treatment R2, residual SD ratio and support bins for all six terms", "sheet": "11_overlap / 18_support_bins", "status": "REPORTED"},
        {"requirement": "Trim sensitivity", "evidence": "0%, 1%, 2%, 5% RF estimates", "sheet": "07_AME_all_trims / 08_terms_all_trims", "status": "REPORTED"},
        {"requirement": "Alternative nuisance learners", "evidence": "RF, Logistic/Ridge, HGB/ET, optional XGB", "sheet": "09-10", "status": "REPORTED"},
        {"requirement": "Unobserved-confounding sensitivity", "evidence": "Partial R2 and robustness values for AMEs and coefficients", "sheet": "02-04", "status": "REPORTED"},
        {"requirement": "Placebo and stability", "evidence": "Within-match placebo, fold and leave-one-team-out checks", "sheet": "19-25", "status": "REPORTED"},
        {"requirement": "Class-imbalance sensitivity", "evidence": "RF outcome nuisance re-estimated after 1:2 majority undersampling inside every inner/outer training fold; validation/test distributions, D|X and second stage unchanged", "sheet": "36-45 undersampling sheets", "status": "REPORTED" if not undersample_target_compare.empty else "NOT_RUN"},
        {"requirement": "External validation", "evidence": "Not available in the current data environment; explicitly separated from this script", "sheet": "33_manifest", "status": "NOT_RUN_no_external_data"},
    ])

    manifest = pd.DataFrame([{
        "input_path": INPUT_PATH,
        "output_path": ALL_PASSED_INTERACTION_OUTPUT_XLSX,
        "rows_input_after_exclusion": len(df),
        "matches_input": df[MATCH_COL].nunique(dropna=True),
        "outcome": OUTCOME_COL,
        "analysis": "rotating joint interaction DML for every KPI passing the strict single-stage screen",
        "main_analysis_relationship": "single-KPI DML screens targets; each passed target then receives a target-specific multivariable interaction validation",
        "single_result_path": SINGLE_RESULT_XLSX,
        "anchor_selection_rule": ANCHOR_SELECTION_RULE,
        "single_screening_rule": (
            f"global FDR; no severe overlap; fold>={SINGLE_SCREEN_FOLD_SIGN_MIN}; "
            f"team-LOO>={SINGLE_SCREEN_TEAM_SIGN_MIN}; placebo<={SINGLE_SCREEN_PLACEBO_P_MAX}; "
            f"all trim directions and {SINGLE_SCREEN_MIN_TRIM_P_LT_05}/{len(TRIM_GRID)} trim p<.05; "
            "all available alternative learner directions and FDR"
        ),
        "model_terms": "three fold-standardized main KPIs + all three pairwise products",
        "interaction_construction": "main KPIs standardized using outer-training matches only; products constructed before treatment nuisance residualization",
        "main_effect_interpretation": "coefficient at other standardized KPIs=0; AME is the primary multivariable KPI effect",
        "conditional_effect_grid": "both other mechanisms at -1, 0, +1 SD",
        "main_y_learner": "RandomForestClassifier + nested match-grouped Platt calibration",
        "main_D_learner": "multi-output RandomForestRegressor for six treatment basis terms",
        "alternative_learners": "Logistic/Ridge, HGB/ExtraTrees, optional XGBoost",
        "undersampling_sensitivity": (
            "RF outcome nuisance only; random majority undersampling inside each inner/outer training subset; "
            "inner validation and outer test rows preserve the original distribution; primary RF D|X residuals reused"
            if RUN_UNDERSAMPLING_SENSITIVITY else "not run"
        ),
        "undersampling_active_stages": " | ".join(UNDERSAMPLING_ACTIVE_STAGES),
        "undersampling_target_minority_to_majority_ratio": UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO,
        "undersampling_magnitude_change_flag": UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG,
        "undersampling_screening_policy": "not an admission gate; strict single-KPI primary-model screening remains unchanged",
        "crossfit": f"{N_SPLITS}-fold grouped by {MATCH_COL}",
        "main_trim": MAIN_TRIM_FRAC,
        "trim_grid": json.dumps(TRIM_GRID),
        "second_stage": "unit-weight WLS on six OOF residualized basis terms + HC3 SE",
        "FDR": "BH for unique target AMEs; full model AMEs, coefficients and Wald blocks also retain model-level BH columns",
        "OVB": "partial R2 and robustness values reported; detailed observed-covariate benchmarking remains in the single-KPI workbook",
        "actual_E_policy": "actual (E) KPIs ignored; L and E' are screened and estimated separately",
        "external_validation": "not run because no external dataset is available",
        "xgboost_available": HAS_XGBOOST,
    }])

    runtime_frame = pd.DataFrame([record.__dict__ for record in runtime_log])
    run_log_frame = pd.DataFrame(run_log)
    tables = OrderedDict([
        ("00_reviewer_checklist", checklist),
        ("01_overview", overview),
        ("02_target_AME_report", target_summary),
        ("03_all_model_AME", main_summary),
        ("04_term_report", main_terms),
        ("05_conditional_effects", conditional),
        ("06_nested_Wald", main_wald),
        ("07_AME_all_trims", ames.loc[ames["learner_set"] == "RF_main"].copy() if not ames.empty else ames),
        ("08_terms_all_trims", terms.loc[terms["learner_set"] == "RF_main"].copy() if not terms.empty else terms),
        ("09_AME_learner_robust", learner_ame_summary),
        ("10_term_learner_robust", learner_term_summary),
        ("11_overlap", overlap),
        ("12_Y_performance", pd.DataFrame(y_performance_rows)),
        ("13_D_performance", pd.DataFrame(d_performance_rows)),
        ("14_raw_main_corr", concat(raw_main_corr_frames)),
        ("15_raw_main_VIF", concat(raw_main_vif_frames)),
        ("16_residual_term_corr", concat(term_corr_frames)),
        ("17_residual_term_VIF", residual_vif),
        ("18_support_bins", support_bins),
        ("19_placebo", placebo),
        ("20_fold_AME_summary", fold_ame_summary),
        ("21_fold_AME_long", concat(fold_ame_long_frames)),
        ("22_team_AME_summary", team_ame_summary),
        ("23_team_AME_long", concat(team_ame_long_frames)),
        ("24_fold_term_summary", concat(fold_term_summary_frames)),
        ("25_team_term_summary", concat(team_term_summary_frames)),
        ("26_fold_audit", concat(fold_audit_frames)),
        ("27_calibration_audit", concat(calibration_frames)),
        ("28_standardization_audit", concat(standardization_frames)),
        ("29_data_audit", pd.DataFrame(data_audit_rows)),
        ("30_single_screening", single_screening_audit),
        ("31_target_model_map", target_map),
        ("32_anchor_plan", pd.DataFrame(anchor_rows)),
        ("33_manifest", manifest),
        ("34_run_log", run_log_frame),
        ("35_runtime", runtime_frame),
        ("36_undersample_target_compare", undersample_target_compare),
        ("37_undersample_all_AME", undersample_ames),
        ("38_undersample_terms", undersample_terms),
        ("39_undersample_conditional", undersample_conditional),
        ("40_undersample_Wald", undersample_wald),
        ("41_undersample_term_compare", undersample_term_compare),
        ("42_undersample_Wald_compare", undersample_wald_compare),
        ("43_undersample_Y_perf", pd.DataFrame(undersample_y_performance_rows)),
        ("44_undersample_sampling", concat(undersample_sampling_frames)),
        ("45_undersample_calibration", concat(undersample_calibration_frames)),
    ])

    output = Path(ALL_PASSED_INTERACTION_OUTPUT_XLSX)
    output.parent.mkdir(parents=True, exist_ok=True)
    used = set()
    final_tables = OrderedDict()
    with pd.ExcelWriter(
        output, engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    ) as writer:
        for requested, table in tables.items():
            if table is None or table.empty:
                table = pd.DataFrame([{"message": "No results produced for this table."}])
            sheet = safe_sheet_name(requested, used)
            table.to_excel(writer, index=False, sheet_name=sheet)
            final_tables[sheet] = table
        format_joint_workbook(writer, final_tables)

    elapsed = time.perf_counter() - total_start
    runtime_csv = output.with_name(output.stem + "_runtime.csv")
    pd.DataFrame([record.__dict__ for record in runtime_log]).to_csv(
        runtime_csv, index=False, encoding="utf-8-sig"
    )
    print("=" * 110)
    print(f"[DONE] {ALL_PASSED_INTERACTION_OUTPUT_XLSX}")
    print(f"[RUNTIME] {format_seconds(elapsed)}")
    print(f"[RUNTIME CSV] {runtime_csv}")
    print("=" * 110)


if __name__ == "__main__":
    all_passed_interaction_main()
