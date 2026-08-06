# -*- coding: utf-8 -*-
"""
Shot · Single-KPI DML · L only · RF+RF primary model
====================================================

Analysis scope
--------------
Only the shot-location stage L is analysed.

No endpoint-stage variable is calculated or used:
- no E treatment;
- no E' treatment;
- no Att(E) or Att(E') covariate;
- no Def(E) or Def(E') covariate;
- no end_location, shot geometry, duration, or endpoint-derived variable in X.

Outcome Y
---------
success_def

Treatment D
-----------
One defending-side KPI measured at the shot location L is analysed at a time.

Covariates X
------------
1. Reduced pre-shot numeric background:
   period
   match_second
   event_team_score_before
   event_opponent_score_before

2. Categorical background, one-hot encoded when available:
   team
   position_name
   play_pattern
   event_score_state_before
   shot_team_home_away

3. Permitted attacking-side L KPIs only.

No defending-side KPI enters X.

Avg/Centroid reciprocal exclusion
---------------------------------
When D is any Avg_*_Def(L):
  remove every Avg_*_Att(L) and DistToAttCentroid(L) from X.

When D is DistToDefCentroid(L):
  remove DistToAttCentroid(L) and every Avg_*_Att(L) from X.

Other same-series rules:
  Area_Def(L) removes Area_Att(L).
  Spr_Def(L) removes Spr_Att(L).
  Adv_5(L) and Adv_10(L) have no Att-only counterpart.

Main nuisance learners
----------------------
Y|X: RandomForestClassifier with nested match-grouped Platt calibration.
D|X: RandomForestRegressor.

Robustness learners
-------------------
HistGradientBoosting + ExtraTrees.
LogisticRegression + Ridge.
Optional XGBoost classifier + regressor.

Diagnostics
-----------
- match-grouped five-fold cross-fitting;
- raw and calibrated OOF ROC-AUC, PR-AUC, Brier, log loss,
  calibration intercept/slope, and ECE;
- treatment OOF R2, RMSE, MAE, and normalized RMSE;
- 0%, 1%, 2%, and 5% trimming;
- continuous-treatment overlap and residual-support checks;
- within-match placebo tests;
- fold and leave-one-team-out stability;
- global and within-cluster BH-FDR;
- HC3 heteroskedasticity-robust second-stage standard errors;
- correlation/VIF, partial-R2, and robustness-value summaries.

Class-imbalance sensitivity
---------------------------
The original-distribution RF/RF DML remains the primary analysis. An additional
reviewer-facing sensitivity branch re-estimates only the outcome nuisance model
using random majority-class undersampling inside every inner and outer TRAINING
fold. Inner calibration-validation rows and outer test rows retain the original
class distribution. The treatment nuisance model and second-stage sample are
not undersampled. The default target minority:majority ratio is 1:2.

This is the single-KPI main analysis. Joint DML and nonlinear treatment-effect
checks remain disabled, matching the final Pass and Carry settings.
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
HOME_AWAY_COL = "shot_team_home_away"

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
ACTIVE_STAGES = ("L",)

# Class-imbalance sensitivity. The primary analysis keeps the observed outcome
# distribution. Only training subsets used to fit Y|X are undersampled.
RUN_UNDERSAMPLING_SENSITIVITY = True
UNDERSAMPLING_ACTIVE_STAGES = ACTIVE_STAGES
UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO = 0.50  # 1:2 minority:majority
UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG = 0.30

# Shot-location-only attacking-control policy:
# - only L treatments are estimated;
# - X may include permitted Att(L) KPIs only;
# - E and E' KPIs are ignored completely;
# - no Def KPI from any stage enters X.
INCLUDE_ATT_EPRIME_IN_L_X = False
INCLUDE_ATT_L_IN_EPRIME_X = False
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

RF_D_N_TREES = 240
RF_D_MIN_SAMPLES_LEAF = 5
RF_D_MAX_FEATURES = 0.80
RF_D_MAX_DEPTH = None

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
# When treatment D is any Avg_*_Def KPI:
#   remove all Avg_*_Att controls and DistToAttCentroid controls
#   from both L and E'.
#
# When treatment D is DistToDefCentroid:
#   remove DistToAttCentroid controls and all Avg_*_Att controls
#   from both L and E'.
#
# This affects attacking-side KPI controls only. Every defending-side KPI is
# already excluded from X under the single-KPI design.
EXCLUDE_AVG_AND_CENTROID_TOGETHER = True

# Families are defined separately for attacking/defending indicators.
# Stage suffixes (L) and (E') are handled automatically.
KPI_FAMILY_PATTERNS = OrderedDict([
    ("avg_def", [r"^Avg_\d+_Def$"]),
    ("avg_att", [r"^Avg_\d+_Att$"]),
    ("local_advantage", [r"^Adv_\d+$"]),
    ("global_structure_def", [
        r"^Area_Def$",
        r"^Spr_Def$",
        r"^DistToDefCentroid$",
    ]),
    ("global_structure_att", [
        r"^Area_Att$",
        r"^Spr_Att$",
        r"^DistToAttCentroid$",
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

# Pre-specified joint models: one representative KPI per tactical mechanism.
# Missing columns are skipped and logged; no p-value-driven selection.
JOINT_MODEL_SPECS = {
    "L": OrderedDict([
        ("main_direct_local_structure", [
            "Avg_1_Def(L)", "Adv_5(L)", "Spr_Def(L)",
        ]),
        ("sensitivity_centroid_local_structure", [
            "DistToDefCentroid(L)", "Adv_5(L)", "Spr_Def(L)",
        ]),
        ("sensitivity_local_adv10", [
            "Avg_1_Def(L)", "Adv_10(L)", "Spr_Def(L)",
        ]),
        ("sensitivity_structure_area", [
            "Avg_1_Def(L)", "Adv_5(L)", "Area_Def(L)",
        ]),
    ]),
    "E'": OrderedDict([
        ("main_direct_local_structure", [
            "Avg_1_Def(E')", "Adv_5(E')", "Spr_Def(E')",
        ]),
        ("sensitivity_centroid_local_structure", [
            "DistToDefCentroid(E')", "Adv_5(E')", "Spr_Def(E')",
        ]),
        ("sensitivity_local_adv10", [
            "Avg_1_Def(E')", "Adv_10(E')", "Spr_Def(E')",
        ]),
        ("sensitivity_structure_area", [
            "Avg_1_Def(E')", "Adv_5(E')", "Area_Def(E')",
        ]),
    ]),
}

INTERACTION_PAIRS = {
    "L": ("Avg_1_Def(L)", "Adv_5(L)"),
    "E'": ("Avg_1_Def(E')", "Adv_5(E')"),
}

# Raw/object/post-event columns are never auto-added as controls.
RAW_OR_POST_EVENT_PATTERNS = [
    r"^id$", r"^type$", r"^pass$", r"^shot$", r"freeze_frame", r"back_ff",
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
    """Return True for an actual endpoint-stage KPI ending in (E)."""
    s = str(c).strip()
    return s.endswith("(E)") and not s.endswith("(E')")


def is_endpoint_stage_kpi(c: str) -> bool:
    """Every E or E' KPI is ignored by the Shot-time-only model."""
    return is_Eprime_col(c) or is_actual_E_col(c)


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

    Base rule
    ---------
    Remove the attacking-side tactical series matching the defending treatment.

    Additional Avg/Centroid reciprocal rule
    ---------------------------------------
    Avg_Def target:
      remove every Avg_Att control and DistToAttCentroid.

    DistToDefCentroid target:
      remove DistToAttCentroid and every Avg_Att control.

    With EXCLUDE_SAME_SERIES_ATT_CROSS_STAGE=True, these exclusions are applied
    to both L and E' controls.
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
        class_weight=None,
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
    """Select training rows for deterministic random majority undersampling.

    All minority rows are retained. This helper must only receive a model-
    training subset; calibration-validation and outer-test rows are never
    passed to it.
    """
    y_arr = np.asarray(y_train, dtype=int).reshape(-1)
    all_idx = np.arange(len(y_arr), dtype=int)
    classes, counts = np.unique(y_arr, return_counts=True)
    audit: Dict[str, object] = {
        "sampling_method": "random_majority_undersampling_training_only",
        "target_minority_to_majority_ratio": float(
            target_minority_to_majority_ratio
        ),
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
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError(
            "UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO must be in (0, 1]."
        )

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
    """Fit RF Y|X after undersampling this training subset only."""
    imputer = CompleteCaseTransformer(strategy="median")
    Xtr_full = imputer.fit_transform(X_train)
    Xte = imputer.transform(X_test)
    sampled_idx, audit = undersample_binary_training_indices(
        y_train,
        target_minority_to_majority_ratio=(
            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
        ),
        seed=seed,
    )
    model = make_main_y_model(seed)
    model.fit(
        Xtr_full[sampled_idx],
        np.asarray(y_train, dtype=int)[sampled_idx],
    )
    return clip_probability(model.predict_proba(Xte)[:, 1]), audit


def nested_group_platt_predict_undersampled(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], pd.DataFrame]:
    """Leakage-safe Platt calibration for the undersampled RF sensitivity."""
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
        "undersampling_target_ratio": (
            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
        ),
        "outer_sampling_applied": outer_sampling["sampling_applied"],
        "outer_original_n": outer_sampling["original_n"],
        "outer_sampled_n": outer_sampling["sampled_n"],
        "outer_original_class0_n": outer_sampling["original_class0_n"],
        "outer_original_class1_n": outer_sampling["original_class1_n"],
        "outer_sampled_class0_n": outer_sampling["sampled_class0_n"],
        "outer_sampled_class1_n": outer_sampling["sampled_class1_n"],
        "outer_achieved_ratio": (
            outer_sampling["achieved_minority_to_majority_ratio"]
        ),
        "inner_splits": 0,
        "inner_oof_rows": 0,
        "inner_oof_coverage": 0.0,
        "platt_intercept": np.nan,
        "platt_slope": np.nan,
        "outer_raw_prediction_mean": float(np.mean(raw_test)),
        "outer_calibrated_prediction_mean": float(np.mean(raw_test)),
    }
    for prefix in ("inner_raw", "inner_calibrated"):
        for metric in (
            "roc_auc", "pr_auc", "brier", "log_loss",
            "calibration_intercept", "calibration_slope", "ece_10bin",
        ):
            audit[f"{prefix}_{metric}"] = np.nan

    def unchanged(reason: str):
        audit["calibration_reason"] = reason
        return raw_test.copy(), raw_test, audit, pd.DataFrame(sampling_rows)

    if not CALIBRATE_MAIN_Y:
        return unchanged("not_requested")
    if len(y_train) < CALIBRATION_MIN_ROWS:
        return unchanged("too_few_training_rows")
    if int(np.bincount(y_train, minlength=2).min()) < CALIBRATION_MIN_CLASS_COUNT:
        return unchanged("too_few_training_cases_in_one_class")

    inner_splits = make_valid_calibration_splits(
        y_train,
        groups_train,
        CALIBRATION_INNER_SPLITS,
        seed + 100_000,
    )
    if not inner_splits:
        return unchanged("no_valid_inner_grouped_splits")

    inner_raw = np.full(len(y_train), np.nan)
    for inner_fold, (itr, iva) in enumerate(inner_splits):
        pred, sample_audit = fit_predict_rf_outcome_undersampled(
            X_train[itr],
            y_train[itr],
            X_train[iva],
            seed=seed + 10_000 + inner_fold,
        )
        inner_raw[iva] = pred
        sampling_rows.append(dict(
            sampling_level="inner_training",
            inner_fold=int(inner_fold),
            **sample_audit,
        ))

    ok = np.isfinite(inner_raw)
    audit["inner_splits"] = len(inner_splits)
    audit["inner_oof_rows"] = int(ok.sum())
    audit["inner_oof_coverage"] = float(np.mean(ok))
    if float(np.mean(ok)) < 0.95 or len(np.unique(y_train[ok])) < 2:
        return unchanged("insufficient_inner_oof_coverage")

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
        calibrator.predict_proba(
            probability_logit(raw_test).reshape(-1, 1)
        )[:, 1]
    )
    for key, value in safe_binary_metrics(y_train[ok], inner_raw[ok]).items():
        audit[f"inner_raw_{key}"] = value
    for key, value in safe_binary_metrics(
        y_train[ok], calibrated_inner
    ).items():
        audit[f"inner_calibrated_{key}"] = value
    audit.update({
        "calibration_used": True,
        "calibration_reason": (
            "nested_grouped_platt_applied_after_training_only_undersampling"
        ),
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_slope": float(calibrator.coef_[0, 0]),
        "outer_calibrated_prediction_mean": float(
            np.mean(calibrated_test)
        ),
    })
    return calibrated_test, raw_test, audit, pd.DataFrame(sampling_rows)


def crossfit_undersampled_rf_y(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> Dict[str, object]:
    """Generate grouped OOF Y predictions for the sensitivity branch."""
    yhat = np.full(len(y), np.nan)
    yhat_raw = np.full(len(y), np.nan)
    calibration_rows = []
    sampling_rows = []
    for fold, (tr, te) in enumerate(splits):
        pred, pred_raw, audit, sample_df = (
            nested_group_platt_predict_undersampled(
                X_train=X[tr],
                y_train=y[tr],
                groups_train=groups[tr],
                X_test=X[te],
                seed=seed + fold,
            )
        )
        yhat[te] = pred
        yhat_raw[te] = pred_raw
        audit.update({
            "outer_fold": int(fold),
            "outer_train_rows": int(len(tr)),
            "outer_test_rows": int(len(te)),
            "outer_train_matches": int(len(pd.unique(groups[tr]))),
            "outer_test_matches": int(len(pd.unique(groups[te]))),
        })
        for key, value in safe_binary_metrics(y[te], pred_raw).items():
            audit[f"outer_raw_{key}"] = value
        for key, value in safe_binary_metrics(y[te], pred).items():
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


def available_joint_specs(stage: str, D_cols: Sequence[str]):
    """Return pre-specified joint KPI sets available for one stage."""
    available = set(D_cols)
    specs = []
    log = []
    stage_specs = JOINT_MODEL_SPECS.get(stage, OrderedDict())
    for spec_name, requested in stage_specs.items():
        selected = [c for c in requested if c in available]
        missing = [c for c in requested if c not in available]
        log.append({
            "stage": stage,
            "spec_name": spec_name,
            "requested": " | ".join(requested),
            "selected": " | ".join(selected),
            "missing": " | ".join(missing),
            "n_selected": len(selected),
        })
        if len(selected) >= 2:
            specs.append((spec_name, selected))
    return specs, log


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
    """Identify only usable L-stage Shot KPIs.

    Every KPI ending in (E) or (E') is recorded for audit but is not made
    available as a treatment or covariate.
    """
    all_L = numeric_usable_columns(
        sub,
        [c for c in sub.columns if is_L_col(c)],
    )

    ignored_endpoint_kpis = numeric_usable_columns(
        sub,
        [c for c in sub.columns if is_endpoint_stage_kpi(c)],
    )

    L_att = [
        c
        for c in all_L
        if is_att_kpi(c)
    ]

    D_L = [
        c
        for c in all_L
        if c not in set(L_att)
    ]

    L_base = numeric_usable_columns(
        sub,
        L_BASE_CONTROL_COLS,
    )

    categorical_dummies, teams, categorical_groups = (
        prepare_context_dummies(sub)
    )

    return {
        "all_L": all_L,
        "all_E": [],
        "L_att": L_att,
        "E_att": [],
        "ignored_endpoint_kpis": ignored_endpoint_kpis,
        "D_by_stage": {
            "L": D_L,
            "E'": [],
        },
        "base_by_stage": {
            "L": L_base,
            "E'": [],
        },
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
    """Build one Shot-location-L single-KPI DML dataset.

    D:
        exactly one defending-side KPI ending in (L).

    X:
        reduced pre-shot background;
        permitted attacking-side KPI controls ending in (L);
        categorical context dummies.

    Every defending KPI and every E/E' KPI is excluded from X.
    """
    stage = str(stage).strip().upper()

    if stage != "L":
        raise ValueError(
            "Shot DML is L-only. E and E' stages are disabled."
        )

    selected = uniq_keep_order(
        list(treatment_cols)
    )

    if len(selected) != 1:
        raise ValueError(
            "This is the single-KPI main model; exactly one "
            "treatment must be supplied."
        )

    missing = [
        c
        for c in selected
        if c not in sub.columns
    ]
    if missing:
        raise ValueError(
            f"Treatment columns missing: {missing}"
        )

    wrong_stage = [
        c
        for c in selected
        if not is_L_col(c)
    ]
    if wrong_stage:
        raise ValueError(
            f"Shot treatments must be L-stage KPIs: {wrong_stage}"
        )

    attacking_targets = [
        c
        for c in selected
        if is_att_kpi(c)
    ]
    if attacking_targets:
        raise ValueError(
            f"Attacking KPIs cannot be treatments: "
            f"{attacking_targets}"
        )

    y_ser = to_num(
        sub[OUTCOME_COL]
    )
    group_ser = sub[MATCH_COL]
    D_df_full = sub[selected].apply(
        to_num
    )

    complete = (
        y_ser.notna()
        & group_ser.notna()
        & D_df_full.notna().all(axis=1)
    )

    work = sub.loc[
        complete
    ].copy()

    D_df = D_df_full.loc[
        complete
    ].copy()

    all_L = list(
        inventory["all_L"]
    )
    L_att_all = list(
        inventory["L_att"]
    )
    ignored_endpoint_kpis = list(
        inventory.get(
            "ignored_endpoint_kpis",
            [],
        )
    )
    base_controls = list(
        inventory["base_by_stage"]["L"]
    )

    L_def = [
        c
        for c in all_L
        if c not in set(L_att_all)
    ]

    # Apply the same-series rule plus the requested Avg/Centroid
    # reciprocal exclusion, but only among L-stage attacking KPIs.
    exclude_from_L_att = (
        attacking_controls_to_exclude_for_targets(
            L_att_all,
            selected,
        )
    )

    excluded_att_set = set(
        exclude_from_L_att
    )

    L_att_controls = [
        c
        for c in L_att_all
        if c not in excluded_att_set
    ]

    requested_numeric = uniq_keep_order(
        base_controls
        + L_att_controls
    )

    excluded_defensive_controls = uniq_keep_order(
        [
            c
            for c in L_def
            if c not in set(selected)
        ]
    )

    estimand = (
        "conditional effect of one defending Shot-location L KPI "
        "given reduced pre-shot background and permitted Att(L) "
        "controls; when D is Avg_Def, all Avg_Att and "
        "DistToAttCentroid controls are removed; when D is "
        "DistToDefCentroid, DistToAttCentroid and all Avg_Att "
        "controls are removed; every other defending KPI and every "
        "E/E' KPI is excluded; exact location, endpoint, shot "
        "geometry, duration, and freeze-frame-count variables do "
        "not enter X"
    )

    usable_numeric = numeric_usable_columns(
        work,
        requested_numeric,
    )

    X_num_df = (
        work[usable_numeric].apply(to_num)
        if usable_numeric
        else pd.DataFrame(index=work.index)
    )

    categorical_dummies_all = inventory[
        "categorical_dummies"
    ]
    categorical_dummies = (
        categorical_dummies_all
        .loc[complete]
        .copy()
    )

    X_df = pd.concat(
        [
            X_num_df,
            categorical_dummies,
        ],
        axis=1,
    )

    X_df = (
        X_df
        .loc[
            :,
            ~X_df.columns.duplicated(),
        ]
        .copy()
    )

    # Hard exclusions.
    X_df = X_df.drop(
        columns=[
            c
            for c in X_df.columns
            if c in set(L_def)
        ],
        errors="ignore",
    )

    X_df = X_df.drop(
        columns=selected,
        errors="ignore",
    )

    X_df = X_df.drop(
        columns=list(excluded_att_set),
        errors="ignore",
    )

    X_df = X_df.drop(
        columns=[
            c
            for c in X_df.columns
            if is_endpoint_stage_kpi(c)
        ],
        errors="ignore",
    )

    leaked_def = [
        c
        for c in X_df.columns
        if c in set(L_def)
    ]
    if leaked_def:
        raise RuntimeError(
            f"Defending KPI leaked into X: {leaked_def}"
        )

    leaked_matched_att = [
        c
        for c in X_df.columns
        if c in excluded_att_set
    ]
    if leaked_matched_att:
        raise RuntimeError(
            "Excluded attacking KPI series leaked into X: "
            f"{leaked_matched_att}"
        )

    leaked_endpoint = [
        c
        for c in X_df.columns
        if is_endpoint_stage_kpi(c)
    ]
    if leaked_endpoint:
        raise RuntimeError(
            "E/E' KPI leaked into Shot-time-only X: "
            f"{leaked_endpoint}"
        )

    teams_all = np.asarray(
        inventory["teams"],
        dtype=object,
    )

    teams = teams_all[
        np.where(complete.values)[0]
    ]

    groups = (
        group_ser.loc[complete]
        .astype(str)
        .values
    )

    y = (
        y_ser.loc[complete]
        .values
        > 0.5
    ).astype(int)

    return {
        "stage": "L",
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
        "earlier_stage_attacking_controls": [],
        "same_stage_attacking_controls": L_att_controls,
        "cross_stage_attacking_controls": [],
        "target_series": sorted(
            {
                get_kpi_series(c)
                for c in selected
            }
        ),
        "treatment_families": sorted(
            {
                get_kpi_family(c)
                for c in selected
            }
        ),
        "excluded_same_series_att_L": (
            uniq_keep_order(
                exclude_from_L_att
            )
        ),
        "excluded_same_series_att_Eprime": [],
        "excluded_same_series_attacking_controls": (
            uniq_keep_order(
                exclude_from_L_att
            )
        ),
        "excluded_defensive_controls": (
            excluded_defensive_controls
        ),
        "ignored_endpoint_kpis": (
            ignored_endpoint_kpis
        ),
        "categorical_control_cols": list(
            categorical_dummies.columns
        ),
        "team_control_cols": list(
            inventory[
                "categorical_groups"
            ].get("team", [])
        ),
        "position_control_cols": list(
            inventory[
                "categorical_groups"
            ].get("position", [])
        ),
        "play_pattern_control_cols": list(
            inventory[
                "categorical_groups"
            ].get("play_pattern", [])
        ),
        "score_state_control_cols": list(
            inventory[
                "categorical_groups"
            ].get("score_state", [])
        ),
        "home_away_control_cols": list(
            inventory[
                "categorical_groups"
            ].get("home_away", [])
        ),
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
            "ignored_endpoint_kpis", "categorical_controls", "estimand"
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
    undersample_oneD_all = []
    undersample_y_perf = []
    undersample_sampling_audits = []
    undersample_calibration_audits = []
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
    print(
        "SHOT · SINGLE-KPI DML · L ONLY · AVG/CENTROID RECIPROCAL "
        "EXCLUSION + TRAINING-FOLD UNDERSAMPLING SENSITIVITY"
    )
    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_XLSX}")
    print("=" * 100)

    with StepTimer(runtime_log, "ALL", "ALL", "read_and_prepare"):
        df = read_input_table(INPUT_PATH, sheet_name=SHEET_NAME)
        df.columns = [str(c).strip() for c in df.columns]
        check_required_columns(df)
        df[OUTCOME_COL] = to_num(df[OUTCOME_COL])

        # Shot analysis is strictly location-stage L only.
        # Do not calculate dx, dy, action length, angle, or any endpoint feature.
        l_def_columns = [
            c
            for c in df.columns
            if is_L_col(c) and not is_att_kpi(c)
        ]
        if not l_def_columns:
            raise ValueError(
                "No usable defending-side L KPI columns were found."
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
                        "ignored_endpoint_kpis": " | ".join(block.get("ignored_endpoint_kpis", [])),
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

                    # Class-imbalance sensitivity: refit Y|X after majority
                    # undersampling inside each inner/outer training fold only.
                    # Reuse primary RF D|X residuals so the comparison isolates
                    # outcome-class handling.
                    if (
                        RUN_UNDERSAMPLING_SENSITIVITY
                        and stage in UNDERSAMPLING_ACTIVE_STAGES
                    ):
                        with StepTimer(
                            runtime_log,
                            cid,
                            stage,
                            f"undersampling_sensitivity::{d_col}",
                            n,
                            n_matches,
                            note=(
                                "RF Y|X majority undersampling inside inner/outer "
                                "training folds; target minority:majority="
                                f"{UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO:.3f}"
                            ),
                        ):
                            us_cf = crossfit_undersampled_rf_y(
                                X=X,
                                y=y,
                                groups=groups,
                                splits=splits,
                                seed=cluster_seed(
                                    cid, stage, 50000 + d_index * 19
                                ),
                            )

                        us_cal = us_cf.get(
                            "calibration_audit", pd.DataFrame()
                        ).copy()
                        if not us_cal.empty:
                            us_cal.insert(0, "treatment", d_col)
                            us_cal.insert(0, "stage", stage)
                            us_cal.insert(0, "cluster", cid)
                            undersample_calibration_audits.append(us_cal)

                        us_sampling = us_cf.get(
                            "sampling_audit", pd.DataFrame()
                        ).copy()
                        if not us_sampling.empty:
                            us_sampling.insert(0, "treatment", d_col)
                            us_sampling.insert(0, "stage", stage)
                            us_sampling.insert(0, "cluster", cid)
                            undersample_sampling_audits.append(us_sampling)

                        yhat_us = us_cf["yhat"]
                        y_res_us = y - yhat_us
                        us_one = run_oneD_all_trims(
                            y_res_us,
                            D_res,
                            [d_col],
                            groups,
                            raw_sd_map,
                            cid,
                            stage,
                            (
                                "rf_outcome_training_fold_"
                                "undersampling_1to2_sensitivity"
                            ),
                        )
                        if not us_one.empty:
                            us_one["n_controls"] = X.shape[1]
                            us_one["undersampling_target_ratio"] = (
                                UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
                            )
                            us_one[
                                "treatment_nuisance_reused_from_primary"
                            ] = True
                            undersample_oneD_all.append(us_one)

                        us_raw_perf = outcome_performance_rows(
                            cid,
                            stage,
                            y,
                            us_cf["yhat_raw"],
                            (
                                "RandomForestClassifier_training_fold_"
                                "undersampled_raw"
                            ),
                        )
                        us_raw_perf["treatment_specific_X"] = d_col
                        us_raw_perf["n_controls"] = X.shape[1]
                        us_raw_perf["undersampling_target_ratio"] = (
                            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
                        )
                        undersample_y_perf.append(us_raw_perf)

                        us_cal_perf = outcome_performance_rows(
                            cid,
                            stage,
                            y,
                            yhat_us,
                            (
                                "RandomForestClassifier_training_fold_"
                                "undersampled_NestedPlatt"
                            ),
                        )
                        us_cal_perf["treatment_specific_X"] = d_col
                        us_cal_perf["n_controls"] = X.shape[1]
                        us_cal_perf["undersampling_target_ratio"] = (
                            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
                        )
                        undersample_y_perf.append(us_cal_perf)

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
                            "ignored_endpoint_kpis": " | ".join(jb.get("ignored_endpoint_kpis", [])),
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
    undersample_df = (
        pd.concat(undersample_oneD_all, ignore_index=True)
        if undersample_oneD_all else pd.DataFrame()
    )
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

    if not undersample_df.empty:
        main_mask_us = np.isclose(
            undersample_df["trim_fraction_rule"], MAIN_TRIM_FRAC
        )
        undersample_main_df = apply_fdr_columns(
            undersample_df.loc[main_mask_us].copy()
        )
        undersample_df = pd.concat(
            [
                undersample_main_df,
                undersample_df.loc[~main_mask_us].copy(),
            ],
            ignore_index=True,
        )
    else:
        undersample_main_df = pd.DataFrame()

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
    undersample_y_perf_df = pd.DataFrame(undersample_y_perf)
    if not undersample_y_perf_df.empty:
        y_perf_df = pd.concat(
            [y_perf_df, undersample_y_perf_df],
            ignore_index=True,
            sort=False,
        )
    undersample_sampling_audit_df = (
        pd.concat(undersample_sampling_audits, ignore_index=True)
        if undersample_sampling_audits else pd.DataFrame()
    )
    undersample_calibration_audit_df = (
        pd.concat(undersample_calibration_audits, ignore_index=True)
        if undersample_calibration_audits else pd.DataFrame()
    )
    d_perf_df = pd.DataFrame(d_perf)
    data_audit_df = pd.DataFrame(data_audit)
    fold_audit_df = pd.concat(fold_audits, ignore_index=True) if fold_audits else pd.DataFrame()
    calibration_audit_df = (
        pd.concat(calibration_audits, ignore_index=True) if calibration_audits else pd.DataFrame()
    )

    # Compare primary and undersampled effects at the prespecified 2% trim.
    if not main_oneD_df.empty and not undersample_main_df.empty:
        key = ["cluster", "stage", "treatment"]
        effect_cols = [
            "theta_per_1sd", "se_per_1sd", "ci_low_per_1sd",
            "ci_high_per_1sd", "p_value", "q_global",
            "fdr_global_pass", "n_used", "n_matches",
        ]
        primary_keep = main_oneD_df[key + effect_cols].rename(columns={
            c: f"primary_{c}" for c in effect_cols
        })
        undersampled_keep = undersample_main_df[key + effect_cols].rename(
            columns={c: f"undersampled_{c}" for c in effect_cols}
        )
        undersample_compare_df = primary_keep.merge(
            undersampled_keep, on=key, how="inner"
        )
        undersample_compare_df["direction_same"] = (
            np.sign(undersample_compare_df["primary_theta_per_1sd"])
            == np.sign(undersample_compare_df["undersampled_theta_per_1sd"])
        )
        undersample_compare_df["absolute_effect_change"] = np.abs(
            undersample_compare_df["undersampled_theta_per_1sd"]
            - undersample_compare_df["primary_theta_per_1sd"]
        )
        undersample_compare_df["relative_absolute_effect_change"] = (
            undersample_compare_df["absolute_effect_change"]
            / undersample_compare_df["primary_theta_per_1sd"].abs().replace(
                0, np.nan
            )
        )

        def imbalance_classification(row):
            if not bool(row["direction_same"]):
                return "DIRECTION_SENSITIVE"
            relative_change = row["relative_absolute_effect_change"]
            if (
                np.isfinite(relative_change)
                and relative_change > UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG
            ):
                return "MAGNITUDE_SENSITIVE_DIRECTION_RETAINED"
            if not bool(row["undersampled_fdr_global_pass"]):
                return "DIRECTION_ROBUST_FDR_NOT_RETAINED"
            return "ROBUST"

        undersample_compare_df["imbalance_sensitivity_class"] = (
            undersample_compare_df.apply(
                imbalance_classification, axis=1
            )
        )
        undersample_compare_df["imbalance_sensitivity_flag"] = (
            undersample_compare_df["imbalance_sensitivity_class"] != "ROBUST"
        )
        undersample_compare_df["undersampling_target_ratio"] = (
            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
        )
    else:
        undersample_compare_df = pd.DataFrame()

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
    if not reviewer_full_df.empty and not undersample_compare_df.empty:
        add_cols = [
            "cluster", "stage", "treatment",
            "undersampled_theta_per_1sd",
            "undersampled_se_per_1sd",
            "undersampled_ci_low_per_1sd",
            "undersampled_ci_high_per_1sd",
            "undersampled_p_value",
            "undersampled_q_global",
            "undersampled_fdr_global_pass",
            "direction_same",
            "absolute_effect_change",
            "relative_absolute_effect_change",
            "imbalance_sensitivity_class",
            "imbalance_sensitivity_flag",
            "undersampling_target_ratio",
        ]
        reviewer_full_df = reviewer_full_df.merge(
            undersample_compare_df[add_cols],
            on=["cluster", "stage", "treatment"],
            how="left",
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
    reviewer_checklist_df = pd.concat([
        reviewer_checklist_df,
        pd.DataFrame([{
            "reviewer_requirement": "Class-imbalance sensitivity",
            "reported_evidence": (
                "RF outcome nuisance refitted after 1:2 majority "
                "undersampling inside every inner/outer training fold; "
                "calibration-validation and outer-test rows retain the "
                "original class distribution"
            ),
            "workbook_location": (
                "29_undersample_oneD / 30_undersample_compare / "
                "31_undersample_y_perf / 32_undersample_sampling / "
                "33_undersample_calibration"
            ),
            "status": (
                "REPORTED"
                if not undersample_compare_df.empty
                else "NOT_RUN"
            ),
        }]),
    ], ignore_index=True)

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
        "analysis_stages": "L only",
        "L_estimand": "one defending Shot-location L KPI conditional on reduced pre-shot background plus permitted Att(L) KPIs; Avg_Def removes all Avg_Att plus DistToAttCentroid; DistToDefCentroid removes DistToAttCentroid plus all Avg_Att; every other Def KPI and every E/E' KPI is excluded from X",
        "Eprime_estimand": "not run",
        "temporal_rule": "only period, match time, pre-shot score/context, categorical controls, and permitted Att(L) KPIs enter X; no endpoint coordinate, E KPI, E' KPI, shot geometry, duration, or endpoint-derived variable is calculated or used",
        "numeric_background_controls_L": " | ".join(L_BASE_CONTROL_COLS),
        "numeric_background_controls_Eprime": "not used",
        "explicitly_excluded_controls": " | ".join(EXCLUDED_EXACT_SPATIAL_CONTROL_COLS),
        "categorical_background_sources": " | ".join(f"{k}:{v}" for k, v in CATEGORICAL_CONTROL_COLS.items()),
        "single_kpi_control_policy": "L only: reduced background + permitted Att(L) only; Avg_Def removes all Avg_Att plus DistToAttCentroid; DistToDefCentroid removes DistToAttCentroid plus all Avg_Att; every other Def KPI, every E/E' KPI, and all exact location/endpoint/geometry/duration/FF-count variables are excluded from X",
        "same_series_att_exclusion": EXCLUDE_SAME_SERIES_ATT_CONTROLS,
        "same_series_att_cross_stage_exclusion": False,
        "avg_centroid_reciprocal_exclusion": EXCLUDE_AVG_AND_CENTROID_TOGETHER,
        "defending_kpis_in_X": "none",
        "kpi_family_patterns_for_reporting_only": json.dumps(KPI_FAMILY_PATTERNS, ensure_ascii=False),
        "same_series_rule": "within L only: Avg_Def -> all Avg_Att plus DistToAttCentroid; DistToDefCentroid -> DistToAttCentroid plus all Avg_Att; Area_Def -> Area_Att; Spr_Def -> Spr_Att; Adv has no Att-only counterpart",
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
        "undersampling_sensitivity": (
            "RF outcome nuisance only; random majority undersampling inside "
            "every inner/outer training fold; inner validation and outer test "
            "rows unchanged"
            if RUN_UNDERSAMPLING_SENSITIVITY else "not run"
        ),
        "undersampling_active_stages": " | ".join(
            UNDERSAMPLING_ACTIVE_STAGES
        ),
        "undersampling_target_minority_to_majority_ratio": (
            UNDERSAMPLING_MINORITY_TO_MAJORITY_RATIO
        ),
        "undersampling_magnitude_change_flag": (
            UNDERSAMPLING_MAGNITUDE_CHANGE_FLAG
        ),
        "undersampling_treatment_model_policy": (
            "reuse primary RF D|X OOF residuals to isolate outcome-class "
            "handling"
        ),
        "treatment_nuisance_metrics": "OOF R2 | RMSE | MAE | normalized RMSE/SD",
        "overlap_diagnostic_note": "no universal continuous-treatment cut-off; FAIL is a prespecified severe residual-support flag; CAUTION is descriptive only",
        "overlap_fail_rule": f"OOF D R2 >= {OVERLAP_R2_HIGH} AND residual SD ratio <= {OVERLAP_RESID_RATIO_LOW}",
        "overlap_caution_rule": f"OOF D R2 >= {OVERLAP_R2_HIGH} AND {OVERLAP_RESID_RATIO_LOW} < residual SD ratio <= {OVERLAP_RESID_RATIO_CAUTION}",
        "common_support_report": f"raw treatment distribution across {SUPPORT_BINS} predicted-treatment quantile bins",
        "support_threshold_sensitivity_grid": f"R2={OVERLAP_R2_SENSITIVITY_GRID}; residual_SD_ratio={OVERLAP_RESID_RATIO_SENSITIVITY_GRID}",
        "second_stage": "unit-weight residual WLS with HC3 heteroskedasticity-robust standard errors",
        "joint_DML": "not run in this single-KPI script",
        "nonlinear_joint_check": "not run in this single-KPI script",
        "interpretation_note": (
            "Only Shot-location L KPIs are analysed. No endpoint-stage, "
            "shot-geometry, duration, or endpoint-derived variable enters "
            "the treatment or covariate sets."
        ),
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
            ("29_undersample_oneD", undersample_df),
            ("30_undersample_compare", undersample_compare_df),
            ("31_undersample_y_perf", undersample_y_perf_df),
            ("32_undersample_sampling", undersample_sampling_audit_df),
            ("33_undersample_calibration", undersample_calibration_audit_df),
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


if __name__ == "__main__":
    main()
