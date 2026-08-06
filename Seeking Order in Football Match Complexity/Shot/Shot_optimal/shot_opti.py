# -*- coding: utf-8 -*-
"""
Shot L · DML-informed independent multi-KPI GA policy optimization
==================================================================

This script treats policy optimization as a NEW stage after DML.
It does NOT refit DML and does NOT use DML coefficients as the GA fitness.

This Shot version is L-stage only. It retains the leakage-safe feature allow-list
and integer decision constraints, and adds the reviewer-facing robustness analyses
required for the independent policy stage: a context-only predictive baseline,
alternative policy learners, both global and within-match conditional
label-permutation diagnostics, explicit optimal-strategy value tables, and GA
random-seed stability. It does not pass every
raw table column into the policy model, because outcome-derived and downstream event
fields can leak defensive success and create an impossible AUC of 1.0.

Workflow
--------
1. Read the completed Shot L single-KPI and rotating joint-DML workbooks.
2. Cross-check them and build a cluster-specific candidate pool containing only
   KPIs that pass both stages at the policy threshold q<0.01. Training-fold
   undersampling is a sensitivity analysis and is not an admission gate when it
   was not triggered because the class ratio was already at least 1:2.
3. Select at most one KPI per available tactical family for the main analysis:
      local advantage    : Adv_5 / Adv_10
      defensive distance : Avg_1/3/5_Def / DistToDefCentroid
      structure/shape    : Area_Def / Spr_Def
   The default rule is the largest absolute joint effect per 1 SD among eligible
   candidates. A cluster may legitimately retain fewer than three families.
4. End the DML stage.
5. Train an independent nonlinear defensive-success prediction model on all
   successful and failed events, using match-grouped outer cross-fitting.
6. Optimize only failed held-out events. A genetic algorithm jointly changes the
   selected KPI(s) to maximize out-of-fold model-predicted defensive-success
   probability under DML-informed direction constraints, empirical quantile
   bounds, an L1 action budget and multivariate support restrictions.
7. Compare the joint action with the best one-KPI action when more than one KPI
   is available, and report
   model performance, calibration, sensitivity to class undersampling, action
   budgets and action bounds.
8. Compare the full policy model against a context-only baseline, rerun policy
   optimization with RF and XGBoost alternatives, perform global and within-match
   conditional label-permutation diagnostics, audit GA random-seed stability, and
   export the exact event-specific and cluster-level optimal strategy values.

Interpretation
--------------
The output is a DML-informed, model-based cooperative policy analysis. DML is
used for KPI admission and direction constraints. The estimated policy gain is
an out-of-fold model-implied probability improvement, not a newly estimated DML
causal effect and not external/prospective validation.

Examples
--------
Run the Shot L policy analysis::

    python shot_L_DML筛选后_独立预测模型_多KPI联合GA策略_v1最终稳健性.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

RAW_INPUT_PATH = (
    r""
)
SINGLE_RESULTS_XLSX = (
    r""
)
ROTATING_JOINT_RESULTS_XLSX = (
    r""
)
OUTPUT_XLSX = (
    r""
)

PIPELINE_VERSION = "shot_L_independent_policy_v1_1_variable_k_2026-08-01"
STAGE = "L"
RANDOM_SEED = 42
POLICY_FDR_ALPHA = 0.01

# Main KPI selection. All passed KPIs form the candidate pool; the main policy
# changes at most one eligible KPI from each tactical family. Missing families
# are skipped rather than filled with DML-ineligible reference companions.
SELECTION_MODE = "max_abs_joint_effect_per_1sd"
EXPECTED_FAMILIES = ("local_advantage", "distance_centroid", "structure_shape")
FINAL_KPI_OVERRIDE: Dict[int, Tuple[str, ...]] = {}
REQUIRE_ALL_FAMILIES = False
ALLOW_NONELIGIBLE_OVERRIDE = False
ACTIVE_SELECTION_RULE = SELECTION_MODE

# Outer match-grouped policy evaluation.
OUTER_FOLDS = 5
INNER_CALIBRATION_FOLDS = 3

# Primary independent policy model.
HGB_MAX_ITER = 300
HGB_LEARNING_RATE = 0.04
HGB_MAX_LEAF_NODES = 31
HGB_MIN_SAMPLES_LEAF = 20
HGB_L2 = 1.0
ONEHOT_MIN_FREQUENCY = 10

# Reviewer-facing predictive and policy robustness. HGB remains the primary
# learner. RF and XGBoost refit the complete out-of-fold policy and independently
# re-optimize the same held-out failed events under identical constraints.
PRIMARY_POLICY_LEARNER = "HGB"
RUN_CONTEXT_ONLY_BASELINE = True
RUN_ALTERNATIVE_POLICY_LEARNERS = True
ALTERNATIVE_POLICY_LEARNERS = ("RF", "XGB")

RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_FEATURES = "sqrt"

XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.04
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE_BYTREE = 0.80
XGB_MIN_CHILD_WEIGHT = 5.0
XGB_REG_LAMBDA = 1.0

# Label-permutation diagnostics. The global permutation is the primary leakage/no-signal
# null: training labels are randomly permuted across the complete training fold, while
# held-out labels remain untouched. Its OOF ROC-AUC should be close to 0.50. The
# within-match permutation is retained as a secondary conditional diagnostic that
# preserves match-level success-rate structure and therefore is NOT expected to be 0.50.
# Neither test is a causal placebo.
RUN_GLOBAL_LABEL_PERMUTATION = True
GLOBAL_LABEL_PERMUTATION_REPS = 100
RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION = True
WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS = 20
GLOBAL_PERMUTATION_NULL_AUC_MAX = 0.60
# Permutation fits are a leakage/no-signal diagnostic rather than the policy
# model itself. Reuse one encoded feature matrix per outer fold, disable HGB's
# additional validation split, cap null-fit iterations, and collect cyclic
# sklearn objects periodically to prevent Windows memory growth across 600 fits.
PERMUTATION_HGB_MAX_ITER = 100
PERMUTATION_GC_INTERVAL = 5

# GA stochastic-search stability on a prespecified held-out subset. Each sampled
# failed event is independently re-optimized under several seeds using the same
# fitted primary model and constraints.
RUN_GA_SEED_STABILITY = True
GA_STABILITY_SEEDS = (101, 202, 303, 404, 505)
GA_STABILITY_MAX_ROWS_PER_FOLD = 100

# Leakage-safe policy feature set. Only variables available as context at the
# tactical decision point are allowed. The selected defensive KPI(s) are
# appended separately as actionable features.
SAFE_NUMERIC_CONTEXT_COLS = [
    "period",
    "match_second",
    "event_team_score_before",
    "event_opponent_score_before",
]
SAFE_CATEGORICAL_CONTEXT_COLS = [
    "team",
    "position_name",
    "play_pattern",
    "event_score_state_before",
    "shot_team_home_away",
]
INCLUDE_SAFE_ATTACKING_KPI_CONTROLS = True
MISSING_CATEGORY_TOKEN = "__MISSING__"

# Stop instead of optimizing when OOF performance is implausibly perfect.
LEAKAGE_STOP_AUC = 0.98
LEAKAGE_STOP_BRIER = 0.05

FORBIDDEN_FEATURE_TOKENS = (
    "success", "outcome", "result", "label", "target", "next_", "post_",
    "after", "possession_change", "turnover", "lost", "loss",
    "pass_outcome", "shot_outcome", "ball_receipt_outcome", "end_reason",
    "defensive_success",
)

# Genetic algorithm.
GA_POPULATION = 48
GA_GENERATIONS = 60
GA_CROSSOVER_PROB = 0.80
GA_MUTATION_PROB = 0.30
GA_ELITISM = 2
GA_TOURNAMENT_SIZE = 3
GA_MUTATION_SCALE = 0.12
GA_EARLY_STOPPING_PATIENCE = 15
GA_EARLY_STOPPING_TOL = 1e-7
GA_BATCH_SIZE = 64

# Main action restrictions.
MAIN_ACTION_QUANTILES = (0.20, 0.80)
MAIN_L1_BUDGET_SD = 3.0
SUPPORT_QUANTILE = 0.95
SUPPORT_RIDGE = 1e-6
PROBABILITY_CLIP_EPS = 1e-6
MIN_POLICY_IMPROVEMENT = 1e-8
SINGLE_KPI_GRID_POINTS = 21

# Count-based actionable KPIs must remain integer-valued during optimization.
# Shot may select either Adv_5 or Adv_10; both are count-valued and integer-valued.
DISCRETE_ACTION_PATTERNS = (
    r"^Adv_5\(L\)$",
    r"^Adv_10\(L\)$",
    r"^Adv_5$",
    r"^Adv_10$",
)
INTEGER_TOLERANCE = 1e-6

# Sensitivity analyses.
RUN_BUDGET_SENSITIVITY = True
BUDGET_SENSITIVITY_GRID = (1.0, 2.0, 3.0)
RUN_BOUND_SENSITIVITY = True
BOUND_SENSITIVITY_GRID = ((0.10, 0.90), (0.20, 0.80), (0.30, 0.70))
RUN_TRAINING_ONLY_UNDERSAMPLING_SENSITIVITY = True
UNDERSAMPLE_TARGET_POS_TO_NEG = 0.50
MATCH_BOOTSTRAP_REPS = 500

# Debug only. Do not use capped rows for manuscript results.
MAX_FAILED_ROWS_PER_CLUSTER: Optional[int] = None
QUICK_SMOKE = False


# =============================================================================
# 2. DATA CLASSES AND GENERAL HELPERS
# =============================================================================

@dataclass
class SupportModel:
    mean: np.ndarray
    sd: np.ndarray
    inverse_covariance: np.ndarray
    mahalanobis_threshold: float
    lower_quantile: np.ndarray
    upper_quantile: np.ndarray


@dataclass
class FittedPolicyModel:
    preprocessor: ColumnTransformer
    estimator: Any
    calibrator: Optional[LogisticRegression]
    learner_name: str
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    action_columns: List[str]
    action_encoded_indices: np.ndarray
    action_numeric_means: np.ndarray
    action_numeric_scales: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.preprocessor.transform(frame[self.feature_columns]), dtype=np.float32)

    def predict_encoded(self, encoded: np.ndarray) -> np.ndarray:
        raw = np.clip(self.estimator.predict_proba(encoded)[:, 1], 1e-6, 1 - 1e-6)
        if self.calibrator is None:
            return raw
        z = logit(raw).reshape(-1, 1)
        return np.clip(self.calibrator.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6)

    def action_raw_to_encoded(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        return (raw - self.action_numeric_means) / self.action_numeric_scales

    def predict_counterfactual_encoded(
        self,
        base_encoded: np.ndarray,
        action_raw: np.ndarray,
    ) -> np.ndarray:
        """Predict one counterfactual action vector per row."""
        out = np.asarray(base_encoded, dtype=np.float32).copy()
        out[:, self.action_encoded_indices] = self.action_raw_to_encoded(action_raw).astype(np.float32)
        return self.predict_encoded(out)


def as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return default
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed", ""}:
        return False
    return default


def safe_float(value, default=np.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")


def cluster_seed(cluster: int, offset: int = 0) -> int:
    return int(RANDOM_SEED + int(cluster) * 1009 + offset * 7919)


def detect_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalize_cluster(value):
    try:
        x = float(value)
        return int(x) if x.is_integer() else x
    except Exception:
        return value


def dml_family_to_policy_family(value: str) -> str:
    text = str(value).strip().lower()
    if "local" in text or "advantage" in text:
        return "local_advantage"
    if "distance" in text or "centroid" in text:
        return "distance_centroid"
    if "structure" in text or "shape" in text:
        return "structure_shape"
    return text


def direction_code_from_row(row: pd.Series) -> int:
    if as_bool(row.get("robust_conditional_sign_change"), False):
        return 2
    label = str(row.get("policy_constraint_class", "")).upper()
    if "NEGATIVE" in label or "DECREASE" in label:
        return -1
    if "POSITIVE" in label or "INCREASE" in label:
        return 1
    estimate = safe_float(row.get("estimate"), 0.0)
    return 1 if estimate > 0 else (-1 if estimate < 0 else 0)


def direction_label(code: int) -> str:
    if code == 2:
        return "two_sided"
    if code > 0:
        return "increase_only"
    if code < 0:
        return "decrease_only"
    return "no_change"


def infer_discrete_action_mask(action_columns: Sequence[str]) -> np.ndarray:
    """Return True for actionable KPI columns that must take integer values."""
    mask = []
    for column in action_columns:
        name = str(column).strip()
        mask.append(any(re.fullmatch(pattern, name) for pattern in DISCRETE_ACTION_PATTERNS))
    return np.asarray(mask, dtype=bool)


def validate_and_round_observed_discrete(
    observed: np.ndarray,
    discrete_mask: np.ndarray,
    action_columns: Sequence[str],
) -> np.ndarray:
    """Validate observed count KPIs and store them as exact integers."""
    values = np.asarray(observed, dtype=float).copy()
    discrete_mask = np.asarray(discrete_mask, dtype=bool)
    for j in np.flatnonzero(discrete_mask):
        rounded = np.rint(values[:, j])
        bad = np.isfinite(values[:, j]) & (np.abs(values[:, j] - rounded) > INTEGER_TOLERANCE)
        if np.any(bad):
            examples = values[bad, j][:5].tolist()
            raise ValueError(
                f"Discrete action KPI {action_columns[j]} contains non-integer observed values: {examples}. "
                "Check the KPI construction before policy optimization."
            )
        values[:, j] = rounded
    return values


def integerize_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    observed: np.ndarray,
    discrete_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert discrete action intervals to valid integer intervals."""
    lo = np.asarray(lower, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    obs = np.asarray(observed, dtype=float)
    for j in np.flatnonzero(np.asarray(discrete_mask, dtype=bool)):
        lo[:, j] = np.minimum(np.ceil(lo[:, j]), obs[:, j])
        hi[:, j] = np.maximum(np.floor(hi[:, j]), obs[:, j])
        invalid = lo[:, j] > hi[:, j]
        lo[invalid, j] = obs[invalid, j]
        hi[invalid, j] = obs[invalid, j]
    return lo, hi


def integerize_population(
    population: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    discrete_mask: np.ndarray,
) -> np.ndarray:
    """Round discrete genes and re-apply row-specific bounds."""
    pop = np.asarray(population, dtype=float).copy()
    discrete_mask = np.asarray(discrete_mask, dtype=bool)
    if np.any(discrete_mask):
        pop[..., discrete_mask] = np.rint(pop[..., discrete_mask])
    return np.clip(pop, lower[:, None, :], upper[:, None, :])


# =============================================================================
# 3. KPI CANDIDATE POOL AND MAIN SELECTION
# =============================================================================



def _single_pass_flag(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct the Shot policy gate and always enforce q<0.01."""
    q = pd.to_numeric(frame.get("q_global"), errors="coerce")
    q_pass = q.lt(POLICY_FDR_ALPHA)
    if "candidate_for_discussion" in frame.columns:
        base = frame["candidate_for_discussion"].map(as_bool)
    elif "reviewer_primary" in frame.columns:
        base = frame["reviewer_primary"].map(as_bool)
    elif "reviewer_evidence_status" in frame.columns:
        base = frame["reviewer_evidence_status"].astype(str).str.upper().eq("PRIMARY")
    elif "q_global" in frame.columns:
        base = pd.Series(True, index=frame.index)
    else:
        raise ValueError(
            "Single-KPI workbook has no recognized screening field "
            "(candidate_for_discussion / reviewer_primary / reviewer_evidence_status / q_global)."
        )
    return base & q_pass


def canonical_stage_label(value: Any) -> str:
    """Normalize workbook labels; the Shot policy subsequently retains L only."""
    text = str(value).strip().replace("′", "'")
    compact = re.sub(r"[\s_-]+", "", text).lower()
    if compact == "l":
        return "L"
    if compact in {"eprime", "e'"}:
        return "E'"
    return text


def _normalize_single_results(path: str) -> pd.DataFrame:
    single = pd.read_excel(path, sheet_name="02_reviewer_full_report")
    required = {"cluster", "stage", "treatment", "theta_per_1sd"}
    missing = sorted(required - set(single.columns))
    if missing:
        raise ValueError(f"Single-KPI workbook is missing required columns: {missing}")

    out = single.copy()
    out["cluster"] = out["cluster"].map(normalize_cluster)
    out["stage"] = out["stage"].map(canonical_stage_label)
    out = out[out["stage"].eq(STAGE)].copy()
    out = out.rename(columns={
        "treatment": "target",
        "theta_per_1sd": "single_effect_per_1sd",
        "q_global": "single_q_global",
    })
    out["single_stage_pass"] = _single_pass_flag(single.loc[out.index])
    out["single_direction_code"] = np.sign(
        pd.to_numeric(out["single_effect_per_1sd"], errors="coerce")
    ).fillna(0).astype(int)

    keep = [
        "cluster", "stage", "target", "single_effect_per_1sd",
        "single_q_global", "single_stage_pass", "single_direction_code",
    ]
    optional = [
        "robustness_value_alpha05", "trim_sign_consistency",
        "direction_same", "imbalance_sensitivity_class",
    ]
    return out[[c for c in keep + optional if c in out.columns]].copy()


def _normalize_joint_results(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheets = set(xls.sheet_names)

    if "02_target_AME_report" in sheets:
        joint = pd.read_excel(path, sheet_name="02_target_AME_report")
        required = {"cluster", "stage", "target", "target_family", "estimate"}
        missing = sorted(required - set(joint.columns))
        if missing:
            raise ValueError(
                "Shot-style rotating joint workbook is missing required columns: "
                + " | ".join(missing)
            )
        out = joint.copy()
        out["joint_schema"] = "rotating_AME_report"
        out["joint_effect_per_1sd"] = pd.to_numeric(out["estimate"], errors="coerce")
        q_source = "target_q_global" if "target_q_global" in out.columns else "q_global"
        out["joint_q_global"] = pd.to_numeric(out[q_source], errors="coerce")

        # Shot undersampling is a sensitivity analysis, not an admission gate.
        # If no training fold was imbalanced enough to trigger sampling,
        # optimizer_eligible_after_undersampling is false by construction.
        if "eligible_for_interaction_aware_optimizer" in out.columns:
            base_joint_pass = out["eligible_for_interaction_aware_optimizer"].map(as_bool)
        elif "eligible_as_average_policy_lever" in out.columns:
            base_joint_pass = out["eligible_as_average_policy_lever"].map(as_bool)
        elif "fdr_global_pass" in out.columns:
            base_joint_pass = out["fdr_global_pass"].map(as_bool)
        else:
            base_joint_pass = pd.Series(True, index=out.index)
        direction_pass = (
            out["single_to_joint_direction_same"].map(as_bool)
            if "single_to_joint_direction_same" in out.columns
            else pd.Series(True, index=out.index)
        )
        out["joint_stage_pass"] = (
            base_joint_pass
            & direction_pass
            & out["joint_q_global"].lt(POLICY_FDR_ALPHA)
        )
        out["direction_code"] = out.apply(direction_code_from_row, axis=1)
        if "joint_kpis" in out.columns:
            out["joint_specification"] = out["joint_kpis"]
        elif "joint_spec" in out.columns:
            out["joint_specification"] = out["joint_spec"]
        else:
            out["joint_specification"] = ""
        out["joint_direction_same_reported"] = (
            out["single_to_joint_direction_same"]
            if "single_to_joint_direction_same" in out.columns else np.nan
        )
        out["joint_robustness_gate_source"] = (
            "primary joint eligibility + target q<0.01 + direction; "
            "undersampling sensitivity is not an admission gate"
        )

    elif "26_joint_selection" in sheets:
        joint = pd.read_excel(path, sheet_name="26_joint_selection")
        required = {
            "cluster", "stage", "target", "target_family",
            "joint_theta_per_1sd",
        }
        missing = sorted(required - set(joint.columns))
        if missing:
            raise ValueError(
                "Legacy joint workbook is missing required columns: "
                + " | ".join(missing)
            )
        out = joint.copy()
        out["joint_schema"] = "pass_joint_selection"
        out["joint_effect_per_1sd"] = pd.to_numeric(
            out["joint_theta_per_1sd"], errors="coerce"
        )
        out["joint_q_global"] = pd.to_numeric(
            out["joint_q_global_target"], errors="coerce"
        ) if "joint_q_global_target" in out.columns else np.nan

        if "selected_for_multi_kpi_optimization" in out.columns:
            out["joint_stage_pass"] = out[
                "selected_for_multi_kpi_optimization"
            ].map(as_bool)
            gate_note = "selected_for_multi_kpi_optimization"
        elif "joint_pass_primary" in out.columns:
            out["joint_stage_pass"] = out["joint_pass_primary"].map(as_bool)
            gate_note = "joint_pass_primary"
        else:
            required_gate = {
                "joint_fdr_global_pass",
                "sign_consistent_with_single",
                "estimability_flag",
            }
            missing_gate = sorted(required_gate - set(out.columns))
            if missing_gate:
                raise ValueError(
                    "26_joint_selection lacks a recognized primary pass flag and "
                    f"cannot reconstruct it; missing {missing_gate}."
                )
            out["joint_stage_pass"] = (
                out["joint_fdr_global_pass"].map(as_bool)
                & out["sign_consistent_with_single"].map(as_bool)
                & ~out["estimability_flag"].map(as_bool)
            )
            gate_note = "global FDR + direction + no estimability flag"

        out["direction_code"] = np.sign(out["joint_effect_per_1sd"]).fillna(0).astype(int)
        if "joint_treatments" in out.columns:
            out["joint_specification"] = out["joint_treatments"]
        elif "model_key" in out.columns:
            out["joint_specification"] = out["model_key"]
        else:
            out["joint_specification"] = ""
        out["joint_direction_same_reported"] = (
            out["sign_consistent_with_single"]
            if "sign_consistent_with_single" in out.columns else np.nan
        )
        out["joint_robustness_gate_source"] = gate_note

    else:
        raise ValueError(
            "Joint workbook must contain either '02_target_AME_report' "
            "(new rotating robust format) or '26_joint_selection' "
            "(legacy unified-family format)."
        )

    out["joint_stage_pass"] = (
        out["joint_stage_pass"].map(as_bool)
        & pd.to_numeric(out["joint_q_global"], errors="coerce").lt(POLICY_FDR_ALPHA)
    )
    out["cluster"] = out["cluster"].map(normalize_cluster)
    out["stage"] = out["stage"].map(canonical_stage_label)
    out = out[out["stage"].eq(STAGE)].copy()
    out["policy_family"] = out["target_family"].map(dml_family_to_policy_family)
    out["direction_label"] = out["direction_code"].map(direction_label)
    return out


def load_candidate_pool(single_path: str, joint_path: str) -> pd.DataFrame:
    """
    Cross-check Shot L against BOTH completed DML analyses.

    A KPI enters the policy candidate pool only when:
      1. it passes the single-KPI DML screening gate;
      2. it passes the available joint-DML primary gate; and
      3. its single and joint effect directions agree.

    The joint loader supports both the newer rotating AME workbook and the
    legacy unified-family joint-selection workbook.
    """
    single = _normalize_single_results(single_path)
    joint = _normalize_joint_results(joint_path)

    # The robust rotating joint workbook can itself contain audit columns named
    # single_stage_pass / single_q_global.  Preserve those as joint-report audit
    # fields, while keeping the independently reloaded single-KPI columns under
    # their canonical unsuffixed names.  Without explicit suffixes, pandas creates
    # _x/_y columns and the admission gate cannot find single_stage_pass.
    out = joint.merge(
        single,
        on=["cluster", "stage", "target"],
        how="left",
        validate="one_to_one",
        suffixes=("_joint_report", ""),
    )
    if "single_stage_pass" not in out.columns:
        raise ValueError(
            "Internal merge error: the normalized single-KPI pass flag was not "
            "preserved. Available columns include: "
            + " | ".join(sorted(map(str, out.columns)))
        )
    if out["single_stage_pass"].isna().any():
        missing = out.loc[out["single_stage_pass"].isna(), ["cluster", "target"]]
        raise ValueError(
            "Some joint targets were not found in the single-KPI workbook: "
            + " | ".join(
                f"C{int(r.cluster)}:{r.target}" for r in missing.itertuples()
            )
        )

    computed_same = (
        np.sign(pd.to_numeric(out["joint_effect_per_1sd"], errors="coerce"))
        == pd.to_numeric(out["single_direction_code"], errors="coerce")
    )
    reported = out["joint_direction_same_reported"]
    if reported.notna().any():
        reported_bool = reported.map(lambda x: as_bool(x, default=True))
        direction_same = computed_same & reported_bool
    else:
        direction_same = computed_same

    out["single_to_joint_direction_same"] = direction_same
    out["passed_single_and_joint"] = (
        out["single_stage_pass"].map(as_bool)
        & out["joint_stage_pass"].map(as_bool)
        & out["single_to_joint_direction_same"].map(as_bool)
    )

    # Standard names used by the common policy code.
    out["estimate"] = out["joint_effect_per_1sd"]
    out["q_global"] = out["joint_q_global"]
    out["target_family"] = out["policy_family"]
    return out.sort_values(
        ["cluster", "policy_family", "target"]
    ).reset_index(drop=True)


def select_main_kpis(candidate_pool: pd.DataFrame) -> Tuple[Dict[int, List[str]], pd.DataFrame]:
    selected: Dict[int, List[str]] = {}
    audit_rows: List[dict] = []

    for cluster in sorted(pd.unique(candidate_pool["cluster"])):
        if cluster in FINAL_KPI_OVERRIDE:
            kpis = list(FINAL_KPI_OVERRIDE[cluster])
            if not kpis:
                raise ValueError(f"FINAL_KPI_OVERRIDE[{cluster}] cannot be empty.")
            cluster_audit: List[dict] = []
            noneligible: List[str] = []
            seen_families: set[str] = set()
            for kpi in kpis:
                match = candidate_pool[
                    (candidate_pool["cluster"] == cluster)
                    & (candidate_pool["target"].astype(str) == str(kpi))
                ]
                if match.empty:
                    raise ValueError(
                        f"Manual override KPI is absent from the DML candidate pool: "
                        f"Cluster {cluster}, {kpi}."
                    )
                if len(match) != 1:
                    raise ValueError(
                        f"Manual override KPI is not unique: Cluster {cluster}, {kpi}."
                    )
                row = match.iloc[0]
                actual_family = str(row["policy_family"])
                if actual_family not in EXPECTED_FAMILIES:
                    raise ValueError(
                        f"Unsupported policy family for Cluster {cluster}, {kpi}: "
                        f"{actual_family}."
                    )
                if actual_family in seen_families:
                    raise ValueError(
                        f"FINAL_KPI_OVERRIDE[{cluster}] contains more than one KPI "
                        f"from family {actual_family}."
                    )
                seen_families.add(actual_family)
                strictly_eligible = as_bool(row["passed_single_and_joint"])
                if not strictly_eligible:
                    noneligible.append(kpi)
                audit = row.to_dict()
                audit.update({
                    "selected_kpi": kpi,
                    "selection_rule": ACTIVE_SELECTION_RULE,
                    "manual_override_used": True,
                    "strictly_eligible_before_override": strictly_eligible,
                    "noneligible_override_authorized": ALLOW_NONELIGIBLE_OVERRIDE,
                    "n_eligible_in_family": int(
                        (
                            (candidate_pool["cluster"] == cluster)
                            & (candidate_pool["policy_family"] == actual_family)
                            & candidate_pool["passed_single_and_joint"].map(as_bool)
                        ).sum()
                    ),
                })
                cluster_audit.append(audit)
            if noneligible and not ALLOW_NONELIGIBLE_OVERRIDE:
                raise RuntimeError(
                    f"Cluster {cluster} manual override contains KPI(s) that failed "
                    f"the strict single+joint DML admission gate: {' | '.join(noneligible)}."
                )
            if noneligible:
                warnings.warn(
                    f"Exploratory Shot L override in Cluster {cluster}: "
                    f"{' | '.join(noneligible)} did not pass every strict DML gate. "
                    "See 01_main_KPI_selection for the failed-gate audit."
                )
            selected[int(cluster)] = kpis
            audit_rows.extend(cluster_audit)
            continue

        cluster_rows = candidate_pool[
            (candidate_pool["cluster"] == cluster)
            & candidate_pool["passed_single_and_joint"]
        ].copy()
        chosen: List[str] = []
        for family in EXPECTED_FAMILIES:
            family_rows = cluster_rows[cluster_rows["policy_family"] == family].copy()
            if family_rows.empty:
                if REQUIRE_ALL_FAMILIES:
                    raise RuntimeError(
                        f"Cluster {cluster} has no eligible KPI in family {family}."
                    )
                continue
            family_rows["abs_joint_effect"] = family_rows["estimate"].abs()
            sort_cols = ["abs_joint_effect"]
            ascending = [False]
            if "robustness_value_alpha05" in family_rows.columns:
                sort_cols.append("robustness_value_alpha05")
                ascending.append(False)
            if "q_global" in family_rows.columns:
                sort_cols.append("q_global")
                ascending.append(True)
            family_rows = family_rows.sort_values(sort_cols, ascending=ascending)
            row = family_rows.iloc[0]
            chosen.append(str(row["target"]))
            audit = row.to_dict()
            audit.update({
                "selected_kpi": row["target"],
                "selection_rule": SELECTION_MODE,
                "manual_override_used": False,
                "strictly_eligible_before_override": True,
                "noneligible_override_authorized": False,
                "n_eligible_in_family": int(len(family_rows)),
            })
            audit_rows.append(audit)
        if not chosen:
            raise RuntimeError(
                f"Cluster {cluster} has no KPI passing both Shot L DML stages at "
                f"q<{POLICY_FDR_ALPHA:.2f}."
            )
        selected[int(cluster)] = chosen

    return selected, pd.DataFrame(audit_rows)


# =============================================================================
# 4. RAW DATA AND POLICY FEATURE CONSTRUCTION
# =============================================================================

DEFENSIVE_STAGE_PATTERN = re.compile(
    r"^(Adv_\d+|Avg_\d+_Def|Area_Def|Spr_Def|DistToDefCentroid)\(L\)$",
    flags=re.IGNORECASE,
)


def _read_raw_table(path: str) -> Tuple[pd.DataFrame, str]:
    """Read a raw Shot table from CSV or Excel and return (frame, source_sheet)."""
    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False), ""

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        xls = pd.ExcelFile(path)
        if not xls.sheet_names:
            raise ValueError(f"Excel raw input contains no worksheets: {path}")

        # Prefer a sheet that visibly contains outcome, cluster and match-group
        # columns.  This makes multi-sheet workbooks safe while remaining fast.
        detection_candidates = []
        for sheet in xls.sheet_names:
            preview = pd.read_excel(path, sheet_name=sheet, nrows=5)
            outcome = detect_column(preview.columns, ["success_def", "outcome", "y"])
            cluster = detect_column(preview.columns, ["cluster_id", "cluster", "Cluster"])
            match = detect_column(
                preview.columns,
                ["__source_file__", "match_id", "match", "source_file", "file", "game_id"],
            )
            if outcome is not None and cluster is not None and match is not None:
                detection_candidates.append(sheet)

        if not detection_candidates:
            raise ValueError(
                "Could not find an Excel worksheet containing outcome, cluster "
                "and match-group columns. Checked sheets: "
                + " | ".join(map(str, xls.sheet_names))
            )
        sheet = detection_candidates[0]
        return pd.read_excel(path, sheet_name=sheet), str(sheet)

    raise ValueError(
        f"Unsupported raw input format '{suffix}'. Use CSV, XLSX, XLS or XLSM."
    )


def load_raw_data(path: str) -> Tuple[pd.DataFrame, str, str, str]:
    df, source_sheet = _read_raw_table(path)
    outcome_col = detect_column(df.columns, ["success_def", "outcome", "y"])
    cluster_col = detect_column(df.columns, ["cluster_id", "cluster", "Cluster"])
    match_col = detect_column(
        df.columns,
        ["__source_file__", "match_id", "match", "source_file", "file", "game_id"],
    )
    if outcome_col is None or cluster_col is None or match_col is None:
        raise ValueError(
            "Could not detect outcome, cluster or match-group column. "
            f"Detected outcome={outcome_col}, cluster={cluster_col}, match={match_col}."
        )
    df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce")
    df[cluster_col] = df[cluster_col].map(normalize_cluster)
    df = df[df[outcome_col].isin([0, 1]) & df[cluster_col].notna() & df[match_col].notna()].copy()
    df["__source_row_index__"] = np.arange(len(df), dtype=int)
    return df, outcome_col, cluster_col, match_col


def _strip_stage_suffix(name: str) -> str:
    text = str(name).strip()
    return re.sub(r"\(L\)$", "", text, flags=re.IGNORECASE)


def _is_safe_attacking_kpi(name: str) -> bool:
    text = str(name).strip()
    if not re.search(r"\(L\)$", text, flags=re.IGNORECASE):
        return False
    base = _strip_stage_suffix(text)
    return bool(
        re.match(r"^Avg_\d+_Att$", base, flags=re.IGNORECASE)
        or re.match(r"^DistToAttCentroid$", base, flags=re.IGNORECASE)
        or re.match(r"^Area_Att$", base, flags=re.IGNORECASE)
        or re.match(r"^Spr_Att$", base, flags=re.IGNORECASE)
    )


def _matched_attacking_proxy_excluded(name: str, selected_kpis: Sequence[str]) -> bool:
    """Mirror the DML same-family attacking-control exclusions."""
    base = _strip_stage_suffix(name)
    selected_bases = [_strip_stage_suffix(x) for x in selected_kpis]
    has_distance = any(
        re.match(r"^Avg_\d+_Def$", x, flags=re.IGNORECASE)
        or re.match(r"^DistToDefCentroid$", x, flags=re.IGNORECASE)
        for x in selected_bases
    )
    if has_distance and (
        re.match(r"^Avg_\d+_Att$", base, flags=re.IGNORECASE)
        or re.match(r"^DistToAttCentroid$", base, flags=re.IGNORECASE)
    ):
        return True
    if any(re.match(r"^Area_Def$", x, flags=re.IGNORECASE) for x in selected_bases):
        if re.match(r"^Area_Att$", base, flags=re.IGNORECASE):
            return True
    if any(re.match(r"^Spr_Def$", x, flags=re.IGNORECASE) for x in selected_bases):
        if re.match(r"^Spr_Att$", base, flags=re.IGNORECASE):
            return True
    return False


def build_feature_columns(
    df: pd.DataFrame,
    selected_kpis: Sequence[str],
    outcome_col: str,
    cluster_col: str,
    match_col: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Create an explicit leakage-safe policy feature allow-list."""
    selected_kpis = list(selected_kpis)
    missing = [c for c in selected_kpis if c not in df.columns]
    if missing:
        raise ValueError(
            f"Selected KPI columns are missing from raw input: {missing}. "
            f"Check that the raw CSV/Excel worksheet contains the full Shot {STAGE} table."
        )

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    excluded_columns: List[str] = []

    for col in SAFE_NUMERIC_CONTEXT_COLS:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if converted.notna().sum() > 0 and converted.nunique(dropna=True) > 1:
                df[col] = converted.astype(float)
                numeric_columns.append(col)

    if INCLUDE_SAFE_ATTACKING_KPI_CONTROLS:
        for col in df.columns:
            if not _is_safe_attacking_kpi(col):
                continue
            if _matched_attacking_proxy_excluded(col, selected_kpis):
                excluded_columns.append(str(col))
                continue
            converted = pd.to_numeric(df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if converted.notna().sum() > 0 and converted.nunique(dropna=True) > 1:
                df[col] = converted.astype(float)
                numeric_columns.append(str(col))

    for kpi in selected_kpis:
        df[kpi] = pd.to_numeric(df[kpi], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).astype(float)
        numeric_columns.append(kpi)

    for col in SAFE_CATEGORICAL_CONTEXT_COLS:
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            # sklearn 1.5's SimpleImputer can raise "boolean value of NA is
            # ambiguous" when a pandas StringDtype column contains pd.NA.
            # Replace missing categories explicitly and use ordinary Python
            # strings before any grouped fold is sliced.
            values = df[col].astype(object)
            values = values.where(pd.notna(values), MISSING_CATEGORY_TOKEN)
            df[col] = values.map(str).astype(object)
            categorical_columns.append(col)

    numeric_columns = list(dict.fromkeys(numeric_columns))
    categorical_columns = list(dict.fromkeys(categorical_columns))
    feature_columns = numeric_columns + categorical_columns

    forbidden = [
        col for col in feature_columns
        if any(token in str(col).lower() for token in FORBIDDEN_FEATURE_TOKENS)
        and col not in selected_kpis
    ]
    if forbidden:
        raise RuntimeError(
            "Leakage audit rejected policy features: " + " | ".join(forbidden)
        )
    if not feature_columns:
        raise RuntimeError("No leakage-safe policy features were available.")

    return feature_columns, numeric_columns, categorical_columns, sorted(excluded_columns)


# =============================================================================
# 5. INDEPENDENT PREDICTIVE POLICY MODEL
# =============================================================================


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n = len(y)
    n1 = max(int((y == 1).sum()), 1)
    n0 = max(int((y == 0).sum()), 1)
    return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(float)


def undersample_training_indices(
    y: np.ndarray,
    target_pos_to_neg: float,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    target_neg = int(round(len(pos) / max(float(target_pos_to_neg), 1e-6)))
    target_neg = min(len(neg), max(target_neg, len(pos)))
    rng = np.random.RandomState(seed)
    keep_neg = rng.choice(neg, size=target_neg, replace=False)
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return idx


def make_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=ONEHOT_MIN_FREQUENCY,
            sparse_output=False,
            dtype=np.float32,
        )),
    ])
    transformers = [("num", numeric_pipeline, list(numeric_columns))]
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, list(categorical_columns)))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def available_policy_learners() -> List[str]:
    learners = ["HGB", "RF"]
    if XGBOOST_AVAILABLE:
        learners.append("XGB")
    return learners


def make_estimator(
    learner_name: str,
    seed: int,
    permutation_mode: bool = False,
):
    learner = str(learner_name).upper()
    if learner == "HGB":
        return HistGradientBoostingClassifier(
            learning_rate=HGB_LEARNING_RATE,
            max_iter=(
                min(HGB_MAX_ITER, PERMUTATION_HGB_MAX_ITER)
                if permutation_mode else HGB_MAX_ITER
            ),
            max_leaf_nodes=HGB_MAX_LEAF_NODES,
            min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
            l2_regularization=HGB_L2,
            random_state=seed,
            early_stopping=not permutation_mode,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
    if learner == "RF":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            class_weight=None,
            bootstrap=True,
            n_jobs=4,
            random_state=seed,
        )
    if learner == "XGB":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost was requested but xgboost is not installed.")
        return XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            min_child_weight=XGB_MIN_CHILD_WEIGHT,
            reg_lambda=XGB_REG_LAMBDA,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(f"Unknown policy learner: {learner_name}")


def fit_encoded_probability_estimator(
    encoded_X: np.ndarray,
    y: np.ndarray,
    seed: int,
    learner_name: str = PRIMARY_POLICY_LEARNER,
    permutation_mode: bool = False,
) -> Any:
    """Fit one classifier on an already encoded matrix."""
    fit_y = np.asarray(y, dtype=int)
    estimator = make_estimator(
        learner_name,
        seed,
        permutation_mode=permutation_mode,
    )
    weights = np.asarray(balanced_sample_weights(fit_y), dtype=np.float32)
    estimator.fit(encoded_X, fit_y, sample_weight=weights)
    return estimator


def fit_raw_probability_model(
    train_df: pd.DataFrame,
    y: np.ndarray,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
    undersample: bool = False,
    learner_name: str = PRIMARY_POLICY_LEARNER,
) -> Tuple[ColumnTransformer, Any]:
    idx = np.arange(len(train_df))
    if undersample:
        idx = undersample_training_indices(y, UNDERSAMPLE_TARGET_POS_TO_NEG, seed)
    fit_df = train_df.iloc[idx]
    fit_y = np.asarray(y)[idx]
    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    X = np.asarray(preprocessor.fit_transform(fit_df[list(feature_columns)]), dtype=np.float32)
    estimator = make_estimator(learner_name, seed)
    # Primary models use balanced outcome weights. In the undersampling
    # sensitivity model, the training distribution has already been changed, so
    # no second class-reweighting is applied.
    weights = np.ones(len(fit_y), dtype=float) if undersample else balanced_sample_weights(fit_y)
    estimator.fit(X, fit_y, sample_weight=weights)
    return preprocessor, estimator


def predict_raw(
    preprocessor: ColumnTransformer,
    estimator: Any,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    X = np.asarray(preprocessor.transform(frame[list(feature_columns)]), dtype=np.float32)
    return np.clip(estimator.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def fit_policy_model(
    train_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    action_columns: Sequence[str],
    seed: int,
    undersample: bool = False,
    learner_name: str = PRIMARY_POLICY_LEARNER,
) -> FittedPolicyModel:
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    unique_groups = pd.unique(groups)
    n_inner = min(INNER_CALIBRATION_FOLDS, len(unique_groups))
    oof_raw = np.full(len(train_df), np.nan, dtype=float)

    if n_inner >= 2 and len(np.unique(y)) == 2:
        inner = GroupKFold(n_splits=n_inner)
        for fold, (tr, va) in enumerate(inner.split(train_df, y, groups)):
            pre, est = fit_raw_probability_model(
                train_df.iloc[tr], y[tr], feature_columns, numeric_columns,
                categorical_columns, seed + 100 + fold, undersample=undersample,
                learner_name=learner_name,
            )
            inner_pred = predict_raw(
                pre,
                est,
                train_df.iloc[va],
                feature_columns,
            )
            oof_raw[va] = inner_pred
            del inner_pred, pre, est
        gc.collect()

    calibrator: Optional[LogisticRegression] = None
    valid = np.isfinite(oof_raw)
    if valid.sum() >= 100 and len(np.unique(y[valid])) == 2:
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(logit(np.clip(oof_raw[valid], 1e-6, 1 - 1e-6)).reshape(-1, 1), y[valid])

    preprocessor, estimator = fit_raw_probability_model(
        train_df, y, feature_columns, numeric_columns, categorical_columns,
        seed + 999, undersample=undersample, learner_name=learner_name,
    )

    numeric_pipeline = preprocessor.named_transformers_["num"]
    scaler: StandardScaler = numeric_pipeline.named_steps["scaler"]
    action_indices = np.array(
        [list(numeric_columns).index(c) for c in action_columns], dtype=int
    ) if action_columns else np.array([], dtype=int)
    action_means = np.asarray(scaler.mean_, dtype=float)[action_indices]
    action_scales = np.asarray(scaler.scale_, dtype=float)[action_indices]
    action_scales[~np.isfinite(action_scales) | (action_scales <= 1e-12)] = 1.0

    return FittedPolicyModel(
        preprocessor=preprocessor,
        estimator=estimator,
        calibrator=calibrator,
        learner_name=str(learner_name).upper(),
        feature_columns=list(feature_columns),
        numeric_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
        action_columns=list(action_columns),
        action_encoded_indices=action_indices,
        action_numeric_means=action_means,
        action_numeric_scales=action_scales,
    )


# =============================================================================
# 6. PREDICTIVE PERFORMANCE DIAGNOSTICS
# =============================================================================


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit(p).reshape(-1, 1), y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def performance_row(y: np.ndarray, p: np.ndarray, **metadata) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y, p)
    row = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_10bin": expected_calibration_error(y, p, 10),
    }
    row.update(metadata)
    return row


# =============================================================================
# 7. SUPPORT, ACTION BOUNDS AND GA
# =============================================================================


def fit_support_model(values: np.ndarray, quantiles: Tuple[float, float]) -> SupportModel:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Main policy support requires an n x k matrix with k >= 1.")
    if len(values) < 30:
        raise ValueError("Insufficient failed training rows for policy support.")
    n_actions = int(values.shape[1])
    mean = np.nanmean(values, axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("At least one selected KPI is entirely missing in failed training rows.")
    sd = np.nanstd(values, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    filled = np.where(np.isfinite(values), values, mean)
    z = (filled - mean) / sd
    cov = np.atleast_2d(np.cov(z, rowvar=False)).astype(float)
    if cov.shape != (n_actions, n_actions):
        raise ValueError(
            f"Unexpected support covariance shape {cov.shape}; expected "
            f"({n_actions}, {n_actions})."
        )
    cov = cov + SUPPORT_RIDGE * np.eye(n_actions)
    inv_cov = np.linalg.pinv(cov)
    md2 = np.einsum("ni,ij,nj->n", z, inv_cov, z)
    lo, hi = quantiles
    return SupportModel(
        mean=mean,
        sd=sd,
        inverse_covariance=inv_cov,
        mahalanobis_threshold=float(np.quantile(md2, SUPPORT_QUANTILE)),
        lower_quantile=np.nanquantile(values, lo, axis=0),
        upper_quantile=np.nanquantile(values, hi, axis=0),
    )


def mahalanobis_squared(raw: np.ndarray, support: SupportModel) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    z = (raw - support.mean) / support.sd
    return np.einsum("...i,ij,...j->...", z, support.inverse_covariance, z)


def directional_bounds(
    observed: np.ndarray,
    directions: np.ndarray,
    support: SupportModel,
    discrete_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    lower = observed.copy()
    upper = observed.copy()
    for j, code in enumerate(directions):
        if code == 2:
            lower[:, j] = np.minimum(observed[:, j], support.lower_quantile[j])
            upper[:, j] = np.maximum(observed[:, j], support.upper_quantile[j])
        elif code > 0:
            upper[:, j] = np.maximum(observed[:, j], support.upper_quantile[j])
        elif code < 0:
            lower[:, j] = np.minimum(observed[:, j], support.lower_quantile[j])
    return integerize_bounds(lower, upper, observed, discrete_mask)


def _enforce_l1_budget_with_discrete(
    population: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    discrete_mask: np.ndarray,
) -> np.ndarray:
    """Project onto the SD-scale L1 budget without fractional count actions."""
    pop = integerize_population(population, lower, upper, discrete_mask)
    obs = observed[:, None, :]
    discrete_mask = np.asarray(discrete_mask, dtype=bool)
    continuous_mask = ~discrete_mask

    if np.any(discrete_mask):
        max_steps = int(np.nanmax(np.abs(pop[..., discrete_mask] - obs[..., discrete_mask]))) + 2
        for _ in range(max(max_steps, 1)):
            discrete_l1 = np.sum(
                np.abs((pop[..., discrete_mask] - obs[..., discrete_mask]) / support.sd[discrete_mask]),
                axis=2,
            )
            bad = discrete_l1 > float(budget_sd) + 1e-12
            if not np.any(bad):
                break
            for j in np.flatnonzero(discrete_mask):
                delta = pop[..., j] - obs[..., j]
                pop[..., j] = np.where(
                    bad & (np.abs(delta) > 0),
                    pop[..., j] - np.sign(delta),
                    pop[..., j],
                )
            pop = integerize_population(pop, lower, upper, discrete_mask)

    discrete_l1 = np.zeros(pop.shape[:2], dtype=float)
    if np.any(discrete_mask):
        discrete_l1 = np.sum(
            np.abs((pop[..., discrete_mask] - obs[..., discrete_mask]) / support.sd[discrete_mask]),
            axis=2,
        )

    if np.any(continuous_mask):
        delta_z = (pop[..., continuous_mask] - obs[..., continuous_mask]) / support.sd[continuous_mask]
        continuous_l1 = np.sum(np.abs(delta_z), axis=2)
        remaining = np.maximum(float(budget_sd) - discrete_l1, 0.0)
        scale = np.minimum(1.0, remaining / np.maximum(continuous_l1, 1e-12))
        pop[..., continuous_mask] = (
            obs[..., continuous_mask]
            + delta_z * scale[..., None] * support.sd[continuous_mask]
        )

    return integerize_population(pop, lower, upper, discrete_mask)


def repair_population(
    population: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    discrete_mask: np.ndarray,
) -> np.ndarray:
    """Repair bounds, integer genes, L1 budget and multivariate support."""
    pop = _enforce_l1_budget_with_discrete(
        population, observed, lower, upper, support, budget_sd, discrete_mask
    )
    obs = observed[:, None, :]
    discrete_mask = np.asarray(discrete_mask, dtype=bool)
    continuous_mask = ~discrete_mask

    observed_md2 = mahalanobis_squared(observed, support)
    allowed = np.maximum(support.mahalanobis_threshold, observed_md2)
    violation = mahalanobis_squared(pop, support) > allowed[:, None] + 1e-10

    if np.any(violation):
        base = np.broadcast_to(obs, pop.shape).copy()
        if np.any(discrete_mask):
            base[..., discrete_mask] = pop[..., discrete_mask]
            max_steps = int(np.nanmax(np.abs(base[..., discrete_mask] - obs[..., discrete_mask]))) + 2
            for _ in range(max(max_steps, 1)):
                base_bad = mahalanobis_squared(base, support) > allowed[:, None] + 1e-10
                if not np.any(base_bad):
                    break
                for j in np.flatnonzero(discrete_mask):
                    delta = base[..., j] - obs[..., j]
                    base[..., j] = np.where(
                        base_bad & (np.abs(delta) > 0),
                        base[..., j] - np.sign(delta),
                        base[..., j],
                    )
                base = integerize_population(base, lower, upper, discrete_mask)
            pop[..., discrete_mask] = base[..., discrete_mask]

        if np.any(continuous_mask):
            original = pop.copy()
            original[..., discrete_mask] = base[..., discrete_mask]
            low = np.zeros(pop.shape[:2], dtype=float)
            high = np.ones(pop.shape[:2], dtype=float)
            for _ in range(20):
                mid = (low + high) / 2.0
                trial = base.copy()
                trial[..., continuous_mask] = (
                    base[..., continuous_mask]
                    + mid[..., None] * (original[..., continuous_mask] - base[..., continuous_mask])
                )
                feasible = mahalanobis_squared(trial, support) <= allowed[:, None] + 1e-10
                low = np.where(feasible, mid, low)
                high = np.where(feasible, high, mid)
            repaired = base.copy()
            repaired[..., continuous_mask] = (
                base[..., continuous_mask]
                + low[..., None] * (original[..., continuous_mask] - base[..., continuous_mask])
            )
            pop = np.where(violation[..., None], repaired, pop)
        else:
            pop = np.where(violation[..., None], base, pop)

    pop = _enforce_l1_budget_with_discrete(
        pop, observed, lower, upper, support, budget_sd, discrete_mask
    )
    return integerize_population(pop, lower, upper, discrete_mask)


def evaluate_population(
    model: FittedPolicyModel,
    base_encoded: np.ndarray,
    population_raw: np.ndarray,
) -> np.ndarray:
    batch, pop_size, n_actions = population_raw.shape
    repeated = np.repeat(base_encoded, pop_size, axis=0)
    flat_raw = population_raw.reshape(batch * pop_size, n_actions)
    repeated[:, model.action_encoded_indices] = model.action_raw_to_encoded(flat_raw).astype(np.float32)
    return model.predict_encoded(repeated).reshape(batch, pop_size)


def tournament_parent_indices(scores: np.ndarray, n_children: int, rng) -> np.ndarray:
    batch, pop_size = scores.shape
    candidates = rng.randint(0, pop_size, size=(batch, n_children, GA_TOURNAMENT_SIZE))
    candidate_scores = np.take_along_axis(scores[:, None, :], candidates, axis=2)
    winner_pos = np.argmax(candidate_scores, axis=2)
    return np.take_along_axis(candidates, winner_pos[:, :, None], axis=2)[:, :, 0]


def best_single_kpi_grid(
    model: FittedPolicyModel,
    base_encoded: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    discrete_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    batch = len(observed)
    candidates = [observed.copy()]
    labels = [-1]
    continuous_grid = np.linspace(0.0, 1.0, SINGLE_KPI_GRID_POINTS)

    for j in range(observed.shape[1]):
        if discrete_mask[j]:
            global_low = int(np.floor(np.nanmin(lower[:, j])))
            global_high = int(np.ceil(np.nanmax(upper[:, j])))
            for level in range(global_low, global_high + 1):
                candidate = observed.copy()
                candidate[:, j] = np.clip(level, lower[:, j], upper[:, j])
                candidate = repair_population(
                    candidate[:, None, :], observed, lower, upper, support,
                    budget_sd, discrete_mask,
                )[:, 0, :]
                candidates.append(candidate)
                labels.append(j)
        else:
            for alpha in continuous_grid[1:]:
                candidate = observed.copy()
                candidate[:, j] = lower[:, j] + alpha * (upper[:, j] - lower[:, j])
                candidate = repair_population(
                    candidate[:, None, :], observed, lower, upper, support,
                    budget_sd, discrete_mask,
                )[:, 0, :]
                candidates.append(candidate)
                labels.append(j)

    stack = np.stack(candidates, axis=1)
    scores = evaluate_population(model, base_encoded, stack)
    idx = np.argmax(scores, axis=1)
    return {
        "optimized_raw": stack[np.arange(batch), idx],
        "optimized_probability": scores[np.arange(batch), idx],
        "best_kpi_index": np.asarray(labels, dtype=int)[idx],
    }


def vectorized_ga(
    model: FittedPolicyModel,
    base_encoded: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    seed: int,
    discrete_mask: np.ndarray,
    seed_candidates: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    batch = len(observed)
    n_actions = int(observed.shape[1])
    if n_actions < 1:
        raise ValueError("GA requires at least one selected KPI.")
    rng = np.random.RandomState(seed)
    population = lower[:, None, :] + rng.rand(batch, GA_POPULATION, n_actions) * (
        upper - lower
    )[:, None, :]
    population[:, 0, :] = observed
    if seed_candidates is not None and GA_POPULATION > 1:
        seeds = np.asarray(seed_candidates, dtype=float)
        if seeds.ndim == 2:
            seeds = seeds[:, None, :]
        n_seed = min(seeds.shape[1], GA_POPULATION - 1)
        population[:, 1:1 + n_seed, :] = seeds[:, :n_seed, :]
    population = repair_population(
        population, observed, lower, upper, support, budget_sd, discrete_mask
    )

    best_score = model.predict_counterfactual_encoded(base_encoded, observed)
    best_raw = observed.copy()
    no_improve = 0
    generations_run = 0

    for generation in range(GA_GENERATIONS):
        generations_run = generation + 1
        scores = evaluate_population(model, base_encoded, population)
        best_idx = np.argmax(scores, axis=1)
        generation_best = scores[np.arange(batch), best_idx]
        improved = generation_best > best_score + GA_EARLY_STOPPING_TOL
        if improved.any():
            best_score[improved] = generation_best[improved]
            best_raw[improved] = population[improved, best_idx[improved], :]
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= GA_EARLY_STOPPING_PATIENCE:
            break

        elite_idx = np.argsort(scores, axis=1)[:, -GA_ELITISM:]
        elites = np.take_along_axis(population, elite_idx[:, :, None], axis=1)
        n_children = GA_POPULATION - GA_ELITISM
        p1_idx = tournament_parent_indices(scores, n_children, rng)
        p2_idx = tournament_parent_indices(scores, n_children, rng)
        p1 = np.take_along_axis(population, p1_idx[:, :, None], axis=1)
        p2 = np.take_along_axis(population, p2_idx[:, :, None], axis=1)
        alpha = rng.rand(batch, n_children, 1)
        crossed = alpha * p1 + (1 - alpha) * p2
        children = np.where(
            rng.rand(batch, n_children, 1) < GA_CROSSOVER_PROB,
            crossed,
            p1,
        )
        individual_mutation = rng.rand(batch, n_children, 1) < GA_MUTATION_PROB
        gene_mutation = rng.rand(batch, n_children, n_actions) < (1.0 / n_actions)
        scale = GA_MUTATION_SCALE * np.maximum(upper - lower, 1e-12)[:, None, :]
        children += rng.normal(size=children.shape) * scale * individual_mutation * gene_mutation
        population = np.concatenate([elites, children], axis=1)
        population = repair_population(
            population, observed, lower, upper, support, budget_sd, discrete_mask
        )

    final_scores = evaluate_population(model, base_encoded, population)
    final_idx = np.argmax(final_scores, axis=1)
    final_best = final_scores[np.arange(batch), final_idx]
    improved = final_best > best_score
    if improved.any():
        best_score[improved] = final_best[improved]
        best_raw[improved] = population[improved, final_idx[improved], :]
    return {
        "optimized_raw": best_raw,
        "optimized_probability": best_score,
        "generations_run": np.full(batch, generations_run, dtype=int),
    }


# =============================================================================
# 8. OUTER-FOLD POLICY ANALYSIS
# =============================================================================


def match_bootstrap_ci(values: np.ndarray, groups: np.ndarray, reps: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=object)
    ok = np.isfinite(values) & pd.notna(groups)
    values, groups = values[ok], groups[ok]
    unique = pd.unique(groups)
    if len(values) == 0 or len(unique) < 2:
        return {"ci_low": np.nan, "ci_high": np.nan, "reps": 0}
    by_group = {g: values[groups == g] for g in unique}
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws.append(np.mean(np.concatenate([by_group[g] for g in sampled])))
    return {
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "reps": int(reps),
    }


def optimize_failed_batch(
    model: FittedPolicyModel,
    eval_frame: pd.DataFrame,
    selected_kpis: Sequence[str],
    directions: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    seed: int,
    discrete_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    base_encoded = model.transform(eval_frame)
    observed = eval_frame[list(selected_kpis)].to_numpy(dtype=float)
    missing = ~np.isfinite(observed)
    observed = np.where(np.isfinite(observed), observed, support.mean)
    # A missing count KPI is imputed to the nearest integer training-support mean.
    for j in np.flatnonzero(discrete_mask):
        observed[missing[:, j], j] = np.rint(support.mean[j])
    observed = validate_and_round_observed_discrete(observed, discrete_mask, selected_kpis)
    lower, upper = directional_bounds(observed, directions, support, discrete_mask)
    single = best_single_kpi_grid(
        model, base_encoded, observed, lower, upper, support, budget_sd, discrete_mask
    )
    ga = vectorized_ga(
        model, base_encoded, observed, lower, upper, support, budget_sd,
        seed=seed, discrete_mask=discrete_mask,
        seed_candidates=single["optimized_raw"],
    )
    baseline = model.predict_counterfactual_encoded(base_encoded, observed)
    optimized = ga["optimized_probability"]
    no_gain = optimized <= baseline + MIN_POLICY_IMPROVEMENT
    ga["optimized_raw"][no_gain] = observed[no_gain]
    optimized[no_gain] = baseline[no_gain]

    if np.any(discrete_mask):
        for name, array in (("joint", ga["optimized_raw"]), ("best_single", single["optimized_raw"])):
            residual = np.abs(array[:, discrete_mask] - np.rint(array[:, discrete_mask]))
            if np.any(residual > INTEGER_TOLERANCE):
                raise RuntimeError(f"{name} policy contains non-integer discrete KPI values.")

    return {
        "observed_raw": observed,
        "lower_raw": lower,
        "upper_raw": upper,
        "baseline_probability": baseline,
        "optimized_raw": ga["optimized_raw"],
        "optimized_probability": optimized,
        "generations_run": ga["generations_run"],
        "best_single_raw": single["optimized_raw"],
        "best_single_probability": single["optimized_probability"],
        "best_single_kpi_index": single["best_kpi_index"],
    }


def global_permute_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Permute binary labels across the complete training fold, preserving prevalence."""
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)
    return rng.permutation(y)


def grouped_permute_labels(y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Permute binary labels within each match group (conditional diagnostic)."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=object)
    out = y.copy()
    rng = np.random.RandomState(seed)
    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def optimize_frame_in_batches(
    model: FittedPolicyModel,
    frame: pd.DataFrame,
    selected_kpis: Sequence[str],
    directions: np.ndarray,
    support: SupportModel,
    budget_sd: float,
    seed_base: int,
    discrete_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    outputs = []
    for start in range(0, len(frame), GA_BATCH_SIZE):
        batch = frame.iloc[start:start + GA_BATCH_SIZE]
        outputs.append(optimize_failed_batch(
            model, batch, selected_kpis, directions, support, budget_sd,
            seed_base + start * 17, discrete_mask,
        ))
    if not outputs:
        return {}
    return {key: np.concatenate([x[key] for x in outputs], axis=0) for key in outputs[0]}


def run_cluster_policy(
    cluster: int,
    cluster_df: pd.DataFrame,
    selected_kpis: Sequence[str],
    directions: np.ndarray,
    outcome_col: str,
    match_col: str,
    feature_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    y_all = cluster_df[outcome_col].to_numpy(dtype=int)
    groups_all = cluster_df[match_col].astype(str).to_numpy(dtype=object)
    discrete_mask = infer_discrete_action_mask(selected_kpis)
    n_splits = min(OUTER_FOLDS, len(pd.unique(groups_all)))
    if n_splits < 2:
        raise RuntimeError(f"Cluster {cluster} has fewer than two matches.")
    splitter = GroupKFold(n_splits=n_splits)

    primary_oof = np.full(len(cluster_df), np.nan)
    baseline_oof = np.full(len(cluster_df), np.nan)
    alternative_names = [
        x for x in ALTERNATIVE_POLICY_LEARNERS
        if x in available_policy_learners() and x != PRIMARY_POLICY_LEARNER
    ] if RUN_ALTERNATIVE_POLICY_LEARNERS else []
    alternative_oof = {name: np.full(len(cluster_df), np.nan) for name in alternative_names}
    global_permutation_oof = {
        rep: np.full(len(cluster_df), np.nan)
        for rep in range(GLOBAL_LABEL_PERMUTATION_REPS)
    } if RUN_GLOBAL_LABEL_PERMUTATION else {}
    conditional_permutation_oof = {
        rep: np.full(len(cluster_df), np.nan)
        for rep in range(WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS)
    } if RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION else {}

    individual_rows: List[dict] = []
    fold_rows: List[dict] = []
    sensitivity_rows: List[dict] = []
    alternative_policy_rows: List[dict] = []
    ga_stability_rows: List[dict] = []

    baseline_numeric = [c for c in numeric_columns if c not in selected_kpis]
    baseline_features = baseline_numeric + list(categorical_columns)
    if RUN_CONTEXT_ONLY_BASELINE and not baseline_features:
        raise RuntimeError("The context-only baseline has no eligible features.")

    for fold, (train_idx, eval_idx) in enumerate(splitter.split(cluster_df, y_all, groups_all)):
        train = cluster_df.iloc[train_idx].copy()
        evaluation = cluster_df.iloc[eval_idx].copy()
        y_train = y_all[train_idx]
        y_eval = y_all[eval_idx]
        train_groups = groups_all[train_idx]

        model = fit_policy_model(
            train, y_train, train_groups, feature_columns,
            numeric_columns, categorical_columns, selected_kpis,
            seed=cluster_seed(cluster, fold), undersample=False,
            learner_name=PRIMARY_POLICY_LEARNER,
        )
        encoded_eval = model.transform(evaluation)
        fold_probability = model.predict_encoded(encoded_eval)
        primary_oof[eval_idx] = fold_probability

        fold_auc = np.nan
        fold_brier = np.nan
        if len(np.unique(y_eval)) == 2:
            fold_auc = float(roc_auc_score(y_eval, fold_probability))
            fold_brier = float(brier_score_loss(y_eval, fold_probability))
            if fold_auc >= LEAKAGE_STOP_AUC and fold_brier <= LEAKAGE_STOP_BRIER:
                raise RuntimeError(
                    f"Predictive leakage guard triggered in Cluster {cluster}, fold {fold}: "
                    f"OOF AUC={fold_auc:.4f}, Brier={fold_brier:.6f}. "
                    "Optimization stopped. Inspect the explicit feature audit and raw data definitions."
                )

        if RUN_CONTEXT_ONLY_BASELINE:
            baseline_model = fit_policy_model(
                train, y_train, train_groups, baseline_features,
                baseline_numeric, categorical_columns, [],
                seed=cluster_seed(cluster, 3000 + fold), undersample=False,
                learner_name=PRIMARY_POLICY_LEARNER,
            )
            baseline_encoded_eval = baseline_model.transform(evaluation)
            baseline_oof[eval_idx] = baseline_model.predict_encoded(
                baseline_encoded_eval
            )
            del baseline_encoded_eval, baseline_model
            gc.collect()

        # Calibration is deliberately omitted for permutation fits. ROC-AUC and
        # PR-AUC are invariant to monotone Platt calibration; Brier is reported as
        # an exploratory raw-probability null.
        permutation_preprocessor = None
        permutation_train_X = None
        permutation_eval_X = None
        if RUN_GLOBAL_LABEL_PERMUTATION or RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION:
            permutation_preprocessor = make_preprocessor(
                numeric_columns,
                categorical_columns,
            )
            permutation_train_X = np.asarray(
                permutation_preprocessor.fit_transform(train[list(feature_columns)]),
                dtype=np.float32,
                order="C",
            )
            permutation_eval_X = np.asarray(
                permutation_preprocessor.transform(evaluation[list(feature_columns)]),
                dtype=np.float32,
                order="C",
            )

        if RUN_GLOBAL_LABEL_PERMUTATION:
            for rep in range(GLOBAL_LABEL_PERMUTATION_REPS):
                perm_y = global_permute_labels(
                    y_train, cluster_seed(cluster, 60000 + rep * 100 + fold)
                )
                perm_est = fit_encoded_probability_estimator(
                    permutation_train_X,
                    perm_y,
                    seed=cluster_seed(cluster, 70000 + rep * 100 + fold),
                    learner_name=PRIMARY_POLICY_LEARNER,
                    permutation_mode=True,
                )
                perm_pred = np.clip(
                    perm_est.predict_proba(permutation_eval_X)[:, 1],
                    1e-6,
                    1 - 1e-6,
                )
                global_permutation_oof[rep][eval_idx] = perm_pred
                del perm_y, perm_pred, perm_est
                if (rep + 1) % PERMUTATION_GC_INTERVAL == 0:
                    gc.collect()

        if RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION:
            for rep in range(WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS):
                perm_y = grouped_permute_labels(
                    y_train, train_groups, cluster_seed(cluster, 80000 + rep * 100 + fold)
                )
                perm_est = fit_encoded_probability_estimator(
                    permutation_train_X,
                    perm_y,
                    seed=cluster_seed(cluster, 90000 + rep * 100 + fold),
                    learner_name=PRIMARY_POLICY_LEARNER,
                    permutation_mode=True,
                )
                perm_pred = np.clip(
                    perm_est.predict_proba(permutation_eval_X)[:, 1],
                    1e-6,
                    1 - 1e-6,
                )
                conditional_permutation_oof[rep][eval_idx] = perm_pred
                del perm_y, perm_pred, perm_est
                if (rep + 1) % PERMUTATION_GC_INTERVAL == 0:
                    gc.collect()

        if permutation_preprocessor is not None:
            del permutation_preprocessor, permutation_train_X, permutation_eval_X
            gc.collect()

        # Alternative policy learners are fitted only after all permutation
        # diagnostics so their RF/XGB objects do not raise the permutation peak.
        alt_models: Dict[str, FittedPolicyModel] = {}
        for alt_idx, learner_name in enumerate(alternative_names):
            alt_model = fit_policy_model(
                train, y_train, train_groups, feature_columns,
                numeric_columns, categorical_columns, selected_kpis,
                seed=cluster_seed(cluster, 4000 + alt_idx * 100 + fold),
                undersample=False, learner_name=learner_name,
            )
            alt_models[learner_name] = alt_model
            alt_encoded_eval = alt_model.transform(evaluation)
            alt_prob = alt_model.predict_encoded(alt_encoded_eval)
            alternative_oof[learner_name][eval_idx] = alt_prob
            if len(np.unique(y_eval)) == 2:
                alt_auc = float(roc_auc_score(y_eval, alt_prob))
                alt_brier = float(brier_score_loss(y_eval, alt_prob))
                if alt_auc >= LEAKAGE_STOP_AUC and alt_brier <= LEAKAGE_STOP_BRIER:
                    raise RuntimeError(
                        f"Leakage guard triggered for {learner_name} in Cluster {cluster}, fold {fold}: "
                        f"AUC={alt_auc:.4f}, Brier={alt_brier:.6f}."
                    )
            del alt_encoded_eval, alt_prob

        failed_train = train[train[outcome_col] == 0]
        if len(failed_train) < 30:
            raise RuntimeError(f"Cluster {cluster}, fold {fold}: insufficient failed training rows.")
        support = fit_support_model(
            failed_train[list(selected_kpis)].to_numpy(dtype=float),
            MAIN_ACTION_QUANTILES,
        )
        failed_eval_mask = evaluation[outcome_col].eq(0).to_numpy()
        failed_eval = evaluation.loc[failed_eval_mask].copy()
        if MAX_FAILED_ROWS_PER_CLUSTER is not None and len(failed_eval) > MAX_FAILED_ROWS_PER_CLUSTER:
            failed_eval = failed_eval.sample(
                n=MAX_FAILED_ROWS_PER_CLUSTER,
                random_state=cluster_seed(cluster, 500 + fold),
            ).copy()
        if failed_eval.empty:
            if alt_models:
                del alt_model
            del alt_models, model, encoded_eval, fold_probability
            gc.collect()
            continue

        main = optimize_frame_in_batches(
            model, failed_eval, selected_kpis, directions, support,
            MAIN_L1_BUDGET_SD, cluster_seed(cluster, 1000 + fold * 100),
            discrete_mask,
        )

        under_baseline = np.full(len(failed_eval), np.nan)
        under_optimized = np.full(len(failed_eval), np.nan)
        if RUN_TRAINING_ONLY_UNDERSAMPLING_SENSITIVITY:
            under_model = fit_policy_model(
                train, y_train, train_groups, feature_columns,
                numeric_columns, categorical_columns, selected_kpis,
                seed=cluster_seed(cluster, 8000 + fold), undersample=True,
                learner_name=PRIMARY_POLICY_LEARNER,
            )
            under_encoded = under_model.transform(failed_eval)
            under_baseline = under_model.predict_counterfactual_encoded(
                under_encoded, main["observed_raw"]
            )
            under_optimized = under_model.predict_counterfactual_encoded(
                under_encoded, main["optimized_raw"]
            )
            del under_encoded, under_model
            gc.collect()

        # Independently re-optimize the same held-out failed rows under each
        # alternative learner and compare both probability gain and action vector.
        for alt_idx, (learner_name, alt_model) in enumerate(alt_models.items()):
            alt_result = optimize_frame_in_batches(
                alt_model, failed_eval, selected_kpis, directions, support,
                MAIN_L1_BUDGET_SD,
                cluster_seed(cluster, 12000 + alt_idx * 1000 + fold * 100),
                discrete_mask,
            )
            for i, (_, row) in enumerate(failed_eval.iterrows()):
                main_change = (main["optimized_raw"][i] - main["observed_raw"][i]) / support.sd
                alt_change = (alt_result["optimized_raw"][i] - alt_result["observed_raw"][i]) / support.sd
                main_sign = np.sign(main_change)
                alt_sign = np.sign(alt_change)
                sign_agreement = float(np.mean(main_sign == alt_sign))
                alt_improvement = (
                    alt_result["optimized_probability"][i]
                    - alt_result["baseline_probability"][i]
                )
                alt_single = (
                    alt_result["best_single_probability"][i]
                    - alt_result["baseline_probability"][i]
                )
                rec = {
                    "cluster": cluster,
                    "outer_fold": fold,
                    "source_row_index": int(row["__source_row_index__"]),
                    "match": str(row[match_col]),
                    "learner": learner_name,
                    "baseline_probability_oof": float(alt_result["baseline_probability"][i]),
                    "optimized_probability_oof": float(alt_result["optimized_probability"][i]),
                    "predicted_probability_improvement": float(alt_improvement),
                    "best_single_improvement": float(alt_single),
                    "joint_minus_best_single_gain": float(alt_improvement - alt_single),
                    "main_HGB_improvement": float(
                        main["optimized_probability"][i] - main["baseline_probability"][i]
                    ),
                    "improvement_positive": bool(alt_improvement > MIN_POLICY_IMPROVEMENT),
                    "action_direction_agreement_with_HGB": sign_agreement,
                    "mean_abs_action_difference_SD_vs_HGB": float(np.mean(np.abs(alt_change - main_change))),
                }
                for j, kpi in enumerate(selected_kpis):
                    rec.update({
                        f"optimized::{kpi}": float(alt_result["optimized_raw"][i, j]),
                        f"change_sd::{kpi}": float(alt_change[j]),
                        f"direction_same_as_HGB::{kpi}": bool(alt_sign[j] == main_sign[j]),
                    })
                alternative_policy_rows.append(rec)
            del alt_result

        if alt_models:
            del alt_model
        del alt_models
        gc.collect()

        # GA random-seed stability on a fixed held-out subset. The reference is
        # the main HGB solution already calculated for the same event.
        if RUN_GA_SEED_STABILITY:
            n_stability = min(GA_STABILITY_MAX_ROWS_PER_FOLD, len(failed_eval))
            rng = np.random.RandomState(cluster_seed(cluster, 16000 + fold))
            positions = np.sort(rng.choice(len(failed_eval), size=n_stability, replace=False))
            stability_frame = failed_eval.iloc[positions].copy()
            reference_actions = main["optimized_raw"][positions]
            reference_improvement = (
                main["optimized_probability"][positions]
                - main["baseline_probability"][positions]
            )
            for seed_id, seed_value in enumerate(GA_STABILITY_SEEDS):
                stability_result = optimize_frame_in_batches(
                    model, stability_frame, selected_kpis, directions, support,
                    MAIN_L1_BUDGET_SD,
                    cluster_seed(cluster, 17000 + fold * 1000 + seed_value),
                    discrete_mask,
                )
                changes = (
                    stability_result["optimized_raw"] - stability_result["observed_raw"]
                ) / support.sd
                reference_changes = (
                    reference_actions - stability_result["observed_raw"]
                ) / support.sd
                for local_i, (_, row) in enumerate(stability_frame.iterrows()):
                    improvement = (
                        stability_result["optimized_probability"][local_i]
                        - stability_result["baseline_probability"][local_i]
                    )
                    rec = {
                        "cluster": cluster,
                        "outer_fold": fold,
                        "source_row_index": int(row["__source_row_index__"]),
                        "match": str(row[match_col]),
                        "ga_seed_label": seed_id,
                        "ga_seed_value": int(seed_value),
                        "predicted_probability_improvement": float(improvement),
                        "reference_main_improvement": float(reference_improvement[local_i]),
                        "improvement_difference_vs_reference": float(
                            improvement - reference_improvement[local_i]
                        ),
                        "action_direction_agreement_vs_reference": float(
                            np.mean(np.sign(changes[local_i]) == np.sign(reference_changes[local_i]))
                        ),
                        "mean_abs_action_difference_SD_vs_reference": float(
                            np.mean(np.abs(changes[local_i] - reference_changes[local_i]))
                        ),
                    }
                    for j, kpi in enumerate(selected_kpis):
                        rec[f"optimized::{kpi}"] = float(stability_result["optimized_raw"][local_i, j])
                        rec[f"change_sd::{kpi}"] = float(changes[local_i, j])
                    ga_stability_rows.append(rec)
                del stability_result

        for i, (_, row) in enumerate(failed_eval.iterrows()):
            observed = main["observed_raw"][i]
            optimized = main["optimized_raw"][i]
            change_sd = (optimized - observed) / support.sd
            improvement = main["optimized_probability"][i] - main["baseline_probability"][i]
            single_improvement = (
                main["best_single_probability"][i] - main["baseline_probability"][i]
            )
            record = {
                "cluster": cluster,
                "stage": STAGE,
                "outer_fold": fold,
                "source_row_index": int(row["__source_row_index__"]),
                "match": str(row[match_col]),
                "observed_outcome": int(row[outcome_col]),
                "baseline_probability_oof": float(main["baseline_probability"][i]),
                "optimized_probability_oof": float(main["optimized_probability"][i]),
                "predicted_probability_improvement": float(improvement),
                "best_single_probability_oof": float(main["best_single_probability"][i]),
                "best_single_improvement": float(single_improvement),
                "joint_minus_best_single_gain": float(improvement - single_improvement),
                "best_single_kpi_index": int(main["best_single_kpi_index"][i]),
                "best_single_kpi": (
                    selected_kpis[int(main["best_single_kpi_index"][i])]
                    if int(main["best_single_kpi_index"][i]) >= 0 else "NO_ACTION"
                ),
                "L1_budget_sd": MAIN_L1_BUDGET_SD,
                "L1_used_sd": float(np.sum(np.abs(change_sd))),
                "budget_binding": bool(np.sum(np.abs(change_sd)) >= MAIN_L1_BUDGET_SD - 1e-4),
                "observed_mahalanobis_sq": float(mahalanobis_squared(observed[None, :], support)[0]),
                "optimized_mahalanobis_sq": float(mahalanobis_squared(optimized[None, :], support)[0]),
                "support_threshold": float(support.mahalanobis_threshold),
                "support_pass": bool(
                    mahalanobis_squared(optimized[None, :], support)[0]
                    <= max(support.mahalanobis_threshold,
                           mahalanobis_squared(observed[None, :], support)[0]) + 1e-8
                ),
                "generations_run": int(main["generations_run"][i]),
                "undersampled_baseline_probability": float(under_baseline[i]) if np.isfinite(under_baseline[i]) else np.nan,
                "undersampled_optimized_probability": float(under_optimized[i]) if np.isfinite(under_optimized[i]) else np.nan,
                "undersampled_improvement": float(under_optimized[i] - under_baseline[i]) if np.isfinite(under_baseline[i]) else np.nan,
                "undersampling_direction_same": bool(
                    np.sign(under_optimized[i] - under_baseline[i]) == np.sign(improvement)
                ) if np.isfinite(under_baseline[i]) else np.nan,
            }
            active_bound_hit = False
            for j, kpi in enumerate(selected_kpis):
                changed = abs(optimized[j] - observed[j]) > 1e-8
                hit = changed and (
                    abs(optimized[j] - main["lower_raw"][i, j]) <= 1e-6
                    or abs(optimized[j] - main["upper_raw"][i, j]) <= 1e-6
                )
                active_bound_hit = active_bound_hit or hit
                record.update({
                    f"observed::{kpi}": float(observed[j]),
                    f"optimized::{kpi}": float(optimized[j]),
                    f"change_raw::{kpi}": float(optimized[j] - observed[j]),
                    f"change_sd::{kpi}": float(change_sd[j]),
                    f"direction_constraint::{kpi}": direction_label(int(directions[j])),
                    f"action_variable_type::{kpi}": "integer" if discrete_mask[j] else "continuous",
                    f"integer_constraint_pass::{kpi}": bool(
                        (not discrete_mask[j])
                        or abs(optimized[j] - round(optimized[j])) <= INTEGER_TOLERANCE
                    ),
                    f"lower_action_bound::{kpi}": float(main["lower_raw"][i, j]),
                    f"upper_action_bound::{kpi}": float(main["upper_raw"][i, j]),
                    f"active_bound_hit::{kpi}": bool(hit),
                })
            record["any_active_bound_hit"] = bool(active_bound_hit)
            individual_rows.append(record)

        fold_rows.append({
            "cluster": cluster,
            "outer_fold": fold,
            "n_train": int(len(train)),
            "n_eval": int(len(evaluation)),
            "n_failed_eval_optimized": int(len(failed_eval)),
            "n_train_matches": int(train[match_col].nunique()),
            "n_eval_matches": int(evaluation[match_col].nunique()),
            "train_positive_rate": float(y_train.mean()),
            "eval_positive_rate": float(y_eval.mean()),
            "primary_learner": PRIMARY_POLICY_LEARNER,
            "alternative_learners": " | ".join(alternative_names),
            "primary_fold_auc": fold_auc,
            "primary_fold_brier": fold_brier,
            "selected_kpis": " | ".join(selected_kpis),
            "discrete_action_kpis": " | ".join(
                [k for k, flag in zip(selected_kpis, discrete_mask) if flag]
            ),
            "action_quantiles": str(MAIN_ACTION_QUANTILES),
            "L1_budget_sd": MAIN_L1_BUDGET_SD,
        })

        sensitivity_specs: List[Tuple[str, Tuple[float, float], float]] = []
        if RUN_BUDGET_SENSITIVITY:
            for budget in BUDGET_SENSITIVITY_GRID:
                if abs(budget - MAIN_L1_BUDGET_SD) > 1e-12:
                    sensitivity_specs.append((f"BUDGET_{budget:g}SD", MAIN_ACTION_QUANTILES, float(budget)))
        if RUN_BOUND_SENSITIVITY:
            for bounds in BOUND_SENSITIVITY_GRID:
                if tuple(bounds) != tuple(MAIN_ACTION_QUANTILES):
                    sensitivity_specs.append((f"BOUNDS_Q{int(bounds[0]*100)}_Q{int(bounds[1]*100)}", tuple(bounds), MAIN_L1_BUDGET_SD))
        for scenario, bounds, budget in sensitivity_specs:
            scenario_support = fit_support_model(
                failed_train[list(selected_kpis)].to_numpy(dtype=float), bounds
            )
            result = optimize_frame_in_batches(
                model, failed_eval, selected_kpis, directions, scenario_support,
                budget,
                cluster_seed(cluster, 20000 + fold * 1000 + int(budget * 10) + int(bounds[0] * 100)),
                discrete_mask,
            )
            for i, (_, row) in enumerate(failed_eval.iterrows()):
                sensitivity_rows.append({
                    "cluster": cluster,
                    "outer_fold": fold,
                    "scenario": scenario,
                    "match": str(row[match_col]),
                    "baseline_probability_oof": float(result["baseline_probability"][i]),
                    "optimized_probability_oof": float(result["optimized_probability"][i]),
                    "predicted_probability_improvement": float(
                        result["optimized_probability"][i] - result["baseline_probability"][i]
                    ),
                    "best_single_improvement": float(
                        result["best_single_probability"][i] - result["baseline_probability"][i]
                    ),
                    "L1_budget_sd": float(budget),
                    "action_quantiles": str(bounds),
                })
            del result, scenario_support

        # Explicitly release the current fold's fitted primary model and large
        # encoded/action arrays before the next fold starts. Otherwise the last
        # RF/XGB/HGB objects can remain bound while the next fold allocates its
        # models, which is particularly costly on Windows.
        del model, encoded_eval, fold_probability, main
        del train, evaluation, failed_train, failed_eval, support
        gc.collect()

    performance_rows = []
    valid = np.isfinite(primary_oof)
    performance_rows.append(performance_row(
        y_all[valid], primary_oof[valid], cluster=cluster, stage=STAGE,
        learner=PRIMARY_POLICY_LEARNER, feature_set="full_policy",
        evaluation="match_grouped_OOF",
    ))
    if RUN_CONTEXT_ONLY_BASELINE:
        valid_b = np.isfinite(baseline_oof)
        performance_rows.append(performance_row(
            y_all[valid_b], baseline_oof[valid_b], cluster=cluster, stage=STAGE,
            learner=PRIMARY_POLICY_LEARNER, feature_set="context_only_baseline",
            evaluation="match_grouped_OOF",
        ))
    for learner_name, pred in alternative_oof.items():
        valid_a = np.isfinite(pred)
        performance_rows.append(performance_row(
            y_all[valid_a], pred[valid_a], cluster=cluster, stage=STAGE,
            learner=learner_name, feature_set="full_policy",
            evaluation="match_grouped_OOF",
        ))

    permutation_rows = []
    for rep, pred in global_permutation_oof.items():
        valid_p = np.isfinite(pred)
        row = performance_row(
            y_all[valid_p], pred[valid_p], cluster=cluster, stage=STAGE,
            learner=PRIMARY_POLICY_LEARNER,
            feature_set="full_policy_global_label_permutation",
            evaluation="match_grouped_OOF", permutation_rep=rep,
            permutation_type="global_training_fold",
            expected_null_interpretation="ROC-AUC approximately 0.50",
        )
        permutation_rows.append(row)
    for rep, pred in conditional_permutation_oof.items():
        valid_p = np.isfinite(pred)
        row = performance_row(
            y_all[valid_p], pred[valid_p], cluster=cluster, stage=STAGE,
            learner=PRIMARY_POLICY_LEARNER,
            feature_set="full_policy_within_match_conditional_permutation",
            evaluation="match_grouped_OOF", permutation_rep=rep,
            permutation_type="within_match_conditional",
            expected_null_interpretation="May exceed 0.50 because match-level outcome structure is preserved",
        )
        permutation_rows.append(row)

    return {
        "individual": pd.DataFrame(individual_rows),
        "fold_audit": pd.DataFrame(fold_rows),
        "performance": pd.DataFrame(performance_rows),
        "sensitivity_long": pd.DataFrame(sensitivity_rows),
        "alternative_policy_long": pd.DataFrame(alternative_policy_rows),
        "ga_stability_long": pd.DataFrame(ga_stability_rows),
        "permutation_long": pd.DataFrame(permutation_rows),
    }


# =============================================================================
# 9. SUMMARIES AND EXCEL OUTPUT
# =============================================================================


def summarize_policy(individual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, g in individual.groupby("cluster", sort=True):
        improvement = g["predicted_probability_improvement"].to_numpy(float)
        ci = match_bootstrap_ci(
            improvement, g["match"].to_numpy(object), MATCH_BOOTSTRAP_REPS,
            cluster_seed(int(cluster), 60000),
        )
        under = g["undersampled_improvement"].to_numpy(float)
        rows.append({
            "cluster": cluster,
            "stage": STAGE,
            "n_failed_evaluated": int(len(g)),
            "n_matches": int(g["match"].nunique()),
            "mean_baseline_probability_oof": float(g["baseline_probability_oof"].mean()),
            "mean_optimized_probability_oof": float(g["optimized_probability_oof"].mean()),
            "mean_probability_improvement": float(improvement.mean()),
            "mean_improvement_match_bootstrap_ci_low": ci["ci_low"],
            "mean_improvement_match_bootstrap_ci_high": ci["ci_high"],
            "bootstrap_reps": ci["reps"],
            "median_probability_improvement": float(np.median(improvement)),
            "p25_probability_improvement": float(np.quantile(improvement, 0.25)),
            "p75_probability_improvement": float(np.quantile(improvement, 0.75)),
            "proportion_positive_improvement": float(np.mean(improvement > 1e-8)),
            "mean_best_single_improvement": float(g["best_single_improvement"].mean()),
            "mean_joint_minus_best_single_gain": float(g["joint_minus_best_single_gain"].mean()),
            "proportion_joint_strictly_better_than_single": float(np.mean(g["joint_minus_best_single_gain"] > 1e-8)),
            "proportion_joint_at_least_as_good_as_single": float(np.mean(g["joint_minus_best_single_gain"] >= -1e-8)),
            "mean_L1_used_sd": float(g["L1_used_sd"].mean()),
            "proportion_budget_binding": float(g["budget_binding"].mean()),
            "proportion_any_active_bound_hit": float(g["any_active_bound_hit"].mean()),
            "support_pass_rate": float(g["support_pass"].mean()),
            "mean_undersampled_model_improvement": float(np.nanmean(under)) if np.isfinite(under).any() else np.nan,
            "undersampling_direction_agreement": float(pd.to_numeric(g["undersampling_direction_same"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def summarize_kpi_changes(individual: pd.DataFrame, selected: Mapping[int, Sequence[str]]) -> pd.DataFrame:
    rows = []
    for cluster, g in individual.groupby("cluster", sort=True):
        for kpi in selected[int(cluster)]:
            change = g[f"change_sd::{kpi}"].to_numpy(float)
            observed_raw = pd.to_numeric(g[f"observed::{kpi}"], errors="coerce")
            optimized_raw = pd.to_numeric(g[f"optimized::{kpi}"], errors="coerce")
            is_discrete = bool(infer_discrete_action_mask([kpi])[0])
            mode_value = np.nan
            mode_share = np.nan
            if is_discrete and optimized_raw.notna().any():
                rounded = optimized_raw.round().astype("Int64")
                counts = rounded.value_counts(dropna=True)
                if len(counts):
                    mode_value = float(counts.index[0])
                    mode_share = float(counts.iloc[0] / rounded.notna().sum())
            rows.append({
                "cluster": cluster,
                "stage": STAGE,
                "kpi": kpi,
                "variable_type": "integer" if is_discrete else "continuous",
                "strategy_scope": "distribution of event-specific OOF optimal values; not one universal cluster optimum",
                "direction_constraint": g[f"direction_constraint::{kpi}"].iloc[0],
                "evaluation_observed_mean": float(observed_raw.mean()),
                "evaluation_observed_median": float(observed_raw.median()),
                "evaluation_optimized_mean": float(optimized_raw.mean()),
                "evaluation_optimized_median": float(optimized_raw.median()),
                "evaluation_optimized_q25": float(optimized_raw.quantile(0.25)),
                "evaluation_optimized_q75": float(optimized_raw.quantile(0.75)),
                "evaluation_optimized_min": float(optimized_raw.min()),
                "evaluation_optimized_max": float(optimized_raw.max()),
                "optimized_mode_if_integer": mode_value,
                "optimized_mode_share_if_integer": mode_share,
                "mean_raw_change": float(g[f"change_raw::{kpi}"].mean()),
                "median_raw_change": float(g[f"change_raw::{kpi}"].median()),
                "mean_SD_change": float(change.mean()),
                "median_SD_change": float(np.median(change)),
                "proportion_increase": float(np.mean(change > 1e-8)),
                "proportion_decrease": float(np.mean(change < -1e-8)),
                "proportion_unchanged": float(np.mean(np.abs(change) <= 1e-8)),
                "proportion_active_bound_hit": float(g[f"active_bound_hit::{kpi}"].mean()),
            })
    return pd.DataFrame(rows)


def summarize_optimal_strategy_values(
    individual: pd.DataFrame,
    selected: Mapping[int, Sequence[str]],
) -> pd.DataFrame:
    """Wide cluster-level table of the event-specific optimal strategy distribution."""
    rows: List[dict] = []
    for cluster, g in individual.groupby("cluster", sort=True):
        row: Dict[str, Any] = {
            "cluster": cluster,
            "stage": STAGE,
            "strategy_scope": "event-specific OOF policy; cluster summaries are distributions, not one universal optimum",
            "n_failed_evaluated": int(len(g)),
            "mean_baseline_probability_oof": float(g["baseline_probability_oof"].mean()),
            "mean_optimized_probability_oof": float(g["optimized_probability_oof"].mean()),
            "mean_probability_improvement": float(g["predicted_probability_improvement"].mean()),
        }
        for kpi in selected[int(cluster)]:
            observed = pd.to_numeric(g[f"observed::{kpi}"], errors="coerce")
            optimized = pd.to_numeric(g[f"optimized::{kpi}"], errors="coerce")
            change_raw = pd.to_numeric(g[f"change_raw::{kpi}"], errors="coerce")
            change_sd = pd.to_numeric(g[f"change_sd::{kpi}"], errors="coerce")
            is_discrete = bool(infer_discrete_action_mask([kpi])[0])
            mode_value = np.nan
            mode_share = np.nan
            if is_discrete and optimized.notna().any():
                rounded = optimized.round().astype("Int64")
                counts = rounded.value_counts(dropna=True)
                if len(counts):
                    mode_value = float(counts.index[0])
                    mode_share = float(counts.iloc[0] / rounded.notna().sum())
            row.update({
                f"variable_type::{kpi}": "integer" if is_discrete else "continuous",
                f"observed_mean::{kpi}": float(observed.mean()),
                f"observed_median::{kpi}": float(observed.median()),
                f"optimized_mean::{kpi}": float(optimized.mean()),
                f"optimized_median::{kpi}": float(optimized.median()),
                f"optimized_q25::{kpi}": float(optimized.quantile(0.25)),
                f"optimized_q75::{kpi}": float(optimized.quantile(0.75)),
                f"optimized_min::{kpi}": float(optimized.min()),
                f"optimized_max::{kpi}": float(optimized.max()),
                f"optimized_mode_if_integer::{kpi}": mode_value,
                f"optimized_mode_share_if_integer::{kpi}": mode_share,
                f"mean_raw_change::{kpi}": float(change_raw.mean()),
                f"median_raw_change::{kpi}": float(change_raw.median()),
                f"mean_SD_change::{kpi}": float(change_sd.mean()),
                f"proportion_changed::{kpi}": float((change_sd.abs() > 1e-8).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def extract_strategy_examples(
    individual: pd.DataFrame,
    selected: Mapping[int, Sequence[str]],
) -> pd.DataFrame:
    """Export exact feasible event-specific vectors: representative and maximum-gain examples."""
    rows: List[dict] = []
    for cluster, g0 in individual.groupby("cluster", sort=True):
        g = g0.reset_index(drop=True)
        kpis = list(selected[int(cluster)])
        optimized = g[[f"optimized::{k}" for k in kpis]].to_numpy(dtype=float)
        target = np.nanmedian(optimized, axis=0)
        scale = np.nanstd(optimized, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        distance = np.sqrt(np.nansum(((optimized - target) / scale) ** 2, axis=1))
        gains = pd.to_numeric(g["predicted_probability_improvement"], errors="coerce").to_numpy(float)
        positive_positions = np.flatnonzero(gains > MIN_POLICY_IMPROVEMENT)
        if len(positive_positions):
            representative_idx = int(positive_positions[np.nanargmin(distance[positive_positions])])
        else:
            representative_idx = int(np.nanargmin(distance))
        max_gain_idx = int(np.nanargmax(gains))
        for example_type, idx in (
            ("representative_feasible_nearest_cluster_median", representative_idx),
            ("maximum_predicted_gain_event", max_gain_idx),
        ):
            source = g.iloc[idx]
            rec: Dict[str, Any] = {
                "cluster": cluster,
                "stage": STAGE,
                "example_type": example_type,
                "interpretation": "Exact feasible event-specific OOF GA solution; not a universal strategy for every event",
                "source_row_index": int(source["source_row_index"]),
                "match": source["match"],
                "outer_fold": int(source["outer_fold"]),
                "baseline_probability_oof": float(source["baseline_probability_oof"]),
                "optimized_probability_oof": float(source["optimized_probability_oof"]),
                "predicted_probability_improvement": float(source["predicted_probability_improvement"]),
                "L1_used_sd": float(source["L1_used_sd"]),
                "support_pass": bool(source["support_pass"]),
                "any_active_bound_hit": bool(source["any_active_bound_hit"]),
            }
            for kpi in kpis:
                rec.update({
                    f"observed::{kpi}": float(source[f"observed::{kpi}"]),
                    f"optimized::{kpi}": float(source[f"optimized::{kpi}"]),
                    f"change_raw::{kpi}": float(source[f"change_raw::{kpi}"]),
                    f"change_sd::{kpi}": float(source[f"change_sd::{kpi}"]),
                })
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_sensitivity(sensitivity_long: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_long.empty:
        return pd.DataFrame()
    rows = []
    for (cluster, scenario), g in sensitivity_long.groupby(["cluster", "scenario"], sort=True):
        ci = match_bootstrap_ci(
            g["predicted_probability_improvement"].to_numpy(float),
            g["match"].to_numpy(object),
            MATCH_BOOTSTRAP_REPS,
            cluster_seed(int(cluster), 70000 + len(rows)),
        )
        rows.append({
            "cluster": cluster,
            "scenario": scenario,
            "n_failed_evaluated": int(len(g)),
            "mean_baseline_probability_oof": float(g["baseline_probability_oof"].mean()),
            "mean_optimized_probability_oof": float(g["optimized_probability_oof"].mean()),
            "mean_probability_improvement": float(g["predicted_probability_improvement"].mean()),
            "mean_improvement_ci_low": ci["ci_low"],
            "mean_improvement_ci_high": ci["ci_high"],
            "mean_best_single_improvement": float(g["best_single_improvement"].mean()),
            "mean_joint_minus_best_single_gain": float(
                (g["predicted_probability_improvement"] - g["best_single_improvement"]).mean()
            ),
            "L1_budget_sd": float(g["L1_budget_sd"].iloc[0]),
            "action_quantiles": g["action_quantiles"].iloc[0],
        })
    return pd.DataFrame(rows)


def summarize_predictive_increment(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if performance.empty:
        return pd.DataFrame()
    for cluster, g in performance.groupby("cluster", sort=True):
        full = g[
            (g["learner"] == PRIMARY_POLICY_LEARNER)
            & (g["feature_set"] == "full_policy")
        ]
        base = g[
            (g["learner"] == PRIMARY_POLICY_LEARNER)
            & (g["feature_set"] == "context_only_baseline")
        ]
        if full.empty or base.empty:
            continue
        f = full.iloc[0]
        b = base.iloc[0]
        rows.append({
            "cluster": cluster,
            "primary_learner": PRIMARY_POLICY_LEARNER,
            "full_roc_auc": f["roc_auc"],
            "baseline_roc_auc": b["roc_auc"],
            "delta_roc_auc_full_minus_baseline": f["roc_auc"] - b["roc_auc"],
            "full_pr_auc": f["pr_auc"],
            "baseline_pr_auc": b["pr_auc"],
            "delta_pr_auc_full_minus_baseline": f["pr_auc"] - b["pr_auc"],
            "full_brier": f["brier"],
            "baseline_brier": b["brier"],
            "brier_improvement_baseline_minus_full": b["brier"] - f["brier"],
            "full_log_loss": f["log_loss"],
            "baseline_log_loss": b["log_loss"],
            "log_loss_improvement_baseline_minus_full": b["log_loss"] - f["log_loss"],
        })
    return pd.DataFrame(rows)


def summarize_alternative_policy(alternative_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if alternative_long.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows = []
    for (cluster, learner), g in alternative_long.groupby(["cluster", "learner"], sort=True):
        gain = g["predicted_probability_improvement"].to_numpy(float)
        joint_single = g["joint_minus_best_single_gain"].to_numpy(float)
        summary_rows.append({
            "cluster": cluster,
            "learner": learner,
            "n_failed_evaluated": int(len(g)),
            "mean_baseline_probability_oof": float(g["baseline_probability_oof"].mean()),
            "mean_optimized_probability_oof": float(g["optimized_probability_oof"].mean()),
            "mean_probability_improvement": float(gain.mean()),
            "median_probability_improvement": float(np.median(gain)),
            "proportion_positive_improvement": float(np.mean(gain > MIN_POLICY_IMPROVEMENT)),
            "mean_best_single_improvement": float(g["best_single_improvement"].mean()),
            "mean_joint_minus_best_single_gain": float(joint_single.mean()),
            "proportion_joint_strictly_better_than_single": float(np.mean(joint_single > MIN_POLICY_IMPROVEMENT)),
            "mean_action_direction_agreement_with_HGB": float(g["action_direction_agreement_with_HGB"].mean()),
            "mean_abs_action_difference_SD_vs_HGB": float(g["mean_abs_action_difference_SD_vs_HGB"].mean()),
            "mean_HGB_improvement_on_same_rows": float(g["main_HGB_improvement"].mean()),
        })
    kpi_rows = []
    change_cols = [c for c in alternative_long.columns if c.startswith("change_sd::")]
    for (cluster, learner), g in alternative_long.groupby(["cluster", "learner"], sort=True):
        for col in change_cols:
            kpi = col.split("::", 1)[1]
            same_col = f"direction_same_as_HGB::{kpi}"
            if col not in g.columns:
                continue
            change = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            kpi_rows.append({
                "cluster": cluster,
                "learner": learner,
                "kpi": kpi,
                "mean_SD_change": float(np.nanmean(change)),
                "median_SD_change": float(np.nanmedian(change)),
                "proportion_increase": float(np.nanmean(change > 1e-8)),
                "proportion_decrease": float(np.nanmean(change < -1e-8)),
                "proportion_unchanged": float(np.nanmean(np.abs(change) <= 1e-8)),
                "direction_agreement_with_HGB": float(pd.to_numeric(g[same_col], errors="coerce").mean()) if same_col in g else np.nan,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(kpi_rows)


def summarize_ga_stability(stability_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if stability_long.empty:
        return pd.DataFrame(), pd.DataFrame()
    seed_rows = []
    for (cluster, seed_value), g in stability_long.groupby(["cluster", "ga_seed_value"], sort=True):
        seed_rows.append({
            "cluster": cluster,
            "ga_seed_value": seed_value,
            "n_events": int(len(g)),
            "mean_probability_improvement": float(g["predicted_probability_improvement"].mean()),
            "mean_difference_vs_reference": float(g["improvement_difference_vs_reference"].mean()),
            "mean_action_direction_agreement_vs_reference": float(g["action_direction_agreement_vs_reference"].mean()),
            "mean_abs_action_difference_SD_vs_reference": float(g["mean_abs_action_difference_SD_vs_reference"].mean()),
        })
    event_rows = []
    for (cluster, source_row), g in stability_long.groupby(["cluster", "source_row_index"], sort=True):
        imp = g["predicted_probability_improvement"].to_numpy(float)
        event_rows.append({
            "cluster": cluster,
            "source_row_index": source_row,
            "n_seeds": int(len(g)),
            "mean_improvement_across_seeds": float(np.mean(imp)),
            "sd_improvement_across_seeds": float(np.std(imp, ddof=1)) if len(imp) > 1 else 0.0,
            "range_improvement_across_seeds": float(np.max(imp) - np.min(imp)),
            "mean_action_direction_agreement_vs_reference": float(g["action_direction_agreement_vs_reference"].mean()),
            "mean_abs_action_difference_SD_vs_reference": float(g["mean_abs_action_difference_SD_vs_reference"].mean()),
        })
    event_df = pd.DataFrame(event_rows)
    cluster_rows = []
    for cluster, g in event_df.groupby("cluster", sort=True):
        cluster_rows.append({
            "cluster": cluster,
            "n_sampled_events": int(len(g)),
            "mean_event_SD_improvement_across_seeds": float(g["sd_improvement_across_seeds"].mean()),
            "p95_event_SD_improvement_across_seeds": float(g["sd_improvement_across_seeds"].quantile(0.95)),
            "mean_event_range_improvement_across_seeds": float(g["range_improvement_across_seeds"].mean()),
            "p95_event_range_improvement_across_seeds": float(g["range_improvement_across_seeds"].quantile(0.95)),
            "mean_action_direction_agreement_vs_reference": float(g["mean_action_direction_agreement_vs_reference"].mean()),
            "mean_abs_action_difference_SD_vs_reference": float(g["mean_abs_action_difference_SD_vs_reference"].mean()),
        })
    return pd.DataFrame(cluster_rows), pd.DataFrame(seed_rows)


def summarize_label_permutation(
    permutation_long: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    if permutation_long.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["cluster", "permutation_type"] if "permutation_type" in permutation_long.columns else ["cluster"]
    for keys, g in permutation_long.groupby(group_cols, sort=True):
        if isinstance(keys, tuple):
            cluster, permutation_type = keys
        else:
            cluster, permutation_type = keys, "legacy"
        actual = performance[
            (performance["cluster"] == cluster)
            & (performance["learner"] == PRIMARY_POLICY_LEARNER)
            & (performance["feature_set"] == "full_policy")
        ]
        if actual.empty:
            continue
        a = actual.iloc[0]
        null_auc = g["roc_auc"].to_numpy(float)
        null_pr = g["pr_auc"].to_numpy(float)
        null_brier = g["brier"].to_numpy(float)
        reps = len(g)
        null_mean_auc = float(np.nanmean(null_auc))
        is_global = permutation_type == "global_training_fold"
        rows.append({
            "cluster": cluster,
            "permutation_type": permutation_type,
            "permutation_reps": reps,
            "expected_null_interpretation": g["expected_null_interpretation"].iloc[0] if "expected_null_interpretation" in g else "",
            "actual_roc_auc": a["roc_auc"],
            "null_mean_roc_auc": null_mean_auc,
            "null_q025_roc_auc": float(np.nanquantile(null_auc, 0.025)),
            "null_q975_roc_auc": float(np.nanquantile(null_auc, 0.975)),
            "actual_exceeds_null_q975": bool(a["roc_auc"] > np.nanquantile(null_auc, 0.975)),
            "empirical_p_auc": float((1 + np.sum(null_auc >= a["roc_auc"])) / (reps + 1)),
            "actual_pr_auc": a["pr_auc"],
            "null_mean_pr_auc": float(np.nanmean(null_pr)),
            "positive_rate_reference": float(a["positive_rate"]),
            "empirical_p_pr_auc": float((1 + np.sum(null_pr >= a["pr_auc"])) / (reps + 1)),
            "actual_brier": a["brier"],
            "null_mean_brier": float(np.nanmean(null_brier)),
            "empirical_p_brier_lower_better": float((1 + np.sum(null_brier <= a["brier"])) / (reps + 1)),
            "global_null_auc_near_random_pass": bool(null_mean_auc <= GLOBAL_PERMUTATION_NULL_AUC_MAX) if is_global else np.nan,
            "primary_leakage_null": bool(is_global),
        })
    return pd.DataFrame(rows)


def write_excel(path: str, sheets: Mapping[str, pd.DataFrame]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#1F4E78",
            "border": 1, "align": "center", "valign": "vcenter",
        })
        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        float_fmt = workbook.add_format({"num_format": "0.0000"})
        for name, frame in sheets.items():
            safe_sheet = name[:31]
            frame.to_excel(writer, sheet_name=safe_sheet, index=False)
            ws = writer.sheets[safe_sheet]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
            for col_idx, col in enumerate(frame.columns):
                ws.write(0, col_idx, col, header_fmt)
                series = frame[col]
                max_len = max([len(str(col))] + [len(str(x)) for x in series.head(200).fillna("")])
                width = min(max(max_len + 2, 10), 38)
                fmt = None
                lower = str(col).lower()
                if "proportion" in lower or "rate" in lower or "agreement" in lower:
                    fmt = percent_fmt
                elif any(token in lower for token in ["probability", "improvement", "auc", "brier", "loss", "change", "mean", "median", "ci_", "slope", "intercept"]):
                    fmt = float_fmt
                ws.set_column(col_idx, col_idx, width, fmt)


def reviewer_checklist(selection_audit: pd.DataFrame) -> pd.DataFrame:
    if "strictly_eligible_before_override" in selection_audit.columns:
        eligible = selection_audit["strictly_eligible_before_override"].map(as_bool)
    elif "passed_single_and_joint" in selection_audit.columns:
        eligible = selection_audit["passed_single_and_joint"].map(as_bool)
    else:
        eligible = pd.Series(False, index=selection_audit.index)
    all_selected_strictly_eligible = bool(len(eligible) > 0 and eligible.all())
    noneligible_override_used = bool(
        "manual_override_used" in selection_audit.columns
        and (
            selection_audit["manual_override_used"].map(as_bool)
            & ~eligible
        ).any()
    )
    admission_note = (
        "All selected KPIs passed the single-KPI, joint-DML and direction gates."
        if all_selected_strictly_eligible
        else
        "At least one selected KPI failed a strict DML admission gate; this output "
        "is exploratory and the exact failed flags are retained in 01_main_KPI_selection."
    )
    return pd.DataFrame([
        ["DML and optimization separated", True, "No DML model is fitted in this script."],
        ["Candidate KPIs passed single and joint DML", all_selected_strictly_eligible, admission_note],
        ["No noneligible manual override used", not noneligible_override_used, "False identifies an explicitly authorized exploratory L table override."],
        ["One KPI per tactical family in main policy", True, "Avoids simultaneous adjustment of redundant same-family proxies."],
        ["Independent nonlinear policy model", True, "HistGradientBoosting with grouped OOF evaluation."],
        ["Context-only predictive baseline", RUN_CONTEXT_ONLY_BASELINE, "The same grouped OOF learner is refit without the actionable KPI(s)."],
        ["Alternative policy learners", RUN_ALTERNATIVE_POLICY_LEARNERS, "RF and XGBoost independently refit and re-optimize held-out failed events under identical constraints."],
        ["Global label permutation", RUN_GLOBAL_LABEL_PERMUTATION, "Primary no-signal/leakage null; training-fold labels are globally permuted and OOF ROC-AUC should return near 0.50."],
        ["Within-match conditional permutation", RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION, "Secondary conditional diagnostic preserving match-level outcome structure; not interpreted as a 0.50 null."],
        ["GA random-seed stability", RUN_GA_SEED_STABILITY, "A fixed held-out subset is re-optimized under five prespecified GA seeds."],
        ["Outcome leakage prevented", True, "Explicit pre-decision feature allow-list; downstream and outcome-derived fields are excluded."],
        ["Match-grouped cross-fitting", True, "Evaluation matches are excluded from model, bounds and support fitting."],
        ["All outcomes used for model training", True, "Successful and failed events train the probability model."],
        ["Failed events only optimized", True, "Only success_def==0 held-out events receive actions."],
        ["Cooperative multi-KPI GA", True, "Three selected KPIs are changed simultaneously."],
        ["Discrete count KPI constraint", True, "Adv_5/Adv_10 bounds and all GA/single-KPI candidates are integer-projected before budget, support and model scoring."],
        ["Joint vs best single-KPI benchmark", True, "Best one-KPI grid action uses the same model and constraints, including integer count actions."],
        ["DML-informed direction restrictions", True, "Directions come from completed joint-DML target results."],
        ["Quantile, L1 and empirical support restrictions", True, "Main Q20-Q80, 3-SD budget and Mahalanobis support."],
        ["Class imbalance sensitivity", RUN_TRAINING_ONLY_UNDERSAMPLING_SENSITIVITY, "Training-fold-only undersampled model evaluates the main action."],
        ["Predictive performance and calibration", True, "ROC-AUC, PR-AUC, Brier, log loss, slope/intercept and ECE."],
        ["Budget and bound sensitivity", RUN_BUDGET_SENSITIVITY and RUN_BOUND_SENSITIVITY, "Prespecified alternative action constraints."],
        ["External/prospective validation", False, "Requires an independent season, league or prospective deployment."],
    ], columns=["review_item", "pass", "note"])


# =============================================================================
# 10. MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=RAW_INPUT_PATH)
    parser.add_argument("--single-results", default=SINGLE_RESULTS_XLSX)
    parser.add_argument("--joint-results", default=ROTATING_JOINT_RESULTS_XLSX)
    parser.add_argument(
        "--output",
        default=OUTPUT_XLSX,
        help="Output workbook for the Shot L policy analysis.",
    )
    parser.add_argument("--global-permutation-reps", type=int, default=None,
                        help="Override the default global training-fold label-permutation repetitions.")
    parser.add_argument("--conditional-permutation-reps", type=int, default=None,
                        help="Override the default within-match conditional permutation repetitions.")
    parser.add_argument("--quick-smoke", action="store_true")
    return parser.parse_args()


def apply_quick_smoke() -> None:
    global OUTER_FOLDS, INNER_CALIBRATION_FOLDS
    global HGB_MAX_ITER, RF_N_ESTIMATORS, XGB_N_ESTIMATORS
    global GA_POPULATION, GA_GENERATIONS, GA_BATCH_SIZE
    global MATCH_BOOTSTRAP_REPS, MAX_FAILED_ROWS_PER_CLUSTER
    global RUN_BUDGET_SENSITIVITY, RUN_BOUND_SENSITIVITY
    global GLOBAL_LABEL_PERMUTATION_REPS, WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS
    global GA_STABILITY_SEEDS, GA_STABILITY_MAX_ROWS_PER_FOLD
    OUTER_FOLDS = 2
    INNER_CALIBRATION_FOLDS = 2
    HGB_MAX_ITER = 40
    RF_N_ESTIMATORS = 40
    XGB_N_ESTIMATORS = 40
    GA_POPULATION = 10
    GA_GENERATIONS = 6
    GA_BATCH_SIZE = 24
    MATCH_BOOTSTRAP_REPS = 20
    MAX_FAILED_ROWS_PER_CLUSTER = 30
    GLOBAL_LABEL_PERMUTATION_REPS = 2
    WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS = 2
    GA_STABILITY_SEEDS = (11, 22)
    GA_STABILITY_MAX_ROWS_PER_FOLD = 8
    RUN_BUDGET_SENSITIVITY = False
    RUN_BOUND_SENSITIVITY = False


def main():
    global GLOBAL_LABEL_PERMUTATION_REPS, WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS
    args = parse_args()
    if args.quick_smoke:
        apply_quick_smoke()
    if args.global_permutation_reps is not None:
        if args.global_permutation_reps < 1:
            raise ValueError("--global-permutation-reps must be >= 1.")
        GLOBAL_LABEL_PERMUTATION_REPS = int(args.global_permutation_reps)
    if args.conditional_permutation_reps is not None:
        if args.conditional_permutation_reps < 1:
            raise ValueError("--conditional-permutation-reps must be >= 1.")
        WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS = int(args.conditional_permutation_reps)

    started = time.time()
    print("=" * 88)
    print("Shot L independent DML-informed multi-KPI GA policy optimization")
    print("Stage          : L only")
    print(f"Selection mode : {ACTIVE_SELECTION_RULE}")
    print(f"Policy FDR     : q < {POLICY_FDR_ALPHA:.2f}")
    print(f"Raw input      : {args.raw}")
    print(f"Single DML     : {args.single_results}")
    print(f"Joint DML      : {args.joint_results}")
    print(f"Output         : {args.output}")
    print("No DML model is refitted in this script.")
    print(f"Available policy learners: {' | '.join(available_policy_learners())}")
    print("=" * 88)

    candidate_pool = load_candidate_pool(args.single_results, args.joint_results)
    selected, selection_audit = select_main_kpis(candidate_pool)
    raw, outcome_col, cluster_col, match_col = load_raw_data(args.raw)

    print("Selected main policy KPIs:")
    for cluster, kpis in selected.items():
        print(f"  Cluster {cluster}: {' | '.join(kpis)}")

    all_individual = []
    all_fold = []
    all_perf = []
    all_sensitivity = []
    all_alternative_policy = []
    all_ga_stability = []
    all_permutation = []
    feature_audit_rows = []

    for cluster, selected_kpis in selected.items():
        cluster_df = raw[raw[cluster_col] == cluster].copy().reset_index(drop=True)
        if cluster_df.empty:
            raise RuntimeError(f"Cluster {cluster} has no raw rows.")
        feature_columns, numeric_columns, categorical_columns, excluded = build_feature_columns(
            cluster_df, selected_kpis, outcome_col, cluster_col, match_col
        )
        directions = []
        for kpi in selected_kpis:
            row = candidate_pool[
                (candidate_pool["cluster"] == cluster)
                & (candidate_pool["target"] == kpi)
            ]
            if row.empty:
                raise RuntimeError(f"No DML target record for Cluster {cluster}, {kpi}.")
            directions.append(int(row.iloc[0]["direction_code"]))
        directions = np.asarray(directions, dtype=int)

        print(
            f"[POLICY] Cluster {cluster}: n={len(cluster_df)}, "
            f"matches={cluster_df[match_col].nunique()}, KPIs={' | '.join(selected_kpis)}"
        )
        result = run_cluster_policy(
            cluster, cluster_df, selected_kpis, directions,
            outcome_col, match_col, feature_columns, numeric_columns,
            categorical_columns,
        )
        all_individual.append(result["individual"])
        all_fold.append(result["fold_audit"])
        all_perf.append(result["performance"])
        if not result["sensitivity_long"].empty:
            all_sensitivity.append(result["sensitivity_long"])
        if not result["alternative_policy_long"].empty:
            all_alternative_policy.append(result["alternative_policy_long"])
        if not result["ga_stability_long"].empty:
            all_ga_stability.append(result["ga_stability_long"])
        if not result["permutation_long"].empty:
            all_permutation.append(result["permutation_long"])

        baseline_numeric = [c for c in numeric_columns if c not in selected_kpis]
        feature_audit_rows.append({
            "cluster": cluster,
            "n_rows": len(cluster_df),
            "n_matches": cluster_df[match_col].nunique(),
            "selected_kpis": " | ".join(selected_kpis),
            "direction_constraints": " | ".join(map(direction_label, directions)),
            "action_variable_types": " | ".join(
                f"{k}:{'integer' if flag else 'continuous'}"
                for k, flag in zip(selected_kpis, infer_discrete_action_mask(selected_kpis))
            ),
            "n_full_model_features": len(feature_columns),
            "n_context_only_features": len(baseline_numeric) + len(categorical_columns),
            "n_numeric_features": len(numeric_columns),
            "n_categorical_features": len(categorical_columns),
            "numeric_feature_allowlist": " | ".join(numeric_columns),
            "context_only_numeric_features": " | ".join(baseline_numeric),
            "categorical_feature_allowlist": " | ".join(categorical_columns),
            "excluded_matched_attacking_proxies": " | ".join(excluded),
            "feature_policy": "explicit_predecision_allowlist",
        })

    individual = pd.concat(all_individual, ignore_index=True)
    fold_audit = pd.concat(all_fold, ignore_index=True)
    performance = pd.concat(all_perf, ignore_index=True)
    sensitivity_long = (
        pd.concat(all_sensitivity, ignore_index=True)
        if all_sensitivity else pd.DataFrame()
    )
    alternative_policy_long = (
        pd.concat(all_alternative_policy, ignore_index=True)
        if all_alternative_policy else pd.DataFrame()
    )
    ga_stability_long = (
        pd.concat(all_ga_stability, ignore_index=True)
        if all_ga_stability else pd.DataFrame()
    )
    permutation_long = (
        pd.concat(all_permutation, ignore_index=True)
        if all_permutation else pd.DataFrame()
    )

    policy_summary = summarize_policy(individual)
    optimal_strategy_values = summarize_optimal_strategy_values(individual, selected)
    strategy_examples = extract_strategy_examples(individual, selected)
    kpi_changes = summarize_kpi_changes(individual, selected)
    sensitivity_summary = summarize_sensitivity(sensitivity_long)
    predictive_increment = summarize_predictive_increment(performance)
    alternative_policy_summary, alternative_kpi_summary = summarize_alternative_policy(
        alternative_policy_long
    )
    ga_stability_summary, ga_stability_by_seed = summarize_ga_stability(ga_stability_long)
    permutation_summary = summarize_label_permutation(permutation_long, performance)
    if not permutation_summary.empty:
        global_null = permutation_summary[
            permutation_summary["permutation_type"].eq("global_training_fold")
        ]
        null_pass = global_null["global_null_auc_near_random_pass"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        failed_null = global_null.loc[~null_pass]
        if not failed_null.empty:
            details = " | ".join(
                f"C{int(r.cluster)} null mean AUC={r.null_mean_roc_auc:.3f}"
                for r in failed_null.itertuples()
            )
            raise RuntimeError(
                "Global label-permutation leakage/null diagnostic failed: " + details
                + f". Expected mean AUC <= {GLOBAL_PERMUTATION_NULL_AUC_MAX:.2f}. "
                  "Inspect feature timing, label construction and permutation implementation."
            )

    selected_rows = []
    for cluster, kpis in selected.items():
        for kpi in kpis:
            row = candidate_pool[
                (candidate_pool["cluster"] == cluster)
                & (candidate_pool["target"] == kpi)
            ].iloc[0]
            selected_rows.append({
                "cluster": cluster,
                "policy_family": row.get("policy_family"),
                "selected_kpi": kpi,
                "joint_effect_per_1SD_used_for_selection_only": row.get("estimate"),
                "single_effect_per_1SD_for_audit": row.get("single_effect_per_1sd"),
                "joint_q_global": row.get("q_global"),
                "direction_constraint": direction_label(int(row["direction_code"])),
                "decision_variable_type": (
                    "integer" if infer_discrete_action_mask([kpi])[0] else "continuous"
                ),
                "passed_single_and_joint": row["passed_single_and_joint"],
                "selection_rule": ACTIVE_SELECTION_RULE,
                "manual_override_used": bool(cluster in FINAL_KPI_OVERRIDE),
                "strictly_eligible_before_override": as_bool(row["passed_single_and_joint"]),
                "noneligible_override_authorized": ALLOW_NONELIGIBLE_OVERRIDE,
            })
    main_selection = pd.DataFrame(selected_rows)

    requested_alternatives = list(ALTERNATIVE_POLICY_LEARNERS) if RUN_ALTERNATIVE_POLICY_LEARNERS else []
    used_alternatives = [x for x in requested_alternatives if x in available_policy_learners()]
    unavailable_alternatives = [x for x in requested_alternatives if x not in available_policy_learners()]
    manifest = pd.DataFrame([{
        "pipeline_version": PIPELINE_VERSION,
        "raw_input": args.raw,
        "single_DML_results": args.single_results,
        "joint_DML_results": args.joint_results,
        "joint_result_schema": " | ".join(sorted(candidate_pool["joint_schema"].astype(str).unique())),
        "output": args.output,
        "stage": STAGE,
        "outcome_column": outcome_col,
        "cluster_column": cluster_col,
        "match_group_column": match_col,
        "primary_policy_learner": PRIMARY_POLICY_LEARNER,
        "alternative_policy_learners_used": " | ".join(used_alternatives),
        "alternative_policy_learners_unavailable": " | ".join(unavailable_alternatives),
        "nested_Platt_calibration": True,
        "context_only_baseline": RUN_CONTEXT_ONLY_BASELINE,
        "global_label_permutation": RUN_GLOBAL_LABEL_PERMUTATION,
        "global_label_permutation_reps": GLOBAL_LABEL_PERMUTATION_REPS,
        "permutation_encoded_features_reused_per_fold": True,
        "permutation_HGB_max_iter": PERMUTATION_HGB_MAX_ITER,
        "permutation_HGB_early_stopping": False,
        "permutation_gc_interval": PERMUTATION_GC_INTERVAL,
        "within_match_conditional_permutation": RUN_WITHIN_MATCH_CONDITIONAL_PERMUTATION,
        "within_match_conditional_permutation_reps": WITHIN_MATCH_CONDITIONAL_PERMUTATION_REPS,
        "global_permutation_null_auc_max": GLOBAL_PERMUTATION_NULL_AUC_MAX,
        "label_permutation_calibration": "none_raw_probability_null; AUC/PR are primary",
        "GA_seed_stability": RUN_GA_SEED_STABILITY,
        "GA_stability_seeds": " | ".join(map(str, GA_STABILITY_SEEDS)),
        "GA_stability_max_rows_per_fold": GA_STABILITY_MAX_ROWS_PER_FOLD,
        "policy_feature_rule": "explicit_predecision_allowlist",
        "safe_numeric_context": " | ".join(SAFE_NUMERIC_CONTEXT_COLS),
        "safe_categorical_context": " | ".join(SAFE_CATEGORICAL_CONTEXT_COLS),
        "leakage_stop_auc": LEAKAGE_STOP_AUC,
        "leakage_stop_brier": LEAKAGE_STOP_BRIER,
        "outer_folds": OUTER_FOLDS,
        "inner_calibration_folds": INNER_CALIBRATION_FOLDS,
        "GA_population": GA_POPULATION,
        "GA_generations": GA_GENERATIONS,
        "GA_crossover": GA_CROSSOVER_PROB,
        "GA_mutation": GA_MUTATION_PROB,
        "GA_elitism": GA_ELITISM,
        "main_action_quantiles": str(MAIN_ACTION_QUANTILES),
        "main_L1_budget_SD": MAIN_L1_BUDGET_SD,
        "support_quantile": SUPPORT_QUANTILE,
        "discrete_action_patterns": " | ".join(DISCRETE_ACTION_PATTERNS),
        "integer_projection_before_scoring": True,
        "selection_mode": SELECTION_MODE,
        "policy_FDR_alpha": POLICY_FDR_ALPHA,
        "require_all_families": REQUIRE_ALL_FAMILIES,
        "missing_families_filled_with_reference_KPIs": False,
        "active_selection_rule": ACTIVE_SELECTION_RULE,
        "manual_override_used": bool(FINAL_KPI_OVERRIDE),
        "noneligible_override_authorized": ALLOW_NONELIGIBLE_OVERRIDE,
        "all_selected_strictly_eligible": bool(
            selection_audit["strictly_eligible_before_override"].map(as_bool).all()
        ),
        "DML_refitted_in_policy_stage": False,
        "optimal_strategy_value_output": "cluster distribution summaries plus exact feasible event-specific examples and complete individual OOF actions",
        "universal_cluster_optimum_claimed": False,
        "runtime_seconds": time.time() - started,
        "quick_smoke": bool(args.quick_smoke),
    }])

    sheets = {
        "00_reviewer_checklist": reviewer_checklist(selection_audit),
        "01_main_KPI_selection": main_selection,
        "02_candidate_pool": candidate_pool,
        "03_feature_audit": pd.DataFrame(feature_audit_rows),
        "04_predictive_performance": performance,
        "05_predictive_increment": predictive_increment,
        "06_primary_policy_summary": policy_summary,
        "07_optimal_strategy_values": optimal_strategy_values,
        "08_exact_strategy_examples": strategy_examples,
        "09_primary_failed_individual": individual,
        "10_optimal_values_long": kpi_changes,
        "11_joint_vs_single": policy_summary[[
            "cluster", "n_failed_evaluated", "mean_probability_improvement",
            "mean_best_single_improvement", "mean_joint_minus_best_single_gain",
            "proportion_joint_strictly_better_than_single",
            "proportion_joint_at_least_as_good_as_single",
        ]],
        "12_alternative_policy_summary": alternative_policy_summary,
        "13_alternative_KPI_summary": alternative_kpi_summary,
        "14_alternative_policy_long": alternative_policy_long,
        "15_label_permutation_summary": permutation_summary,
        "16_label_permutation_long": permutation_long,
        "17_GA_seed_stability": ga_stability_summary,
        "18_GA_seed_by_seed": ga_stability_by_seed,
        "19_GA_seed_long": ga_stability_long,
        "20_fold_audit": fold_audit,
        "21_constraint_sensitivity": sensitivity_summary,
        "22_manifest": manifest,
    }
    write_excel(args.output, sheets)
    print("=" * 88)
    print("Completed independent policy optimization with final reviewer robustness analyses and explicit optimal-strategy values.")
    print(f"Saved: {args.output}")
    print(f"Runtime: {time.time() - started:.1f} seconds")
    print("=" * 88)


if __name__ == "__main__":
    main()
