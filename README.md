# Seeking-Order-in-Football-Match-Complexity
Seeking Order in Football Match Complexity: From Causal Identification to Strategic Optimization
# Reproducibility Code for Scenario-Specific Causal Analysis in Football

## Overview

This repository contains the analytical code used for:

1. KPI calculation
2. Spatial clustering and clustering-sensitivity analysis
3. Single-KPI Double Machine Learning (DML)
4. Joint and interaction DML
5. Identification diagnostics and robustness analyses
6. Constrained strategy optimization

The repository does not include the complete raw-data preprocessing pipeline. Users should prepare the analytical input data before running the scripts in this repository.

The required input data should contain the event-level variables, player-location information, outcome variable, contextual covariates, and scenario- or stage-specific identifiers required by the corresponding analytical module.

## Data Availability

The complete analytical dataset used in the study was supplied by StatsBomb under a commercial licence and cannot be redistributed through this repository.

Public football event data may be used to understand and adapt the analytical workflow. However, reproducing the complete KPI calculations requires event-time player-location information sufficient to construct distance, numerical-configuration, centroid, area, and spread measures.

## Repository Scope

### 1. KPI Calculation

The KPI scripts calculate indicators describing:

- local numerical configuration;
- nearest-defender and nearest-attacker distances;
- distances to team centroids;
- defensive and attacking convex-hull areas; and
- defensive and attacking team spread.

The required coordinate fields and player-team labels may need to be adapted to the structure of a new dataset.

### 2. Spatial Clustering and Sensitivity Analysis

Spatial clustering is conducted separately by action scenario using standardized event-location or origin–destination coordinates.

The repository includes the reference clustering procedure and sensitivity analyses across alternative values of K and random initializations.

Cluster structures are data dependent and should be re-estimated rather than transferred directly to another dataset.

### 3. Single-KPI DML

The single-KPI DML scripts estimate the effect of one numerical defensive KPI at a time while adjusting for the specified contextual and attacking-side covariates.

The scripts include match-grouped cross-fitting, nuisance-model estimation, probability calibration, treatment-support diagnostics, trimming, and statistical inference.

### 4. Joint and Interaction DML

The joint-DML scripts evaluate selected KPIs simultaneously and include their pairwise interactions where applicable.

These models are used to examine whether single-KPI effects persist when multiple tactical mechanisms are considered jointly.

### 5. Identification Diagnostics and Robustness Analyses

The diagnostic and robustness modules include, where applicable:

- outcome-model performance;
- treatment-model R²;
- residual treatment variation;
- treatment-support bins;
- alternative trimming levels;
- alternative nuisance learners;
- training-fold undersampling;
- fold-direction consistency;
- leave-one-team-out analysis;
- placebo tests;
- robustness values for unmeasured confounding;
- residual-space VIFs; and
- condition indices.

These diagnostics improve transparency but do not prove conditional exchangeability or eliminate unmeasured confounding.

### 6. Constrained Strategy Optimization

The strategy scripts use a separately fitted probability model and a genetic algorithm to explore feasible joint KPI configurations among unsuccessful defensive events.

Candidate adjustments are restricted by:

- DML-supported directions;
- percentile-based candidate bounds;
- a standardized total-adjustment budget;
- integer restrictions for count-valued KPIs; and
- an empirical-support constraint.

The resulting configurations are model-based simulations and should not be interpreted as interventionally validated tactical instructions.

## Input-Data Preparation

Raw-data preprocessing is not included in this repository. Before running the provided scripts, users should prepare an analytical dataset with the required event, player-location, outcome, grouping, and contextual variables.

### Player-location completeness

In the original study, observations were retained only when locations were available for at least 20 players. This rule was used to reduce measurement error in numerical, distance, centroid, convex-hull-area, and spread KPIs.

The 20-player threshold is specific to the original dataset and is not a universal requirement. Users may modify or remove this threshold according to:

- the player-location coverage of their data;
- whether goalkeeper locations are included;
- the frequency and pattern of missing locations;
- the sensitivity of the selected KPIs to missing players; and
- the balance between sample size and measurement reliability.

For datasets with complete continuous tracking, a different completeness rule may be more appropriate. For partially observed event-time spatial data, users should assess how alternative thresholds affect the analytical sample and KPI stability.

### Contextual covariates

The original analysis used available match-context variables such as:

- match period;
- match time;
- pre-event score;
- home status; and
- the playing position of the player in possession.

The contextual adjustment set may be modified according to the variables available in a new dataset and the causal question being studied.

Variables should be included only when they are temporally eligible and substantively justified as possible common causes of the target KPI and outcome. Variables measured after the focal KPI, or variables considered consequences of the treatment, should not be included as ordinary adjustment covariates.

When some contextual variables are unavailable, they may be omitted, but the resulting identification assumptions and possible unmeasured confounding should be stated explicitly.

## Adaptation to Other Datasets

When applying the code to another dataset, users may need to modify:

- variable and column names;
- coordinate conventions;
- player and team identifiers;
- KPI calculation fields;
- scenario and stage labels;
- match-grouping variables;
- the player-location completeness threshold;
- contextual covariates;
- clustering inputs;
- treatment and outcome definitions; and
- optimization constraints.

All modifications should be documented because they may affect KPI distributions, empirical treatment support, spatial clusters, effect estimates, and optimized configurations.

## Interpretation

The analytical framework may be transferred to other football datasets, but the preprocessing rules, covariate set, cluster structure, estimated effects, and optimized configurations are context and dataset dependent.

DML estimates rely on conditional exchangeability, consistency, and sufficient empirical support. The diagnostic analyses make these assumptions more visible but cannot verify that all relevant confounders have been measured.
