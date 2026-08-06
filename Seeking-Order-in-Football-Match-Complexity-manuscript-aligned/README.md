# Seeking Order in Football Match Complexity

Reproducibility code accompanying **“Seeking Order in Football Match Complexity: From Causal Identification to Strategic Optimization.”**

## Repository scope

The repository contains the analytical modules used for:

1. KPI construction;
2. spatial clustering and clustering-sensitivity analysis;
3. single-KPI Double Machine Learning (DML);
4. joint and interaction DML;
5. identification diagnostics and robustness analyses; and
6. constrained strategy optimization.

The complete provider-specific raw-data preprocessing and event-sequence construction pipeline is **not** included. Users must prepare an analytical input table containing event identifiers, match identifiers, the defensive-success outcome, event coordinates, event-time player locations, contextual covariates, and the scenario/configuration labels required by the relevant module.

## Analytical configurations and temporal eligibility

The manuscript uses four configuration names:

- **pass-origin configuration**: the pass-initiation player snapshot evaluated relative to the passer location;
- **pass-destination-referenced configuration**: the same pass-initiation snapshot evaluated relative to the pass destination;
- **carry-endpoint-referenced configuration**: the carry-initiation snapshot evaluated relative to the carry endpoint; and
- **shot-time configuration**: the player snapshot observed at the shot location and shot time.

For compatibility with the analysis files, some scripts retain the internal suffixes `(L)` and `(E')`. In this repository:

- `(L)` corresponds to the pass-origin configuration in the Pass modules and the shot-time configuration in the Shot modules;
- `(E')` corresponds to the pass-destination-referenced configuration in the Pass modules and the carry-endpoint-referenced configuration in the Carry modules.

No post-initiation player-location snapshot is used for passing or carrying. The destination or endpoint coordinate is used only as the focal spatial reference. This distinction is important because using the actual receiving-time or carry-end-time spatial snapshot would introduce post-initiation information into the analysis.

## Data availability

The complete analytical dataset was supplied by StatsBomb under a commercial licence and cannot be redistributed. Public football event data may be used to understand or adapt the workflow, but exact reproduction of the reported numerical results requires licensed event data containing the required event-time player-location information.

## Input-data preparation

### Player-location completeness

The manuscript retained observations with locations available for at least 20 players. This rule was used to reduce measurement error in local numerical, distance, centroid, convex-hull-area, and spread KPIs. The threshold is dataset-specific rather than universal. Users may adapt it according to player-location coverage, goalkeeper recording, missingness patterns, KPI sensitivity, and the desired balance between sample size and measurement reliability.

### Contextual covariates

The manuscript used available pre-event or contemporaneous contextual variables, including match period, match time, pre-event scores, home status, and the playing position of the player in possession. The adjustment set may be adapted to another dataset, but variables must be temporally eligible and substantively justified as possible common causes of the target KPI and outcome. Post-treatment variables and consequences of the target KPI must not be included as ordinary adjustment covariates.

### Complete-case rule

The analytical sample used complete-case screening. The scripts therefore use a fail-fast complete-case transformer rather than statistical imputation. Input rows with missing required outcomes, treatments, grouping variables, contextual covariates, or KPI fields should be removed before estimation or will trigger an explicit error.

## KPI definitions

The manuscript analyses the following defensive KPIs:

- `Adv_5`, `Adv_10`;
- `Avg_1_Def`, `Avg_3_Def`, `Avg_5_Def`;
- `DistToDefCentroid`;
- `Area_Def`; and
- `Spr_Def`.

`Area_Def` and `Spr_Def` use outfield defenders. `Spr_Def` is the mean squared Euclidean distance of the outfield defenders from their defensive centroid and is measured in square metres. The KPI scripts do not construct `Avg_2` indicators.

## Spatial clustering settings

- K-means after within-scenario Z-standardization;
- `K = 3` as the prespecified main solution;
- `K = 2–5` as sensitivity solutions;
- random seed `42` for the reference solution;
- 20 random-seed solutions for stability analysis;
- stability summarized using mean pairwise adjusted Rand index (ARI);
- cluster quality summarized using silhouette, Calinski–Harabasz, Davies–Bouldin, inertia, and cluster-size diagnostics.

Passing and carrying use standardized origin–destination coordinates. Shooting uses standardized shot-location coordinates only.

## Single-KPI DML settings

- one numerical defensive KPI as the treatment at a time;
- five-fold cross-fitting grouped by match;
- Random Forest nuisance learners with 240 trees for both the binary outcome and numerical treatment;
- nested three-fold match-grouped Platt calibration within each outer training fold;
- training-fold Z-standardization;
- residual weighted least squares with unit weights and HC3 heteroskedasticity-robust inference;
- 2% two-sided trimming as the primary estimate, with 0%, 1%, and 5% sensitivity analyses;
- Benjamini–Hochberg correction within scenario, configuration, and spatial cluster (`q < 0.05` for primary screening);
- severe-overlap flag when `R²(D|X) >= 0.90` and the residual/observed treatment-SD ratio is `<= 0.25`;
- alternative nuisance learners, training-fold undersampling, fold-direction consistency, leave-one-team-out consistency, match-preserving placebos, and robustness values for unmeasured confounding;
- primary retention requires a 95% CI excluding zero, `q < 0.05`, no severe-overlap flag, at least 80% fold-direction consistency, at least 90% leave-one-team-out consistency, and no direction reversal across the prespecified sensitivity analyses.

Attacking-side KPIs may enter the adjustment set only when temporally eligible and conceptually distinct from the target treatment. Other defensive KPIs do not enter the ordinary single-KPI adjustment set.

## Joint and interaction DML

Eligible KPIs are evaluated in three-mechanism joint models containing, where available, one local-numerical KPI, one defensive-distance KPI, and one spatial-organisation KPI. The joint basis contains three main effects and three pairwise interactions. The same grouped cross-fitting, calibration, trimming, diagnostics, robustness analyses, and HC3 inference are retained.

## Constrained strategy optimization

- primary learner: `HistGradientBoostingClassifier`;
- alternative learners: Random Forest and XGBoost;
- five-fold match-grouped out-of-sample evaluation with nested three-fold Platt calibration;
- optimization restricted to unsuccessful held-out events;
- DML-supported adjustment directions;
- training-fold 20th–80th percentile candidate bounds;
- total L1 adjustment budget of 3 SD units;
- integer restrictions for count-valued KPIs;
- 95% Mahalanobis empirical-support restriction;
- genetic algorithm with population 48, 60 generations, two elites, tournament size 3, and seed 42;
- sensitivity bounds of 10th–90th and 30th–70th percentiles and alternative L1 budgets of 1, 2, and 4 SD units.

The resulting values are constrained, out-of-sample, model-based simulations. They are not interventionally validated tactical policies.

## Running the code

Each script contains a user-configuration section or command-line arguments for input and output paths. No author-specific absolute paths are retained. A typical order is:

1. construct KPIs;
2. run the scenario-specific clustering script;
3. run single-KPI DML;
4. run joint and interaction DML;
5. run constrained strategy optimization.

Because the licensed analytical data are not distributed, the repository has been checked statically for syntax and manuscript-setting consistency but the full numerical pipeline cannot be rerun from the public repository alone.
