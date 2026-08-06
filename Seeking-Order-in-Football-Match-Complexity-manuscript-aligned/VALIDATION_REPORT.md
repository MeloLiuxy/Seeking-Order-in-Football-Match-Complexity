# Validation report

## Static manuscript-setting checks

```text
PASS - RF nuisance learners specify 240 trees
PASS - Primary DML trimming is 2%
PASS - DML trimming sensitivity is 0/1/2/5%
PASS - No match-clustered second-stage covariance remains
PASS - HC3 second-stage covariance is implemented
PASS - No statistical imputation remains
PASS - No Avg_2 KPI is constructed or referenced
PASS - No author-specific Windows absolute path remains
PASS - Strategy budget sensitivity uses 1, 2, and 4 SD
PASS - GA uses population 48 and 60 generations
PASS - Primary screening requires at least 80% fold-direction consistency
PASS - Primary screening requires at least 90% leave-one-team-out consistency
PASS - Primary screening uses the prespecified placebo threshold
PASS - Severe-overlap thresholds match the manuscript
PASS - Complete-case fail-fast handling is implemented
PASS - Clustering sensitivity includes K=2-5
PASS - K=3 is the prespecified reference clustering
```

## Synthetic smoke checks

```text
PASS - pass KPI module imported
PASS - carry KPI module imported
PASS - shot KPI module imported
PASS - pass_dml HC3 WLS smoke test
PASS - carry_dml HC3 WLS smoke test
PASS - shot_dml HC3 WLS smoke test
```

## Scope and limitation

All Python files were compiled successfully after modification. The KPI modules were imported, and the HC3 unit-weight residual-WLS second stage was exercised with synthetic data for the Pass, Carry, and Shot single-KPI DML modules.

The licensed StatsBomb analytical data are not included in this archive. Therefore, the complete pipeline and the manuscript's numerical tables were not regenerated. Before using newly generated outputs in a submission, the full workflow should be rerun with the licensed analytical data and the resulting tables should be checked against the manuscript.
