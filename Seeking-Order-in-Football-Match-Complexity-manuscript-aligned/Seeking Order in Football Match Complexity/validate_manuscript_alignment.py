from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
checks = []

def check(condition, message):
    checks.append((bool(condition), message))

all_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in ROOT.rglob("*.py") if path.name != Path(__file__).name)
check("n_estimators=240" in all_text or "RF_Y_N_TREES = 240" in all_text, "RF nuisance learners specify 240 trees")
check("MAIN_TRIM_FRAC = 0.02" in all_text, "Primary DML trimming is 2%")
check("TRIM_GRID = (0.00, 0.01, 0.02, 0.05)" in all_text, "DML trimming sensitivity is 0/1/2/5%")
check("cov_type=\"cluster\"" not in all_text, "No match-clustered second-stage covariance remains")
check("cov_type=\"HC3\"" in all_text, "HC3 second-stage covariance is implemented")
check("SimpleImputer" not in all_text, "No statistical imputation remains")
check("Avg_2_" not in all_text, "No Avg_2 KPI is constructed or referenced")
check(not re.search(r'[A-Za-z]:\\\\', all_text), "No author-specific Windows absolute path remains")
check("BUDGET_SENSITIVITY_GRID = (1.0, 2.0, 4.0)" in all_text, "Strategy budget sensitivity uses 1, 2, and 4 SD")
check("GA_POPULATION = 48" in all_text and "GA_GENERATIONS = 60" in all_text, "GA uses population 48 and 60 generations")
check("FOLD_DIRECTION_MIN = 0.80" in all_text, "Primary screening requires at least 80% fold-direction consistency")
check("TEAM_LOO_DIRECTION_MIN = 0.90" in all_text, "Primary screening requires at least 90% leave-one-team-out consistency")
check("PLACEBO_ALPHA = 0.05" in all_text, "Primary screening uses the prespecified placebo threshold")
check("OVERLAP_R2_HIGH = 0.90" in all_text and "OVERLAP_RESID_RATIO_LOW = 0.25" in all_text, "Severe-overlap thresholds match the manuscript")
check("CompleteCaseTransformer" in all_text, "Complete-case fail-fast handling is implemented")
check("K_VALUES = (2, 3, 4, 5)" in all_text or "K_VALUES = [2, 3, 4, 5]" in all_text, "Clustering sensitivity includes K=2-5")
check("MAIN_K = 3" in all_text or "K3_main_for_DML" in all_text, "K=3 is the prespecified reference clustering")

for passed, message in checks:
    print(("PASS" if passed else "FAIL") + " - " + message)
if not all(passed for passed, _ in checks):
    sys.exit(1)
