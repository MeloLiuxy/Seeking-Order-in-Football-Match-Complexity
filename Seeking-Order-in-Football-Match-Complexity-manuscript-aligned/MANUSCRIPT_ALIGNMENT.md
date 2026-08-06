# Manuscript-alignment notes

This copy was revised so that code-level settings follow the current manuscript rather than older exploratory scripts.

Main changes:

- clarified the pass-origin, pass-destination-referenced, carry-endpoint-referenced, and shot-time configurations;
- removed calculation of actual receiving/end-time spatial snapshots from the KPI scripts;
- removed `Avg_2` KPI construction and restricted defensive treatment candidates to the eight manuscript KPIs;
- implemented squared-distance spread for outfield players;
- added a carry-endpoint-referenced KPI construction script;
- replaced the incorrectly uploaded Shot clustering file with a shot-location K = 2–5 sensitivity workflow;
- set Carry DML to the endpoint-referenced configuration only and prevented pass-origin models from using destination-referenced attacking KPIs;
- replaced median/mode imputation with fail-fast complete-case checking;
- changed second-stage inference to unit-weight residual WLS with HC3 standard errors;
- set match-preserving placebo repetitions to 50;
- aligned strategy sensitivity budgets with 1, 2, and 4 SD alternatives around the primary 3-SD budget;
- removed author-specific absolute file paths;
- reduced `requirements.txt` to project dependencies.

The licensed data were not included, so this revision was syntax checked and statically audited but was not used to regenerate the manuscript tables. A full rerun with the licensed analytical data is required before replacing archived numerical outputs.
