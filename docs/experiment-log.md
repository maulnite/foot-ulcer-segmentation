# Experiment log

**No full reproduction training experiments have been run yet.** Two one-step CPU smoke runs have verified the pipeline; their metrics are diagnostic only. This file indexes substantial experiments; detailed findings and permitted figures belong in `reports/`.

A reference source audit and CPU-only diagnostic checks have been completed. See [the audit report](reference-audit.md); raw local probe results are in `outputs/reference-audit/report.json`. These checks do not provide reproduction accuracy scores.

Pipeline verification is recorded in [the readiness report](../reports/cp1-pipeline-readiness.md). Final smoke artifacts: `outputs/smoke/cp1-pipeline-02/`. Both source data splits and test-set isolation were preserved; temporary smoke checkpoints were removed after reload verification.

## Run index

| Run ID | Objective | Protocol / split | Seed | Configuration | Results | Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |

Use a stable ID such as `YYYYMMDD-baseline-01`. Keep failed and interrupted runs documented when they affect interpretation or reproducibility.

## Run record template

```text
Run ID:
Date:
Objective / hypothesis:
Git commit:
Environment and dependency versions:
Hardware:
Dataset source / version / license:
Split manifest and overlap audit:
Seed and determinism settings:
Configuration path:
Pretrained weights source:
Changes from the baseline:
Training duration and peak GPU memory:
Checkpoint selection criterion:
Evaluation split and aggregation:
Dice / IoU / precision / recall:
Boundary metrics, if applicable:
Artifact paths:
Findings and limitations:
Next action:
```

Report validation, held-out test, and official challenge scores separately. Include repeat-run variability when repeat runs are performed; do not imply statistical significance from a single run.
