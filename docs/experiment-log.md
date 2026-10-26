# Experiment log

**The first full baseline run and final test evaluation are complete.** The source-data run achieves 92.29% data-based test Dice using offline evaluator semantics. See [the run report](../reports/cp1-baseline-01.md) for protocol limits and the separate legacy result.

A reference source audit and CPU-only diagnostic checks have been completed. See [the audit report](reference-audit.md); raw local probe results are in `outputs/reference-audit/report.json`. These checks do not provide reproduction accuracy scores.

Pipeline verification is recorded in [the readiness report](../reports/cp1-pipeline-readiness.md). Final smoke artifacts: `outputs/smoke/cp1-pipeline-02/`. Both source data splits and test-set isolation were preserved; temporary smoke checkpoints were removed after reload verification.

## Run index

| Run ID | Objective | Protocol / split | Seed | Configuration | Results | Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cp1-baseline-01` | Reproduce the public FUSegNet implementation | Chronic Wound source data; 707 train / 124 validation / 278 test | 42 | `configs/baseline.json` | Completed: test data-based Dice 92.29%; image-based Dice 86.42% | `outputs/runs/cp1-baseline-01/` |

## Completed run: cp1-baseline-01

- Launched: **2026-10-26 11:31 UTC** on the idle A100 80 GB GPU, using `cuda:0`.
- Completed: **2026-10-26 13:16:59 UTC**, including final test evaluation. Both stages exited successfully.
- Training stopped at epoch **82** after 30 epochs without source-policy improvement; selected checkpoint: **epoch 52**. Training duration: **105.36 minutes**.
- Final test scores, offline semantics: data-based Dice **92.29%**, IoU **85.68%**, precision **93.47%**, recall **91.13%**; image-based Dice **86.42%**.
- Pipeline commit: `e7c312fcc2db46db34978b7ce3926ef1effb232e`; working tree was clean at run initialization.
- Author source: `60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc`.
- Software: `.venv-training`, Python 3.10.4, PyTorch 1.11.0+cu113 / torchvision 0.12.0+cu113; full environment saved with the run.
- Limits: maximum 200 epochs, early-stopping patience 30, original loss and selection quirks retained.
- Config fingerprint: `86ee03891cb5da1bda31efcba4332b52f345f5a8ae8057b1fb600e6e11dcbc8b`.
- Split fingerprint: `7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925`.
- Supervisor: `scripts/run_cp1_baseline.py`; detached from the interactive terminal. It runs final test evaluation once after successful training, without adjusting thresholds or selecting models against test results.
- Progress: `outputs/runs/cp1-baseline-01/status.json` and `history.json`.
- Process / exit metadata: `outputs/runs/cp1-baseline-01.launch.json`.
- Log: `outputs/runs/cp1-baseline-01.console.log`.
- Checkpoints: `last.pth` and source-policy-selected `best.pth` in the run directory.
- Limitations: source counts differ from the paper, original split order is unrecovered, source duplicates are retained, and evaluator variants are labeled separately.

Do not interpret in-training validation-history scores as held-out test results. The final validation and test reports are now available. The [completed run report](../reports/cp1-baseline-01.md) distinguishes paper numbers, local offline metrics, and legacy inline metrics.

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
