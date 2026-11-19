# Experiment log

**The first full baseline run and final test evaluation are complete.** The source-data run achieves 92.29% data-based test Dice using offline evaluator semantics. See [the run report](../reports/cp1-baseline-01.md) for protocol limits and the separate legacy result.

**Initial CP1 analysis is also complete.** The [analysis report](../reports/cp1-analysis-01.md) and [notebook](../notebooks/README.md) cover learning curves, wound-size errors, empty-GT false positives, representative predictions, and exact RGB overlap with the actual 707-image training split. All 402 cached validation/test masks match final evaluation counts. This is inference and analysis of checkpoint 52, without a new training run or test-based tuning.

A reference source audit and CPU-only diagnostic checks have been completed. See [the audit report](reference-audit.md); raw local probe results are in `outputs/reference-audit/report.json`. These checks do not provide reproduction accuracy scores.

Pipeline verification is recorded in [the readiness report](../reports/cp1-pipeline-readiness.md). Final smoke artifacts: `outputs/smoke/cp1-pipeline-02/`. Both source data splits and test-set isolation were preserved; temporary smoke checkpoints were removed after reload verification.

## Run index

**October 29, x-FUSegNet completed:** all five folds finished at 124/200/156/129/111 epochs, selecting checkpoints 94/198/126/99/81. Total training time was **22.57 hours**. All 1,010 internal OOF predictions and 200 five-model test means/masks were verified. Internal OOF: **88.03% image Dice, 92.72% data Dice, 0.7315 PFOM**. Official test metrics remain unavailable because test masks are absent locally. [Completed report](../reports/cp1-xfusegnet-01.md) · [Fully executed notebook](../notebooks/03_cp1_xfusegnet.ipynb).

**October 28, x-FUSegNet reproduction:** source loop from `xfusegnet_train.py` is executed with recording/resume hooks; numeric AST parity, five-fold native-resolution GPU smoke, and frozen training dependencies are checked before launch. The 1,010-image labeled FUSeg pool uses 808/202 per fold; 200 unlabeled test images are reserved for mean-probability ensemble inference. OOF validation is reported separately from official challenge test metrics, which cannot be computed locally. [Protocol and live status](xfusegnet.md).

**October 27, attention branch ablations:** predeclared additive-only and maxout-only runs against the completed full baseline. Only decoder stages 0–3 change. Matched initial tensors, split, frozen training dependencies, source engine, and evaluator were verified. All 26 tests and both GPU smoke tests passed. [Protocol and monitoring](attention-ablations.md); run results are recorded by the sequential workflow after training, without treating diagnostic fixture outputs as results.

**October 27, PFOM addition:** evaluated the same 124 validation / 278 test cached mask pairs at epoch 52 and threshold 0.5, without inference or training. Mean PFOM with the documented 1/0 empty-edge policy: **0.7244 validation / 0.7782 test**. Sentinel-2 proxy: **0.8696 / 0.8933**. Nine analytic/oracle tests passed. Dependency additions are isolated to `.venv-analysis`; frozen training files and region metrics are unchanged. [Protocol](pfom.md) · [PFOM report](../reports/cp1-pfom-01.md).

| Run ID | Objective | Protocol / split | Seed | Configuration | Results | Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cp1-baseline-01` | Reproduce the public FUSegNet implementation | Chronic Wound source data; 707 train / 124 validation / 278 test | 42 | `configs/baseline.json` | Completed: test data-based Dice 92.29%; image-based Dice 86.42% | `outputs/runs/cp1-baseline-01/` |
| `cp1-attention-additive-01` | Measure contribution of the maxout branch by removing it | Same frozen Chronic Wound split | 42 | `configs/ablation/additive.json` | See workflow / final comparison report; no pre-training accuracy claim | `outputs/runs/cp1-attention-additive-01/` |
| `cp1-attention-maxout-01` | Measure contribution of the additive branch by removing it | Same frozen Chronic Wound split | 42 | `configs/ablation/maxout.json` | See workflow / final comparison report; no pre-training accuracy claim | `outputs/runs/cp1-attention-maxout-01/` |
| `cp1-xfusegnet-01` | Reproduce five-fold FUSegNet and probability-mean ensemble | FUSeg 512×512; 1,010 labeled pool, 200 unlabeled test | 42, continuous RNG across folds | `configs/xfusegnet.json` | Completed: internal OOF image Dice 88.03%, data Dice 92.72%, PFOM 0.7315; official test score unavailable | `outputs/runs/cp1-xfusegnet-01/` |

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
