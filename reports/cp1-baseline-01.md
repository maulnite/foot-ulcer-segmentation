# CP1 baseline: first completed run

**Training and final test evaluation completed successfully.** This run reproduces the public FUSegNet implementation on the downloaded Chronic Wound source dataset; exact equivalence to the paper's sample selection is not established.

## Run record

| Item | Value |
| :--- | :--- |
| Run ID | `cp1-baseline-01` |
| Pipeline commit | `e7c312fcc2db46db34978b7ce3926ef1effb232e` |
| Author revision | `60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc` |
| Seed | 42 |
| Split | 707 train / 124 validation / 278 test |
| Training start | October 26, 2026, 11:31:11 UTC |
| Training end | October 26, 2026, 13:16:32 UTC |
| Workflow end, including test | October 26, 2026, 13:16:59 UTC |
| Training duration | 105.36 minutes |
| Epochs completed | 82 of at most 200 |
| Stop reason | 30 consecutive epochs without a source-policy checkpoint selection |
| Selected checkpoint | Epoch 52, using the audited author's loss/IoU policy |
| Final learning rate | `1e-5` |
| Hardware | NVIDIA A100 80 GB |
| Peak allocated CUDA memory | 2,386,764,288 bytes (approximately 2.22 GiB) |
| Software | Python 3.10.4, PyTorch 1.11.0+cu113, torchvision 0.12.0+cu113 |

Training and test subprocesses both exited with code 0. Checkpoint reload passed. Final evaluation uses the selected checkpoint, not the final epoch's weights or the epoch maximizing another metric.

## Primary evaluation: offline HARD_LINE semantics

All values below are percentages. These are final evaluation metrics; training-history metrics use a different batch-averaging convention.

| Metric | Validation, 124 images | Test, 278 images |
| :--- | ---: | ---: |
| Data-based Dice | 85.90 | **92.29** |
| Data-based IoU | 75.29 | **85.68** |
| Data-based precision | 81.63 | **93.47** |
| Data-based recall | 90.65 | **91.13** |
| Image-based Dice | 82.34 | **86.42** |
| Image-based IoU | 75.27 | **80.01** |
| Image-based precision | 83.63 | **88.17** |
| Image-based recall | 83.78 | **86.81** |

Test aggregate counts: TP 354,409; TN 13,535,296; FP 24,743; FN 34,480. Prediction threshold is `>=0.5`. There was one final test evaluation after the validation-based selection policy finished; test results did not determine the selected checkpoint or threshold.

## Comparison with paper reference numbers

| Metric | Paper, Chronic Wound | This source-data run | Difference, percentage points |
| :--- | ---: | ---: | ---: |
| Data-based Dice | 92.70 | 92.29 | −0.41 |
| Image-based Dice | 86.05 | 86.42 | +0.37 |

The scores are numerically close, but this is **not an identical-protocol comparison**. The paper states 810 source training / 200 test images, while the public archive contains 831 / 278. The authors' test-category JSON matches our 278 filenames, but original train/validation assignments and the evaluator used for the headline paper table remain unknown. Source cross-split duplicates are retained. These limits preclude interpreting the score difference as a controlled performance gain or loss.

## Legacy inline evaluator: separate result

For the same predictions, the author's inline perfect-match counting behavior produces **96.66% test data-based Dice**, compared with **92.29%** under offline semantics. Validation legacy Dice is **90.85%**, compared with **85.90%** offline. These values must not be interchanged. The primary results above use the offline evaluator; see the [audit](../docs/reference-audit.md) for the confirmed counting discrepancy.

## Artifacts and remaining CP1 work

- `outputs/runs/cp1-baseline-01/config.json`, `environment.json`, and `split-manifest.json`: exact run setup and provenance.
- `history.json`: all 82 completed epochs.
- `best.pth`: selected epoch 52; `last.pth`: epoch 82. Both remain local.
- `validation.json` / `.csv`: final validation metrics and per-image results.
- `evaluation-test.json` / `.csv`: held-out test metrics and per-image results.
- `status.json` and `outputs/runs/cp1-baseline-01.launch.json`: successful completion records.

Config fingerprint: `86ee03891cb5da1bda31efcba4332b52f345f5a8ae8057b1fb600e6e11dcbc8b`.
Split fingerprint: `7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925`.

The first baseline training, test evaluation, and [initial error analysis](cp1-analysis-01.md) are complete. A [fully executed local notebook](../notebooks/README.md) provides learning curves, size-based analysis, empty-mask errors, prediction galleries, and actual-split overlap counts. Method ablations have not been run; the final CP1 report and presentation remain to be prepared. No novelty claim, private FUSeg challenge score, or tissue-level segmentation result is made by this run.
