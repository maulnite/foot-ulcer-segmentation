# Running the CP1 baseline

The baseline pipeline imports the authors' hash-verified package and uses their dataset, augmentation, loss, and epoch-runner definitions. Scientific quirks identified in the [reference audit](reference-audit.md) are preserved and labeled. The [readiness report](../reports/cp1-pipeline-readiness.md) records verification completed locally.

## Isolated environment

Python **3.10** is required. The prepared `.venv-training` has no access to system site packages and uses **PyTorch 1.11.0+cu113 / torchvision 0.12.0+cu113**, matching the authors' pinned release pair. Scientific dependency versions are recorded in [`requirements-training.txt`](../requirements-training.txt); the complete resolved training + test environment is [`requirements-training.lock.txt`](../requirements-training.lock.txt).

On a fresh checkout:

```bash
python3.10 -m venv .venv-training
.venv-training/bin/python -m pip install --no-cache-dir -r requirements-training.lock.txt
.venv-training/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv-training/bin/python scripts/fetch_reference.py
.venv-training/bin/python -m pip check
```

The lock includes CUDA 11.3 wheel builds, which also execute on CPU. GPU execution still requires a compatible host driver. The server's shared packages are unchanged. The runtime dependency subset omits unrelated original notebook / visualization packages and uses the headless build of OpenCV 4.6.0.66; training uses no GUI operations. Albumentations 1.3.0 is an explicitly pinned version satisfying the original lower bound; the exact version used by the authors was not published.

The original audit-only `.venv` is separate and is not the training environment.

## Data and persisted splits

Both raw datasets are already present locally. On a new checkout, prepare them with `python scripts/prepare_datasets.py` in a suitable environment. To re-audit existing data without downloading compressed archives:

```bash
.venv-training/bin/python scripts/prepare_datasets.py --audit-only
.venv-training/bin/fuseg prepare-splits --config configs/baseline.json
```

The baseline uses Chronic Wound, preserving 831 source training images and 278 test images. It captures the local unsorted image filename order, applies the authors' seed-42 shuffle, and holds out 15%: **707 train / 124 validation / 278 test**.

The manifest is stored at `data/splits/chronic_wound_author_code/manifest.json`. It contains initial file order, final assignments, checksums, and a fingerprint. Reuse verifies the fingerprint and file contents; changed data or split settings cause an error. Preserve this file with the experiment artifacts. It is ignored by Git along with the datasets.

Directories such as `.ipynb_checkpoints` are excluded from image enumeration while retaining image order. The initial local train image folder contains such a directory. No image samples are removed or deduplicated. The published author's original ordering and the paper's 810 / 200 selection remain unrecovered. Python / NumPy / PyTorch training seed 42 is an explicit reproducibility addition; the author's NumPy / PyTorch seeds were not specified.

## Smoke test

```bash
.venv-training/bin/fuseg smoke --device cpu
.venv-training/bin/python -m pytest
```

Smoke mode uses the original ImageNet weights, original augmentation, batch size 2, **one optimizer step on two training images**, and two validation images. It checks finite loss / gradients, an actual encoder weight update, checkpoint save/load, optimizer/scheduler reload, and exact prediction restoration after a deliberate parameter perturbation. It does not perform inference on the test set.

Temporary smoke checkpoints are deleted after successful verification. The status, environment, split manifest, configuration, history, and diagnostic evaluation remain under `outputs/smoke/<timestamp>/`. Their metrics are explicitly marked `diagnostic_smoke_only`; they are not reproduction scores or useful trained models. Smoke mode uses zero loader workers; a separate local check also verified both full-run loaders with four workers.

ImageNet weights are cached once under `outputs/cache/torch/hub/checkpoints/` (approximately 255 MiB). The full training environment occupies approximately 3.9 GiB locally. Dataset archives remain deleted; neither environment nor cached weights is committed.

## Full training

Once GPU resources are available, the runnable command is:

```bash
.venv-training/bin/fuseg train \
  --config configs/baseline.json \
  --device cuda:0 \
  --run-dir outputs/runs/cp1-baseline-01
```

The first full run was launched on **2026-10-26 at 11:31 UTC**, after the GPU became idle and a CUDA smoke test passed. Current progress is recorded in the [experiment log](experiment-log.md) and local run artifacts. Full-run convergence is still being established. The CLI defaults to CPU unless a device is selected explicitly.

The active run uses `scripts/run_cp1_baseline.py` as a detached supervisor. After successful training it runs the final test evaluation once. It stores process / exit state outside the run directory in `outputs/runs/cp1-baseline-01.launch.json` and console output in `outputs/runs/cp1-baseline-01.console.log`. This process continues independently of the chat or interactive terminal.

Check local progress with `python scripts/training_status.py`. It reads the run records without loading a model or using GPU memory.

The baseline uses the reference optimizer, loss, batch size, scheduler, and early-stopping rule, with up to 200 epochs. It preserves sigmoid-probability input to the focal-with-logits function and the original `if loss ... elif IoU` checkpoint policy. Each run records pipeline file hashes, reference revision, Git revision / dirty status, package versions, configuration, and split fingerprints.

`last.pth` stores epoch-boundary model / optimizer / scheduler / policy / RNG state. `best.pth` preserves the selected checkpoint using a hard link, so updating `last.pth` does not overwrite an earlier selected model. Both can occupy roughly 0.8 GiB each once their epochs differ. Checkpoint epoch metadata uses completed-epoch counts rather than the original helper's duplicate increment; this does not change training or selection.

Resume with the same code, environment, configuration, and split:

```bash
.venv-training/bin/fuseg train \
  --config configs/baseline.json \
  --device cuda:0 \
  --run-dir outputs/runs/cp1-baseline-01 \
  --resume
```

Resume checks the saved pipeline hashes and installed package versions. CPU unit tests verify exact next-step continuation on a small stochastic model. GPU bitwise determinism and cross-device equivalence are not claimed. Use one training process per run directory.

## Evaluation

Full training evaluates the selected checkpoint on validation only. Final test inference is a separate explicit command:

```bash
.venv-training/bin/fuseg evaluate \
  --run-dir outputs/runs/cp1-baseline-01 \
  --split test \
  --device cuda:0
```

The evaluator writes JSON and per-image CSV with:

- **`offline_hard_line`**: source offline evaluator semantics, data-based and image-based Dice / IoU / precision / recall.
- **`author_inline_legacy`**: separately labeled data-based scores retaining the inline evaluator's perfect-match counting quirk.

Scores use fractions in `[0,1]`. Aggregate Dice is JSON `null` if the entire collection has no positive ground truth or prediction; the corresponding original formula is undefined. No binary threshold is silently applied to fractional ground truth. The model prediction threshold is `>= 0.5`; training-history metrics retain the original strict `> 0.5` rule.

Every report must state that the source dataset counts differ from the paper and original split order is unavailable. Neither evaluator is established as the one used for the paper's headline table. FUSeg five-fold training, its private challenge test evaluation, PFOM, and novelty experiments are outside this initial Chronic Wound pipeline.
