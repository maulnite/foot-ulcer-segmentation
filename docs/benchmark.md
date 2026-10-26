# Benchmark protocol

This document separates settings reported in **FUSegNet, arXiv:2305.02961v2** from decisions that still need verification. It is a starting protocol, not a claim of completed reproduction.

## Dataset identity and split integrity

| Property | Chronic Wound | FUSeg Challenge 2021 |
| :--- | :--- | :--- |
| Total images in the paper | 1,010 | 1,210 |
| Training images | 810 | 1,010 |
| Test images | 200 | 200 |
| Input resolution | 224 × 224 | 512 × 512 |
| Image view | Cropped wound regions | Full images with more complex backgrounds |
| Test masks in the paper | Publicly available | Held privately by challenge organizers |

The paper says 1,010 images occur in both datasets, with different views. **Do not treat them as independent datasets.** Before any cross-dataset experiments, map corresponding samples and audit overlap. Verify current access and licensing when obtaining data; the table describes the paper's datasets.

**Local source audit:** the linked Chronic Wound ZIP actually contains 831 train and 278 test samples, differing from the paper counts above. Exact image duplicates also cross both datasets' provided splits, and one FUSeg validation mask has nonbinary pixels. See [dataset preparation and audit](datasets.md) for verified counts and details. The downloaded source split is not yet an exact paper reproduction protocol.

For each experiment, save a split manifest listing sample IDs and roles. If patient identifiers are available, audit patient overlap and use patient-grouped splits for additional experiments. If this differs from the reference protocol, report it as a separate protocol. Do not invent patient IDs or claim patient independence without evidence.

The paper does not clearly specify the Chronic Wound validation split. The code uses a 15% holdout after a Python seed-42 shuffle of an unsorted directory listing. On our 831 source training images, this gives 707 train / 124 validation, with the separate 278-image test set preserved. The author's original filename order / exact split was not published. Their test-category JSON matches all 278 local test filenames. See the [reference audit](reference-audit.md) for evidence and the unresolved paper count discrepancy. Keep the source test set reserved for final evaluation rather than tuning.

## Model

- Encoder: pretrained EfficientNet-B7, updated during training rather than used solely for zero-shot inference.
- Decoder: upsampling with encoder skip connections and P-scSE attention in the middle of decoder stages.
- P-scSE: parallel additive and max-out scSE branches, followed by elementwise addition.
- Final decoder stage: shorted P-scSE, which bypasses max-out and combines the input feature map with the additive branch output.
- Post-attention processing: verified Conv → BatchNorm → ReLU in the source; wording in the paper varies.
- x-FUSegNet: five separately trained fold models; average pixelwise prediction probabilities, then threshold at 0.5.

The [reference audit](reference-audit.md) verifies channel counts, nearest-neighbor upsampling, reduction ratio 16, separate attention branch weights, and Conv-BN-ReLU ordering. A CPU check gives the expected output shape and 64,898,248 parameters in modules used by forward, matching the paper's rounded 64.90M. Unused registered layers bring the full parameter count to 70,968,939.

## Training

| Setting | Reported value |
| :--- | :--- |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Batch size | 2 |
| Epoch limit | 200 |
| Loss | Dice loss + focal loss, equal weights |
| LR scheduler | ReduceLROnPlateau; factor 0.1, patience 10 |
| Early stopping | Patience 30 |
| Checkpoint | Updated when validation loss decreases or validation IoU increases |
| Hardware used by authors | One NVIDIA RTX 2080 Ti |

The [reference audit](reference-audit.md) resolves augmentation composition, ImageNet normalization, mask division by 255, Dice epsilon 1.0, focal gamma 2 / alpha None, scheduler monitor / minimum LR, and checkpoint branching. It also confirms source quirks: sigmoid probabilities are passed to focal-with-logits loss, and the checkpoint IoU tracker is stale after loss improvements. Label source behavior in the primary code-faithful run and identify corrected variants separately. Original training seeds / split manifests remain unavailable.

The server exposes an A100 80 GB GPU. The initial code audit used CPU and a different runtime pair. The subsequent isolated `.venv-training` uses the original PyTorch 1.11.0 / torchvision 0.12.0 pair and passed CPU training / checkpoint smoke tests; see the [readiness report](../reports/cp1-pipeline-readiness.md). GPU execution and full training remain untested while shared GPU resources are occupied.

## Evaluation

**Data-based metrics:** sum TP, FP, and FN over all images before computing the metric. For Dice, use `2 × TP / (2 × TP + FP + FN)` on the aggregate counts.

**Image-based metrics:** compute each metric per image, then average. Establish the behavior for empty ground truth and empty predictions explicitly; it can materially change reported averages.

Report Dice, IoU, precision, and recall, with both aggregation strategies where applicable. Include boundary analysis using a verified PFOM implementation if reproducing the corresponding paper experiments. The paper uses a nonstandard special-case replacement of infinity with 2 for its PFOM analysis. Document this convention separately from standard metric behavior and verify it in code before reproducing the plot.

Never label a local validation score as an official FUSeg test score. If private test evaluation cannot be accessed, report the locally reproducible portion and the remaining limitation.

The authors' inline inference evaluator and offline `utils/eval.py` disagree on perfect matches, including empty masks. The audit executes both source branches and documents the discrepancy. Record evaluator identity rather than treating their scores as interchangeable. No PFOM implementation was found in the selected upstream Python sources.

## Targets reported by the paper

| Dataset | Model | Data-based Dice | Image-based Dice |
| :--- | :--- | ---: | ---: |
| Chronic Wound | FUSegNet | 92.70% | 86.05% |
| FUSeg Challenge 2021 | x-FUSegNet | 89.23% | Not reported in its challenge table |

These are reference values, not acceptance thresholds guaranteeing an identical run. Report deviations in data, code, protocol, and results explicitly.

## Reproduction audit checklist

- [x] Identify the authors' repository, revision, dependency requirements, and license.
- [ ] Verify dataset download sources, licensing, counts, masks, and overlap.
- [x] Persist the Chronic Wound local train / validation / test manifest and file checksums; original author assignments remain unrecovered.
- [x] Audit model details, pretrained weight source, and parameter counting; pass a CPU shape check.
- [x] Resolve source preprocessing, augmentation, loss, and checkpoint behavior; document conflicts.
- [x] Validate metrics against original offline code on edge cases, including empty masks.
- [x] Pass a CPU training / checkpoint / evaluation smoke check before full training.
- [ ] Record complete run metadata and compare results under matching protocols.
