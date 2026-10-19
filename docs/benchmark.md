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

For each experiment, save a split manifest listing sample IDs and roles. If patient identifiers are available, audit patient overlap and use patient-grouped splits for additional experiments. If this differs from the reference protocol, report it as a separate protocol. Do not invent patient IDs or claim patient independence without evidence.

The paper does not clearly specify the Chronic Wound validation split. Resolve it from the reference implementation if possible. Keep the 200-image test set reserved for final evaluation rather than tuning.

## Model

- Encoder: pretrained EfficientNet-B7, updated during training rather than used solely for zero-shot inference.
- Decoder: upsampling with encoder skip connections and P-scSE attention in the middle of decoder stages.
- P-scSE: parallel additive and max-out scSE branches, followed by elementwise addition.
- Final decoder stage: shorted P-scSE, which bypasses max-out and combines the input feature map with the additive branch output.
- Post-attention processing: convolution, batch normalization, and ReLU; confirm ordering against code because descriptions in the paper vary.
- x-FUSegNet: five separately trained fold models; average pixelwise prediction probabilities, then threshold at 0.5.

Audit channel counts, interpolation, padding, scSE reduction ratios, branch weight sharing, normalization, and pretrained weight source before asserting architectural equivalence.

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

The paper uses geometric and intensity augmentations, listed in Table 1. Its nested probabilities and composition must be checked against code before implementation. Verify exact image standardization, binary mask handling, loss smoothing / reduction, focal alpha / gamma, scheduler monitor, and checkpoint logic. Seeds and fold assignments also need verification.

The machine used for this project has not been audited yet; the authors' hardware is not a statement about our environment.

## Evaluation

**Data-based metrics:** sum TP, FP, and FN over all images before computing the metric. For Dice, use `2 × TP / (2 × TP + FP + FN)` on the aggregate counts.

**Image-based metrics:** compute each metric per image, then average. Establish the behavior for empty ground truth and empty predictions explicitly; it can materially change reported averages.

Report Dice, IoU, precision, and recall, with both aggregation strategies where applicable. Include boundary analysis using a verified PFOM implementation if reproducing the corresponding paper experiments. The paper uses a nonstandard special-case replacement of infinity with 2 for its PFOM analysis. Document this convention separately from standard metric behavior and verify it in code before reproducing the plot.

Never label a local validation score as an official FUSeg test score. If private test evaluation cannot be accessed, report the locally reproducible portion and the remaining limitation.

## Targets reported by the paper

| Dataset | Model | Data-based Dice | Image-based Dice |
| :--- | :--- | ---: | ---: |
| Chronic Wound | FUSegNet | 92.70% | 86.05% |
| FUSeg Challenge 2021 | x-FUSegNet | 89.23% | Not reported in its challenge table |

These are reference values, not acceptance thresholds guaranteeing an identical run. Report deviations in data, code, protocol, and results explicitly.

## Reproduction audit checklist

- [ ] Identify the authors' repository, revision, dependencies, and license.
- [ ] Verify dataset download sources, licensing, counts, masks, and overlap.
- [ ] Record train / validation / test manifests and fold assignments.
- [ ] Verify model details, pretrained weights, and parameter count.
- [ ] Resolve preprocessing, augmentation, loss, and checkpoint ambiguities.
- [ ] Validate metrics on meaningful edge cases, including empty masks.
- [ ] Run a small pipeline smoke check before full training.
- [ ] Record complete run metadata and compare results under matching protocols.
