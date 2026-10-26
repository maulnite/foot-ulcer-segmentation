# FUSegNet reference implementation audit

The primary CP1 reproduction should follow the authors' implementation and disclose differences from the paper. This audit establishes source behavior; no training or segmentation accuracy evaluation has been performed.

Subsequent implementation status: CPU/GPU smoke tests and the first baseline training/test run have completed. The [run report](../reports/cp1-baseline-01.md) records results and limits. The findings below describe the earlier source audit.

## Reference and evidence

- Repository: [mrinal054/FUSegNet](https://github.com/mrinal054/FUSegNet).
- Pinned revision: [`60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc`](https://github.com/mrinal054/FUSegNet/tree/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc), May 30, 2023.
- Local snapshot: `source/upstream/FUSegNet/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/`.
- 101 source / documentation files were verified against upstream Git blob hashes. Subsequent runs check local SHA-256 hashes. The upstream MIT license is preserved in the snapshot.
- Machine-readable settings: [`configs/code_reference.json`](../configs/code_reference.json).
- Local probe results: `outputs/reference-audit/report.json`.

The audit executes selected original definitions in isolation, without running top-level training scripts or importing the complete vendored package. The CPU model probe uses original encoder, decoder, head, and model classes with no pretrained weights. Its encoder factory is restricted to B7 using the reference registry's values.

## Dataset count discrepancy: new evidence

The authors' [`categorized_oldDfu.json`](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/categorized_oldDfu.json) contains **278 distinct test filenames**, matching **all 278 local Chronic Wound test images exactly**. Category sizes are `17, 36, 35, 37, 24, 35, 29, 38, 20, 7`.

The dataset repository's commit history lists one change to the linked ZIP, [its addition in May 2020](https://github.com/uwm-bigdata/wound-segmentation/commit/c968e47848dbdce4036a1ec73f7f2144e92758fd). The current test inventory is therefore supported by an artifact in the authors' code repository, rather than just a newly substituted download.

**Verified:** the public code artifacts reference our local test filenames. **Unresolved:** why the paper states 200 test images, the exact training inventory used for its tables, and whether samples were excluded. Matching filenames does not prove that the reported 92.70% score used all 278 images or that mask bytes match the authors' local copies.

Preserve source data for code-faithful reproduction and disclose the count discrepancy. Do not arbitrarily trim to 810 / 200 and claim to have recovered the paper split.

## Split behavior

[`fusegnet_train.py`](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/fusegnet_train.py#L152) and `fusegnet_all.py` read **unsorted** `os.listdir`, calculate `n_val = int(N * 0.15)`, shuffle with `random.seed(42)`, then use the first `N - n_val` samples for training and the rest for validation. The separate test directory is unchanged.

For our 831-image source training folder this yields **707 train + 124 validation**, with **278 separate test images**. The original directory order and final filename lists were not published. Seed 42 alone cannot recover the author's exact split. The inspected training entry points do not set NumPy / PyTorch seeds or deterministic settings.

Persist the initial order and resulting manifests for a local run. If sorting or deduplication is introduced, label it as a change. Existing cross-split duplicates remain documented in the [data audit](datasets.md); no samples were removed during this audit.

[`xfusegnet_train.py`](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/xfusegnet_train.py#L150) shuffles the unsorted list once, then uses `KFold(n_splits=5, shuffle=False)`. It expects the complete labeled pool in one training directory. A five-fold FUSeg run requires explicit assembly of the downloaded train + validation pool; no such run has started.

## Preprocessing and augmentation

- Images: OpenCV BGR read → RGB → ImageNet preprocessing: divide by 255, mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`.
- Masks: OpenCV grayscale read → divide by 255 → add channel dimension. No loading-time binary threshold; raw value 32 becomes approximately 0.12549.
- Tensors: HWC → CHW, float32.
- No resizing is performed by the loader. `IMAGE_SIZE=224` controls the model summary. Validation augmentation is an empty `Compose`.

The [augmentation function](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/fusegnet_train.py#L73) uses outer `Compose(p=0.9)` with four nested `OneOf` blocks:

| Block | Probability | Alternatives |
| :--- | ---: | :--- |
| Flip | 0.5 | Horizontal (`p=0.8`), vertical (`p=0.4`) |
| Affine | 0.9 | Scale, rotate, shift, or combination; each child `p=1` |
| Appearance | 0.2 | Perspective, Gaussian noise, sharpen, blur, motion blur |
| Color | 0.2 | CLAHE, brightness / contrast, gamma, hue / saturation |

Scale limit is 0.5, rotation limit 30 degrees, shift limit 0.1, affine border mode 0, blur limit 3, brightness / contrast limits 0.2. Preserve `OneOf` composition. Library defaults affect unspecified parameters; upstream only specifies `albumentations>=1.3.0`, so a resolved version must be recorded before training.

## Architecture and CPU verification

Use the authors' **modified, vendored** `segmentation_models_pytorch`; an upstream library install alone does not supply their P-scSE implementation. Sources: [decoder](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/segmentation_models_pytorch/decoders/unet/decoder.py), [attention modules](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/segmentation_models_pytorch/base/modules.py), and [model defaults](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/segmentation_models_pytorch/decoders/unet/model.py).

| Item | Verified behavior |
| :--- | :--- |
| Encoder | EfficientNet-B7; `efficientnet-pytorch==0.7.1`; ImageNet weights for training |
| Encoder feature channels | `(3, 64, 48, 80, 224, 640)`; depth 5 |
| Decoder channels | `(256, 128, 64, 32, 16)` |
| Upsampling | Nearest-neighbor, factor 2 |
| Attention | Separate max-out and additive scSE branch weights; reduction ratio 16 |
| Fusion | Sum attention branches after concatenating upsampled and skip features |
| Post-attention | One **Conv → BatchNorm → ReLU** block per stage |
| Final decoder stage | Input + additive scSE, max-out bypassed |
| Segmentation head | 3×3 convolution to one channel, then sigmoid |

CPU probe passed: `[1,3,224,224]` → finite output `[1,1,224,224]`. No pretrained weights were downloaded and no accuracy was evaluated.

The model registers **70,968,939 parameters**. Modules called during forward contain **64,898,248 parameters**, matching the paper's rounded **64.90M**. The remaining **6,070,691** belong to unused registered layers: decoder `conv2` blocks, concatenation projections, the bypassed final max-out branch, and encoder head layers. Removing them changes state-dict structure. The apparent parameter discrepancy is explained by counting convention.

## Loss and checkpoint quirks

The training loss is `utils.DiceLoss() + utils.FocalLoss()`: soft batch-level Dice with epsilon 1.0 / beta 1, plus binary focal with gamma 2, alpha **None**, mean reduction, no normalization.

**Confirmed:** sigmoid model outputs are passed directly to a focal function using `binary_cross_entropy_with_logits`. They are treated as logits. Replacing this with a correctly paired logits/probability loss changes the public code's objective. For probabilities `[0.01,0.99]` and targets `[0,1]`, the original focal loss is **0.09973956**, versus **0.000001005** after converting probabilities to logits. These are diagnostic values, not research results. Sources: [training setup](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/fusegnet_train.py#L229), [loss class](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/segmentation_models_pytorch/utils/losses.py#L89), [focal function](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/segmentation_models_pytorch/losses/_functional.py#L36).

Training uses Adam over all parameters, LR `1e-4`, weight decay `1e-5`, batch size 2, four loader workers, and up to 200 epochs. ReduceLROnPlateau monitors validation loss with factor 0.1, patience 10, and **minimum LR `1e-5`**. Encoder weights are not frozen.

The [checkpoint condition](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/fusegnet_train.py#L341) is **`if loss improves ... elif IoU improves`**. The stored best IoU is not updated on loss improvement. Executing the original conditional confirms that it can later save a checkpoint with worse loss and worse IoU than a preceding epoch because the stored IoU is stale. Early stopping patience 30 resets on either save branch. Replacing this with independent all-time-best trackers is a change.

Validation history consists of batch-level scores averaged across batches, which differs from final dataset-level metrics.

## Evaluation implementations disagree

Inference thresholds predictions at **`>=0.5`**, while training metric code uses **`>0.5`**. The distinction at exactly 0.5 was verified. x-FUSegNet includes both mean-probability averaging and majority voting; the averaging branch matches the paper's description.

| Evaluator | Perfect matching masks | Empty / empty masks |
| :--- | :--- | :--- |
| Inline `fusegnet_test.py` / `fusegnet_all.py` | Counts every pixel as TP, including background | Image-size TP |
| Offline `utils/eval.py`, `HARD_LINE=True` | Ordinary confusion counts for nonempty masks | Per-image scores 100%; all pixels TN |

The original conditionals were executed on synthetic masks. A four-pixel empty pair yields `(TN,FP,FN,TP)=(0,0,0,4)` inline and `(4,0,0,0)` offline. A matching two-background / two-foreground pair yields `(0,0,0,4)` inline and `(2,0,0,2)` offline. This can alter aggregated scores. The offline evaluator assigns per-image scores 0% for empty GT with any positive prediction and includes false positives in aggregate counts. Sources: [inline](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/fusegnet_test.py#L292), [offline](https://github.com/mrinal054/FUSegNet/blob/60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc/utils/eval.py#L53).

The offline evaluator matches the paper's empty-mask description more closely, but source publication does not establish which evaluator produced the headline table. Record evaluator identity and keep their scores separately labeled. No PFOM / Pratt / Canny implementation was found in the selected Python source; boundary evaluation still needs implementation and verification.

## Environment and next steps

The server has PyTorch **1.13.1+cu116** / torchvision **0.14.1+cu116**; upstream pins **1.11.0 / 0.12.0**. Albumentations, OpenCV, timm, and pretrainedmodels were absent. An A100 80 GB GPU is exposed and was already active when checked; this audit used CPU only.

A project-local `.venv` with access to existing system packages was created for these probes. Only missing `efficientnet-pytorch==0.7.1` was installed into it, with `--no-deps`. No shared packages were replaced. This is **not a validated full training environment** or software-version parity.

Before full training: assemble an isolated training environment; persist local file order, split manifests, and seeds; implement the audited source behavior with any compatibility changes documented; label evaluator differences; smoke-test a training step, checkpoint reload, and evaluation. Corrected loss, checkpoint, or split variants belong in separate experiments.

Rerun the prepared audit environment with:

```bash
.venv/bin/python scripts/audit_reference.py --model-check
```

On a new checkout, supply PyTorch, NumPy, scikit-learn, and `efficientnet-pytorch==0.7.1` in a compatible environment. The script fetches the pinned source if missing, without retaining a compressed archive. Omitting `--model-check` skips the full model check but still runs the focused loss, threshold, checkpoint, and evaluator probes.
