# Hugging Face · Orie and Ganta checkpoint publication

**COMPLETED — all available checkpoints published and verified.** No training, inference, model conversion, epoch reselection, or shared-environment changes.

Archive: [maulnite/foot-ulcer-segmentation-weights](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights).
Final Hub revision: `b9cf18ef27a2583f89dc3a93614721c74926504c`.
Completed: **21 November 2026, 10:49:39 UTC / 17:49:39 WIB**.

## Contents

| Collection | Checkpoint paths |
| :--- | ---: |
| orie/cp2 | 118 |
| orie/cp1 | 16 |
| ganta/chronic-wound | 11 |
| ganta/fuseg-2021 | 26 |

Total **171 checkpoint paths**, **141 unique file contents**, 119.52 GiB logical / 100.89 GiB before Hub chunk-level deduplication. These are checkpoint counts, not 171 different architectures.

Roles: 73 historical selected checkpoints, 38 final/resume checkpoints, 11 initializers, 45 engineering-smoke checkpoints, and four original Cascade stage checkpoints. Roles and recorded run status are explicit in the [catalog](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights/blob/main/CATALOG.md).

Both original Ganta Cascade stages are present for FUSeg 2021 and Chronic Wound. The CP2 Cascade bundle keeps frozen B1 and the selected refiner in their original run directories; `RECIPES.md` identifies both. The five original x-FUSegNet fold checkpoints remain separate.

## Verification

- Every checkpoint was deserialized on CPU in the existing torch 1.11 project environment; original bytes were not rewritten.
- SHA256 and size matched Hub LFS metadata for all 171 checkpoint paths; 962 additional documentation/source/metadata files matched remote Git blobs.
- Source checkpoints were rehashed after upload. All 610 protected research source/report/notebook files retained their original hashes.
- All 26 frozen external-registry entries and 49 selected-checkpoint status digests matched the publication inventory.
- Anonymous access and SHA256/size checks passed for a B1 checkpoint and a Ganta Cascade refiner, without downloading whole checkpoint files again.
- **254 tests passed**, including 11 new publication tests; 29 existing warnings, zero errors/failures. Python compilation and strict JSON checks passed. Ruff was unavailable and no dependency installation was performed.
- No raw dataset images, GT masks, prediction caches, credentials, or GPU work were included. Machine-specific paths in copied text metadata were sanitized.

[Machine-readable publication record](huggingface-weights-publication.json) · [Hub verification record](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights/blob/main/publication-validation.json) · [Reproduction instructions](../docs/huggingface-weights.md).

## Timing and execution

- Preparation: 2026-11-21T10:27:29.503171+00:00 → 2026-11-21T10:30:17.797663+00:00.
- Full upload: 2026-11-21T10:30:41.090431+00:00 → 2026-11-21T10:47:43.731599+00:00 (about 17 minutes).
- Final checksum/source-preservation checks completed 2026-11-21T10:49:39.822910+00:00.
- Three SDK workers; existing Xet backend; resumable task state and persistent logs remain in `outputs/model-publication/20261121T102717Z`. Staging uses symbolic links, avoiding a second 120 GiB checkpoint copy.

## Completeness and limitations

Eight historical Ganta directories contain no checkpoint. They are listed below rather than being retrained or represented as uploaded weights. No model is claimed to exist when its file is absent.

- `ganta/output_chronic_wound/checkpoints/cl_dpsc_chronic_wound_efficientnet-b7_2026-11-08_13-21-03`
- `ganta/output_fuseg_2021/checkpoints/bw_ftl_efficientnet-b7_2026-11-06_20-26-27`
- `ganta/output_fuseg_2021/checkpoints/lg_cl_bw_efficientnet-b7_2026-11-06_20-26-58`
- `ganta/output_fuseg_2021/checkpoints/lg_cl_bw_efficientnet-b7_2026-11-15_13-12-21`
- `ganta/output_fuseg_2021/checkpoints/lg_pscse_efficientnet-b7_2026-11-06_20-16-55`
- `ganta/output_fuseg_2021/checkpoints/xFUSegNet_efficientnet-b7_2026-10-29_20-38-42`
- `ganta/output_fuseg_2021/checkpoints/xFUSegNet_efficientnet-b7_2026-10-29_20-50-44`
- `ganta/output_fuseg_2021/checkpoints/xFUSegNet_efficientnet-b7_2026-10-30_00-26-09`

The excluded upstream EfficientNet download cache and 20 RNG-only files are listed in the inventory; they are not experiment model weights. Partial historical Ganta runs retain their identities. Different contributors, phases and datasets use different protocols, so the archive is not a combined performance ranking. Original source notices remain in force; this publication does not establish a new dataset or weight license.

No historical training/evaluation artifacts, model selections, canonical metrics or pre-existing user notebook changes were overwritten. The publication adds its runner, tests, documentation and this report on the existing CP2 branch.
