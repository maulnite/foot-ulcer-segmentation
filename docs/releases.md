# Releases and research artifacts

## Current package

[CP2 final research · 21 November 2026](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp2-final-2026-11-21)
is the current research package. Its tag identifies the merged code, completed
reports, configurations, tests, and presentation evidence on `main`.

| Resource | Location |
| :--- | :--- |
| All available original weights | [Hugging Face archive](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights) |
| Browse each model/run | [Checkpoint catalog](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights/blob/main/CATALOG.md) |
| Download/load/verify | [Model downloads](model-checkpoints.md) |
| Final research results | [Cascade](../reports/cp2-cascade-clean.md) · [All phases](../reports/README.md) |
| Figures for the presentation | [Final slides 13–16 index](../reports/cp2-presentation-final-visual-index.md) |
| Partner source implementations | [Ganta branch](https://github.com/maulnite/foot-ulcer-segmentation/tree/ganta/notebook) |

The release attaches these compact artifacts:

- `MODEL_ARCHIVE.md`: collection links, checkpoint roles, loading context, and limitations.
- `model-index.json`: every available checkpoint path, original metadata, SHA256, byte size, and download URL.
- `SHA256SUMS.txt`: all 171 checkpoint checksums, using paths inside the Hub archive.
- `publication-validation.json`: verified Hub revision and checkpoint/source-preservation evidence.
- `cp2-cascade-clean-results.json`: unchanged canonical final Cascade results.
- `cp2-presentation-final.zip`: the 13 final scientific figures, companion metadata, visual index, and supporting context.
- `integration-validation.json`: release preparation checks, tests, merge identity, and preservation evidence.
- `release-assets-SHA256SUMS.txt`: checksums for the other release assets.

All 171 checkpoint binaries remain on Hugging Face; the GitHub release links
them and provides integrity metadata. Downloading GitHub's automatic source
archive gives the committed code/reports, not the raw datasets or model binaries.
Pin Hub revision `b9cf18ef27a2583f89dc3a93614721c74926504c` when reproducing
the published model collection. See the archive's `RECIPES.md` for the correct
two-file Cascade bundles and five-fold x-FUSegNet groups.

The archive has 134 Orie and 37 Ganta checkpoint paths. Selected checkpoints,
last/resume states, initializers, and smoke outputs are labeled separately.
Eight historical Ganta directories without weights are documented; no missing
weight was synthesized or retrained. Counts describe checkpoint paths, not
distinct architectures. Different experiment populations and selection rules
must not be treated as a single comparable leaderboard.

## Historical CP1 releases

These releases and their original assets remain available:

- [Orie x-FUSegNet five-fold checkpoints](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-orie-weights-v1).
- [Ganta x-FUSegNet five-fold checkpoints](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-ganta-weights-v1).

They retain their original cohorts, implementation differences, and metadata.
The new release does not change historical checkpoints, metrics, split manifests,
research conclusions, or source notices. Neither the release nor the archive
establishes a new license for the underlying datasets or weights.
