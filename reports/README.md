# Research reports

This directory contains reviewed experiment summaries, aggregate tables, permitted figures, and report / presentation sources.

## Completed CP2 research

The controlled source cohort is **671 train / 118 validation / 261 source-test**,
using exact-RGB grouping rather than patient identities. Research outcomes and
phase-specific validation guards remain in the original reports.

| Topic | Report |
| :--- | :--- |
| Protocol / corrected baseline | [P](cp2-phase-p.md) |
| Loss / attention | [L](cp2-phase-l.md) · [A screening review](cp2-phase-a-review.md) |
| Auxiliary / presence supervision | [H](cp2-phase-h.md) · [N](cp2-phase-n.md) |
| Combinations | [A1+H2](cp2-phase-c.md) · [B1+A6+H2 and L3+A6+H2](cp2-phase-c-combinations.md) |
| Seed / recipe sensitivity | [Three-seed confirmation](cp2-confirmation.md) · [CP1 recipe](cp2-cp1-recipe.md) · [Matched CP1](cp1-matched.md) |
| Final Cascade | [Completed controlled Cascade](cp2-cascade-clean.md) · [Original Ganta replay](ganta-cascade-replay.md) |
| External transfer | [CO2Wounds frozen-checkpoint evaluation](cp2-external-evaluation.md) |
| Presentation | [Original visual index](cp2-presentation-visual-index.md) · [Final slides 13–16](cp2-presentation-final-visual-index.md) |
| All weights | [Orie + Ganta publication and integrity report](huggingface-weights-publication.md) |

Cascade improves external aggregate scores but increases GT-positive total
misses (143→153); this limitation remains part of the final findings. Different
CP1/CP2/partner protocols are not pooled into one ranking.

[Latest downloadable release](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp2-final-2026-11-21)
· [Model archive](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights)
· [Release and artifact guide](../docs/releases.md).

## Historical CP1 research

- [CP1 pipeline readiness](cp1-pipeline-readiness.md): pre-training implementation and verification.
- [Completed baseline](cp1-baseline-01.md): training, final evaluation, paper comparison, and protocol limits.
- [CP1 result analysis](cp1-analysis-01.md): wound-size errors, empty masks, representative predictions, and exact RGB overlap.
- [CP1 PFOM boundary evaluation](cp1-pfom-01.md): per-image and per-category boundary scores, empty-edge conventions, and verification.
- [Completed attention branch ablations](cp1-attention-ablation-01.md): both single-branch variants, full baseline, and verified comparison.
- [Completed x-FUSegNet reproduction](cp1-xfusegnet-01.md): five-fold training, internal OOF validation, and 200 verified test ensemble predictions.
- [Actual-train EDA and original preprocessing](cp1-eda-01.md): 707 saved-manifest training pairs, foreground/background imbalance, mask-area distribution, deterministic photo–GT examples, and paired augmentation demos. Includes the [executed notebook](../notebooks/04_cp1_eda.ipynb), [PNG/SVG figures](figures/cp1-eda-01/), tables, manifest copy, replay metadata, and verification. Descriptive EDA does not establish method superiority.

The [fully executed analysis notebook](../notebooks/01_cp1_analysis.ipynb) includes all tables, aggregate figures, and five prediction galleries in Git. See the [notebook instructions](../notebooks/README.md) to regenerate it.

The [attention ablation workflow](../docs/attention-ablations.md) generates `cp1-attention-ablation-01.md`, per-image/aggregate tables, and a fully executed comparison notebook after both new training runs finish. Diagnostic analysis fixtures remain local and are not reported as ablation results.

The [x-FUSegNet reproduction workflow](../docs/xfusegnet.md) completed five-fold training, OOF/test inference, and the [executed notebook](../notebooks/03_cp1_xfusegnet.ipynb). Internal OOF image Dice is 88.03%, data Dice 92.72%, and mean PFOM 0.7315. Official challenge test metrics remain unavailable locally; OOF scores are labeled internal validation.

Every report should identify its experiment IDs, dataset split, metric aggregation, and limitations. Attribute selected dataset examples to their source and keep bulk datasets, checkpoints, and personal documents local.
