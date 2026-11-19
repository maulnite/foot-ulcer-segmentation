# Research reports

This directory contains reviewed experiment summaries, aggregate tables, permitted figures, and report / presentation sources.

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
