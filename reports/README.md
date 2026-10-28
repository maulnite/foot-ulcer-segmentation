# Research reports

This directory contains reviewed experiment summaries, aggregate tables, permitted figures, and report / presentation sources.

- [CP1 pipeline readiness](cp1-pipeline-readiness.md): pre-training implementation and verification.
- [Completed baseline](cp1-baseline-01.md): training, final evaluation, paper comparison, and protocol limits.
- [CP1 result analysis](cp1-analysis-01.md): wound-size errors, empty masks, representative predictions, and exact RGB overlap.
- [CP1 PFOM boundary evaluation](cp1-pfom-01.md): per-image and per-category boundary scores, empty-edge conventions, and verification.
- [Completed attention branch ablations](cp1-attention-ablation-01.md): both single-branch variants, full baseline, and verified comparison.

The [fully executed analysis notebook](../notebooks/01_cp1_analysis.ipynb) includes all tables, aggregate figures, and five prediction galleries in Git. See the [notebook instructions](../notebooks/README.md) to regenerate it.

The [attention ablation workflow](../docs/attention-ablations.md) generates `cp1-attention-ablation-01.md`, per-image/aggregate tables, and a fully executed comparison notebook after both new training runs finish. Diagnostic analysis fixtures remain local and are not reported as ablation results.

The [x-FUSegNet reproduction workflow](../docs/xfusegnet.md) generates `cp1-xfusegnet-01.md`, fold/OOF tables, and a fully executed notebook after five-fold training and test ensemble inference. Official challenge test metrics remain unavailable locally; OOF scores are labeled internal validation.

Every report should identify its experiment IDs, dataset split, metric aggregation, and limitations. Attribute selected dataset examples to their source and keep bulk datasets, checkpoints, and personal documents local.
