# Research reports

This directory contains reviewed experiment summaries, aggregate tables, permitted figures, and report / presentation sources.

- [CP1 pipeline readiness](cp1-pipeline-readiness.md): pre-training implementation and verification.
- [Completed baseline](cp1-baseline-01.md): training, final evaluation, paper comparison, and protocol limits.
- [CP1 result analysis](cp1-analysis-01.md): wound-size errors, empty masks, representative predictions, and exact RGB overlap.
- [CP1 PFOM boundary evaluation](cp1-pfom-01.md): per-image and per-category boundary scores, empty-edge conventions, and verification.

The [fully executed analysis notebook](../notebooks/01_cp1_analysis.ipynb) includes all tables, aggregate figures, and five prediction galleries in Git. See the [notebook instructions](../notebooks/README.md) to regenerate it.

Every report should identify its experiment IDs, dataset split, metric aggregation, and limitations. Attribute selected dataset examples to their source and keep bulk datasets, checkpoints, and personal documents local.
