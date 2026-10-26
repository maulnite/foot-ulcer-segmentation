<div align="center">

<img src="assets/hero.svg" alt="Foot Ulcer Segmentation — reproduce, rethink, reinvent" width="100%" />

<br />

**Reproducing FUSegNet. Understanding its limits. Exploring what comes next.**

![Task](https://img.shields.io/badge/task-binary_segmentation-0F766E?style=flat-square)
![Domain](https://img.shields.io/badge/domain-computer_vision-2563EB?style=flat-square)
![Stage](https://img.shields.io/badge/stage-baseline_training_in_progress-D97706?style=flat-square)

[Overview](#overview) · [Research plan](#research-plan) · [Benchmark](#benchmark) · [Repository](#repository) · [Getting started](#getting-started)

</div>

## Overview

This project investigates **foot ulcer segmentation** through reproduction of FUSegNet and controlled experiments extending the baseline. It is developed for the **Intelligent Systems Lab FILKOM 2026** internship program.

The research follows three steps: **reproduce** the reference method, **rethink** its behavior through analysis and ablations, and **reinvent** through a testable extension. FUSegNet is our benchmark; the project's scope includes both reproduction and further development.

> [!NOTE]
> **Current status: first full baseline training in progress.** CPU/GPU smoke tests and 12 automated checks passed. Run `cp1-baseline-01` uses the audited author implementation and persisted source-data split; final test evaluation follows successful training. See the [experiment log](docs/experiment-log.md), [training guide](docs/training.md), and [reference audit](docs/reference-audit.md). No final reproduction scores are available yet; numbers below are reported by the paper.

## Research plan

| Phase | Focus | Evidence to produce |
| :--- | :--- | :--- |
| **01 · Reproduce** | Audit data, inspect the reference implementation, and reproduce FUSegNet | Documented protocol, trained baseline, metrics, and prediction examples |
| **02 · Rethink** | Examine failure cases and run controlled ablations | Analysis by wound size, boundary quality, and false positives / negatives |
| **03 · Reinvent** | Select and evaluate an extension | Fair comparisons, ablation evidence, and discussion of gains and tradeoffs |

- [ ] **Checkpoint 1 · Tentative late October 2026:** reproduction, a working deep learning model, and initial analysis / ablations.
- [ ] **Checkpoint 2 · Tentative mid-November 2026:** extension experiments, new findings, and final research artifacts.

The extension is **not selected yet**. Candidate research questions and decision criteria live in the [roadmap](docs/roadmap.md).

## Benchmark

**FUSegNet: A Deep Convolutional Neural Network for Foot Ulcer Segmentation**<br />
Mrinal Kanti Dhar, Taiyu Zhang, Yash Patel, Sandeep Gopalakrishnan, and Zeyun Yu · 2023<br />
[Read the reference paper ↗](https://arxiv.org/abs/2305.02961v2)

FUSegNet combines a pretrained **EfficientNet-B7 encoder** with a decoder using **parallel spatial and channel squeeze-and-excitation (P-scSE)** attention. The attention module combines additive and max-out scSE branches, with a shorted variant in the final decoder stage. **x-FUSegNet** averages predictions from five models trained using 5-fold cross-validation.

| Dataset / evaluation | Model | Paper Dice | Our Dice |
| :--- | :--- | ---: | ---: |
| Chronic Wound · data-based | FUSegNet | **92.70%** | — |
| Chronic Wound · image-based | FUSegNet | **86.05%** | — |
| FUSeg Challenge 2021 · data-based | x-FUSegNet | **89.23%** | — |

**Evaluation details matter.** Data-based Dice aggregates pixel counts across the dataset; image-based Dice averages per-image scores. The Chronic Wound images overlap with the FUSeg dataset, and the paper describes the FUSeg test masks as private. These results need separate, clearly identified evaluation protocols.

See the [benchmark protocol](docs/benchmark.md) and [paper reference configuration](configs/paper_reference.yaml) for reported settings and details still to verify.

## Repository

```text
foot-ulcer-segmentation/
├── assets/                 # README visuals
├── configs/                # Reference settings and future run configs
├── data/                   # Local datasets; contents excluded from Git
├── docs/
│   ├── benchmark.md        # Data, architecture, training, and evaluation protocol
│   ├── datasets.md         # Download provenance, inventory, and data audit
│   ├── experiment-log.md   # Run records and results index
│   ├── references.bib      # Reference paper citation
│   ├── reference-audit.md  # Source behavior, CPU probes, and paper discrepancies
│   └── roadmap.md          # Checkpoints, deliverables, and research questions
├── notebooks/              # Future exploratory analysis
├── outputs/                # Local checkpoints and predictions; excluded from Git
├── reports/                # Shareable research findings and figures
├── scripts/                # Reproducible dataset preparation and audit
├── source/                 # Local reference documents; excluded from Git
├── src/                    # Reference-code training and evaluation package
└── tests/                  # Metric parity, checkpoint resume, and data integrity
```

## Getting started

```bash
git clone https://github.com/maulnite/foot-ulcer-segmentation.git
cd foot-ulcer-segmentation
```

1. Read the [roadmap](docs/roadmap.md) and [benchmark protocol](docs/benchmark.md).
2. Read the [dataset audit](docs/datasets.md). Both datasets are prepared locally; use `python scripts/prepare_datasets.py` to download or re-audit them on another checkout. Review the count mismatch, split overlap, mask policy, and licensing before training.
3. Read the [reference implementation audit](docs/reference-audit.md) and [code settings](configs/code_reference.json); carry unresolved discrepancies into the reproduction record.
4. Follow the [training guide](docs/training.md) to create the isolated environment, prepare persisted splits, and run the smoke test.

The runnable baseline is [`configs/baseline.json`](configs/baseline.json). [`requirements-training.lock.txt`](requirements-training.lock.txt) records the validated training/test environment. The paper YAML and code-reference JSON remain separate evidence records. Full training and final test evaluation are separate commands; smoke metrics are diagnostic only.

## Experiment discipline

- Keep validation and test evaluation separate. Select models and tune thresholds using validation data.
- Track seeds, split manifests, environment versions, configuration, metrics, and artifact paths for every substantial run.
- Distinguish **paper results**, **local validation**, and **official challenge evaluation**.
- Change one factor at a time for ablations and compare extensions under the same protocol.
- Keep raw datasets, patient images, model weights, and personal documents out of Git. Publish aggregate results and permitted figures deliberately.

Use the [experiment log](docs/experiment-log.md) to record runs. Weekly updates should cover completed work, decisions, next steps, and blockers.

## Planned deliverables

Public research code and documentation · reproducible experiment records · project report · presentation slides for both checkpoints · best model artifact

Model artifacts can be distributed separately once dataset and model licensing have been checked. No model artifacts are available yet.

<details>
<summary><strong>Reference citation</strong></summary>

```bibtex
@misc{dhar2023fusegnet,
  title         = {FUSegNet: A Deep Convolutional Neural Network for Foot Ulcer Segmentation},
  author        = {Dhar, Mrinal Kanti and Zhang, Taiyu and Patel, Yash and
                   Gopalakrishnan, Sandeep and Yu, Zeyun},
  year          = {2023},
  eprint        = {2305.02961},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2305.02961v2}
}
```

</details>
