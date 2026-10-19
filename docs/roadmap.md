# Research roadmap

## Objective

Reproduce FUSegNet as the reference benchmark for foot ulcer segmentation, analyze its behavior, and evaluate an extension supported by controlled experiments. The internship guidebook frames this as **Reproduce → Rethink → Reinvent**.

## Checkpoint 1 · Reproduction and initial analysis

Tentative presentation: **late October 2026**, subject to the program schedule.

1. Confirm the paper choice with the mentor and check that it does not duplicate another group's paper.
2. Audit dataset access, reference code, environment requirements, and compute availability.
3. Finalize the reproducible baseline protocol and document unresolved differences from the paper.
4. Implement data preparation, model, training, and evaluation; validate the pipeline before substantial training.
5. Train and evaluate the baseline, saving configurations, splits, seeds, metrics, and model artifacts.
6. Analyze initial failures and conduct feasible ablations.
7. Prepare the checkpoint report and presentation with clearly attributed paper and local results.

The first target is a defensible reproduction and working deep learning pipeline. Exact numerical equivalence depends on access to the original protocol and data.

## Checkpoint 2 · Extension and new findings

Tentative presentation: **mid-November 2026**, subject to the program schedule.

1. Choose a research question based on baseline evidence and mentor discussion.
2. Define a hypothesis, evaluation protocol, and experiment budget before launching experiments.
3. Compare the extension with the reproduced baseline using matching splits and evaluation.
4. Ablate the extension and analyze uncertainty, failure cases, and compute costs where feasible.
5. Consolidate the final report, presentation, best model artifact, and reproducibility instructions.

## Candidate questions · Not committed contributions

| Direction | Motivation | Evidence needed before selection |
| :--- | :--- | :--- |
| Boundary-sensitive learning | The paper reports missed or incomplete wound boundaries | Boundary error analysis and a reproducible boundary metric |
| Small / superficial wound performance | Small and low-contrast wounds remain difficult | Sufficient samples and a fixed size-based evaluation protocol |
| Accuracy versus compute | The reference uses a large encoder and a five-model ensemble | Baseline memory, latency, and accuracy measured on our hardware |
| Tissue-level segmentation | The paper proposes this as future work | Suitable tissue labels, a compatible dataset, and feasible scope |

These are hypotheses to investigate, not established novelty claims. Review related work and dataset feasibility before choosing a direction. Tissue segmentation cannot be evaluated from binary wound masks alone.

## Deliverables

- Public GitHub repository with research code and documentation.
- Report covering background, objectives, data, methodology, results, discussion, conclusions, future work, and artifact links.
- Presentation slides for each checkpoint.
- Best model artifact and its matching configuration / preprocessing specification.
- Experiment records and weekly progress updates.

## Weekly progress template

```text
Group: [to be filled]
Week: [number / dates]

Completed work:
Learning and findings:
Decisions and rationale:
Challenges / blockers:
Next week's targets:
Experiment IDs and artifact links:
```

The guidebook requires a weekly mentor meeting attended by all members. Written progress is sent over the weekend; its exact format can follow the mentor's preference.
