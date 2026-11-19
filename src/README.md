# Baseline implementation

`foot_ulcer_segmentation` provides the `fuseg` CLI for split preparation, CPU smoke tests, reference-code training, checkpoint resume, and explicitly labeled evaluation. It loads the hash-verified author model and training definitions from the ignored `source/upstream/` snapshot.

See [training instructions](../docs/training.md), [runnable baseline settings](../configs/baseline.json), and [verification results](../reports/cp1-pipeline-readiness.md). The first full baseline run is complete; see the [results and limitations](../reports/cp1-baseline-01.md).
