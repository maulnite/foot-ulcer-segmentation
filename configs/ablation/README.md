# Attention ablation configurations

This directory contains the configuration files for the CP1 attention branch
ablation of the FUSegNet reproduction.

## Experiment overview

The experiment compares the completed full P-scSE baseline with two controlled
variants:

| Configuration | Variant | Changed stages |
| :--- | :--- | :--- |
| [`additive.json`](additive.json) | Additive branch only | Decoder stages 0-3 |
| [`maxout.json`](maxout.json) | Maxout branch only | Decoder stages 0-3 |
| [`cp1-attention-01.json`](cp1-attention-01.json) | Experiment manifest | Defines both runs |

The full baseline is reused from
`outputs/runs/cp1-baseline-01`. The final decoder stage remains the original
shorted P-scSE implementation for every variant. The experiment changes only
which regular-stage attention branch is used during the forward pass.

The purpose is to test whether the additive and maxout branches provide
complementary information. Removing one branch changes the active capacity and
compute, so this is a component ablation rather than an operator comparison
with identical active parameter counts.

## Frozen controls

Both variants use the same controls as the baseline:

- Chronic Wound source data with 707 training, 124 validation, and 278 test
  images.
- Split manifest fingerprint:
  `7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925`.
- Seed `42` and the same pretrained EfficientNet-B7 ImageNet cache.
- Input size `224 x 224`, batch size `2`, and four data-loader workers.
- Adam training with learning rate `0.0001` and weight decay `0.00001`.
- Maximum 200 epochs with early stopping patience of 30.
- The original loss and checkpoint policy:
  `author_dice_plus_focal_with_probability_input` and
  `author_if_loss_elif_iou`.
- Inference threshold `0.5` and the offline `offline_hard_line` evaluator.
- PFOM settings and empty-edge policy documented in
  [`docs/pfom.md`](../../docs/pfom.md).

The source loss, checkpoint-selection behavior, and final-stage implementation
are intentionally preserved. These choices keep the ablation comparable with
the baseline, including known source-code quirks.

## Verification

Run the model and branch-isolation checks before training:

```bash
python scripts/run_attention_ablation.py verify
```

The verification checks include model reconstruction, unchanged initialization
and RNG state, selected-branch gradients, inactive-branch gradients, and the
unchanged final decoder stage.

## Running the experiment

Run the two configurations sequentially on one GPU:

```bash
python scripts/run_ablation_batch.py --device cuda:0
```

The batch runner:

1. Trains the additive-only variant.
2. Evaluates its selected checkpoint on the test split.
3. Trains and evaluates the maxout-only variant.
4. Builds the comparison analysis and publication artifacts.

Check progress without starting a new run:

```bash
python scripts/ablation_status.py
```

If a workflow is interrupted, inspect its status and checkpoints before
resuming:

```bash
python scripts/run_ablation_batch.py --device cuda:0 --resume
```

Do not start a second batch while another supervisor is active on the same
GPU.

## Outputs

The workflow uses these output locations:

```text
outputs/ablations/cp1-attention-01/
  verification.json
  workflow.json
  comparison.csv
  figures/
  published/

outputs/runs/cp1-attention-additive-01/
outputs/runs/cp1-attention-maxout-01/
```

The run directories contain the resolved configuration, environment metadata,
split manifest, training history, checkpoints, validation results, test
evaluation, and status information. Training checkpoints and local datasets
are excluded from Git.

## Interpretation

The primary endpoint is final image-based test Dice. Validation selects
checkpoints; the test split is used only for the predeclared comparison of all
variants. Secondary endpoints include data-based Dice, IoU, precision, recall,
PFOM, small-wound performance, fully missed positive images, false positives
on empty masks, forward-called parameters, training time, and peak CUDA
memory.

This is a preliminary one-seed ablation. A difference between variants should
not be treated as evidence of stable superiority without repeated seeds. The
dataset count discrepancy, possible source duplicates, unresolved original
split-order details, and patient-independence uncertainty remain limitations.

For the complete protocol and predeclared hypothesis, see
[`docs/attention-ablations.md`](../../docs/attention-ablations.md). For the
experiment history, see [`docs/experiment-log.md`](../../docs/experiment-log.md).
