# CP1 pipeline readiness

**Status: implementation, CPU smoke tests, and GPU smoke test passed.** Full-run progress is recorded separately in the experiment log.

## Verified locally

| Check | Result |
| :--- | :--- |
| Isolated environment | Python 3.10.4; system site packages disabled |
| Reference runtime pair | PyTorch 1.11.0+cu113 / torchvision 0.12.0+cu113 |
| Full source-package import | Pinned, hash-verified author package loads successfully |
| Dependencies | `pip check` passes; resolved environment locked |
| Dataset split | 707 train / 124 validation / 278 test; original source samples retained |
| Data preprocessing | Original functions produce float32 `[3,224,224]` image / `[1,224,224]` mask |
| Standard loaders | First train and validation batches pass with four workers |
| Training smoke | Original ImageNet initialization; one batch / optimizer step, two images |
| Backward pass | Gradients finite; encoder weights actually updated |
| Validation smoke | Two validation images; no test-set inference |
| Checkpoint round trip | Model, optimizer, scheduler reload; predictions restore exactly |
| Unit tests | **12 passed**: original evaluator parity, aggregation, thresholds, empty masks, source checkpoint policy, stochastic optimizer continuation, and manifest integrity |
| Temporary smoke checkpoints | Deleted after successful checks |
| GPU smoke | A100; finite gradients, encoder update, exact checkpoint prediction restoration; peak allocated 1,987,409,408 bytes |

The final local smoke record is `outputs/smoke/cp1-pipeline-02/status.json`. Its configuration and split fingerprints are:

```text
config: 86ee03891cb5da1bda31efcba4332b52f345f5a8ae8057b1fb600e6e11dcbc8b
split:  7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925
```

The ImageNet encoder weights are cached locally with SHA-256 `dcc49843b8ec5a097a8cb166eded1d7a09291093eff7735f603bcfcafce12c8e`. Smoke metrics are diagnostic and are not included as research performance results.

## Remaining for Checkpoint 1

- Complete full baseline training and its held-out test evaluation; one-step CUDA behavior is verified, while convergence remains to be established.
- Perform validation analysis and separate final test evaluation, naming both source evaluators explicitly.
- Report the paper / source dataset count discrepancy and unavailable original split order.
- Produce initial error analysis and feasible ablations, then the report and presentation.

The source focal-loss and checkpoint quirks are preserved. Cross-split duplicates remain in this source-protocol baseline and must be disclosed. Any corrected objective, deduplicated split, or changed selection policy belongs to a separately labeled experiment. See [training instructions](../docs/training.md) and [reference audit](../docs/reference-audit.md).
