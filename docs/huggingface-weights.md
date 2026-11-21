# Hugging Face checkpoint archive

The shared archive for **Orie and Ganta** is
[maulnite/foot-ulcer-segmentation-weights](https://huggingface.co/maulnite/foot-ulcer-segmentation-weights).
Its landing page reports publication status; `publication-validation.json` is
available after complete remote integrity verification.

## Organization

| Directory | Contents |
| :--- | :--- |
| `orie/cp1/` | Original baseline, attention runs, and five-fold x-FUSegNet |
| `orie/cp2/` | P/L/A/H/N/C, recipe checks, matched CP1 runs, seed confirmation, and Cascade |
| `ganta/fuseg-2021/` | Available FUSeg 2021 experiment checkpoints and historical partial runs |
| `ganta/chronic-wound/` | Available Chronic Wound experiments, including both original Cascade stages |
| `reproducibility/` | Matching project utilities, configurations, dependency records, and upstream source notices |
| `evidence/` | Existing scientific reports and available partner numeric results |

Use `CATALOG.md` to browse individual files. `model-index.json` records original
relative paths, SHA256, byte size, checkpoint container, stored epoch metadata,
checkpoint roles, recorded run status, and download URLs. `SHA256SUMS.txt`
contains checkpoint checksums. Source-side text metadata is sanitized for
machine-specific home paths; checkpoint bytes are preserved exactly.

`best.pth` and `best_model*.pth` retain historical selection. `last.pth` is a
training-resume checkpoint, not a newly selected result. Initializers and smoke
checkpoints remain explicitly labeled. Some original checkpoint dictionaries
include optimizer/RNG/history state and are consequently much larger than their
model tensors. RNG-only files and downloaded upstream pretraining caches are
excluded from the experiment-weight archive and listed with reasons.

The Cascade CP2 bundle uses frozen phase-P B1 plus its separate CP2-trained
refiner. The original Ganta Cascade uses `stage1_coarse.pth` and
`stage2_refiner.pth` from the same dataset/run directory. Do not mix these
protocols. Phase-L L0 reuses phase-P B1, rather than requiring another checkpoint.

## Reproduce publication

The publication runner never trains, converts weights, or imports partner
training entrypoints. It stages links to original files, reads checkpoints on
CPU, checks the frozen external-evaluation registry, and records a preservation
snapshot. Run preparation in the existing project environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .venv-xfuseg/bin/python scripts/publish_huggingface_weights.py prepare \
  --run outputs/model-publication/NEW_UNIQUE_RUN_ID
```

The prepared folder can be uploaded resumably using the isolated identity:

```bash
hf-orie upload-large-folder maulnite/foot-ulcer-segmentation-weights \
  --repo-type model outputs/model-publication/NEW_UNIQUE_RUN_ID/hub \
  --num-workers 3
```

The runner's `upload` stage coordinates this with remote verification; invoke
it using the existing Hub-SDK interpreter with `HF_HOME`, `HF_TOKEN_PATH`,
`HF_HUB_CACHE`, `HF_XET_CACHE`, and `HF_ASSETS_CACHE` pointing to the `hf-orie`
profile, then pass the same `--run`. Authentication is read only from
`~/.config/hf-orie/token`, never embedded in arguments, logs, or Git.
No shared packages are installed or changed.

Verification compares every remote checkpoint's LFS SHA256 and byte size,
checks text files against remote Git blob hashes, rehashes source checkpoints,
and checks original research documents/notebooks against the preservation
inventory. The final report records the verified Hub revision and completion
revision. Local logs and resumable Hub task metadata remain under the new
publication directory.

## Scope and limitations

Only checkpoints actually present at inventory time can be archived. Empty
historical partner directories are explicitly listed; no missing weights are
manufactured or retrained. Different contributors and experiment families use
different cohorts, implementations, selection rules, and environments; this
archive is not one comparable leaderboard. No dataset photos, raw masks,
probability caches, private tokens, or clinical validation claims are included.
Original source/data notices still apply; publication does not assign new
licensing rights to the weights or underlying datasets.
