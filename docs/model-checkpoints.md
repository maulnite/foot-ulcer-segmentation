# Best-model checkpoint downloads

The original x-FUSegNet checkpoints are distributed as **GitHub Release assets**.
Each release contains **five best-model checkpoints**, one per fold; use all five
for the averaging ensemble. They are not added to Git history.

| Release | Training / evaluation scope | Total checkpoint bytes |
| --- | --- | ---: |
| [Orie · cp1-xfusegnet-orie-weights-v1](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-orie-weights-v1) | 1,010 labeled pool, 808/202 folds; internal OOF; official test has no local GT | 4,026,547,645 |
| [Ganta · cp1-xfusegnet-ganta-weights-v1](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-ganta-weights-v1) | 810 training pool, configured 648/162 folds; ensemble on 200 provided validation images | 4,027,619,325 |

Every release includes `best_model_1.pth` through `best_model_5.pth`,
`SHA256SUMS.txt`, `model-manifest.json`, `MODEL_CARD.md`, and the upstream
reference-code MIT notice. Sizes and SHA-256 hashes are checked against GitHub's
server asset digests before publication. No raw datasets or prediction caches
are included. Ganta and Orie results have different evaluation scopes; the
reported scores are not a direct ranking of their implementations.

## Download and verify

Download from the release page in a browser, or use the configured GitHub CLI:

```bash
gh-orie release download cp1-xfusegnet-orie-weights-v1 \
  --repo maulnite/foot-ulcer-segmentation \
  --pattern 'best_model_*.pth' --pattern SHA256SUMS.txt \
  --pattern model-manifest.json --pattern MODEL_CARD.md \
  --dir downloaded-checkpoints/orie
cd downloaded-checkpoints/orie
sha256sum -c SHA256SUMS.txt
```

For Ganta, select tag `cp1-xfusegnet-ganta-weights-v1` and a separate download
directory. Keep the two sets of five models separate. Each file is approximately
805 MB; no single merged x-FUSegNet checkpoint is provided.

## Format and matching implementation

These are byte-identical original `torch.save` dictionaries containing `epoch`,
`state_dict`, and `optimizer`, rather than reduced weights-only exports.
Load model parameters from `checkpoint['state_dict']` into the matching model
implementation. The source's saved epoch metadata is preserved; no model is
reselected and no weights are changed for publication.

- **Orie:** PyTorch 1.11.0+cu113, author's SMP 0.3.0 fork, Albumentations 1.3.0.
  Use the pinned environment and source snapshot documented in
  [the x-FUSegNet protocol](xfusegnet.md). The normal local checkpoint directory
  is `outputs/runs/cp1-xfusegnet-01/author-checkpoints/`.
- **Ganta:** recorded PyTorch 2.7.1+cu118, official SMP 0.6.0 with the copied/adapted
  P-scSE decoder, Albumentations 2.0.8. See the
  [Ganta branch source](https://github.com/maulnite/foot-ulcer-segmentation/tree/ganta/notebook)
  and [protocol comparison](https://github.com/maulnite/foot-ulcer-segmentation/blob/ganta/reports/ganta-xfusegnet-comparison.md).
  The source run directory is
  `output_fuseg_2021/checkpoints/xFUSegNet_efficientnet-b7_2026-10-29_22-39-16/`.
  A complete run-linked split/RNG record is unavailable; this limitation remains
  in the release manifest. Importing the entire partner entrypoint can also run
  training/inference, so use its architecture definitions through a safe loader.

Inference uses native 512×512 RGB images, pretrained ImageNet normalization,
CHW float32, and no training augmentation. Compute the arithmetic mean of the
five raw sigmoid maps, then apply `>=0.5`. These downloads do not include all
auxiliary run state needed for a fully reproducible training resume.

## Attribution and provenance

Reference: Dhar et al., [FUSegNet](https://arxiv.org/abs/2305.02961v2),
[author repository](https://github.com/mrinal054/FUSegNet), and Pavel Iakubovskii's
Segmentation Models PyTorch. Training data:
[UWM Foot Ulcer Segmentation Challenge 2021](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge).
The included MIT notice covers reference code; it does not establish or change
the underlying dataset redistribution terms. See the existing
[dataset audit](datasets.md) and each release's model card for provenance.

Release hosting supports files below 2 GiB; regular Git commits block files over
100 MiB. See [GitHub's release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
and [large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
