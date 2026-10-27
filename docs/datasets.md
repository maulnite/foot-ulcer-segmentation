# Dataset preparation and audit

Both datasets are available under `data/raw/` with their original bytes and provided splits. No training, resizing, augmentation, mask conversion, sample removal, or new train / validation split has been performed.

## Source and reproducibility

- Repository: [uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation).
- Pinned commit: `42a272dfe0679f20675e826385925cb7562934b6`.
- Chronic Wound: `data/wound_dataset/azh_wound_care_center_dataset_patches.zip` in the source repository.
- FUSeg: `data/Foot Ulcer Segmentation Challenge/` in the source repository.
- Download archive SHA-256: `6a6977dce562430dff415cbe481fa30f71c7f49b45e018d87f9cb464795c0968`.

Run from the project root:

```bash
python scripts/prepare_datasets.py
```

The script uses `gh-orie` and the existing Pillow / NumPy installation. It verifies the cached archive checksum, checks selected archive files against upstream Git blob hashes, safely extracts the two requested datasets, and reads every PNG. It can be rerun without downloading an intact cached archive. It refuses to overwrite raw files that differ from the source.

The download includes a repository archive because this retrieves the challenge's many small files in one request. Only the requested datasets and source documentation are extracted; unrelated models and datasets are not unpacked.

## Verified local inventory

| Dataset | Split | Images | Masks | Dimensions | Empty masks |
| :--- | :--- | ---: | ---: | :--- | ---: |
| Chronic Wound | train | 831 | 831 | 224 × 224 | 52 |
| Chronic Wound | test | 278 | 278 | 224 × 224 | 17 |
| FUSeg 2021 | train | 810 | 810 | 512 × 512 | 19 |
| FUSeg 2021 | validation | 200 | 200 | 512 × 512 | 5 |
| FUSeg 2021 | test | 200 | 0 | 512 × 512 | — |

All **2,319 images and 2,119 masks** decode successfully. Every labeled sample has a matching filename and matching image / mask dimensions. Images and masks are RGB; all three mask channels agree for every mask.

## Findings that affect experiments

### Chronic Wound counts differ from the reference paper

The linked ZIP contains **831 train + 278 test = 1,109 paired samples**. FUSegNet reports **810 train + 200 test = 1,010 samples** for its Chronic Wound experiment. The source archive was preserved without arbitrary filtering. This dataset version cannot yet be labeled an exact match to the paper's split. Resolve sample selection or report the difference explicitly before interpreting reproduction scores.

The [reference code audit](reference-audit.md) subsequently found that the authors' category JSON lists exactly the same 278 test filenames as this download. This supports source-code alignment, while the paper's differing counts and exact train / validation sample selection remain unresolved.

### Exact image duplicates cross provided split boundaries

SHA-256 of decoded RGB pixel arrays detects duplicates even if PNG file encoding differs:

| Dataset | All duplicate pixel groups | Groups crossing splits | Samples in crossing groups |
| :--- | ---: | ---: | ---: |
| Chronic Wound | 30 | **17 train–test** | 35 |
| FUSeg 2021 | 29 | **7 train–validation** | 14 |

A group contains two or more pixel-identical images. Group counts are not counts of images to remove. The local report lists their sample IDs and splits. These findings establish exact image overlap; they do not establish patient identities or rule out other near-duplicates.

Keep the provided split for a clearly attributed source-protocol comparison if needed, and define a separately labeled leakage-controlled protocol before using these datasets to assess generalization. No deduplication has been applied automatically.

### Mask encoding requires an explicit policy

All Chronic Wound masks and all FUSeg training masks contain only `0` and `255`. FUSeg validation mask **`0233.png`** additionally contains **84 pixels with value `32`**. Its RGB channels agree. The original file is unchanged.

The inspected authors' `fusegnet_train.py` reads masks in grayscale and divides by `255.0`; it does not explicitly binarize them at loading. Confirm the relevant training and evaluation code when deciding whether to preserve fractional targets or use a documented binary threshold. Do not silently convert this mask and call the result identical preprocessing.

### Other protocol limits

- FUSeg test masks are absent from the linked source; local test Dice cannot be computed from this download.
- The paper describes Chronic Wound images as crops of images also present in FUSeg. Cross-dataset correspondence has not been resolved; do not combine them as independent samples.
- Patient independence is not established from filenames.
- No standalone license file was found at the source repository root; dataset redistribution terms remain unresolved. The repository does not mirror the full datasets or standalone mask caches. The [executed CP1 notebook](../notebooks/01_cp1_analysis.ipynb) includes selected source-image, ground-truth, and prediction panels for analysis, with source attribution; this does not establish a dataset license.

## Local audit artifacts

- `data/manifests/chronic_wound.csv` and `data/manifests/fuseg_2021.csv`: one row per image, with paths relative to the project root, dimensions, modes, checksums, mask paths where available, and nonzero mask counts.
- `data/metadata/preparation-report.json`: inventory, provenance, duplicate groups, mask anomalies, and manifest checksums.
- `data/metadata/upstream-tree.json`: pinned source file inventory with Git blob hashes.

These files are local and ignored by Git along with the data. Original evaluation sets remain intact. No experiment scores have been generated.

The compressed download caches were deleted at the user's request after extracted files were checked. Provenance checksums still document the downloads; their paths in the preparation report are historical. Re-running the default preparation command downloads the archives again if absent. Use `python scripts/prepare_datasets.py --audit-only` to audit existing raw data without downloading them.
