# Local datasets

Both upstream datasets have been downloaded and audited locally. Dataset contents are excluded from Git; this README is the only tracked file in this directory.

```text
data/
├── downloads/                  # Optional cache; compressed archives deleted locally
├── raw/
│   ├── chronic_wound/
│   │   ├── train/{images,labels}/    # 831 paired samples, 224 × 224
│   │   └── test/{images,labels}/     # 278 paired samples, 224 × 224
│   └── fuseg_2021/
│       ├── train/{images,labels}/       # 810 paired samples, 512 × 512
│       ├── validation/{images,labels}/  # 200 paired samples, 512 × 512
│       └── test/images/                # 200 images, no public masks
├── manifests/
│   ├── chronic_wound.csv
│   └── fuseg_2021.csv
└── metadata/
    ├── upstream-tree.json
    ├── upstream-README.md
    └── preparation-report.json
```

Original file bytes, image dimensions, and upstream splits are preserved. CSV paths are relative to the project root. Manifests contain sample IDs, image / mask paths, dimensions, modes, checksums, and nonzero mask counts. Nonzero counts are inventory measurements, not a selected training binarization policy.

Recreate the download with `python scripts/prepare_datasets.py` from the project root, or audit existing raw data without downloading archives using `python scripts/prepare_datasets.py --audit-only`. Downloading requires the configured `gh-orie` CLI; both modes require Pillow and NumPy. No dependencies are installed automatically. The compressed archives were deleted after extraction at the user's request. Existing changed raw files cause an error instead of being overwritten during extraction.

**Before training:** read [the dataset audit](../docs/datasets.md). The Chronic Wound archive differs from the paper's sample counts, both datasets have cross-split duplicates, and one FUSeg validation mask has nonbinary values. No corrected / deduplicated splits have been created. Licensing, patient independence, and cross-dataset correspondence still need review.
