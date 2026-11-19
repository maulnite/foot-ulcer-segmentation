# Foot Ulcer Segmentation (FUSeg) - Experiment Notebooks & Scripts

This directory contains research implementations, ablation studies, and empirical probes for automated foot ulcer segmentation using deep convolutional architectures (based on FUSegNet and EfficientNet-B7 backbones).

Experiments are evaluated across two primary datasets:
1. **FUSeg 2021** (benchmark dataset with single split seed 42)
2. **Chronic Wound** (clinical wound dataset with heterogeneous wound sizes and contrast)

---

## Directory Structure

```
notebook/
├── README.md               # This documentation
├── *.ipynb                 # Interactive Jupyter notebooks for analysis & visualization
├── scripts/                # Standalone, reproducible Python training scripts (*.py)
└── logs/                   # Full execution logs holding training output and metrics (*.log)
```

> **Note on Logs:** Because training runs were executed in batch headless mode on GPU nodes rather than through interactive Jupyter kernels, the execution traces and epoch metrics are preserved inside `notebook/logs/`.

---

## Detailed Experiment Breakdown (01 - 14)

### 1. Novelty 1: Local-Global Parallel scSE (LG-P-scSE)
- **Files**:
  - `scripts/01_lg_pscse.py` (FUSeg 2021)
  - `scripts/05_lg_pscse_chronic_wound.py` (Chronic Wound)
- **Concept**: Enhances standard spatial and channel Squeeze-and-Excitation (scSE) blocks by adding parallel paths capturing both local detail features and dilated/global contextual pooling, combined via additive and max-out fusion.
- **Goal**: Address boundary ambiguity and intra-wound texture heterogeneity.

### 2. Novelty 2: Cross-Level Detail-Preserving Skip Connection (CL-DPSC)
- **Files**:
  - `scripts/02_cross_level_skip.py` (FUSeg 2021)
  - `scripts/06_cross_level_skip_chronic_wound.py` (Chronic Wound)
- **Concept**: Bypasses deep encoder downsampling loss by routing multi-scale low-level edge features directly into decoder stage 0 with detail-preserving convolution blocks.
- **Empirical Finding**: Demonstrates consistent improvement on FUSeg 2021 (mean Dice delta +0.010, and +0.016 on small wounds <1% area).

### 3. Novelty 3: Boundary-Weighted Focal Tversky Loss (BW-FTL)
- **Files**:
  - `scripts/03_boundary_weighted_ftl.py`
  - `scripts/07_boundary_weighted_ftl_chronic_wound.py`
  - `scripts/07_boundary_weighted_ftl_fuseg_2021.py`
- **Concept**: Integrates Euclidean Distance Transform (EDT) boundary penalty maps into a Focal Tversky Loss objective to penalize contour errors more heavily than interior pixels.
- **Empirical Finding**: Recovered images from total failure on FUSeg 2021 (+0.008 mean Dice delta, +0.012 on small wounds).

### 4. Tri-Novelty Combined (LG-P-scSE + CL-DPSC + BW-FTL)
- **Files**:
  - `scripts/04_combined_lg_cl_bw.py`
  - `scripts/08_combined_lg_cl_bw_chronic_wound.py`
  - `scripts/08_combined_lg_cl_bw_fuseg_2021.py`
- **Concept**: Unified architecture (Eksperimen E) combining all three novelties in a single end-to-end trained model.

### 5. Baseline: FUSegNet PyTorch Reconstruction
- **Files**:
  - `scripts/09_baseline_fusegnet_chronic_wound.py`
  - `scripts/09_baseline_fusegnet_fuseg_2021.py`
- **Concept**: Clean, pure PyTorch re-implementation of the published FUSegNet architecture (Dhar et al.) without custom novelty modules, serving as the benchmark standard.

### 6. Probe M1: Detail Skip Injections (Raw & Frequency)
- **Files**:
  - `scripts/10_raw_detail_skip_chronic_wound.py` / `_fuseg_2021.py`: **Raw Detail Skip (RDS)** using space-to-depth operations directly from the raw input tensor.
  - `scripts/11_freq_detail_skip_chronic_wound.py` / `_fuseg_2021.py`: **Frequency Detail Skip (FDS)** extracting high-frequency wavelets/Laplacian sub-bands to retain microscopic wound margins.

### 7. Probe M2: Guided Offset Upsampling
- **Files**:
  - `scripts/12_guided_upsampling_chronic_wound.py` / `_fuseg_2021.py`
- **Concept**: Replaces standard nearest-neighbor or bilinear upsampling in decoder stages with learnable feature-guided sampling offsets (deformable kernel upsampling).

### 8. Probe M3: Cascade Crop-Refine
- **Files**:
  - `scripts/13_cascade_crop_refine_chronic_wound.py` / `_fuseg_2021.py`
- **Concept**: Two-stage cascaded architecture where a coarse segmentation model detects region of interest (ROI), followed by bounding-box crop, zoom, and a specialized refinement sub-network.
- **Key Empirical Finding**: Achieves statistically significant improvement on Chronic Wound (+0.022 overall mean Dice delta, p=0.000, 159 wins vs 102 losses), with exceptional gain on low-contrast ulcer boundaries (+0.036 mean Dice gain, p=0.000).

### 9. Probe M4: Size-Aware Focal Tversky + Boundary Loss
- **Files**:
  - `scripts/14_size_aware_tversky_chronic_wound.py` / `_fuseg_2021.py`
- **Concept**: Dynamic loss weighting adjusting alpha/beta coefficients and boundary weights dynamically based on GT lesion size distributions.

---

## Summary of Empirical Findings (from analisis_empat_ide/)

The comprehensive pairwise statistical evaluation (reported in `analisis_empat_ide/chronic_wound/laporan.md` and `analisis_empat_ide/fuseg_2021/laporan.md`) evaluated:
1. **Baseline failure diagnosis**: Small lesions (<0.1% of image area) suffer from high false positive rates and low Dice (~0.623).
2. **Cascade Crop-Refine (Probe M3)** is the strongest architectural performer on complex chronic wounds, significantly mitigating boundary loss on low-contrast tissue.
3. **CL-DPSC (Novelty 2)** provides consistent structural skip enhancements, especially on small wound subsets.
4. **BW-FTL (Novelty 3)** stabilizes training convergence and prevents total catastrophic miss on fine margins.

---

## Execution & Environment

To run scripts or inspect notebooks:

```bash
# Activate virtual environment
source ~/venvs/ganta/bin/activate

# Navigate to project directory
cd ~/fuseg/ganta

# Example: Run baseline or novelty script
python notebook/scripts/09_baseline_fusegnet_fuseg_2021.py
```

---

# Ganta · script x-FUSegNet

[Python source asli](x-FUSegNet%20Reproduce.py) disalin byte-identik dari working directory partner. Script berisi training lima fold, inference average/voting, evaluasi, dan pembuatan figure. Mengimpor/mengeksekusi script dapat menjalankan rangkaian tersebut; audit publikasi hanya memakai AST dan cache PNG, tanpa mengimpor entrypoint.

Source mengharapkan CWD folder `notebook/` (`BASE_DIR=".."`), raw data di `../data/raw/fuseg_2021/`, dan menulis ke `../output_fuseg_2021/`. **`TEST_DIR` menunjuk `validation`, bukan challenge test.** Lima fold memakai 810 provided train; evaluasi ensemble memakai 200 provided validation.

Environment yang tercatat pada run: PyTorch 2.7.1+cu118, SMP 0.6.0, Albumentations 2.0.8. Full dependency lock dan state RNG run tidak tersedia. Jangan menganggap environment frozen Orie identik dengan environment partner.

Lihat [perbandingan dan batas verifikasi](../reports/ganta-xfusegnet-comparison.md), [hasil asli](../output_fuseg_2021/results/), dan [MIT notice upstream](UPSTREAM_LICENSE.txt). Dua notebook `.ipynb` partner yang tersimpan mengandung training terinterupsi; publikasi ini berfokus pada script `.py` dan hasil yang diminta, tanpa mengeksekusi ulang training.

[Lima checkpoint terbaik Ganta](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-ganta-weights-v1) tersedia sebagai aset Release terpisah, dilengkapi SHA-256. Ikuti [petunjuk unduh dan format checkpoint](../docs/model-checkpoints.md).
