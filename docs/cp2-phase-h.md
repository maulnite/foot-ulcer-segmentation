# CP2 Fase H — standalone auxiliary supervision

Preregistration sebelum full training, branch `feat/cp2-phase-h`, berasal dari fase A commit `23189284e0ad13a4101461078f19c2382c26c5d2`. Satu seed 42. [Config](../configs/cp2/phase_h.json).

## Pertanyaan, landasan dan batas cakupan

Fase A belum menghasilkan kandidat yang memenuhi seluruh guard validation. Karena itu H diuji pada **P-scSE asli dengan objective utama B1**, tanpa cabang attention baru atau loss L3. H0 adalah cached B1, tidak dilatih ulang. Seluruh CP1/P/L/A tetap utuh.

Pertanyaan: apakah supervisi tambahan pada fitur resolusi tinggi membantu mask utama? Jika ya, apakah target batas lebih membantu dibanding supervisi segmentasi biasa? Ini eksperimen auxiliary/deep supervision, bukan reproduksi Gated-SCNN atau klaim novelty global.

| ID | Target auxiliary | Kontras |
|---|---|---|
| H0 | Tidak ada head | B1 cached, pembanding |
| H1 | Mask segmentasi | Kontrol deep supervision umum; dijalankan pertama |
| H2 | Morphological boundary | Perubahan target saja terhadap H1, setelah diagnosis validation H1 |
| H3 | Boundary-guided local branch | DEFERRED; memerlukan kombinasi dengan A, tidak otomatis dijalankan |

Hipotesis H1−H0: auxiliary gradient ke decoder stage 3 dapat membantu small/boundary metrics, dengan risiko FP dan konflik gradient. Hipotesis H2−H1: target batas memberi manfaat tambahan di atas deep supervision umum. Seluruh hipotesis dapat tidak didukung hasil. Tidak memilih head/weight/bandwidth dari source-test.

## Model, target dan skala objective

Tap **output** zero-based `decoder.blocks[3]`: `[B,32,112,112]` untuk input `[B,3,224,224]`. Temporary read-only hook menangkap fitur tanpa mengubah source forward dan selalu dilepas, termasuk saat error. Head `Conv2d(32,1,1,bias=True)` mempunyai **33 parameter**; logit di-upsample bilinear ke 224×224, `align_corners=False`. H1/H2 mempunyai head, jumlah parameter, seed dan initial tensors yang identik. Semua source tensors dimuat dari shared initial state fase P, bukan checkpoint yang telah dilatih.

Model mengembalikan pasangan eksplisit `HeadOutputs(main_logits, auxiliary_logits)`. `.sigmoid()` pada pasangan itu berarti sigmoid **mask utama** untuk kompatibilitas evaluator frozen. Tidak ada GT di forward. `auxiliary=False` melewati head; prediksi utama harus sama persis. Source decoder stage terakhir tidak berubah. Auxiliary head dapat dibuang saat inference, karena tidak memberi input ke forward mask utama.

Loss utama tetap **B1**, termasuk source Dice/focal primitives dan seluruh alpha/gamma/epsilon/reduction. Kedua kandidat memakai:

`L_total = L_B1(main, mask) + 0.1 * L_B1(auxiliary, target_auxiliary)`.

Untuk H1 target auxiliary adalah mask yang sudah mengalami augmentasi. Untuk H2 target adalah **dilation3×3(mask) − erosion3×3(mask)**, binary morphological gradient dengan exterior zero eksplisit. Target dibuat **setelah augmentasi geometrik**, dari tensor mask batch yang benar-benar masuk training. Empty mask menghasilkan empty band dan tetap mendapat penalti FP dari Dice/focal. Mask nonfinite/nonbinary ditolak; tidak ada relabeling otomatis. Full foreground mempunyai band di tepi kanvas sesuai kebijakan exterior.

Target boundary mencakup kedua sisi contour sekitar dua pixel pada batas lurus; detail mengikuti kernel square 3×3, bukan jarak fisik. Ini **berbeda** dari inner band radius 3 pixel pada evaluator Boundary IoU. Auxiliary logits resolusi 112 di-upsample; ini bukan batas berpresisi klinis. Auxiliary objective dan bobot sama agar H2−H1 mengisolasi perubahan target sejauh struktur memungkinkan; distribusi target dan gradient tetap dapat berbeda. Bobot 0,1 tidak dipilih melalui search dan tidak menjamin gradient auxiliary kecil.

## Kontrol, eksekusi bertahap dan seleksi

Manifest yang sama **671 train / 118 validation / 261 eligible source-test**, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Raw inventory tetap 831/278. Duplicate-RGB-group split, **bukan patient-level**. Pasien/crop independence belum diketahui.

Adam 1e-4, weight decay 1e-5, batch 2, workers 4, augmentasi/preprocessing source, float32/AMP off/TF32 flags warisan sama dengan P/L/A. Maksimal 200 epoch, early stopping 30 nonimproving; checkpoint highest validation **main Image Dice**, earliest tie. Scheduler ReduceLROnPlateau atas metrik sama, factor 0,1/patience 10/min LR 1e-5. Total validation loss mencakup auxiliary, tetapi **bukan monitor checkpoint/scheduler**. Dedicated loader seeds dan aturan `42+10007*zero_based_epoch` tetap; actual input order/augmented bytes dibandingkan B1 pada common epochs.

Preregister kode/config sebelum full training. Jalankan semua tests, profile initial forward, oracle target boundary dan smoke real-data (empty+positive, 3 optimizer steps, validation, finite gradients, update encoder/head, checkpoint replay). Dua diagnostic smoke tidak dipresentasikan sebagai hasil penelitian. Satu trainer GPU berurutan, monitor GPU keseluruhan dan `/dev/shm`, tanpa perubahan package/batch/precision otomatis.

Selesaikan **H1 dahulu**, simpan diagnosis validation dan keputusan gate sebelum H2. Gate kelayakan: overall/small Dice masing-masing ≥B1−0,03; total misses ≤B1+1; empty-FP images ≤B1+2. Jika gagal, stop dan H2 NOT RUN. Gate ini bukan penerimaan peningkatan.

Setelah semua kondisi yang diizinkan selesai, kandidat eligible jika **seluruh** guard sama seperti fase A terpenuhi: main validation Dice >B1, small Dice/Boundary IoU ≥B1, total misses dan empty-FP images ≤B1, active training params ≤1,01×B1. Pilih overall validation Dice tertinggi; exact tie H1 lebih awal daripada H2. Jika tidak ada, pertahankan H0. Simpan selection sebelum semua test probabilities baru. Source-test sudah diekspos CP1/P/L/A, hasilnya deskriptif. Tidak memilih ulang dari test atau mengubah threshold ≥0,5/empty rules.

## Artefak, diagnosis dan reproduksi

Run baru append-only `outputs/cp2/phase_h/<timestamp>/`. Config/source/manifest/initialization fingerprints, checkpoints/optimizer/scheduler/RNG, predictions, curves, waktu dan peak allocator disimpan. Source P/L/A tidak diubah. Periksa auxiliary task pada validation setelah selection sebagai diagnosis; tidak memakai auxiliary mask sebagai output segmentasi utama. Laporkan overall, small, boundary, PFOM, GT-positive, empty FP, total misses dan biaya, termasuk train-time head versus removable inference head.

```bash
.venv-training/bin/python scripts/run_cp2_phase_h.py --root <new-run> --stage prepare
.venv-training/bin/python scripts/run_cp2_phase_h.py --root <new-run> --stage preflight --device cuda:0
# Setelah preregistration.json dan pretraining_checks.json cocok dengan kode/config:
.venv-training/bin/python scripts/run_cp2_phase_h.py --root <new-run> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_h.py --root <run> --stage status
```

Gunakan interpreter virtualenv tanpa meresolusikan symlink ke base Python. `--resume` hanya setelah inspeksi kegagalan dan verifikasi source/config/input/checkpoint/environment; completed stages tidak dilatih ulang. Seluruh foto/GT/clinical overlays tetap lokal; hanya hasil numerik dan grafik agregat masuk Git. Tidak ada dataset/weights tambahan atau environment upgrade.

H3, L3 combinations, attention combinations, multiseed, boundary-guided inference dan external evaluation **belum termasuk eksekusi ini**. Satu-seed screening, validation reuse dan test exposure tetap membatasi interpretasi.

## Hasil eksekusi 12 November 2026 — tambahan setelah preregistration

Preregistration di atas tetap; kode/config training dibekukan pada commit `db1d4a212a84dea9b79c2af43d6cf014bc9b4c6d`. Run `outputs/cp2/phase_h/20261112T105427Z` menyelesaikan **H1 55 epoch (best 25)** dan **H2 80 epoch (best 50)**. H0 tidak dilatih ulang. H1 lolos gate kelayakan H2, tetapi tidak memenuhi guard penerimaan.

| Model | Validation Image Dice | Small val | Boundary IoU val | Empty-FP validation | Source-test Image Dice |
|---|---:|---:|---:|---:|---:|
| H0 = B1 cached | 0.8428 | 0.7728 | 0.5393 | 4/7 | 0.8329 |
| H1 segmentation auxiliary | 0.8360 | 0.7850 | 0.5342 | 4/7 | 0.8333 |
| H2 boundary auxiliary | 0.8488 | 0.8056 | 0.5577 | 5/7 | 0.8477 |

**Keputusan tetap H0/B1:** H2 gagal guard no-extra-empty-FP; H1 gagal overall/boundary. Selection disimpan sebelum test baru. Source-test bersifat deskriptif dan tidak digunakan untuk memilih ulang. [Laporan lengkap](../reports/cp2-phase-h.md), [review ilmiah](../reports/cp2-phase-h-review.md), [QA](../reports/cp2-phase-h-validation.json).

Untuk memeriksa run yang selesai tanpa retraining:

```bash
.venv-training/bin/python scripts/run_cp2_phase_h.py \
  --root outputs/cp2/phase_h/20261112T105427Z --stage status
.venv-xfuseg/bin/python -m pytest
```

Posthoc helper terpisah dari frozen training fingerprints:

- `analyze_cp2_phase_h.py`: reload checkpoint, replay primary validation, diagnosis auxiliary dan mask-only latency; hanya sesudah selection. Output directory baru diperlukan jika analisis diulang.
- `report_cp2_phase_h.py`: membaca artefak tervalidasi untuk tabel/kurva/grafik CPU; tidak melatih atau memilih ulang.
- `validate_cp2_phase_h.py`: checksum, manifest, raw GT, threshold, metrics, chronology, paired initialization/inputs dan historical preservation. Gunakan `--output <nama-qa-baru.json>` jika mengulang agar bukti QA lama tetap tersimpan.
- `finish_cp2_phase_h.py --root <run> --device cuda:0 --wait`: menunggu trainer selesai/keluar lalu menjalankan ketiganya berurutan; completed stages di-load, tidak dilatih ulang. Completed workflow tidak menampilkan ulang gallery atau menggantikan direct visual review.
- `cp2_phase_h_cases.py`: gallery lokal deterministik dari prediksi checksum-validated; menolak overwrite direktori gallery selesai. Numeric case provenance boleh publik, clinical imagery tetap lokal.

Artefak model/cache/raw data tidak masuk Git; clone GitHub saja dapat membaca hasil publik, tetapi tidak otomatis menjalankan replay tanpa aset lokal yang dirujuk manifest dan checksum. Fresh full run tetap memerlukan preregistration/gate tests baru, direktori baru, shared initial state serta baseline historis yang sama. Jangan menyalin timestamp/claim preregistration run ini untuk run lain.
