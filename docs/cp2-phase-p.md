# CP2 FASE P — protokol B0/B1

Roadmap: `Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md`, revisi 7 November 2026. Cakupan eksekusi ini hanya audit, persiapan, smoke test dan dua baseline. Tidak menjalankan matriks loss/attention/boundary head, seed tambahan, atau evaluasi eksternal.

## Data dan keputusan sebelum training

Seluruh 831 source-train dan 278 source-test Chronic Wound diperiksa: pasangan nama, decoding, dimensi, channel mask dan nilai biner. Group memakai SHA256 **decoded RGB pixels**, bukan hash encoding PNG. Audit menemukan 30 kelompok duplikat, 28 kelompok dengan anotasi mask berbeda, dan 17 kelompok lintas source train–test.

- Karantina seluruh kelompok dengan RGB identik tetapi mask berbeda. Tidak memilih anotasi dengan model score, tidak menggabungkan/merata-ratakan mask, tidak menganggap salah satunya ground truth yang pasti lebih benar.
- Source-train RGB yang juga terdapat pada source-test tidak boleh masuk train/validation. Dalam data ini, seluruh kelompok lintas split sudah termasuk kelompok anotasi berkonflik.
- Satu wakil leksikografis untuk duplikat RGB+mask tanpa konflik; semua file mentah tetap ada.
- Atas sorted eligible RGB groups, shuffle dengan `random.Random(42)`, ambil `round(15%)` kelompok terakhir sebagai validation. Tidak memilih split berdasarkan skor model atau dataset eksternal.

Manifest akhir: **671 train / 118 validation / 261 source-test**. Karantina 57 baris berkonflik dan ringkas dua duplikat redundan; jumlah akhirnya bukan 707/124 CP1. Manifest menyimpan seluruh exclusion reason, source/image/mask hashes, group, luas GT, membership, parameter split dan fingerprint. Train/validation/test tidak memiliki overlap RGB group yang diketahui.

Ini **duplicate-group split, bukan patient-level split**. Tidak ada ID pasien terverifikasi. Near-duplicate screening menggunakan RGB-only nonblack content crop, pHash Hamming ≤2, thumbnail RGB MAE ≤3 dan SSIM ≥0,995; tidak ada kandidat di bawah aturan sempit ini. Screening full canvas sebelumnya dipengaruhi padding hitam dan disimpan sebagai audit pendahuluan, bukan bukti duplikasi. Screening tidak meniadakan kemungkinan crop/foto pasien berulang atau perbedaan viewpoint.

Source-test pernah dianalisis dalam CP1. Hasil CP2 atas source-test adalah evaluasi deskriptif setelah checkpoint terkunci, bukan bukti konfirmatori pada test yang sama sekali belum disentuh. Tidak ada data eksternal yang digunakan atau diunduh pada fase ini.

## Kontrol model dan objective

Konfigurasi: [`configs/cp2/phase_p.json`](../configs/cp2/phase_p.json).

Arsitektur FUSegNet–EfficientNet-B7 source dipertahankan, termasuk parameter terdaftar yang tidak dipakai forward. Activation akhir diekspos sebagai logits tanpa mengubah parameter. Uji dengan bobot dan input yang sama memastikan sigmoid eksplisit sama persis dengan head sigmoid source.

| Baseline | Dice menerima | Focal-with-logits menerima |
|---|---|---|
| B0 | `sigmoid(z)` | `sigmoid(z)` — mismatch source dipertahankan |
| B1 | `sigmoid(z)` | `z` — pasangan input diperbaiki |

Primitive Dice dan focal memakai source terverifikasi: alpha `None`, gamma 2, Dice epsilon 1/beta 1, focal epsilon 1e-6, mean reduction, normalisasi focal off dan bobot 1+1. Dice tetap agregasi seluruh batch. Tidak ada class weighting, clipping gradient, objective baru, atau perubahan augmentasi.

Kedua baseline membaca **satu berkas state_dict awal** (ImageNet encoder + decoder baru, bukan CP1 fine-tuned weights). Hash per named tensor dan buffer diverifikasi saat load. Seed saja tidak dianggap bukti inisialisasi yang sama. Urutan sample dan hash input aktual setelah augmentasi disimpan per epoch, lalu dibandingkan untuk seluruh epoch bersama B0/B1.

Adam LR 1e-4/weight decay 1e-5, batch size 2, empat loader workers, maksimal 200 epoch. Preprocessing dan paired augmentation memakai fungsi source yang sama seperti CP1. Seed training 42; epoch e memakai `42 + 10007*e` untuk Python/NumPy/torch dan generator loader khusus; worker seeds eksplisit.

Environment training `.venv-training`: PyTorch 1.11.0+cu113, torchvision 0.12.0+cu113. Float32 dan flag TF32 bawaan CP1 dipertahankan; **tanpa AMP atau perubahan precision otomatis**. Plotting memakai environment `.venv-analysis` yang sudah ada. Dependency tidak di-upgrade.

## Evaluator dan model selection

- Prediksi positif jika probabilitas **>=0,5**. Inference mengaplikasikan sigmoid satu kali dari logits.
- Ordinary TP/TN/FP/FN, region evaluator offline CP1; Data Dice dari aggregate counts dan Image Dice dari mean per gambar. GT/prediksi sama-sama kosong = 1 pada per-image scores; tepat satu kosong = 0. Aggregate Dice `null` bila seluruh koleksi tidak mempunyai GT/prediksi positif.
- Checkpoint: **Image Dice validation tertinggi**, improvement ketat, tie memilih epoch lebih awal.
- ReduceLROnPlateau memantau **validation Image Dice** (max), factor 0,1, patience 10, threshold absolut 0, min LR 1e-5. Scheduler tidak memantau loss yang berbeda definisinya. Jadwal LR aktual tetap bisa berbeda akibat kurva validation; keduanya dicatat.
- Early stopping: 30 epoch berturut-turut tanpa perbaikan metrik checkpoint, maksimal 200 epoch. Test tidak memilih epoch, LR atau threshold.
- Subgroups: GT kosong, GT positif, kecil `0<area<=1%`, sedang `>1–5%`, besar `>5%` kanvas. Catat TP=0 pada GT positif dan jumlah prediksi foreground pada GT kosong.
- Boundary IoU tambahan: inner erosion 3×3, zero-padded exterior; radius `max(1, round(.01*hypot(H,W)))`, yakni 3 pixel pada 224×224. Keduanya kosong=1, salah satu kosong=0. PFOM menggunakan primitive CP1 yang telah diuji, bukan klaim identik dengan implementasi paper yang tidak tersedia.

## Eksekusi dan monitoring

Semua direktori CP2 terpisah: `outputs/cp2/phase_p/<timestamp>/`. `outputs/cp2/latest_phase_p.json` menunjuk run saat ini. Dataset, artefak CP1 dan notebook pengguna tidak ditimpa.

```bash
.venv-training/bin/python scripts/run_cp2_phase_p.py \
  --root outputs/cp2/phase_p/<timestamp> --stage prepare
.venv-training/bin/python scripts/run_cp2_phase_p.py \
  --root outputs/cp2/phase_p/<timestamp> --stage smoke --device cuda:0
# Hanya setelah pretraining_checks.json dan kedua smoke test lulus:
.venv-training/bin/python scripts/run_cp2_phase_p.py \
  --root outputs/cp2/phase_p/<timestamp> --stage train --device cuda:0
# Lanjutkan interrupted epoch-boundary checkpoint setelah memeriksa kegagalan:
# tambahkan --resume; config/input/training sources dan packages harus cocok.
```

Training B0 lalu B1 berjalan **sekuensial dalam satu proses GPU**. Tidak ada proses kedua untuk mengejar VRAM. Loss/gradient nonfinite menyebabkan error yang tercatat; tidak ada silent fallback precision/config. `last.pth` menyimpan optimizer, scheduler, RNG, history dan checkpoint policy secara atomik; completed stages tidak dilatih ulang. `best.pth` memakai hard link sehingga checkpoint terpilih tidak ditimpa saat `last.pth` bergerak.

Smoke memakai enam gambar train termasuk GT kosong dan lima positif; tiga optimizer steps, validation, encoder-update check, serta exact checkpoint replay setelah deliberate perturbation. Hasilnya diagnostik, bukan skor baseline. Final export validation/test hanya berjalan setelah **kedua** baseline selesai dan checkpoint terkunci, satu probability map dan prediksi per gambar.

`console.log`, `launch.json`, `train_workflow.json`, `B0/status.json`, `B1/status.json`, histories dan GPU sampling setiap lima detik memberi status persisten. Peak CUDA allocated/reserved berasal dari proses kita; sampling GPU keseluruhan juga mencakup job lain. Catat shared-load timing, bukan menganggap latency GPU eksklusif.

## QA

```bash
# Semua tests pada environment existing yang juga mempunyai dependency plotting CP1:
.venv-xfuseg/bin/python -m pytest -o addopts='' -q
# CP2 + tests training pada environment training tanpa menginstal plotting:
.venv-training/bin/python -m pytest --ignore=tests/test_xfusegnet.py
# Re-render hanya laporan selesai, tanpa training atau inference ulang:
.venv-analysis/bin/python scripts/report_cp2_phase_p.py --root outputs/cp2/phase_p/<timestamp>
```

Tests mencakup parity nilai/gradien loss B0 terhadap source, formula focal B1, finite gradients pada mask kosong, sigmoid saturation source, threshold inklusif, empty rules, Boundary IoU analitik, konflik label/overlap grup, hash manifest, pemilihan checkpoint, actual parameter hashes dan ekspor subgroup. Snapshot checksum CP1 serta perubahan notebook lokal pengguna dicatat sebelum perubahan; validasi akhir memastikan preservasinya.

[Hasil B0/B1](../reports/cp2-phase-p.md) akan tersedia setelah kedua run selesai. Keputusan fase L/A/H dan evaluasi eksternal menunggu review hasil fase P.
