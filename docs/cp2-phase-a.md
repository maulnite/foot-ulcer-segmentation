# CP2 Fase A — local attention dan fusion antarcabang

Dokumen dan [config](../configs/cp2/phase_a.json) ditulis **sebelum training A1**. Branch berasal dari `feat/cp2-phase-l`, commit `59bdb6f`. Ini screening satu seed 42, bukan konfirmasi multiseed, klaim novelty global, atau evaluasi klinis.

## Acuan dan hipotesis

[Fase P](../reports/cp2-phase-p.md) menetapkan protokol dan B1. [Fase L](../reports/cp2-phase-l-review.md) menemukan trade-off luka kecil–FP dan mempertahankan L0/B1. **A0 adalah B1 cached**, tidak dilatih ulang. **Loss semua A tetap B1:** source Dice(sigmoid(z)) + source FocalWithLogits(z), alpha/gamma/epsilon/reduction tidak berubah. L3 dan seluruh model lama dipertahankan untuk kombinasi mendatang; tidak dipakai training fase ini.

Pertanyaan bertahap:

1. Apakah respons spasial lokal tambahan membantu luka kecil/batas pada fusion tetap?
2. Apakah belajar bobot konstan cukup, atau gate per gambar memberikan manfaat tambahan?
3. Apakah efeknya dapat dijelaskan oleh tambahan kapasitas atau attention mapan?

P-scSE sudah mempunyai cSE/sSE yang bergantung input. Yang diuji adalah **fusion antarcabang**, bukan penemuan attention adaptif pertama. [Selective Kernel Networks](https://arxiv.org/abs/1903.06586) menjadi landasan pemilihan cabang; [CBAM](https://arxiv.org/abs/1807.06521) pembanding yang telah dikenal. Kedua paper bukan bukti metode ini berhasil pada luka kita.

## Lokasi dan skala yang diaudit

Penomoran **zero-based `decoder.blocks`**. Stage 2 menerima `[B,176,56,56]`; stage 3 menerima `[B,128,112,112]`, sebelum source `conv1`, setelah upsample dan concatenation skip. Stage 0–1 dan stage terakhir 4 tetap source. `conv2` yang dormant tetap terdaftar dan tidak diaktifkan diam-diam.

Definisikan `a=attention2(x)` (additive), `m=attention1(x)` (max-out persis kode sumber), dan `l=x*sigmoid(PW(DW(x)))`. DW adalah depthwise 3×3, groups=C; PW adalah pointwise C→1. Peta lokal `[B,1,H,W]` dibroadcast ke channel, tanpa aktivasi/normalisasi tambahan di antara DW/PW.

Source menghasilkan `a+m`. Kandidat A1–A3 menggunakan:

\[
y=3\left(w_a\frac{a}{2}+w_m m+w_l l\right),\quad
\mathbf w=\operatorname{softmax}(\mathbf z).
\]

Prior `(4/9,2/9,1/3)` mempertahankan rasio 2:1 dari kontribusi source pada ruang cabang ternormalisasi, sambil mengalokasikan sepertiga massa kepada cabang lokal. A1 ekuivalen dengan `2/3*(a+m)+l`. Bukan sekadar menambahkan `l` tanpa koreksi gain. Semua cabang ternormalisasi mempertahankan tanda dan memiliki magnitude ≤|x|; source dan fusion sama-sama dibatasi 3|x|. Jika seluruh sigmoid bernilai 0,5, gain nominal keduanya 1,5. Untuk input negatif, max-out tetap `torch.maximum` pada hasil cabang, bukan mengganti operasi dengan max atas attention weights.

Kontrol ini menyamakan konvensi/batas skala, **bukan menjamin RMS/distribusi aktivasi identik**. RMS sebelum/sesudah fusion dicatat pada validation. Memodifikasi fusion juga dapat mengubah optimisasi dan statistik BatchNorm downstream.

## Matriks terbatas dan urutan eksekusi

| ID | Kondisi | Status rencana / kontras |
|---|---|---|
| A0 | P-scSE asli + B1 | CACHED; checkpoint epoch 25, tidak diulang |
| A1 | Local branch + fusion tetap prior | Pertama dijalankan dan dianalisis sendiri |
| A2 | A1 + satu vector logits fusion terlatih per stage | Setelah gate kelayakan A1; bobot tidak tergantung gambar |
| A3 | A1 + GAP→MLP→3-logit gate per gambar | Kontras utama A3−A2; last gate layer zero-init sehingga initial fusion sama |
| A4 | Gate per lokasi | **DEFERRED**, tidak otomatis dalam screening awal |
| A5 | P-scSE + residual conv block tanpa multiplicative attention | Kontrol kapasitas terhadap A3; bukan gating |
| A6 | CBAM pada lokasi sama menggantikan dua cabang P-scSE | Pembanding attention mapan, bukan isolasi satu faktor |

A5 menggunakan DW3×3 C→C, PW C→h, ReLU, PW h→C, `h=round(max(8,C//16)/2)`, lalu ditambahkan pada source P-scSE. Last PW zero-init menjaga initial function source. Tambahan parameter aktif ditargetkan dalam 5% tambahan A3; angka aktual diverifikasi. Output koreksi conv tidak dibatasi seperti sigmoid, sehingga kapasitas/operasi/RMS aktual harus dibaca, bukan disebut identik.

A6 mengikuti channel avg+max/shared MLP (reduction 16), lalu spatial max+mean/Conv7×7/BatchNorm/sigmoid pada [kode penulis yang dipin](https://github.com/Jongchan/attention-module/tree/459efad0e05ee7dde50c41ca10a3d0800bc3792a). License MIT disimpan. Output memakai gain 3 untuk menyamakan upper bound source; midpoint dua sigmoid memberi 0,75x, bukan 1,5x. Perbedaan bentuk/distribusi gain tetap keterbatasan comparator. Tidak mengklaim reproduksi benchmark asli CBAM.

### Gate perluasan setelah A1

Simpan diagnosis **validation-only** sebelum meluncurkan A2/A3/A5/A6. Perluasan diizinkan jika A1 completed dan integrity/finite-gradient checks lulus, overall/small validation Dice masing-masing tidak turun lebih dari 0,03 terhadap B1, total miss bertambah maksimal satu gambar, serta empty FP bertambah maksimal dua gambar. Ini gate kelayakan, **bukan kriteria menerima peningkatan**. Fixed fusion yang sedikit menurun masih dapat menguji apakah learned fusion menyelesaikan trade-off-nya.

Jika gagal, hentikan perluasan dan laporkan kondisi tersisa sebagai NOT RUN. Jika lulus, jalankan hanya A2→A3→A5→A6 secara sekuensial. Tidak menambah A4, seluruh decoder stages, kombinasi L3, seed, atau H berdasarkan skor test. Diagnosis dan gate tersimpan sebelum perluasan; seluruh evaluasi test baru menunggu akhir matriks yang benar-benar dijalankan.

## Kontrol training dan inisialisasi

Manifest sama: **671 train / 118 validation / 261 source-test**, hash `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Ini duplicate-group split, **bukan patient-level split**. Karantina anotasi dan data mentah tidak berubah; tidak menggunakan target eksternal.

Seluruh named tensors source dimuat dari shared ImageNet encoder + decoder initial state Fase P, **bukan fine-tuning checkpoint B1**. Verifikasi nilai/nama/shape dilakukan sesudah menambah modul. Local parameters A1/A2/A3 dipasangkan persis. Modul baru menggunakan CPU `fork_rng` per komponen/stage agar tidak menggeser bobot bersama; aturan seed lengkap ada di config. A1/A2/A3 mempunyai forward initial yang sama. A5/A6 yang berbeda struktur tidak dapat dipasangkan seluruh parameternya; perbedaannya dicatat.

Adam LR 1e-4, WD 1e-5, batch 2, empat workers, source preprocessing/augmentasi, float32 tanpa AMP dan TF32 flags warisan tetap. Maksimal 200 epoch, early stopping 30 nonimproving epochs. Checkpoint highest validation Image Dice, earliest tie. ReduceLROnPlateau max/factor 0,1/patience 10/min LR 1e-5. Urutan dan actual augmented bytes dibandingkan dengan B1 pada common epochs; dedicated data-loader generators tetap.

Smoke enam gambar nyata termasuk GT kosong harus mencakup forward/backward, finite gradients, optimizer update, validation, checkpoint save/load setelah perturbasi parameter, serta gradien modul baru. Smoke diagnostic bukan hasil ilmiah. Satu proses training; shared GPU, `/dev/shm`, RAM/disk dan peak allocated/reserved dipantau. Tidak mengubah environment/batch/precision ketika resource bermasalah.

## Aturan pemilihan sebelum test

Kandidat eligible jika **semua** terpenuhi pada validation: overall Image Dice >A0; small Image Dice ≥A0; Boundary IoU ≥A0; total miss ≤A0; empty FP-image count ≤A0; active forward parameters ≤1,01×A0. Pilih overall Dice tertinggi; exact tie memakai urutan A1,A2,A3,A5,A6. Jika tidak ada, pertahankan A0. Ini guard eksploratif, bukan clinical acceptance threshold.

Latency dan peak memori dilaporkan dengan same-image benchmark termasuk inference A0 cached pada perangkat yang sama. Latency GPU bersama bukan parameter untuk tuning atau memilih seed. Selection ditulis sebelum semua test probabilities baru; seluruh test hasil model yang dijalankan dilaporkan, termasuk yang tidak lolos. Source-test telah dianalisis CP1/P/L dan tetap eksploratif.

## Analisis dan artefak

Laporkan keseluruhan, GT positif/kosong, small wounds, Boundary IoU, PFOM, precision/recall, TP=0 dan empty FP pixels/images. Simpan prediksi, GT, per-image CSV, config, source fingerprints, optimizer/scheduler/RNG, checkpoint, kurva LR/loss, waktu, active/registered params dan MACs conv/linear. MAC count tidak mencakup seluruh pooling/elementwise cost; latency diukur terpisah.

Gate A3 dianalisis per stage/gambar dan subgroup: mean/std bobot, normalized entropy, proporsi max weight>0,95; GT subgroup hanya untuk evaluasi. Sesudah primary selection, evaluasi intervensi validation A1/A3 drop-local (renormalisasi sisa bobot) dan A3 fixed-prior-gate. Ini inference intervention tanpa retraining, bukan causal proof atau model yang dilatih robust terhadap hilangnya cabang. Attention/gate map saja bukan bukti mekanisme klinis.

Run terpisah `outputs/cp2/phase_a/<timestamp>/`; historical CP1/P/L dan notebook pengguna dilindungi checksum. Completed checkpoints/runs tidak ditimpa. Source, config, aggregate figures dan numeric results dipublikasikan; foto klinis, raw data dan model besar tetap lokal. Laporan akhir membedakan implementasi, smoke, full training, kondisi deferred, kegagalan dan keterbatasan satu seed.

## Reproduksi dan pemantauan

Gunakan existing environment proyek dan baseline assets lokal yang checksum-nya sesuai config. Tidak ada download dataset/weights otomatis. Preflight mengaudit preservasi dan initialization, membandingkan CBAM dengan kode penulis, menjalankan seluruh tests/dependency checks serta lima real-data diagnostic smoke tests; skor smoke tidak menjadi dasar perluasan ilmiah.

```bash
.venv-training/bin/python scripts/run_cp2_phase_a.py --root <run> --stage preflight --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_a.py --root <run> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_a.py --root <run> --stage status
```

Training berjalan sekuensial: A1 → ekspor/diagnosis validation → keputusan perluasan tersimpan → kandidat perluasan yang diizinkan → seluruh validation → selection tersimpan → seluruh test. `--resume` hanya setelah memeriksa penyebab interupsi dan kecocokan source/config/input/environment/checkpoint. Completed stages tidak dilatih ulang. Jalankan Python lewat path virtualenv aslinya, tanpa meresolusikan symlink ke base interpreter. Simpan stdout/stderr ke log persisten; runner menulis workflow/status JSON dan GPU/shared-memory samples untuk pemantauan.
