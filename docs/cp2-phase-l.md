# CP2 Fase L — screening objective untuk foreground dan batas

Protokol ini ditulis sebelum training L1–L4. [Konfigurasi tetap](../configs/cp2/phase_l.json) memakai satu seed 42. Ini screening eksploratif, bukan konfirmasi multi-seed atau klaim novelty loss.

## Pertanyaan dan acuan

Apakah perubahan objective membantu menemukan luka kecil dan memperbaiki batas, tanpa menambah kegagalan total atau false positive pada anotasi kosong?

L0 adalah **B1 Fase P yang sudah completed**, checkpoint epoch 25 dari 55 epoch. Validation Image Dice 0,84276478 dan luka kecil 0,77284880. B1 tetap acuan sesuai roadmap walaupun source-test B0 sebelumnya sedikit lebih tinggi; source-test tidak dipakai mengganti baseline.

Seluruh artefak CP1 dan Fase P dilindungi checksum. Kode lama tidak diedit. Trainer L memakai data loader, preprocessing, model, evaluator, checkpoint I/O dan RNG helpers Fase P; loop objective-aware terpisah diperlukan agar trainer historis tetap utuh.

## Data dan kontrol bersama

- Manifest Fase P yang sama: **671 train / 118 validation / 261 source-test**, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`.
- Karantina konflik anotasi dan duplicate-group membership tidak berubah. Tidak ada ID pasien; ini bukan patient-level split. GT kosong bukan diagnosis sehat.
- Inisialisasi adalah salinan persis state ImageNet encoder + decoder baru yang dipakai B0/B1, **bukan** fine-tuning lanjutan dari checkpoint B1. Semua named parameters/buffers diverifikasi, bukan hanya seed.
- FUSegNet EfficientNet-B7/P-scSE tetap: 70.968.939 parameter registered, 64.898.248 aktif forward. Tidak ada auxiliary boundary head.
- Adam LR 1e-4, weight decay 1e-5, batch 2, empat workers, source augmentation, seed/worker RNG dan epoch input hashing sama.
- Environment existing `.venv-training`, torch 1.11.0+cu113, float32, tanpa AMP; TF32 flags CP1 tetap. Tidak menginstal/mengubah dependency.
- Maksimal 200 epoch. Checkpoint highest validation Image Dice, tie earliest. ReduceLROnPlateau pada metrik itu, factor 0,1/patience 10/min LR 1e-5. Early stopping 30 epoch tanpa perbaikan. Jadwal aktual/jumlah epoch dapat berbeda sebagai akibat validation.
- Threshold **probabilitas ≥0,5** dan evaluator/subgroup/Boundary IoU/PFOM sama dengan Fase P. Source-test telah dianalisis CP1/P; evaluasi berikutnya deskriptif, bukan untouched confirmatory test.

## Empat objective yang dikunci

| ID | Regional term | Boundary term | Kontras utama |
|---|---|---|---|
| L0 | Source Dice + corrected-logit focal | Tidak ada | Cached B1; tidak dilatih ulang |
| L1 | Tversky, q=1 | Tidak ada | Penalti FN > FP |
| L2 | Focal Tversky, q=0,75 | Tidak ada | Hanya eksponen terhadap L1 |
| L3 | Objective B1 yang sama | λ=0,01 tetap | Hanya tambahan boundary terhadap L0 |
| L4 | Objective L2 yang sama | λ=0,01 tetap | Hanya tambahan boundary terhadap L2 |

Soft TP/FP/FN memakai probabilitas sigmoid, diagregasi pada seluruh batch/piksel:

\[
TI=\frac{TP_s+1}{TP_s+0.3 FP_s+0.7 FN_s+1},\qquad
L_q=(1-TI+\delta)^q-\delta^q,\quad \delta=10^{-6}.
\]

Delta adalah stabilisasi numerik eksplisit supaya gradien q<1 tetap finite pada TI=1. Untuk q=1 ekspresi kembali ke 1−TI, selain pembulatan floating point. Ini bukan klaim formula implementasi paper yang identik. Jangan menukar q dengan gamma dari parameterisasi lain. Kurva/gradien q=0,75 tidak boleh dijelaskan sebagai jaminan perbaikan semua kasus sulit. [Focal Tversky paper](https://arxiv.org/abs/1810.07842).

### Boundary term dan kebijakan GT kosong

Implementasi binary foreground diadaptasi dari [kode Kervadec](https://github.com/LIVIAETS/boundary-loss/tree/171c32d88a4ce59af8be46fb88b96d3637b9515b), MIT, dengan referensi [Boundary Loss](https://arxiv.org/abs/1812.07032). License disimpan di `licenses/boundary-loss-MIT.txt`. Primitive distance dan product-mean diperiksa terhadap fungsi penulis yang dipin.

Untuk mask G yang **sudah mengalami augmentasi geometrik**, distance map φ menggunakan EDT Euclidean: positif di luar foreground dan `−(EDT(G)−1)` di dalam, sehingga inner boundary bernilai nol. Konvensi tepi/full mask mengikuti fungsi referensi SciPy tanpa padding tambahan.

\[
L_{boundary}=\operatorname{mean}_{batch,pixel}(p\,\phi_G),\qquad
L_{total}=L_{regional}+0.01L_{boundary}.
\]

Jarak dalam pixel native 224×224; **tanpa normalisasi distance**, class/size weighting, clipping atau warm-up. λ dipraspesifikasikan, bukan dituning berdasarkan test. Nilai boundary dapat negatif dan tidak diubah menjadi absolute value. Pada mask kosong φ=0; term regional tetap memberi penalti FP. Reduction boundary tetap mean seluruh batch, termasuk zero-map kosong. Ini adaptasi foreground binary, bukan reproduksi studi Kervadec secara lengkap.

Distance map dibuat dari augmented mask setiap batch. Tidak menghitung map pada mask asli lalu memperlakukannya seolah jarak Euclidean masih benar setelah resize/warp. GT/distance tidak dipakai saat inference.

## Pemilihan kandidat sebelum source-test

Setelah seluruh training selesai, ekspor validation dari selected checkpoint tiap model. Kandidat lolos screen konservatif bila:

1. Overall validation Image Dice **lebih tinggi** dari L0.
2. Small-wound validation Image Dice tidak lebih rendah dari L0.
3. Jumlah GT positif dengan TP=0 tidak bertambah.
4. Jumlah GT kosong dengan foreground terprediksi tidak bertambah.

Pilih overall validation Image Dice tertinggi dari kandidat eligible; exact tie memakai urutan L1, L2, L3, L4. Jika tidak ada yang lolos, pertahankan L0. Ini aturan screening, bukan ambang kelayakan klinis. Seluruh trade-off tetap ditampilkan, termasuk kandidat yang gagal guard.

Simpan `selection.json` **sebelum** menghasilkan test predictions baru. Setelah itu evaluasi semua empat kandidat pada 261 source-test yang sama; jangan hanya menampilkan pemenang. Tidak ada threshold/parameter tuning dari test.

L5–L7 adalah pengembangan **bersyarat setelah diagnosis**, bukan matriks otomatis: aggregation per-image, distance normalization/size weighting, dan fixed vs warm-up perlu kontras terisolasi serta protokol baru. Pada fixed vs warm-up, anggaran langkah harus dibuat sama. Multi-seed confirmation, Fase A/H dan evaluasi eksternal belum termasuk screening ini.

## Eksekusi dan artefak

```bash
.venv-training/bin/python scripts/run_cp2_phase_l.py \
  --root outputs/cp2/phase_l/<timestamp> --stage prepare
.venv-xfuseg/bin/python -m pytest -o addopts='' -q
.venv-training/bin/python scripts/run_cp2_phase_l.py \
  --root outputs/cp2/phase_l/<timestamp> --stage smoke --device cuda:0
# Setelah tests, reference oracle dan empat real-data smoke lulus:
.venv-training/bin/python scripts/run_cp2_phase_l.py \
  --root outputs/cp2/phase_l/<timestamp> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_l.py \
  --root outputs/cp2/phase_l/<timestamp> --stage status
```

Satu proses mengeksekusi L1→L2→L3→L4 secara berurutan. Smoke masing-masing memakai enam gambar nyata, termasuk GT kosong, untuk forward/backward/optimizer/validation/checkpoint replay. Skor smoke bersifat diagnostik.

`--resume` hanya setelah memeriksa kegagalan; config, source, package versions dan input identities harus sama. Completed stages tidak dilatih ulang. Checkpoint/NPZ/JSON memakai atomic replace; partial export dipindahkan ke arsip lokal sebelum replay, tidak dihapus. Log persisten, status, component losses, LR, waktu, peak CUDA allocated/reserved dan pemakaian keseluruhan GPU disimpan. GPU sharing memengaruhi waktu; pekerjaan pengguna lain tidak disentuh.

Run ada di `outputs/cp2/phase_l/<timestamp>/`, pointer lokal `outputs/cp2/latest_phase_l.json`. Model/data/gambar klinis tetap lokal. Hanya source, config, agregat metrik, membership tanpa foto, dan grafik penelitian yang masuk Git.
