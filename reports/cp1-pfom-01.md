# CP1 · PFOM dan ketepatan batas luka

PFOM dihitung pada **27 Oktober 2026** dari mask prediksi yang sudah diverifikasi untuk `cp1-baseline-01`. Checkpoint tetap epoch **52**, threshold **≥0.5**, split **124 validation / 278 test**. Tidak ada training atau inference ulang. Dice, IoU, precision, recall, dan hitungan piksel sebelumnya tidak berubah.

## Hasil per gambar

PFOM berikut memakai rumus Persamaan 13 pada pasangan edge nonkosong, dengan ekstensi kasus kosong: kedua kosong → 1, tepat satu kosong → 0. Rentang 0–1, semakin tinggi semakin baik. Mean utama mencakup semua gambar.

| Ringkasan | Validation | Test |
| :--- | ---: | ---: |
| Jumlah gambar | 124 | 278 |
| Mean PFOM, kebijakan kosong 1/0 | **0.7244** | **0.7782** |
| Median PFOM | 0.7966 | 0.8558 |
| Mean PFOM pada GT positif | 0.7637 | 0.7905 |
| Kedua edge nonkosong: jumlah gambar | 114 | 257 |
| Mean pada subset kedua edge nonkosong | 0.7704 | 0.8029 |

Subset kedua edge nonkosong mengecualikan kasus tanpa edge dan karena itu tidak menggantikan mean utama. Pada test, empat luka yang terlewat tetap mendapat skor 0 dalam mean utama dan mean GT positif.

## Kasus khusus yang disebut paper

| Keadaan | Validation | Test |
| :--- | ---: | ---: |
| Kedua edge map kosong | 2 | 10 |
| Hanya edge GT kosong | 7 | 7 |
| Hanya edge prediksi kosong | 1 | 4 |
| Mean proxy sentinel 2 | **0.8696** | **0.8933** |

Bagian 4.2 [paper FUSegNet](https://arxiv.org/pdf/2305.02961v2#page=13) menyebut pengganti 2 untuk nilai tak hingga pada kasus tanpa edge, tetapi kode dan pemicu persisnya tidak tersedia. Proxy di sini mengartikan kalimat tersebut sebagai **sedikitnya satu edge map kosong → 2**. Hasilnya bukan skor akurasi 200% dan tidak boleh dicampur dengan PFOM 0–1. Selisih mean berasal dari kebijakan kasus kosong pada prediksi yang sama.

## Per kategori test

Kategori mengikuti keanggotaan JSON penulis C0–C9 yang juga dipakai pada analisis Dice. C0 mempunyai GT kosong.

| Kategori | n | Mean PFOM 1/0 | Median PFOM | Mean proxy sentinel 2 |
| :--- | ---: | ---: | ---: | ---: |
| C0 | 17 | 0.5882 | 1.0000 | 2.0000 |
| C1 | 36 | 0.7227 | 0.8557 | 0.8893 |
| C2 | 35 | 0.8233 | 0.8551 | 0.8233 |
| C3 | 37 | 0.7899 | 0.8339 | 0.8440 |
| C4 | 24 | 0.7819 | 0.7939 | 0.7819 |
| C5 | 35 | 0.7798 | 0.8229 | 0.7798 |
| C6 | 29 | 0.8332 | 0.8324 | 0.8332 |
| C7 | 38 | 0.7970 | 0.8649 | 0.7970 |
| C8 | 20 | 0.8018 | 0.8787 | 0.8018 |
| C9 | 7 | 0.8183 | 0.8770 | 0.8183 |

PFOM menunjukkan informasi batas yang melengkapi Dice. Mean kategori tidak naik secara monoton dengan luas GT. C1 memiliki tiga kasus edge prediksi kosong, sementara C3 memiliki satu; C0 tidak mengevaluasi lokalisasi batas luka. Hasil ini deskriptif dan belum menguji penyebab error atau perbaikan metode.

## Protokol, pengujian, dan kesetaraan

Beta **1/9** dan Canny sigma **0.1** mengikuti paper. Implementasi lokal memakai **scikit-image 0.19.3 / SciPy 1.8.1**, input float64 0/1, threshold Canny 0.1/0.2, border constant 0, dan jarak Euclidean prediksi-ke-GT pada kanvas penuh 224 × 224. Library, threshold, dan rincian border Canny tidak dipublikasikan penulis. Kebijakan kosong 1/0 dan interpretasi sentinel 2 merupakan keputusan lokal yang eksplisit. Karena itu, ini implementasi yang diuji dan terdokumentasi, **belum merupakan reproduksi numerik Figure 6 yang terverifikasi**.

Sembilan uji sintetis lulus, termasuk jarak analitik, kebijakan kosong, dan pembanding brute-force 40 pasangan acak. Seluruh 402 skor juga diperiksa melalui environment training yang dibekukan: jumlah edge/kasus sama dan selisih skor maksimum sekitar **5 × 10⁻¹⁶**. Versi paket analysis sebelumnya dipertahankan ketika dependency PFOM ditambahkan; environment training tidak diubah. Protokol lengkap dan cara menjalankan ulang ada di [docs/pfom.md](../docs/pfom.md).

Grafik dan tabel tersedia pada [notebook bagian PFOM](../notebooks/01_cp1_analysis.ipynb#pfom). Bukti numerik publik: [ringkasan CSV](tables/cp1-pfom-summary.csv), [kategori CSV](tables/cp1-pfom-categories.csv), [kasus edge kosong](tables/cp1-pfom-empty-edges.csv), [402 skor per gambar](tables/cp1-pfom-per-image.csv), dan [metadata](tables/cp1-pfom-metadata.json). Sumber ekspor dan PNG/SVG berada di `outputs/analysis/cp1-baseline-01/`. Penilaian ini hanya untuk satu baseline, tanpa menjalankan ulang model scSE pembanding pada Figure 6 paper. Keterbatasan jumlah data, split asli, overlap, dan satu seed pada [laporan baseline](cp1-baseline-01.md) tetap berlaku.
