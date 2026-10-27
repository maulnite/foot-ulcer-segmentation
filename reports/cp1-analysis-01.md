# CP1 · Analisis baseline FUSegNet

Analisis ini membaca checkpoint **epoch 52** dari `cp1-baseline-01`, memakai threshold **≥0.5** dan evaluator **offline HARD_LINE**. Tidak ada training, tuning threshold, atau pemilihan checkpoint ulang. Seluruh 124 mask validation dan 278 mask test diekspor melalui environment training asli; setiap TP/TN/FP/FN cocok dengan evaluasi akhir.

[Notebook lengkap yang sudah dieksekusi](../notebooks/01_cp1_analysis.ipynb) dipublikasikan di GitHub dengan seluruh tabel, grafik, serta lima galeri foto/GT/prediksi/overlay. Semua output langsung terlihat tanpa menjalankan cell. Lihat [petunjuk menjalankan ulang dan ekspor](../notebooks/README.md).

## Hasil utama

| Metrik | Validation, 124 gambar | Test, 278 gambar |
| :--- | ---: | ---: |
| Data-based Dice | 85.90% | 92.29% |
| Image-based Dice | 82.34% | 86.42% |
| Data-based IoU | 75.29% | 85.68% |

Paper melaporkan 92.70% data-based dan 86.05% image-based Dice untuk Chronic Wound. Selisih lokal masing-masing −0.41 dan +0.37 percentage points, dengan **protokol yang belum dapat dianggap identik**. Arsip publik berisi 831/278 source train/test dibanding 810/200 di paper; split train/validation dan evaluator tabel asli belum pasti. Detail run dan delapan metrik ada di [laporan baseline](cp1-baseline-01.md).

Kurva mencakup 82 epoch. Puncak validation F-score pada log ada di epoch 20, sementara aturan loss/IoU penulis memilih epoch 52. F-score di log memakai rata-rata batch dengan smoothing; metrik akhir memakai agregasi piksel/per gambar dan kebijakan GT kosong. Keduanya tidak langsung dapat disamakan.

## Temuan error yang terukur

| Subset test | Jumlah | Temuan |
| :--- | ---: | :--- |
| GT berisi luka | 261 | Median Dice 92.83%; 4 gambar memiliki TP=0 |
| C1, kelompok luka terkecil penulis | 36 | Luas GT 0.04–0.60% gambar; mean Dice 73.22% |
| C9, kelompok luka terbesar penulis | 7 | Luas GT 11.59–30.88% gambar; mean Dice 97.36% |
| GT kosong | 17 | 10 benar kosong; 7 menghasilkan FP sebanyak 56–1,229 piksel |

Hubungan ukuran luka dengan Dice bersifat deskriptif; jumlah sampel tidak seimbang dan faktor gambar lain belum dikontrol. Luas memakai seluruh kanvas 224 × 224 termasuk padding hitam, sehingga dipengaruhi skala/crop foto sumber dan tidak menunjukkan ukuran fisik luka. GT kosong adalah keterangan anotasi, bukan diagnosis pasien.

HARD_LINE memberi skor per gambar 1 pada GT/prediksi sama-sama kosong dan 0 pada GT kosong dengan FP berapa pun. Karena itu, notebook menampilkan jumlah piksel FP di samping jumlah gambar bermasalah. Skor keseluruhan tetap mencakup semua gambar.

## Contoh prediksi

Lima galeri menampilkan masing-masing tiga foto: Dice tertinggi, dekat median, terendah, acak seed 42, dan GT kosong. Setiap baris berisi gambar asli, GT, prediksi, dan overlay TP/FP/FN.

Galeri men-zoom tampilan untuk menghilangkan margin hitam kosong, dengan crop yang sama pada keempat kolom dan margin 4 piksel. Seluruh piksel foto nonhitam, GT, serta prediksi tetap masuk tampilan. Metrik dan luas dihitung pada kanvas penuh, tanpa perubahan mask atau data.

Sampel acak dipilih lebih dahulu dari foto RGB unik dengan GT positif. Galeri berikutnya menghindari foto yang sudah dipilih. Kasus terburuk diurutkan menurut Dice terendah, GT terbesar, lalu nama file. Galeri GT kosong menampilkan satu benar kosong dan dua FP terbesar. Aturan serta file pilihan dicatat dalam `outputs/analysis/cp1-baseline-01/tables/selected_examples.csv` dan `analysis-summary.json`. Kasus ekstrem bukan estimasi frekuensi error.

## Audit overlap pada split aktual

Hash seluruh piksel RGB dibandingkan dengan **707 gambar train aktual**, bukan seluruh 831 source train sebelum holdout.

| Subset | Gambar dengan RGB identik di train | Gambar tanpa kecocokan RGB di train |
| :--- | ---: | ---: |
| Validation | 2 | 122 |
| Test | 16 | 262 |

Pada subset test tanpa kecocokan RGB, data-based Dice 92.17% dan image-based Dice 86.05%. Angka ini merupakan pembacaan subset dari prediksi yang sama, bukan eksperimen baru dengan training yang sudah didedup. Tidak adanya kecocokan RGB juga belum membuktikan independensi pasien; foto lain dari pasien/luka yang sama tidak terdeteksi oleh audit ini.

## Implikasi CP1 dan CP2

Training baseline, evaluasi, analisis awal, dan visualisasi CP1 tersedia. Laporan dan slide masih perlu dirangkai dari bukti ini. Belum ada ablasi metode baru, estimasi variasi antar-seed, atau klaim peningkatan yang terkontrol.

Luka kecil dan false positive pada background merupakan kandidat pertanyaan CP2. Perubahan loss, attention, atau penanganan ketidakseimbangan harus diuji melalui validation dan ablasi dengan protokol tetap. Analisis saat ini belum membuktikan intervensi mana yang efektif. Bila pengembangan mengikuti temuan test, ungkapkan penggunaan test tersebut; jangan menyebut evaluasi berikutnya sebagai test baru yang sepenuhnya tidak tersentuh.

Artefak lokal: `outputs/analysis/cp1-baseline-01/` memuat lima grafik agregat PNG/SVG, lima galeri prediksi PNG, tabel CSV, ringkasan JSON, hash cache prediksi, dan catatan eksekusi notebook. Contoh foto, GT, dan prediksi terpilih tertanam pada notebook publik dengan atribusi sumber; dataset lengkap, checkpoint, dan cache mask tetap lokal.
