# CP2 — Visual index presentasi P0

Paket ini menambahkan bukti visual dari eksperimen **yang sudah completed dan tervalidasi**, tanpa training, perubahan threshold, seleksi ulang checkpoint, atau perubahan hasil historis. Foto, GT, mask prediksi, serta overlay berasal dari data dan cache asli. Dua kasus Cascade Ganta direplay menggunakan checkpoint asli di **CPU**, dengan refiner input dan proses stitch yang diperiksa langsung.

Penomoran slide mengikuti dokumen `KC_CP2_Visual_Asset_Audit_and_Agent_Brief.md` yang diunggah. File blueprint terpisah yang disebut dokumen tersebut tidak tersedia saat audit; indeks ini tidak mengasumsikan isi slide yang tidak diberikan.

## Cara memakai gambar

Semua PNG berukuran **2400×1350 (16:9)**. Galeri memiliki tiga tampilan: **full** mempertahankan kanvas evaluasi 224×224; **zoom** memperbesar area isi foto dan seluruh GT/prediksi, dengan batas tampilan yang sama antarmodel; **mask** memperlihatkan prediksi biner sebenarnya pada kanvas penuh. Zoom hanya pendamping tampilan, tidak digunakan untuk inference atau penghitungan metrik.

Warna overlay: **hijau = TP**, **jingga/vermillion = FP**, **biru = FN**. FP berarti foreground di luar *anotasi GT*, bukan bukti bahwa pasien tidak memiliki luka. Gambar yang mempunyai GT kosong tidak boleh disebut sebagai pasien sehat. Padding hitam pada foto asli dipertahankan pada versi full.

Kasus dipilih **posthoc dengan aturan deterministik**: gain/regresi terbesar sesuai populasi yang dinyatakan, atau kegagalan yang memenuhi definisi; tie dipecahkan berdasarkan sample ID. Contoh ekstrem tersebut menjelaskan mekanisme dan error, bukan estimasi frekuensi keberhasilan atau dasar pemilihan model.

## Pemetaan slide → gambar → pesan

| Slide | Figure yang tersedia | Caption / klaim yang didukung | Batas interpretasi / status |
|---|---|---|---|
| 03 — masalah CP1 | [Full](figures/cp2-presentation/03_cp1_failure_gallery.png) · [Zoom](figures/cp2-presentation/03_cp1_failure_gallery_zoom.png) · [Mask](figures/cp2-presentation/03_cp1_failure_gallery_masks.png) | CP1 epoch52 melewatkan seluruh 64 piksel GT pada satu luka kecil, menghasilkan FP pada GT kosong, dan melakukan undersegmentation pada contoh lain. | **VERIFIED.** Test CP1 historis n=278; bukan cohort CP2. Tiga kasus dipilih berdasarkan kegagalan, bukan sampel acak. |
| 06 — Fase P | [Full](figures/cp2-presentation/06_phase_p_case_gallery.png) · [Zoom](figures/cp2-presentation/06_phase_p_case_gallery_zoom.png) · [Mask](figures/cp2-presentation/06_phase_p_case_gallery_masks.png) | B0 dan B1 memprediksi foto validation yang sama. Koreksi input focal dapat mengurangi FP pada satu contoh, tetapi tidak memperbaiki semua gambar. | **VERIFIED, tambahan P0.** B0 epoch17 / B1 epoch25; gain/regresi B1−B0. Ini perbandingan checkpoint terpilih dari training historis, bukan klaim efek kausal dari satu gambar. |
| 07 — Fase L | [Full](figures/cp2-presentation/07_phase_l_case_gallery.png) · [Zoom](figures/cp2-presentation/07_phase_l_case_gallery_zoom.png) · [Mask](figures/cp2-presentation/07_phase_l_case_gallery_masks.png) | Pada luka kecil yang sama, Dice B1 1,42%, L2 82,94%, L3 67,92%; contoh kedua memperlihatkan regresi L3. | **VERIFIED.** Aturan pemilihan berdasarkan delta L3−B1, bukan masing-masing model mencari gambar terbaiknya sendiri. Gain lokal tidak menggantikan hasil validation keseluruhan. |
| 08 — Fase A | [Full](figures/cp2-presentation/08_phase_a_case_gallery.png) · [Zoom](figures/cp2-presentation/08_phase_a_case_gallery_zoom.png) · [Mask](figures/cp2-presentation/08_phase_a_case_gallery_masks.png) | B1/A1/A6 dapat dibandingkan pada piksel yang sama; A6 membantu contoh pertama, tetapi memperluas FP pada contoh regresi. | **VERIFIED.** A6 tetap tidak meloloskan seluruh guard historis; gambar ini tidak memilih ulang kandidat. |
| 09 — Fase H | [Full](figures/cp2-presentation/09_phase_h_case_gallery.png) · [Zoom](figures/cp2-presentation/09_phase_h_case_gallery_zoom.png) · [Mask](figures/cp2-presentation/09_phase_h_case_gallery_masks.png) | B1/H1/H2 memperlihatkan perubahan **mask utama** setelah auxiliary supervision; ada gain dan regresi. | **VERIFIED.** Tidak melabeli mask utama sebagai output boundary head. Boundary target/auxiliary map tidak ditambahkan karena tidak diperlukan untuk galeri P0 ini. |
| 10 — Fase N | [Full](figures/cp2-presentation/10_phase_n_case_gallery.png) · [Zoom](figures/cp2-presentation/10_phase_n_case_gallery_zoom.png) · [Mask](figures/cp2-presentation/10_phase_n_case_gallery_masks.png) | Pada satu GT kosong, B1 FP=0 dan N1 FP=33; pada contoh lain keduanya masih memiliki FP. Mean Image Dice validation B1 84,28%, N1 82,69%. | **VERIFIED.** Hanya tujuh mask kosong di validation. Presence head tidak terbukti menyelesaikan masalah FP. |
| 11 — konteks kombinasi completed | [Konteks A1H2](figures/cp2-presentation/11_combinations_completed_context.png) | Matriks faktor B1/A1/H2/A1H2 dan metrik validation menunjukkan bahwa menggabungkan komponen tidak otomatis meningkatkan skor/guard. | **VERIFIED, PARTIAL CONTEXT.** A1H2 adalah kombinasi historis completed, bukan hasil final dua kombinasi A6+H2 terbaru. |
| 11 — final seluruh kombinasi | `11_combinations_final_comparison.png` **belum dibuat** | Menunggu laporan final dan QA B1+A6+H2 serta L3+A6+H2. | **DEFERRED.** Workflow terbaru belum memiliki laporan final yang lolos QA saat audit. Tidak diperbaiki, dijalankan ulang, atau dimasukkan ke ranking dalam pekerjaan visual ini. |
| 12 — Cascade Ganta asli | [Full walkthrough](figures/cp2-presentation/12_cascade_original_walkthrough.png) · [Zoom walkthrough](figures/cp2-presentation/12_cascade_original_walkthrough_zoom.png) | Foto → coarse → proposal/crop aktual → mask refiner dan stitch. Contoh gain: Dice 18,54%→75,90%, dua crop (tiga forward). Contoh GT kosong: FP 724→755, satu crop (dua forward). | **VERIFIED, REAL/FRESH CPU.** Coarse epoch41 / refiner epoch27 adalah checkpoint **Ganta asli**. Test historis n=278, berbeda dengan CP2. Crop ditentukan dari prediksi coarse, tidak dari GT. |
| 13 — Cascade CP2 final | `13_cascade_cp2_final_walkthrough.png` **belum dibuat** | Memerlukan checkpoint dan evaluasi final dari training Cascade CP2 yang masih berjalan saat audit. | **DEFERRED.** Tidak membaca checkpoint interim untuk produksi figure ini. Audit epoch27 CP2 yang lama tetap merupakan hasil interim, bukan final. |
| 14 — unified comparison | [PNG scoreboard](figures/cp2-presentation/14_cp2_same_cohort_scoreboard.png) · [SVG](figures/cp2-presentation/14_cp2_same_cohort_scoreboard.svg) | Perbandingan 15 model completed pada **118 validation / 261 source-test yang identik**, seed42, Mean Image Dice dalam persen. Epoch checkpoint dicantumkan. | **VERIFIED, PARTIAL UNTIL FINAL RUNS.** Dua kombinasi terbaru dan Cascade CP2 final belum dimasukkan. Ganta asli, CP1, dan CO2 tidak dicampur dalam cohort ini. Source-test sudah terekspos dan bersifat deskriptif. |
| 14 — guard/trade-off | [PNG heatmap](figures/cp2-presentation/14_cp2_guard_tradeoffs.png) · [SVG](figures/cp2-presentation/14_cp2_guard_tradeoffs.svg) | Delta validation terhadap B1 untuk Image Dice, Small Dice, Boundary IoU, foreground miss dan empty FP memperlihatkan biaya dari kenaikan skor. | **VERIFIED.** Guard asli masing-masing fase dipertahankan; definisinya berbeda antar-fase. Tidak menerapkan guard baru. B1 tetap baseline yang dipertahankan. |
| 15 — diskusi kasus nyata | [Full](figures/cp2-presentation/15_cross_model_real_cases.png) · [Zoom](figures/cp2-presentation/15_cross_model_real_cases_zoom.png) · [Mask](figures/cp2-presentation/15_cross_model_real_cases_masks.png) | B1/A6/H2/A1H2 memprediksi tiga gambar yang identik: gain luka kecil, GT-empty FP, dan regresi kombinasi historis. | **VERIFIED, PARTIAL UNTIL FINAL CASCADE.** Tidak menambahkan kolom Cascade CP2 atau menganggap A1H2 sebagai kombinasi final terbaik. |

Slide 01/02/04/05/16 dan diagram tambahan termasuk prioritas desain/P1; tidak dibuat dalam pekerjaan P0 ini. Seluruh galeri yang dapat dibangun dari completed runs tersedia. Kedua aset final yang bergantung pada eksperimen belum selesai tetap ditunda.

## Angka dan definisi yang dipertahankan

- Semua CP2 comparison di sini memakai manifest clean **671 train / 118 validation / 261 source-test**, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Pengelompokan berdasarkan decoded-RGB identik, **bukan patient-level**.
- Threshold mask **probability ≥0,5**; GT dan prediksi kosong bersama memberi Image Dice=1, sedangkan GT kosong dengan prediksi foreground memberi Image Dice=0.
- Small: **0 < GT ≤1% kanvas berpadded 224×224**, sesuai subgroup CP2 historis, bukan pengganti kategori penulis. Boundary IoU mengikuti implementasi CP2 yang terkunci; bukan PFOM atau Boundary F1.
- B1 validation Image Dice **84,2765%**, A6 **85,1686%**, H2 **84,8825%**, A1H2 **84,1838%**. Semua angka berasal dari cache completed yang dihitung ulang. Kenaikan satu metrik belum berarti kandidat meloloskan guard.
- Cascade Ganta menggunakan dua model, masing-masing **64.898.248 parameter**, total **129.796.496**. Jumlah forward per foto adalah **1 + jumlah proposal yang direfine**. Waktu dua replay CPU dengan hook bukan benchmark latency deployment.

## Reproducibility dan provenance

[Manifest paket](cp2-presentation-manifest.json) menyimpan SHA256 kode/config, checkpoint, sumber foto/GT, cache, epoch, seed, sample ID, metrik, aturan seleksi dan hash setiap figure. Tiap PNG/SVG mempunyai companion `<nama-file>.json` di folder figure. [Laporan QA](cp2-presentation-validation.json) mencatat pemeriksaan metrik, preservation, pengujian dan visual review.

Whitespace akhir baris pada kedua SVG hasil Matplotlib dinormalisasi saat publikasi. Hash sebelum/sesudah tercatat, dan hasil rasterisasi Inkscape terbukti identik piksel demi piksel dengan versi yang diperiksa secara visual; nilai dan geometri figure tidak berubah.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-xfuseg/bin/python -u scripts/build_cp2_presentation.py \
  --config configs/cp2/presentation_p0.json \
  --run-root outputs/presentation/cp2_p0/replay-01
```

Runner membutuhkan dataset/cache/checkpoint lokal yang tercantum di config, membuat direktori render bertimestamp baru, memeriksa hash sumber, menghitung ulang metrik dan tidak memanggil training. Aset yang hilang/berubah menyebabkan kegagalan yang jelas. Source Ganta direlokasi ke `notebook/scripts/13_cascade_crop_refine_chronic_wound.py`; hash tetap sama dengan replay historis. Tidak ada penulisan ke folder partner.

Run produksi lokal: `outputs/presentation/cp2_p0/20261119T155448Z/`. Input CP1, seluruh run CP2 sebelumnya, notebook yang telah dimodifikasi pengguna, serta arsip dokumentasi upload dilindungi inventory awal. Data mentah lengkap, cache NPZ, crop tensor dan checkpoint tidak dipush; figure analisis terpilih ikut dipublikasikan sesuai instruksi pengguna.

## Sumber dan batas publikasi

Foto/GT: **Chronic Wound / AZH wound-care-center patches**, melalui [uwm-bigdata/wound-segmentation, revision 42a272d](https://github.com/uwm-bigdata/wound-segmentation/tree/42a272dfe0679f20675e826385925cb7562934b6). Transformasi yang ditampilkan: overlay TP/FP/FN, mask hasil model, zoom tampilan, dan crop inference Cascade. Sample ID adalah nama berkas dataset; tidak ada nama pasien atau metadata identitas pasien.

Pengguna secara eksplisit menginstruksikan foto/overlay analisis terpilih ikut dipush. Hal tersebut dicatat sebagai otorisasi publikasi proyek, **tidak dinyatakan sebagai lisensi dataset baru**; audit lisensi historis tidak diubah. Galeri foto CO2Wounds bersifat opsional dan tidak dibuat pada cakupan P0 ini. [Hasil agregat CO2Wounds yang sudah completed](cp2-external-evaluation.md) tetap dapat digunakan sebagai konteks eksternal terpisah, dengan caveat cross-source/cross-pathology dan tanpa menjadikannya dasar seleksi model.
