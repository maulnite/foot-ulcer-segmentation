# CP2 — visual final untuk slide 13–16

Paket ini melengkapi [P0 historis](cp2-presentation-visual-index.md) setelah dua kombinasi dan Cascade CP2 selesai. **Tidak ada retraining, perubahan konfigurasi eksperimen, seleksi ulang checkpoint, perubahan threshold, atau penggantian hasil lama.** Slide PowerPoint tidak dibuat di sini.

## Peta figure ke slide

| Slide | Figure | Yang ditunjukkan dan cara membacanya | Laporan ilmiah |
|---|---|---|---|
| 13 | [Walkthrough utuh](figures/cp2-presentation-final/13_cascade_cp2_final_walkthrough.png) | Enam tahap aktual: foto → coarse → proposal → crop tensor aktual → mask refiner → stitched mask. Satu gain luka kecil dan satu regresi ditampilkan. | [Cascade CP2](cp2-cascade-clean.md) |
| 13 | [Walkthrough zoom](figures/cp2-presentation-final/13_cascade_cp2_final_walkthrough_zoom.png) | Pasangan kasus yang sama; viewport foto mempertahankan seluruh GT/prediksi dan ROI. Crop/refiner tetap pada grid crop224, bukan grid foto. | [Cascade CP2](cp2-cascade-clean.md) |
| 13 | [Walkthrough masks](figures/cp2-presentation-final/13_cascade_cp2_final_walkthrough_masks.png) | Coarse, mask refiner, dan mask akhir biner yang benar-benar dihasilkan. Putih foreground, hitam background. | [Cascade CP2](cp2-cascade-clean.md) |
| 13 | [Arsitektur SVG](figures/cp2-presentation-final/13_cascade_cp2_architecture.svg) | Diagram implementasi: coarse B1 frozen; refiner trainable hanya saat training. Training crop memakai GT training; inference proposal hanya coarse. | [Konfigurasi terkunci](../configs/cp2/cascade_clean.json) |
| 14 | [Scoreboard PNG](figures/cp2-presentation-final/14_cp2_final_scoreboard.png) | Semua 18 kondisi seed42, termasuk dua kombinasi terbaru dan Cascade, pada validation118 yang sama. Urutan keluarga, bukan pemilihan ulang berdasarkan test. | [Fase C](cp2-phase-c.md), [kombinasi](cp2-phase-c-combinations.md), [Cascade](cp2-cascade-clean.md) |
| 14 | [Scoreboard SVG](figures/cp2-presentation-final/14_cp2_final_scoreboard.svg) | Versi vektor dengan angka dan urutan identik. | Sama seperti PNG |
| 14 | [Guard/trade-off PNG](figures/cp2-presentation-final/14_cp2_final_guard_tradeoffs.png) | Nilai absolut Image Dice, Small Dice, Boundary IoU, foreground miss, dan GT-empty FP; guard asli tiap fase dan applicability budget dipisahkan. | [Review A](cp2-phase-a-review.md), [H](cp2-phase-h.md), [kombinasi](cp2-phase-c-combinations.md) |
| 14 | [Guard/trade-off SVG](figures/cp2-presentation-final/14_cp2_final_guard_tradeoffs.svg) | Versi vektor; Cascade tidak dilabeli lolos budget attention1,01×. | Sama seperti PNG |
| 15 | [External comparison](figures/cp2-presentation-final/15_external_cascade_comparison.png) | CO2Wounds: mean/pooled/small/boundary membaik, tetapi total miss meningkat143→153. Keduanya harus dibahas. | [Cascade final](cp2-cascade-clean.md), [audit eksternal](cp2-external-evaluation.md) |
| 15 | [External trade-offs](figures/cp2-presentation-final/15_external_cascade_tradeoffs.png) | Transisi miss pada foto yang sama dan jumlah foto dengan Dice naik/sama/turun, dihitung dari seluruh558 pasangan. | [Cascade final](cp2-cascade-clean.md) |
| 16 | [Cross-model utuh](figures/cp2-presentation-final/16_cross_model_real_cases.png) | Input/GT/B1/A6/H2/Cascade pada empat foto validation yang identik; metrik tetap pada seluruh kanvas224. | [Metadata figure](figures/cp2-presentation-final/16_cross_model_real_cases.png.json) |
| 16 | [Cross-model zoom](figures/cp2-presentation-final/16_cross_model_real_cases_zoom.png) | Viewport sama antarmodel untuk setiap foto, mempertahankan seluruh GT dan semua prediksi termasuk FP jauh. Zoom tidak digunakan menghitung skor. | Sama seperti galeri utuh |
| 16 | [Cross-model masks](figures/cp2-presentation-final/16_cross_model_real_cases_masks.png) | Mask aktual threshold≥0,5; tidak ada rekonstruksi ilustratif atau mask yang diedit. | Sama seperti galeri utuh |

Grafik training dan comparison Cascade yang sudah ada **digunakan ulang**, bukan ditimpa: [training76epoch](figures/cp2-cascade-clean/training.png), [source/external comparison](figures/cp2-cascade-clean/comparison.png).

## Slide 13 — mekanisme dan kasus aktual

Coarse adalah **B1 epoch25**, SHA256 `e5f147018231b0e13926daa5d0158957e42831404ab05bd3151c1115592aeeeb`. Refiner CP2 terpilih **epoch46** dari76epoch, SHA256 `a30cc65200e3a39be9a7ef60e2e1c8f5aebe79aa3278bfdecfa16949832a52a9`. Checkpoint terlatih Ganta asli tidak dipakai. Seluruh crop input dan output refiner ditangkap lewat forward hook dalam eval/no-grad CPU, lalu hasil stitch dibandingkan bit-exact dengan probability cache final.

| Kasus | Sample ID validation | Coarse Dice | Cascade Dice | Proposal xyxy, batas kanan/bawah eksklusif |
|---|---|---:|---:|---|
| Gain luka kecil terbesar | `train/4d0ae09266e45178c65b2c217d2ccd43_1.png` | 1,42% | 88,00% | `[0,0,56,56)` |
| Regresi terbesar pada GT positif | `train/4ed0412508164104bc694a0ce0f5bffd_1.png` | 95,71% | 85,76% | `[0,4,86,90)` |

Awalan `train/` pada sample ID menyimpan folder asal dataset mentah. Kedua foto di atas merupakan anggota **validation CP2**, mengikuti manifest bersih; bukan foto yang masuk training refiner.

**Yang terlihat:** coarse dapat hanya menemukan sebagian kecil luka, tetapi probabilitas≥0,3 masih menghasilkan ROI yang membantu refiner. Pada kasus lain, refinement justru memperburuk segmentasi yang sudah baik. **Artinya:** refinement memberi kesempatan tambahan, bukan jaminan setiap foto membaik. **Perlu diingat:** dua kasus ekstrem dipilih posthoc dengan aturan deterministik, bukan estimasi frekuensi keberhasilan atau dasar pemilihan checkpoint.

Proposal memakai threshold≥0,3, komponen8-connected minimal4px, maksimal5ROI, margin0,5 dan minimum sisi0,25. Crop di-resize224×224. Refiner menghasilkan probability crop, di-resize kembali ke ROI; overlap mengambil max **antarcrop**, bukan max terhadap coarse. Di luar ROI tetap coarse; final threshold≥0,5. Tanpa proposal, output sama dengan coarse. Kedua kasus terpilih mempunyai tepat satu crop, dan semuanya ditampilkan. GT validation tidak masuk fungsi proposal/refinement.

## Slide 14 — perbandingan yang dapat dipertanggungjawabkan

Cohort tetap **671 train /118 validation /261 source-test**, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Ini pengelompokan decoded-RGB identik, **bukan patient-level split**. Scoreboard mempertahankan seluruh15 kandidat awal, menambah B1+A6+H2, L3+A6+H2 dan Cascade. CP1, Cascade Ganta asli, serta eksternal tidak masuk ranking ini. Konfirmasi tiga seed tetap merupakan [analisis terpisah](cp2-confirmation.md); grafik final bukan pengganti bukti multiseed.

**Yang terlihat:** A6/H2 menjanjikan pada sebagian skor, kombinasi tidak otomatis lebih baik, dan Cascade meningkatkan overall/small/boundary tanpa menambah validation miss/empty FP. **Artinya:** dampak pada error harus diperiksa bersama mean Dice. **Perlu diingat:** aturan penelitian antar-fase memang berbeda; tabel tidak menerapkan satu guard baru ke seluruh eksperimen.

| Keluarga | Guard historis yang dipertahankan |
|---|---|
| P | B0 diagnostik; B1 working baseline. Tidak direseleksi oleh renderer. |
| L | Overall meningkat, small tidak turun, total miss/GT-empty FP images tidak bertambah. Boundary bukan guard seleksi fase L. |
| A/H | Overall meningkat; small/boundary tidak turun; miss/empty FP tidak bertambah; budget parameter≤1,01×. |
| N | Guard A/H ditambah recall positif, empty-FP pixels dan target pengurangan FP sesuai keputusan asli. |
| C A1H2 | Guard asli terhadap B1 plus tuntutan melampaui standalone relevan. |
| C dua kombinasi terbaru | Guard asli terhadap B1 plus melampaui A6/H2; resep L3 juga dibandingkan standalone L3. |
| Cascade | Lima guard validation terhadap coarse CPU: overall meningkat; small/boundary tidak turun; miss dan empty-FP images tidak bertambah. **Tidak memakai budget single-model1,01×.** |

Cascade berisi **129.796.496 parameter aktif** (dua×64.898.248), membutuhkan **1+jumlah proposal forward**. Tambahan kapasitas/komputasi menjadi confound yang dilaporkan, bukan dianggap setara model tunggal. Timing CPU shared pada laporan final bukan benchmark latency deployment/paper.

Baseline B1 pada scoreboard mempertahankan cache historis GPU (Image Dice84,2765%). Pembanding Cascade berpasangan memakai coarse CPU yang sama selama validation training/final (84,2773%). Selisih kecil floating-point/threshold ini direkam, bukan disembunyikan atau dipakai memilih ulang epoch. Galeri memakai coarse CPU tersebut; A6/H2 mempertahankan cache GPU aktualnya.

## Slide 15 — generalisasi eksternal dan error

| Split | Model | Image Dice | Data Dice | Small Dice | Boundary IoU | GT+ TP=0 | GT-empty FP |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation118 | Coarse B1 CPU | 84,28% | 91,51% | 77,29% | 53,93% | 0/111 | 4/7 |
| Validation118 | Cascade CP2 | 87,37% | 92,88% | 84,71% | 59,04% | 0/111 | 4/7 |
| Source-test261 | Coarse B1 CPU | 83,29% | 90,77% | 76,21% | 53,54% | 4/245 | 9/16 |
| Source-test261 | Cascade CP2 | 85,86% | 92,78% | 80,64% | 57,53% | 3/245 | 8/16 |
| External558 | Coarse B1 CPU | 39,33% | 54,69% | 9,52% | 10,85% | 143/558 | N/A, n=0 |
| External558 | Cascade CP2 | 42,27% | 58,99% | 12,33% | 12,68% | 153/558 | N/A, n=0 |

**Yang terlihat:** external Image Dice bertambah2,94pp tetapi total miss naik10foto. Ada9 miss coarse yang pulih,19 miss baru, dan134 miss pada keduanya. Dice meningkat pada253foto, sama pada134foto, dan menurun pada171foto. **Artinya:** peningkatan rerata dapat terjadi bersamaan dengan memburuknya jenis error penting. Transisi miss dan distribusi delta dihitung pada seluruh558 gambar yang sama. TP=0 tidak selalu berarti mask prediksi kosong: prediksi dapat berada seluruhnya di luar GT.

**Perlu diingat:** CO2Wounds-V2 mengubah sumber sekaligus patologi (luka kronis pasien kusta); evaluasi memakai koordinat native, berbeda dari source224. Semua558 GT positif, sehingga tidak ada pengujian GT-empty eksternal. Target sudah pernah diperiksa dan bukan untouched confirmatory test. Hasil tidak dipakai memilih model atau threshold. Figur hanya agregat; tidak ada foto pasien, mask, overlay atau derived clinical panels CO2 yang diterbitkan. Registry26checkpoint historis tidak diubah atau digabung seolah Cascade bagian dari run lama.

## Slide 16 — empat contoh, aturan seleksi tetap

Tie dipecahkan menurut full sample ID; setiap kasus harus berbeda. Urutannya:

1. **Small Cascade gain:** delta Dice Cascade−coarse terbesar di antara0<GT≤1% kanvas224.
2. **Standalone gain:** max(A6,H2)−B1 historis terbesar pada GT positif yang belum dipilih.
3. **GT-empty FP:** jumlah maksimum FP pixels di antara B1/A6/H2/Cascade pada GT kosong.
4. **Cascade regression:** delta Cascade−coarse paling negatif pada GT positif yang tersisa.

Kasus standalone adalah `train/5492fe099502ce4c418e8d6686fac07d_0.png` (B1 sekitar58,08%, A6 91,41%, H2 38,14%); kasus GT kosong adalah `train/3da2dcfdb655b741e1b97119f964e460_0.png`. **Tidak tersedia** contoh total miss B1/Cascade pada validation: keduanya0/111. Tidak mencari sampel dari split lain untuk mengisi kategori yang tidak ada. Detail GT, FP/FN, cache hash dan angka presisi penuh tersedia pada companion metadata.

**Yang terlihat:** A6 dapat membantu ketika H2 memburuk; GT kosong tetap memicu FP; Cascade tidak selalu lebih baik. **Artinya:** setiap tambahan modul mempunyai pola error, tidak sekadar satu skor agregat. **Perlu diingat:** galeri ekstrem posthoc tidak menyatakan prevalensi atau signifikansi; anotasi GT kosong bukan diagnosis pasien sehat dan tidak diubah setelah melihat prediksi.

Overlay konsisten: **hijau TP / jingga FP / biru FN**. Full-image companion selalu tersedia; zoom mempertahankan union RGB-content, GT dan seluruh prediksi sehingga tidak menyembunyikan FP. Geometri/aspect ratio asli dipertahankan. Skor seluruh versi tetap dihitung pada full224×224.

## Reproduksi, provenance, dan publikasi

Renderer: [`scripts/build_cp2_final_presentation.py`](../scripts/build_cp2_final_presentation.py), [config](../configs/cp2/presentation_final.json). Helper baru memakai evaluator, pemeriksa cache, fungsi proposal dan stitching asli; tidak mengedit kode training frozen. Tiap PNG/SVG memiliki companion `<filename>.json`: hash figure/kode/config/manifest/CSV/checkpoint, epoch, sample IDs, aturan kasus, GT/prediction provenance, metrik dan caveat. [QA final](cp2-presentation-final-validation.json) mencatat pengujian, actual visual inspection dan preservation.

```bash
# Gunakan run-root baru untuk reproduksi. Runner merekam inventory sebelum render.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-xfuseg/bin/python -u scripts/build_cp2_final_presentation.py \
  --config configs/cp2/presentation_final.json \
  --run-root outputs/presentation/cp2_final/20261121T043840Z
```

Runner menyimpan `protected_before.json` pada run-root baru dan tidak mengganti inventory yang sudah ada. Produksi paket ini menggunakan inventory yang dibuat **sebelum perubahan kode**. Completed checkpoints/data/cache yang tercantum pada config harus tersedia lokal; runner tidak mengunduh data atau weights. Setiap render membuat direktori attempt bertimestamp baru; file lama tidak ditimpa. Untuk koreksi tipografi saja, `--reuse-verified <attempt-yang-sudah-lengkap>` memuat payload lokal yang hash-nya cocok dan memeriksa ulang hash source manifest/CSV. Final preservation tetap memverifikasi file historis dan cache; bukan sekadar keberhasilan plotting.

Foto source/GT: [Chronic Wound / AZH patches, uwm-bigdata revision42a272d](https://github.com/uwm-bigdata/wound-segmentation/tree/42a272dfe0679f20675e826385925cb7562934b6). Publikasi panel analisis source terpilih telah diinstruksikan pengguna; ini **tidak menciptakan lisensi dataset baru** atau mengubah audit lisensi lama. Full dataset, cache, tensor crop dan checkpoint tetap lokal. Tidak ada klaim patient independence, clinical validation, novelty global, atau superioritas statistik.
