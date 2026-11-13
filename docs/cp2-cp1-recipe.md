# CP2 — Uji terarah resep training CP1

## Mengapa eksperimen ini diperlukan?

CP2 memperbaiki integritas split dan memperjelas prosedur penelitian, tetapi
belum membuktikan bahwa paket optimisasinya lebih efektif dari CP1. Eksperimen
ini menguji kandidat terbatas dengan **kebijakan training CP1**, mempertahankan
data bersih dan objective sumber B0 (termasuk focal input mismatch). Ini tidak mengulang seluruh fase dan tidak
memakai trained checkpoint CP1 sebagai inisialisasi.

Pada manifest yang sudah diverifikasi, 103/118 gambar validation CP2 pernah
masuk training CP1. Karena itu checkpoint CP1 lama bukan pembanding validation
yang sah pada split ini. Ketiga arm dilatih dari shared initial weights fase P.

## Preregistrasi dan hipotesis

| Arm | Arsitektur | Main objective | Pertanyaan |
|---|---|---|---|
| R0 | P-scSE asli | B0 sumber | Bagaimana paket kebijakan CP1 berperilaku pada split CP2? |
| RH2 | P-scSE + auxiliary boundary head | B0 sumber + 0,1×B0 auxiliary | Apakah supervisi batas membantu tanpa menambah empty FP? |
| RA6 | CBAM pada decoder stage 2–3 | B0 sumber | Apakah pembanding attention mapan mempertahankan manfaatnya? |

H2 dipilih karena meningkatkan overall/small/boundary validation, walaupun gagal
guard empty FP. A6 dipilih karena overall validation tertinggi pada fase A,
juga gagal guard empty FP. Pemilihan tidak didasarkan pada ranking test.
Tidak ada kombinasi H2+A6, perubahan stage/gain, loss baru, multiseed, atau
evaluasi eksternal dalam matriks ini.

Kandidat berasal dari validation yang sudah digunakan sebelumnya. Seluruh
eksperimen ini **eksploratif/adaptif**, bukan konfirmasi independen.

## Apa yang mengikuti CP1?

- Adam LR 1e-4, weight decay 1e-5, batch 2, 4 workers, maksimum 200 epoch.
- Augmentasi, preprocessing, dan FP32/TF32 historis; AMP tetap mati.
- Checkpoint `if validation loss improves ... elif batch-mean IoU improves`.
  Perilaku stale IoU tracker sumber sengaja dipertahankan dan diuji.
- ReduceLROnPlateau memantau main validation loss, mode min, factor 0,1,
  patience 10, min LR 1e-5; threshold default CP1 1e-4 relatif.
- Early stopping setelah 30 epoch tanpa save menurut kebijakan sumber.
- Seed 42 sekali; evolving RNG dan DataLoader shuffle sumber, tanpa reseed per
  epoch. Wrapper mencatat indices tanpa mengubah konsumsi RNG. State RNG
  disimpan setiap epoch untuk resume.

Source IoU menggunakan threshold **>0,5**, smoothing 1e-7 dan batch averaging.
Pelaporan akhir memakai evaluator CP2 dengan threshold **≥0,5**, ordinary
confusion counts, both-empty=1 dan exactly-one-empty=0. Kedua fungsi berbeda
dan tidak disajikan sebagai satu metrik yang sama.

Pada RH2, checkpoint/scheduler memakai **main regional validation loss**, tanpa
auxiliary term. Dengan demikian ketiga arm memantau task mask utama yang sama;
training RH2 tetap menggunakan auxiliary term. Ini adaptasi eksplisit karena
CP1 tidak mempunyai head auxiliary.

## Apa yang tetap mengikuti kontrol CP2?

- Manifest 671 train / 118 validation / 261 test, hash
  `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`.
- Tidak mengembalikan duplikat/konflik yang dikecualikan. Raw data tidak dihapus.
- Sesuai pilihan eksplisit pengguna, objective sumber B0: Dice(sigmoid(logits)) + focal-with-logits(sigmoid(logits)). Input mismatch sengaja dipertahankan; ini bukan loss logits yang benar. Pada RH2, auxiliary memakai primitive B0 yang sama dan bobot 0,1.
- Semua pairable source tensors sama persis dengan initial weights fase P;
  head/gain/CBAM sama dengan fase H/A. Bukan warm-start bobot hasil training.
- Seluruh arm dimulai dari common post-construction CPU/CUDA/Python/NumPy RNG
  state. Lazy reference imports diselesaikan sebelum seed bersama karena import pertama dapat mengonsumsi RNG. Actual train order dan augmented bytes diverifikasi pada common epochs.
- Forward mask utama tidak menerima GT; target boundary berasal dari mask
  sesudah augmentasi geometrik, morphological gradient 3×3, exterior zero.
- Evaluator dan guard historis terhadap **R0 baru**: overall Image Dice lebih
  tinggi, small Dice/Boundary IoU tidak lebih rendah, tidak ada tambahan total
  foreground misses atau empty-FP images, active parameters ≤1,01×R0.
  Kandidat eligible tertinggi overall dipilih; tie RA6 sebelum RH2; bila tidak
  ada kandidat eligible, pertahankan R0. Guard ini aturan riset, bukan klinis.

Semua validation dan keputusan seleksi disimpan sebelum test inference baru.
Source-test sudah digunakan sejak CP1; hasilnya deskriptif, bukan untouched
confirmatory test. Tidak ada threshold tuning dari test.

## Batas interpretasi

Perbandingan R0 dengan B0 historis mengubah **paket** checkpoint, scheduler,
early stopping dan RNG sekaligus; tidak mengisolasi kontribusi setiap faktor.
Perbandingan RH2/RA6 terhadap R0 menguji arsitektur di dalam paket yang sama.
Jumlah epoch dapat berbeda karena early stopping/checkpoint validation.
Gain A6 tetap 3 seperti eksperimen sebelumnya; menyamakan gain maksimum tidak
menyamakan distribusi aktivasi dengan P-scSE. Tidak ada klaim novelty CBAM.

Satu seed, hanya tujuh GT kosong validation, identitas pasien tidak tersedia,
serta kemungkinan relasi crop/pasien tetap menjadi keterbatasan. Ini
duplicate-RGB-group split, bukan patient-level split.

## Eksekusi dan artefak

```bash
.venv-training/bin/python scripts/run_cp2_cp1_recipe.py \
  --root outputs/cp2/cp1_recipe/<new-run> --stage prepare
.venv-training/bin/python scripts/run_cp2_cp1_recipe.py \
  --root outputs/cp2/cp1_recipe/<new-run> --stage preflight --device cuda:0
.venv-training/bin/python scripts/run_cp2_cp1_recipe.py \
  --root outputs/cp2/cp1_recipe/<new-run> --stage train --device cuda:0
```

`prepare` membekukan protokol, source hashes, snapshot kode dan inventori
preservasi. Preflight menjalankan tests, dependency checks dan tiga smoke
real-data (forward/backward/optimizer, empty/positive masks, validation,
checkpoint reload). Satu trainer menjalankan R0 → RH2 → RA6 berurutan.

`--resume` hanya untuk run terinterupsi dengan konfigurasi/kode/environment yang
sama; stage completed diverifikasi dan tidak dilatih ulang. Checkpoint, raw
foto dan probability maps tetap lokal. Run baru berada pada direktori
append-only terpisah; checkpoint `last` hanya diperbarui dalam run yang aktif.

Lihat `status.json`, `history.json`, `train_workflow.json`, dan console log di
root run untuk progress. Laporan final dan grafik dibuat **setelah** ketiga
arm benar-benar selesai dan validasi independen lulus. Grafik memerlukan
review visual langsung sebelum hasil dinyatakan final.

Jika laporan publik sudah berasal dari run lain, reproduksi berikutnya menyimpan
laporan pada `<new-run>/presentation/reports/` agar publikasi sebelumnya tetap
utuh. Pemanggilan ulang renderer pada completed run yang sama memverifikasi
metrics/statuses lalu memakai cache. Analisis berpasangan hasil final tersedia
melalui `scripts/analyze_cp2_cp1_recipe.py`; tidak menjalankan model atau training.

[Hasil final dan review ilmiah](../reports/cp2-cp1-recipe-review.md).

## Audit preflight awal

Protokol B1 sementara pada commit `5747b50` hanya menjalankan engineering smoke, bukan full training. Preflight menangkap ketidaksamaan urutan/input pada arm pertama akibat lazy import sebelum common seed. Gate memblokir training penuh. Catatan lama dipertahankan; regression test baru memeriksa first-import RNG, dan permintaan pengguna mengubah protokol efektif menjadi B0 sumber pada run terpisah.
