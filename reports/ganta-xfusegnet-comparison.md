# x-FUSegNet Ganta vs Orie · audit kode dan hasil

**Kesimpulan: keluarga metode sama, tetapi run dan protokol evaluasi berbeda.** Ganta memakai model EfficientNet-B7 + P-scSE dan ensemble lima fold. Implementasinya mengadaptasi kode penulis ke dependency baru. Run Orie memakai snapshot dependency/kode lama penulis, pool training yang berbeda, dan evaluasi berbeda. Angka keduanya tidak boleh dijadikan peringkat langsung.

Audit ini membaca folder Ganta tanpa mengubahnya. Tidak menjalankan script partner, training, inference, atau deserialisasi checkpoint/pickle. Metrik ensemble dihitung ulang dari **prediksi PNG yang sudah ada**, bukan membuat prediksi baru.

## Artefak yang diaudit dan dipublikasikan

- [Script Python asli](../notebook/x-FUSegNet%20Reproduce.py), disalin byte-identik. SHA-256: `a2b865920432360a4deadafaf8361ec28f19246fc1754650de40f7869626e836`.
- [Folder hasil asli](../output_fuseg_2021/results/): seluruh 15 file hasil PNG/CSV/JSON di luar cache `.ipynb_checkpoints` disalin byte-identik, termasuk plot run lama.
- Run hasil lengkap yang dianalisis: **`xFUSegNet_efficientnet-b7_2026-10-29_22-39-16`**.
- [Audit mesin / hash sumber](tables/ganta-xfusegnet-audit.json), [hitungan ulang per gambar](tables/ganta-xfusegnet-audit-per-image.csv), dan [script audit](../scripts/audit_ganta_xfusegnet.py).
- Referensi Orie: commit `90970d5ad7496c38fb06ce340caa0c828498859e`, [laporan x-FUSegNet](https://github.com/maulnite/foot-ulcer-segmentation/blob/main/reports/cp1-xfusegnet-01.md), dan [protokol Orie](https://github.com/maulnite/foot-ulcer-segmentation/blob/main/docs/xfusegnet.md).

File asli dan label metrik partner dipertahankan. Kata **“test”** dalam nama CSV/source partner merujuk **provided validation**, sebagaimana dijelaskan di bawah. Script dan hasil analisis berada di branch ini; [lima checkpoint terbaik tersedia sebagai GitHub Release terpisah](https://github.com/maulnite/foot-ulcer-segmentation/releases/tag/cp1-xfusegnet-ganta-weights-v1) dengan [petunjuk unduh](../docs/model-checkpoints.md). Raw dataset, seluruh cache prediksi, dan notebook partner yang belum selesai dieksekusi tidak dipublikasikan.

## 1 · Perbedaan yang paling menentukan: data dan evaluasi

| Komponen | Ganta | Orie |
| --- | --- | --- |
| Raw data yang tersedia | 810 train + 200 validation berlabel + 200 test tanpa mask | Sama; seluruh filename dan SHA-256 foto/mask yang tersedia cocok |
| Pool untuk lima fold | **810 provided train** | **1.010 berlabel**, gabungan provided train + validation |
| Ukuran fold | **648 train / 162 validation**, diturunkan dari konfigurasi KFold pada 810 gambar; assignment aktual belum tersimpan secara terverifikasi | **808 train / 202 validation**, assignment aktual dipersistenkan |
| Yang dinamai “test” oleh script | **200 provided validation dengan GT**, ditahan dari pool 810 | **200 challenge test tanpa GT**, terpisah dari pool 1.010 |
| Skor yang tersedia | Ensemble average/voting pada held-out provided validation 200 gambar | OOF internal 1.010 gambar, satu model yang menahan masing-masing gambar; test ensemble hanya prediksi |
| Skor resmi test challenge | Belum diverifikasi | Belum diverifikasi |

Source Ganta [baris 686–702](../notebook/x-FUSegNet%20Reproduce.py#L686) mengatur `TEST_DIR = os.path.join(DATASET_ROOT, "validation")`. Sementara itu folder `data/raw/fuseg_2021/test` mempunyai 200 gambar **tanpa mask**. Jadi Dice 92,64% Ganta **bukan** skor pada 200 test resmi challenge dan tidak dapat dibandingkan langsung dengan angka test paper **89,23%** ([paper §3.2.4 dan evaluasi FUSeg](https://arxiv.org/abs/2305.02961v2)).

Audit memeriksa filename serta SHA-256 semua foto pada train/validation/test dan mask train/validation terhadap manifest Orie dan raw file Orie. Dengan demikian perbedaan utama di sini berasal dari **cara memakai raw dataset yang sama**, bukan sekadar nama dataset berbeda. Resolusi native semuanya **512×512**. Mask berlabel mengandung nilai 0, 32, 255; training kedua source tetap memakai grayscale/255, termasuk target fractional yang ada. Tidak ada resize aktif tambahan dalam definisi augmentasi yang dibandingkan.

Seluruh 200 gambar evaluasi Ganta termasuk pool berlabel Orie, dan tidak beririsan filename dengan 200 test resmi. Setiap gambar tersebut masuk training empat model Orie dan validation satu model. **Menilai ensemble lima model Orie pada 200 gambar ini akan mencampurkan prediksi dari model yang pernah melatihnya**; itu bukan perbandingan held-out yang setara. Audit ini tidak menjalankan evaluasi tersebut.

## 2 · Bagian metode yang sama

- EfficientNet-B7 pretrained ImageNet, satu kanal segmentasi, sigmoid, decoder P-scSE termasuk perilaku stage terakhir tanpa skip.
- Lima model dilatih melalui `KFold(n_splits=5, shuffle=False)` setelah shuffle daftar gambar.
- Adam, batch 2, empat data-loader workers, LR `1e-4`, weight decay `1e-5`, maksimum 200 epoch/fold, early stopping patience 30.
- ReduceLROnPlateau factor 0.1, patience 10, min LR `1e-5`.
- Dice + Focal dan alur sigmoid probability ke focal-with-logits sumber dipertahankan; ini termasuk quirks sumber yang dicatat pada [audit Orie](https://github.com/maulnite/foot-ulcer-segmentation/blob/main/docs/reference-audit.md).
- Kebijakan checkpoint `if validation loss improves ... elif IoU improves`, bukan hanya memilih puncak Dice.
- Average ensemble: rata-rata lima probabilitas, kemudian threshold `>=0.5`. Ganta juga menyimpan alternatif **majority voting**; run Orie melaporkan average sesuai protokol paper.

Pemeriksaan AST, mengabaikan komentar/docstring, menunjukkan definisi **Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing, SCSEModule, SCSEModule0, Attention, DecoderBlock, dan CenterBlock** sama dengan snapshot penulis Orie revisi `60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc`.

Ini bukti kemiripan definisi kode, bukan bukti identitas bobot, tensor intermediate, RNG, atau hasil numerik lintas environment.

## 3 · Implementasi, seed, dan library berbeda

| Komponen | Ganta | Orie |
| --- | --- | --- |
| PyTorch tercatat | **2.7.1+cu118** | **1.11.0+cu113** |
| Segmentation Models PyTorch | **0.6.0 resmi**, decoder P-scSE diganti dengan salinan/adaptasi lokal | **0.3.0 fork penulis**, snapshot utuh |
| Albumentations | **2.0.8** | **1.3.0** |
| Input order sebelum shuffle | `sorted(...)`, difilter ekstensi gambar | Urutan direktori lokal ditangkap dan dipersistenkan |
| Python shuffle seed | **200** | **42**, sesuai source |
| Global torch/NumPy seed | Tidak ditetapkan di script yang dipilih; state RNG run tidak dicatat | 42 sekali sebelum loop; state berlanjut antar-fold dan disimpan untuk resume |
| P-scSE integration | Buat SMP Unet standar, kemudian ganti decoder dan inisialisasi decoder lokal | Bangun model langsung dari fork penulis; loop asli dijalankan dengan hook pencatatan |
| Catatan membership | Source membaca daftar file langsung, tanpa membaca manifest | Manifest aktual dipakai dan diverifikasi terhadap run |

Walaupun AST fungsi augmentasi sama, dependency, default library, dan RNG yang berbeda berarti distribusi/hasil numerik transformasi **tidak dapat diasumsikan identik**. Audit tidak memasang dependency baru atau menjalankan training untuk menguji kesetaraan runtime.

Adaptasi lokal yang teridentifikasi:

- `to_tensor` menambahkan guard untuk mask 2D sebelum HWC→CHW.
- `UnetDecoder.forward` menerima kedua bentuk pemanggilan: `decoder(*features)` dan `decoder(features)`.
- `Conv2dReLU` menghilangkan alternatif InPlaceABN; konfigurasi aktif partner tetap `use_batchnorm=True`, memakai BatchNorm biasa.
- Pembuatan model, epoch runner/loss compatibility layer, dan scheduler disesuaikan dengan SMP/PyTorch baru. Smoke test di script membuat model dan memakai RNG sebelum training; state RNG tidak dipulihkan secara eksplisit.

Kesimpulan yang tepat adalah **implementasi adaptasi dari metode yang sama**, bukan kode dan protokol yang persis identik dengan run Orie.

## 4 · Hasil partner yang berhasil diverifikasi

Perhitungan ulang memakai 200 GT pada provided validation serta 200 PNG prediksi untuk masing-masing average/voting. Semua enam kolom metrik pada kedua baris ensemble **persis cocok** dengan CSV asli (selisih maksimum 0 percentage points pada environment audit).

| Evaluasi | Cakupan | Data Dice (%) | Image Dice (%) | Data IoU (%) |
| --- | --- | ---: | ---: | ---: |
| Ganta · Average | Ensemble, 200 provided validation | **92.644239** | **86.411004** | **86.296476** |
| Ganta · Voting | Ensemble, 200 provided validation | **92.639994** | **86.399783** | **86.289110** |
| Orie · OOF | Satu model penahan/gambar, 1.010 internal OOF | **92.719254** | **88.032267** | **86.426742** |
| Orie · Test ensemble | 200 challenge test tanpa GT | Tidak tersedia | Tidak tersedia | Tidak tersedia |

Tabel menunjukkan **jenis hasil yang tersedia**, bukan peringkat metode. Tidak ada selisih performa yang diatribusikan pada arsitektur/augmentasi karena cohort, ukuran pool, seed, library, dan evaluasi berbeda. PFOM partner tidak tersedia pada hasil yang diaudit; audit ini tidak menambah perhitungan PFOM.

Untuk average: TP **594.688**, FP **48.242**, FN **46.192**. Untuk voting: TP **594.708**, FP **48.324**, FN **46.172**. Ada **5 GT kosong** pada 200 gambar evaluasi.

Definisi sesuai source partner:

```text
GT            = grayscale / 255 >= 0.5
prediksi PNG  = grayscale > 127
data Dice     = (2ΣTP + eps) / (2ΣTP + ΣFP + ΣFN + eps)
image Dice    = mean((2TP_i + eps) / (2TP_i + FP_i + FN_i + eps))
eps           = 1e-7
```

Seluruh kanvas mask dihitung, termasuk padding. Kedua mask kosong memberi skor 1 menurut rumus smoothing tersebut. Statistik audit berasal dari cache yang ada; **tidak ada model dijalankan ulang**. Lima baris single-model pada CSV asli dipublikasikan sebagai **reported**, belum dihitung ulang karena raw prediction setiap model hanya dipakai di memori oleh source, tidak tersedia dalam folder PNG ensemble yang diaudit. Average probabilitas/voting dari lima raw maps juga belum diverifikasi secara independen; yang diverifikasi adalah konsistensi **PNG akhir ↔ GT ↔ metrik CSV**.

## 5 · Batas bukti dan catatan untuk presentasi

1. **Duplikasi tetap ada.** Tujuh gambar provided validation memiliki RGB identik dengan gambar provided train. Filename held-out saja tidak menjamin bebas overlap isi gambar atau independensi pasien. Tidak ada duplikat dihapus atau split diubah. Run Orie juga mempertahankan serta mendokumentasikan duplikasi antar-fold; tidak ada klaim independensi pasien pada keduanya.
2. **Ada manifest di folder Ganta, tetapi bukan bukti assignment run ini.** `data/splits/fuseg_2021_xfusegnet/manifest.json` byte-identik dengan manifest Orie: pool 1.010, seed 42, fingerprint `93aac316e596dcc34b8d55174ea1b10a564cb07f7398e4616c78409552076e2a`. Script Ganta yang diaudit memakai 810 gambar, seed 200, dan tidak membaca manifest itu. Manifest tersebut tidak dipakai sebagai pengganti daftar fold aktual Ganta; fold tidak direkonstruksi dari daftar file sekarang.
3. **Source→run belum terikat penuh.** Run-info tidak memuat hash script, run-linked fold IDs, full dependency lock, atau state RNG. Dua notebook sumber tersimpan masing-masing dengan 10 dari 18 code cell dieksekusi dan `KeyboardInterrupt` pada cell training. Itu tidak membatalkan kecocokan CSV dengan PNG akhir, tetapi notebook tersimpan tidak merupakan rekaman eksekusi lengkap dari seluruh run hasil akhir. Lima checkpoint akhir tidak dimuat/dihash dalam audit kode/metrik ini; ukuran dan mtime/inode diperiksa tetap. Publikasi bobot berikutnya menghitung SHA-256 seluruh checkpoint dan mencocokkannya dengan digest aset GitHub; bukti tersebut tersedia dalam manifest dan checksum Release, tanpa mengubah keterbatasan source→run di atas.
4. **Plot lama dan label kurva.** Folder `results/plots/` berisi lima PNG dari run lebih lama dan lima PNG dengan ID run akhir; semuanya disalin tanpa diubah. Pada source plot per-fold, panel berjudul “Dice curve” menggunakan array IoU lagi. Kurva gabungan [`..._val_curves.png`](../output_fuseg_2021/results/xFUSegNet_efficientnet-b7_2026-10-29_22-39-16_val_curves.png) memakai `train_dice/val_dice` dan lebih tepat untuk presentasi. Jangan menyebut PNG lama sebagai hasil run akhir atau membaca panel IoU yang salah label sebagai Dice.
5. **Tidak ada klaim melampaui paper atau mengungguli run lain.** Gunakan label presentasi “x-FUSegNet adaptasi library baru; ensemble pada provided validation 200 gambar”. Perbandingan adil memerlukan protokol dan data evaluasi sama yang benar-benar ditahan dari semua model terkait; audit ini tidak memulai eksperimen baru tersebut.

## 6 · Menjalankan ulang audit, tanpa training

Gunakan environment analisis proyek yang sudah dipin. Perintah berikut hanya membaca dua checkout dan cache hasil yang sudah ada:

```bash
.venv-analysis/bin/python scripts/audit_ganta_xfusegnet.py \
  --ganta-root /path/to/original/fuseg/ganta \
  --orie-root /path/to/original/fuseg/orie \
  --output-dir /path/to/separate/audit-output
```

Raw dataset, snapshot source penulis, manifest Orie aktual, metadata run, serta PNG prediksi partner diperlukan untuk audit ulang lengkap. Checkout GitHub saja menyediakan script/hasil publik dan catatan audit; tidak menyertakan data atau model lokal. Script partner sendiri adalah entrypoint training/inference dan tidak diimpor oleh script audit.

Audit SHA-256 dan stat memastikan 16 file sumber yang dipublikasikan serta lima checkpoint tidak berubah selama audit. File source dan hasil partner disalin identik, sedangkan laporan, CSV hitungan ulang, dan script audit adalah tambahan Orie yang diberi label terpisah.

## Atribusi

Kode acuan: Mrinal Kanti Dhar, Taiyu Zhang, Yash Patel, Sandeep Gopalakrishnan, Zeyun Yu, [FUSegNet](https://arxiv.org/abs/2305.02961v2), [repository penulis](https://github.com/mrinal054/FUSegNet), dan Segmentation Models PyTorch oleh Pavel Iakubovskii. [MIT notice upstream](../notebook/UPSTREAM_LICENSE.txt) disertakan untuk salinan kode. Adaptasi partner dipublikasikan pada branch **`ganta`** dengan atribusi sumber tetap.

Foto/GT pada figure berasal dari [dataset publik Foot Ulcer Segmentation Challenge UWM](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge). Figure dan angka asli partner dipertahankan; raw dataset lengkap tidak diunggah. Hasil partner tidak digabung ke skor, notebook, atau kesimpulan eksperimen Orie pada `main`.
