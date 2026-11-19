# CP1 · Ablasi cabang attention

Eksperimen ini menguji apakah dua cabang P-scSE memberi manfaat yang saling melengkapi dibanding memakai salah satu cabang saja. Baseline penuh menggunakan hasil `cp1-baseline-01` yang sudah selesai; dua varian dilatih ulang dari pretrained encoder dengan protokol yang sama.

## Alasan eksperimen

Pada [analisis baseline](../reports/cp1-analysis-01.md), mean image Dice kelompok luka terkecil C1 adalah **73.22%**, sedangkan C9 **97.36%**. Empat dari 261 gambar dengan GT positif terlewat seluruhnya, dan tujuh dari 17 gambar dengan GT kosong menghasilkan false positive. [Mean PFOM test 0.7782](../reports/cp1-pfom-01.md) juga memberi alasan untuk memeriksa kualitas batas, selain luas overlap.

Temuan tersebut menunjukkan bagian hasil yang perlu diperiksa. Temuan itu **belum membuktikan attention sebagai penyebabnya**. Additive menggabungkan respons spatial/channel melalui penjumlahan; maxout mengambil maksimum elemen. Menghapus satu cabang menguji kontribusinya terhadap Dice, batas luka, luka kecil, dan false positive. Hasil yang membaik maupun memburuk sama-sama dilaporkan.

## Varian dan kontrol

| Model | Decoder stage 0–3 | Stage 4 | Training |
| :--- | :--- | :--- | :--- |
| P-scSE penuh | `conv1(maxout(x) + additive(x))` | Shorted asli penulis | Baseline selesai, dipakai ulang |
| Additive-only | `conv1(additive(x))` | Shorted asli penulis | Run baru |
| Maxout-only | `conv1(maxout(x))` | Shorted asli penulis | Run baru |

`x` pada tabel adalah fitur sesudah upsampling dan concat skip. Pada stage terakhir, implementasi asli memakai `conv1(x + additive(x))`. Stage ini tetap identik di semua model. Posisi attention dan satu konvolusi aktif per stage juga dipertahankan; opsi scSE bawaan sumber mengubah hal lain sehingga tidak digunakan untuk eksperimen ini.

Seluruh model dibangun dengan constructor P-scSE lengkap terlebih dahulu. Cabang yang tidak dipakai tetap terdaftar agar konsumsi RNG dan bobot awal sama; adapter hanya mengubah forward stage 0–3. Rekonstruksi baseline dan dua varian dalam proses Python terpisah memiliki SHA256 state awal identik:

```text
fd3fe8b007a897b5c3a977b8bbafec662fed4bdf6489d8a51a18376a4bb7e031
```

Ini pemeriksaan rekonstruksi inisialisasi baseline; baseline asli tidak menyimpan checkpoint epoch 0. Semua model memiliki **70,968,939 parameter terdaftar**. Parameter pada modul yang dipanggil saat forward berjumlah **64,898,248** untuk baseline dan **64,781,798** untuk setiap varian. Ablasi juga mengubah kapasitas aktif dan komputasi; hasilnya tidak mengisolasi operator agregasi pada jumlah parameter aktif identik.

Kontrol yang dikunci dalam [protokol JSON](../configs/ablation/cp1-attention-01.json):

- Chronic Wound yang sama, **707 train / 124 validation / 278 test**, kanvas 224 × 224, manifest split identik.
- Seed **42**, EfficientNet-B7 ImageNet, cache pretrained identik, augmentasi dan seluruh versi dependency training identik.
- Batch 2, Adam dan scheduler asli, maksimum 200 epoch, early stopping 30, kebijakan checkpoint loss/IoU penulis.
- Loss asli beserta perilaku probability-input focal dipertahankan. Tidak memperbaiki loss atau kebijakan checkpoint dalam ablasi ini.
- Threshold inference `>= 0.5`, evaluator offline HARD_LINE yang sama; skor inline sumber tetap terpisah.
- [PFOM](pfom.md) dengan parameter Canny dan kebijakan edge kosong 1/0 yang sama; proxy sentinel 2 dilaporkan terpisah.

Split fingerprint:

```text
7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925
```

## Endpoint dan pembacaan hasil

Endpoint utama adalah **image-based Dice** akhir. Checkpoint dipilih memakai validation dan kebijakan sumber. Test digunakan untuk mendeskripsikan **seluruh** varian yang telah ditetapkan, tanpa penyesuaian threshold atau pemilihan model ulang dari test.

Endpoint tambahan: data-based Dice, IoU, precision/recall, PFOM, Dice/PFOM C1, gambar GT positif dengan TP=0, false positive pada GT kosong, waktu training, peak CUDA memory, dan jumlah parameter forward. Enam contoh visual ditetapkan sebelum training: dua worst-positive, dua median-positive, dan dua empty-FP dari galeri baseline. Gambar tidak dipilih ulang berdasarkan hasil ablasi.

Baseline test telah dianalisis sebelum hipotesis ini dibuat. Perbandingan ini bersifat eksploratif dengan **satu seed**. Variasi antar-training belum diukur; perubahan kecil belum mendukung klaim keunggulan yang stabil. Ketidakpastian split paper, duplikasi RGB antar-split, dan independensi pasien yang belum terverifikasi tetap berlaku. C1–C9 menggunakan luas pada kanvas termasuk padding, bukan ukuran fisik luka.

## Eksekusi dan monitoring

Konfigurasi: [additive](../configs/ablation/additive.json), [maxout](../configs/ablation/maxout.json). Adapter dan runner baru menggunakan engine baseline serta snapshot sumber yang tetap utuh.

Sebelum training, **26 test lulus**, termasuk uji jalur forward, gradien cabang aktif, state/RNG, dan stage terakhir. Probe tiga model dengan constructor pretrained penuh lulus. Kedua varian juga lulus smoke test GPU: gradien finite, encoder diperbarui, dan prediksi identik sesudah checkpoint dimuat ulang. Pemeriksaan terpisah menguji seluruh alur analisis dengan prediksi baseline sebagai fixture; output diagnostik itu disimpan lokal dan **bukan hasil ablasi**.

Antrean detached menjalankan additive-only, lalu maxout-only secara berurutan pada satu GPU. Sebelum setiap proses, supervisor menunggu GPU kosong; proses pengguna lain tidak dihentikan. Setiap run mengevaluasi checkpoint terpilih pada validation dan test, sekaligus menyimpan 402 mask prediksi. Setelah kedua run selesai, analisis CPU memverifikasi semua mask terhadap CSV/JSON evaluasi, menghitung PFOM, lalu mengeksekusi notebook perbandingan.

Cek progres dari root proyek:

```bash
python scripts/ablation_status.py
```

Perintah tersebut hanya membaca status dan log, tanpa memulai training. Artefak utama:

Pemantau tambahan `scripts/monitor_ablations.py` mencatat pemeriksaan kesehatan setiap **30 menit**: progres epoch, metrik validation terbaru, proses supervisor/trainer, error log, dan penggunaan GPU. Catatan terakhir berada di `outputs/ablations/cp1-attention-01/monitor/latest.json`; riwayat lengkap ada di `checks.jsonl`. Pemantau berhenti saat workflow selesai/gagal atau supervisor hilang. Pemantau hanya membaca proses training, tidak melakukan restart atau mengirim notifikasi chat.

```bash
cat outputs/ablations/cp1-attention-01/monitor/latest.json
```

Artefak training dan analisis:

```text
outputs/ablations/cp1-attention-01/
  verification.json       # State awal, route modul, parameter, hash runtime
  workflow.json           # Status antrean dan publikasi
  supervisor.log          # Log supervisor
  comparison.csv          # Hasil sesudah kedua run selesai
  figures/                # Grafik dan galeri perbandingan
  published/              # Notebook executed, laporan, tabel untuk publikasi
outputs/runs/cp1-attention-{additive,maxout}-01/
  config.json, environment.json, split-manifest.json, model-ablation.json
  history.json, status.json, best.pth, last.pth
  validation.json, validation.csv, evaluation-test.json, evaluation-test.csv
outputs/analysis/cp1-attention-{additive,maxout}-01/
  predictions/, prediction-manifest.json
```

Supervisor mempublikasikan hasil ke `notebooks/02_cp1_attention_ablation.ipynb`, `reports/cp1-attention-ablation-01.md`, dan `reports/tables/cp1-attention-*`. Commit/push otomatis hanya dilakukan jika working tree bersih, branch `main`, dan tidak menimpa file hasil yang berbeda. Jika ada perubahan lain, hasil tetap tersimpan di `published/` dan alasan penundaan dicatat dalam `workflow.json`.

Untuk mereproduksi antrean pada workspace dengan baseline/data/cache sumber yang sama:

```bash
.venv-training/bin/python scripts/run_attention_ablation.py verify
.venv-training/bin/python scripts/run_ablation_batch.py --device cuda:0
```

Jangan menjalankan antrean kedua saat supervisor masih aktif. `--resume` disediakan untuk melanjutkan workflow yang terputus setelah status dan checkpoint diperiksa. Menjalankan ulang analisis menggunakan `.venv-analysis/bin/python scripts/build_ablation_analysis.py`; ini tidak melakukan training atau inference.

Sumber contoh dataset: [uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/wound_dataset). Paper dan kode referensi tercantum di [audit sumber](reference-audit.md).
