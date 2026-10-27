# CP1 · Analisis hasil FUSegNet

Notebook berbahasa Indonesia untuk membaca hasil baseline: protokol, kurva training, metrik akhir, error menurut ukuran luka, PFOM untuk ketepatan batas, kasus GT kosong, visualisasi prediksi, dan audit overlap.

**[Buka notebook CP1 yang sudah dieksekusi →](01_cp1_analysis.ipynb)**

Seluruh **24 code cell** sudah dieksekusi. Tabel, enam grafik analisis, dan lima galeri prediksi berisi **15 foto contoh** tersedia langsung di GitHub. Setiap contoh menampilkan gambar, ground truth, prediksi, dan overlay TP/FP/FN. Tidak perlu menjalankan cell untuk melihat output.

File utama `01_cp1_analysis.ipynb` menyimpan seluruh output dan dilacak Git. Builder juga menghasilkan salinan lokal identik `01_cp1_analysis.executed.ipynb` untuk kerja di server; salinan tambahan ini diabaikan Git. Untuk menjalankan ulang secara interaktif, buka notebook utama di VS Code/Jupyter dan pilih interpreter `.venv-analysis/bin/python` sebagai kernel.

## Menjalankan ulang

Jalankan dari root proyek. Python 3.10 dan artefak run `outputs/runs/cp1-baseline-01/`, data lokal, serta snapshot kode penulis diperlukan. File checkpoint dan data tidak didistribusikan melalui Git.

Environment analisis terpisah dari environment training:

```bash
python -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.lock.txt
```

Mask prediksi untuk 124 validation dan 278 test sudah tersedia secara lokal. Jika belum ada di checkout lain, ekspor sekali menggunakan environment training yang dipin:

```bash
.venv-training/bin/python scripts/export_analysis_predictions.py --device cpu
```

Ini **inference saja** dari `best.pth`, tanpa training. Batch inference mengikuti evaluasi asli: validation 2, test 1. Setiap hitungan TP/TN/FP/FN harus cocok dengan CSV evaluasi akhir. Ekspor berhenti jika ada perbedaan. Perangkat CPU tersedia sebagai pilihan konservatif; `--device cuda:0` dapat dipakai saat GPU tersedia. Mask yang sudah tersimpan diverifikasi dan digunakan kembali.

Buat dan eksekusi notebook dari awal sampai akhir:

```bash
.venv-analysis/bin/python scripts/build_cp1_notebook.py --execute
```

Bagian PFOM menghitung ulang skor dari mask cache pada CPU memakai SciPy/skimage, tanpa inference. Rumus, pilihan Canny, dan dua kebijakan edge kosong dijelaskan di [protokol PFOM](../docs/pfom.md). Jalankan uji analitik dengan `.venv-analysis/bin/python -m unittest discover -s tests -p test_pfom.py -v`.

Perintah ini membuat kernel di folder `outputs/` proyek, mengeksekusi semua cell tanpa GPU, lalu menyimpan notebook utama beserta seluruh output dan salinan lokal identiknya. Galeri foto dan prediksi dipertahankan saat notebook dibangkitkan ulang. Setiap eksekusi memeriksa SHA256 checkpoint, konfigurasi, split, source training, data, mask prediksi, dan kecocokan metrik. Tidak ada tuning atau pemilihan checkpoint ulang.

`scripts/cp1_analysis.py` menyimpan fungsi analisis dan plotting agar notebook tetap ringkas. `scripts/build_cp1_notebook.py` merupakan sumber susunan notebook; ubah builder jika ingin membangkitkan ulang susunannya. Menjalankan builder tanpa `--execute` hanya membuat template tanpa output.

## Ekspor untuk laporan dan slide

```text
outputs/analysis/cp1-baseline-01/
├── figures/                    # 6 grafik agregat PNG 300 dpi + SVG; 5 galeri PNG
├── tables/                     # Metrik, PFOM per gambar/kategori, GT kosong, overlap, pilihan contoh
├── predictions/                # Mask biner validation dan test
├── prediction-manifest.json    # Identitas checkpoint, hash mask, verifikasi parity
├── pfom-metadata.json          # Parameter, versi, hash implementasi, dan ringkasan PFOM
├── analysis-summary.json       # Metrik, aturan pemilihan, keterbatasan
└── notebook-execution.json     # Catatan eksekusi seluruh code cell
```

Contoh foto, GT, dan prediksi terpilih dipublikasikan sebagai output tertanam pada notebook, dengan atribusi sumber dataset di bagian referensi. Dataset lengkap, checkpoint, dan cache mask tetap berada di folder lokal sesuai [kebijakan repositori](../README.md#experiment-discipline). Hasil harus disertai keterbatasan: jumlah data berbeda dari paper, split asli belum diketahui, evaluator diberi label eksplisit, satu seed, dan overlap piksel RGB antar-split.
