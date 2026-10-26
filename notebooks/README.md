# CP1 · Analisis hasil FUSegNet

Notebook berbahasa Indonesia untuk membaca hasil baseline: protokol, kurva training, metrik akhir, error menurut ukuran luka, kasus GT kosong, visualisasi prediksi, dan audit overlap.

| File | Isi |
| :--- | :--- |
| `01_cp1_analysis.executed.ipynb` | **Buka ini di workspace lokal.** Seluruh cell sudah dieksekusi, termasuk lima galeri prediksi. File ini diabaikan Git. |
| [01_cp1_analysis.ipynb](01_cp1_analysis.ipynb) | Salinan publik dengan output tabel dan grafik agregat. Output foto klinis dilepas sesuai kebijakan repo. Semua kode galeri tetap tersedia. |

Untuk langsung melihat hasil, buka file `.executed.ipynb` di VS Code/Jupyter. Tidak perlu menjalankan cell lagi. Untuk menjalankan ulang secara interaktif, pilih interpreter `.venv-analysis/bin/python` sebagai kernel.

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

Perintah ini membuat kernel di folder `outputs/` proyek, mengeksekusi semua cell tanpa GPU, menyimpan notebook lengkap lokal, lalu menghasilkan salinan publik dengan output foto dilepas. Setiap eksekusi memeriksa SHA256 checkpoint, konfigurasi, split, source training, data, mask prediksi, dan kecocokan metrik. Tidak ada tuning atau pemilihan checkpoint ulang.

`scripts/cp1_analysis.py` menyimpan fungsi analisis dan plotting agar notebook tetap ringkas. `scripts/build_cp1_notebook.py` merupakan sumber susunan notebook; ubah builder jika ingin membangkitkan ulang susunannya. Menjalankan builder tanpa `--execute` hanya membuat template tanpa output.

## Ekspor untuk laporan dan slide

```text
outputs/analysis/cp1-baseline-01/
├── figures/                    # 5 grafik agregat PNG 300 dpi + SVG; 5 galeri PNG
├── tables/                     # Metrik, kategori ukuran, GT kosong, overlap, pilihan contoh
├── predictions/                # Mask biner validation dan test
├── prediction-manifest.json    # Identitas checkpoint, hash mask, verifikasi parity
├── analysis-summary.json       # Metrik, aturan pemilihan, keterbatasan
└── notebook-execution.json     # Catatan eksekusi seluruh code cell
```

Foto, mask, dan galeri klinis tetap lokal sesuai [kebijakan repositori](../README.md#experiment-discipline). Hasil harus disertai keterbatasan: jumlah data berbeda dari paper, split asli belum diketahui, evaluator diberi label eksplisit, satu seed, dan overlap piksel RGB antar-split.
