# PFOM · Protokol analisis batas

PFOM ditambahkan pada 27 Oktober 2026 untuk melengkapi analisis `cp1-baseline-01`. Penghitungan memakai **402 pasangan mask yang sudah tersimpan**: 124 validation dan 278 test, checkpoint epoch 52, threshold prediksi `>=0.5`. Tidak ada training, inference, tuning, atau pemilihan checkpoint ulang. Metrik region dan evaluator asli tetap sama.

## Definisi yang diimplementasikan

Untuk kedua edge map nonkosong, implementasi mengikuti Persamaan 13 [paper FUSegNet](https://arxiv.org/pdf/2305.02961v2#page=8):

```text
PFOM = sum_i[1 / (1 + beta * d_i²)] / max(N_GT, N_pred)
beta = 1/9
```

Setiap `d_i` adalah jarak Euclidean dalam piksel dari satu edge prediksi ke edge GT terdekat. Skor berarah dari prediksi ke GT, tidak harus simetris. Nilai berada di 0–1 untuk pasangan nonkosong dan lebih tinggi berarti lebih baik. Skor dihitung per gambar, lalu dirata-ratakan; tidak menggunakan agregasi TP/FP/FN data-based Dice.

Edge diekstrak dari mask GT dan prediksi pada **seluruh kanvas 224 × 224**, termasuk padding hitam. Crop pada galeri hanya untuk tampilan. Parameter lengkap dan asumsi ada di [`configs/pfom.json`](../configs/pfom.json).

| Komponen | Pengaturan |
| :--- | :--- |
| Edge detector | `skimage.feature.canny`, scikit-image 0.19.3 |
| Input | Mask biner, dikonversi ke float64 bernilai 0/1 |
| Gaussian sigma | **0.1**, dinyatakan di bagian 4.2 paper |
| Low / high threshold | **0.1 / 0.2**, pilihan eksplisit analisis memakai default Canny untuk float |
| Border | `mode="constant"`, `cval=0`, `mask=None`; perilaku border bawaan scikit-image |
| Quantile threshold | `False` |
| Jarak | `scipy.ndimage.distance_transform_edt`, SciPy 1.8.1; sampling 1 piksel per sumbu |
| Normalisasi | Maksimum jumlah edge GT dan edge prediksi pada gambar yang sama |

Beta, sigma, dan rumus berasal dari paper. Penulis tidak memublikasikan kode PFOM, library Canny, semua threshold, atau rincian border. Pengaturan lainnya adalah **keputusan implementasi lokal yang terdokumentasi**, sehingga kesetaraan numerik dengan Figure 6 belum terverifikasi. Sumber primitive: [Canny v0.19.3](https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/feature/_canny.py), [EDT SciPy v1.8.1](https://github.com/scipy/scipy/blob/v1.8.1/scipy/ndimage/_morphology.py).

## Kasus edge kosong: dua kebijakan terpisah

| Keadaan edge map | PFOM 0–1, ekstensi lokal | Proxy sentinel 2 |
| :--- | ---: | ---: |
| Kedua map nonkosong | Rumus di atas | Rumus di atas |
| Kedua map kosong | 1 | 2 |
| Hanya GT kosong | 0 | 2 |
| Hanya prediksi kosong | 0 | 2 |

Bagian 4.2 paper menyebut mengganti nilai tak hingga dengan **2** ketika tidak ada edge. Pemicu persisnya ambigu; nilai 2 juga bukan keluaran rumus pada pasangan nonkosong. Kita memilih interpretasi **sedikitnya satu edge map kosong → 2**, memberi nama kolom `pfom_paper_sentinel2_proxy`, dan mencatat jumlah kasus yang terpengaruh. Ini bukan klaim parity kode penulis atau akurasi 200%.

Kebijakan bounded 1/0 adalah ekstensi lokal untuk menangani kasus kosong, bukan klaim bahwa paper memakai kebijakan tersebut. Seluruh mean, median, subset GT positif, subset kedua edge nonkosong, serta kategori C0–C9 ditampilkan agar efek kebijakan terlihat. Mean utama tidak membuang luka yang seluruhnya terlewat. Edge kosong diperiksa dari keluaran Canny, bukan hanya dari ada tidaknya piksel GT; mask konstan juga dapat tidak mempunyai edge.

## Menjalankan dan menguji

Environment analysis mempunyai SciPy/skimage dengan versi yang sama seperti environment training. Paket training dan source yang dipakai training tidak diubah.

```bash
.venv-analysis/bin/python -m pip install -r requirements-analysis.lock.txt
.venv-analysis/bin/python -m unittest discover -s tests -p test_pfom.py -v
.venv-analysis/bin/python scripts/build_cp1_notebook.py --execute
```

[`scripts/pfom_metrics.py`](../scripts/pfom_metrics.py) berisi rumus dan Canny; [`scripts/cp1_analysis.py`](../scripts/cp1_analysis.py) menghitung seluruh pasangan setelah hash data, checkpoint, dan mask diverifikasi. Parameter tidak dipilih dari hasil test.

Sembilan pengujian di [`tests/test_pfom.py`](../tests/test_pfom.py) mencakup kecocokan sempurna, jarak analitik 1/3/5 piksel, penalti edge tambahan/hilang, arah prediksi ke GT, kebijakan kosong, Canny pada mask konstan, perpindahan mask, serta input tidak valid. Uji oracle membandingkan distance transform dengan pencarian jarak brute-force pada 40 pasangan edge acak seed 42.

Seluruh **402 pasangan** juga dihitung melalui `.venv-training` yang tetap dibekukan (NumPy 1.22.4) dan dibandingkan dengan hasil analysis (NumPy 1.24.1). Semua jumlah edge dan klasifikasi kasus sama; selisih skor maksimum sekitar **5 × 10⁻¹⁶**, di bawah toleransi 10⁻¹². Catatan pemeriksaan lokal: `outputs/reference-audit/pfom-frozen-environment-parity.json`.

Artefak lokal di `outputs/analysis/cp1-baseline-01/`:

- `tables/pfom_per_image.csv`: skor, jumlah edge, dan klasifikasi kasus kosong untuk seluruh 402 gambar.
- `tables/pfom_summary.csv`, `pfom_categories.csv`, `pfom_empty_edges.csv`: agregasi yang ditampilkan di notebook.
- `pfom-metadata.json`: parameter, fingerprint protokol, hash implementasi, versi dependency, hash CSV, dan identitas prediksi.
- `figures/07_pfom_boundaries.{png,svg}`: hubungan Dice/PFOM dan sebaran per kategori.
- `analysis-summary.json`: menyertakan ringkasan PFOM di samping metrik region.

[Hasil dan batas interpretasi](../reports/cp1-pfom-01.md) · [CSV numerik dan metadata publik](../reports/tables/) · [Notebook yang sudah dieksekusi](../notebooks/01_cp1_analysis.ipynb#pfom).
