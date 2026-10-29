# CP1 · Reproduksi x-FUSegNet

**Training lima fold, inference ensemble, dan analisis selesai.**

[Notebook executed](../notebooks/03_cp1_xfusegnet.ipynb) · [Protokol dan audit kesetaraan](../docs/xfusegnet.md)

## Cakupan

FUSeg 512×512: 1.010 gambar berlabel, lima fold 808 train / 202 validation, dan 200 gambar test tanpa mask. Loop training asli penulis dipakai dengan hook pencatatan/resume. Rata-rata lima probabilitas float32 mendahului threshold >=0.5.

## Validasi internal OOF

- Image Dice: **88.03%**.
- Data Dice: **92.72%**.
- Mean PFOM (empty-edge 1/0): **0.7315**; sentinel-2 proxy **0.7780**.

Setiap gambar hanya diprediksi model fold yang menahannya. Checkpoint dipilih pada validation fold tersebut. Skor ini bukan evaluasi ensemble lima model pada independent test set.

| Fold | Epoch | Checkpoint | OOF image Dice (%) | OOF data Dice (%) | PFOM | Menit training |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 124 | 94 | 84.95 | 91.00 | 0.7044 | 233.0 |
| 2 | 200 | 198 | 88.99 | 93.70 | 0.7366 | 330.2 |
| 3 | 156 | 126 | 87.66 | 92.36 | 0.7162 | 296.1 |
| 4 | 129 | 99 | 87.47 | 92.19 | 0.7279 | 303.8 |
| 5 | 111 | 81 | 91.09 | 94.04 | 0.7723 | 191.1 |

## Test ensemble

Tersedia **200 prediksi ensemble test**. Semua mean probabilitas lima model dan PNG threshold diverifikasi. Tidak ada skor Dice/PFOM test lokal karena mask test tidak tersedia. **Skor paper 89.23% belum diverifikasi secara lokal.**

## Batas kesetaraan

Urutan direktori/fold penulis tidak dipublikasikan; urutan lokal dipersistenkan sebelum training. Seed torch/numpy 42 adalah tambahan reproducibility, dengan RNG berlanjut antar-fold. Seluruh versi dependency training baseline tetap sama; dependency plotting tambahan berada di environment khusus.

Augmentasi dan mask fractional training mengikuti xfusegnet_train.py secara langsung. Evaluasi OOF/PFOM membinarisasi GT pada >=0.5 secara terpisah. Loss probability-input, checkpoint loss/IoU elif, dan metadata epoch off-by-one sumber dipertahankan. Tabel melaporkan jumlah epoch training aktual.

Duplikasi RGB antar-fold dipertahankan dan dihitung; independensi pasien belum terverifikasi. Validasi OOF memakai fold yang juga memilih checkpoint. Perbedaan dataset/resolusi/protokol menghalangi perbandingan langsung dengan hasil test Chronic Wound.

Bukti: [folds](tables/cp1-xfusegnet-folds.csv), [OOF per gambar](tables/cp1-xfusegnet-oof-per-image.csv), [keanggotaan fold](tables/cp1-xfusegnet-split-membership.csv), [metadata](tables/cp1-xfusegnet-metadata.json).
