# CP1 · Ablasi cabang attention FUSegNet

Semua run selesai. Stage 0–3 memakai kedua cabang, additive-only, atau maxout-only; stage terakhir tetap shorted.
Split 707/124/278, seed 42, konfigurasi training/evaluator dan inisialisasi constructor dikontrol.

[Notebook lengkap yang sudah dieksekusi](../notebooks/02_cp1_attention_ablation.ipynb) · [Protokol](../docs/attention-ablations.md)

## Validation

| Model | Epoch | Checkpoint | Image Dice (%) | Data Dice (%) | PFOM 1/0 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| P-scSE penuh | 82 | 52 | 82.34 | 85.90 | 0.7244 |
| Additive-only | 68 | 38 | 80.58 | 89.11 | 0.7058 |
| Maxout-only | 176 | 146 | 82.58 | 91.27 | 0.7266 |

## Test

| Model | Epoch | Checkpoint | Image Dice (%) | Data Dice (%) | PFOM 1/0 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| P-scSE penuh | 82 | 52 | 86.42 | 92.29 | 0.7782 |
| Additive-only | 68 | 38 | 84.81 | 91.83 | 0.7554 |
| Maxout-only | 176 | 146 | 84.65 | 92.35 | 0.7613 |

## Perubahan terhadap baseline pada test

- Additive-only: delta image Dice -1.61 pp; delta PFOM -0.0228; GT positif TP=0 3; GT kosong dengan FP 8/17.
- Maxout-only: delta image Dice -1.77 pp; delta PFOM -0.0169; GT positif TP=0 6; GT kosong dengan FP 9/17.

## Verifikasi dan batas interpretasi

Semua 1.206 pasangan mask dari tiga model diverifikasi terhadap CSV dan JSON evaluasi akhirnya. PFOM memakai protokol baseline yang sama. Tidak ada pemilihan threshold atau varian berdasarkan test.

Rekonstruksi baseline dan kedua varian memiliki hash bobot awal yang sama. Model dibangun lengkap sebelum jalur forward cabang diubah. Cabang dorman tetap terdaftar; parameter forward aktif baseline 64,898,248 dan varian 64,781,798. Posisi attention, satu conv aktif per stage, dan shorted stage terakhir tetap sama.

Satu seed berarti variasi antar-training belum diukur. Ablasi mengubah komponen sekaligus kapasitas aktif; ini tidak mengisolasi operator agregasi pada kapasitas parameter identik. Test baseline sudah dianalisis sebelum eksperimen; seluruh hipotesis dan kedua varian ditetapkan sebelum run baru. Ketidakpastian split asli paper, duplikasi sumber, dan konvensi PFOM tetap berlaku.

Contoh notebook memakai enam nama yang ditetapkan dari galeri baseline, tanpa pemilihan ulang berdasarkan hasil varian baru.

Bukti numerik: [ringkasan](tables/cp1-attention-summary.csv), [kategori](tables/cp1-attention-categories.csv), [per gambar](tables/cp1-attention-per-image.csv), [metadata](tables/cp1-attention-metadata.json).
