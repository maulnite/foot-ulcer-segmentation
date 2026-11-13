# Review hasil — resep CP1 pada kandidat CP2

**Ketiga training dan evaluasi selesai. RH2 dipilih melalui validation sebelum
test dibuka.** Hasil mendukung manfaat numerik supervisi batas terhadap baseline
baru dalam run ini, tetapi tidak membuktikan resep CP1 lebih baik secara umum.

Run: `outputs/cp2/cp1_recipe/20261113T082016Z`. Seed 42. Duplicate-RGB-group
split **671 train / 118 validation / 261 test**, bukan patient-level split.
Semua arm memakai loss sumber B0, termasuk probabilitas yang diteruskan ke
focal-with-logits, sesuai pilihan eksplisit pengguna. Tidak ada warm-start
trained CP1 checkpoint, perubahan label, atau pengembalian data konflik.

## Perbandingan terkontrol yang baru

| Arm | Epoch dipilih / selesai | Validation Image Dice | Test Image Dice | Test Data Dice | Test small Image Dice | Test Boundary IoU | Test miss / 245 GT positif | Test empty FP / 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 — P-scSE sumber | 24 / 54 | 82,63% | 83,84% | 91,04% | 82,17% | 0,5385 | 2 | 11 |
| RH2 — auxiliary boundary | 95 / 125 | **84,44%** | **84,36%** | **92,46%** | **83,02%** | **0,5580** | **1** | 13 |
| RA6 — CBAM stage 2–3 | 81 / 111 | 84,24% | 83,11% | 91,26% | 77,40% | 0,5427 | 4 | 11 |

RH2 dan RA6 lolos seluruh guard validation terhadap **R0 baru**. Validation
empty FP masing-masing R0/RH2/RA6 adalah **6/7, 6/7, 5/7**; tidak ada total
foreground miss. RH2 dipilih karena overall validation Dice tertinggi di
antara kandidat eligible. Keputusan tersimpan **16:00:06 UTC, 13 November
2026**, sebelum semua ekspor test baru. Guard tidak diubah setelah hasil.

Pada test, RH2−R0 menghasilkan delta Image Dice **+0,520 poin persentase**,
Data Dice **+1,421 poin**, small Image Dice **+0,844 poin**, dan Boundary IoU
**+1,957 poin**. Namun empty-FP images naik **11→13**, sementara empty-FP
pixels turun **3.162→2.447**. Jumlah gambar dengan FP dan luas FP tidak
mengukur hal yang sama. Test tidak dipakai memilih ulang kandidat atau threshold.

RA6 mempunyai keuntungan validation, tetapi test Image Dice turun **0,728
poin** terhadap R0, small Dice turun **4,776 poin**, dan total miss naik **2→4**.
Ini tidak mendukung klaim CBAM unggul pada konfigurasi/seed ini.

## Apa arti kurva dan aturan checkpoint?

Untuk RH2, Image Dice validation tertinggi selama training tercatat **85,50%
pada epoch 125**, tetapi kebijakan sumber memilih **epoch 95, 84,44%**.
Checkpoint memantau main validation loss melalui `if`, lalu source batch-IoU
melalui `elif`; scheduler juga memantau main loss. Jadi puncak Image Dice
memang tidak harus dipilih. Epoch/checkpoint tidak diganti setelah melihat
hasil, dan tidak dilakukan test inference pada checkpoint alternatif.

R0 menurunkan LR setelah epoch 35, RH2 setelah 74, dan RA6 setelah 39 menurut
history yang tersimpan. Scheduler/early stopping mengikuti aturan sama, tetapi
jumlah update aktual berbeda. Kontras ini bukan eksperimen fixed-step budget.

## Ketidakpastian dan variasi per gambar

Bootstrap berpasangan 2.000 replikasi, seed 42, menggunakan exact-RGB group
(satu gambar tersisa per group). Interval berikut **deskriptif**, bukan
signifikansi konfirmatori:

| Kontras | Split | Delta Image Dice (poin persentase) | Interval persentil 95% | Membaik / memburuk / sama |
|---|---|---:|---:|---:|
| RH2−R0 | Validation | +1,809 | [+0,804; +2,846] | 73 / 38 / 7 |
| RH2−R0 | Test | +0,520 | [−1,124; +2,063] | 150 / 96 / 15 |
| RA6−R0 | Validation | +1,618 | [−0,087; +3,952] | 70 / 42 / 6 |
| RA6−R0 | Test | −0,728 | [−2,926; +1,350] | 136 / 112 / 13 |

Patient/crop dependence, pemilihan kandidat/checkpoint, reuse validation dan
variasi seed tidak dicakup interval ini. Validation sudah digunakan berulang
dan source-test telah dianalisis sejak CP1. Interval RH2 test melintasi nol:
belum ada bukti peningkatan yang stabil atau manfaat klinis.

## Perbandingan historis pada cohort CP2 yang sama

| Run | Validation Image Dice | Test Image Dice | Test Data Dice |
|---|---:|---:|---:|
| B0 fase P — source loss, kebijakan CP2 | 83,61% | 83,73% | 91,08% |
| R0 baru — source loss, resep training CP1 | 82,63% | 83,84% | 91,04% |
| B1 fase P — corrected loss, kebijakan CP2 | 84,28% | 83,29% | 90,77% |
| H2 fase H — corrected loss, kebijakan CP2 | 84,88% | 84,77% | 91,50% |
| RH2 baru — source loss, resep training CP1 | 84,44% | 84,36% | 92,46% |
| A6 fase A — corrected loss, kebijakan CP2 | 85,17% | 84,41% | 92,04% |
| RA6 baru — source loss, resep training CP1 | 84,24% | 83,11% | 91,26% |

**Interpretasi:** resep CP1 tidak otomatis memulihkan skor tinggi pada data
bersih. R0 vs B0 memberi validation lebih rendah, test Image Dice hampir sama,
dan Data Dice sedikit lebih rendah. RH2 baru menaikkan Data Dice dibanding H2
lama, tetapi Image Dice sedikit lebih rendah; empty FP test juga naik dari
10 menjadi 13. Lolos terhadap baseline baru bukan berarti melampaui semua
hasil CP2 sebelumnya.

Perbandingan historis ini deskriptif. B0 vs R0 mengubah satu paket kebijakan
checkpoint, scheduler, early stopping dan RNG; tidak mengisolasi faktor tunggal.
H2/A6 historis vs versi baru juga mengubah B1 menjadi B0, termasuk objective
auxiliary RH2. Tidak boleh mengatribusikan seluruh perbedaan kepada attention
atau checkpoint saja. Semua historical model/checkpoint/metrics tetap utuh.

## QA, biaya, dan publikasi

- **136 tests** dan tiga smoke real-data lulus; smoke pertama yang provisional
  tetap disimpan sebagai engineering failure sebelum training penuh. Lazy-import
  RNG diperbaiki dan regression test ditambahkan sebelum run final.
- Source tensors pasangan identik. Actual train order dan augmented bytes cocok
  pada seluruh common epochs: R0/RH2 **54**, R0/RA6 **54**, RH2/RA6 **111**.
- Checkpoint save/reload ketiga model lulus. Source checkpoint policy dan seleksi
  validation dihitung ulang; **1.137 exports** (354 validation + 783 test)
  diverifikasi terhadap checksum, raw GT, threshold, region/boundary/subgroup
  metrics dan per-image CSV. Tidak ada retraining atau inference tambahan saat
  analisis bootstrap.
- **13.511 historical/user files** identik, termasuk notebook pengguna yang
  sudah dimodifikasi sebelum pekerjaan ini. Tidak mengedit notebook atau
  raw dataset dan tidak menimpa run/checkpoint lama.
- Jumlah waktu tiga arm sekitar **7,61 jam**, satu trainer GPU A100 berurutan.
  Peak allocated **2,21–2,22 GiB**, reserved **2,36 GiB**; FP32, batch 2, AMP off.
  Ini allocator proses sendiri, bukan seluruh pemakaian GPU bersama.
- R0/RH2/RA6 active parameters **64.898.248 / 64.898.281 / 64.891.593**.
  Registered parameters **70.968.939 / 70.968.972 / 70.975.382** mencakup modul
  sumber yang tidak aktif di forward. Head RH2 dapat dilepas saat inference;
  tidak ada GT sebagai input inference.
- Ketiga PNG diperiksa langsung. Legend plot area/boundary yang awalnya
  menutupi batang dipindahkan ke bawah; grafik asli dan source renderer lama
  dipertahankan di `presentation_v1/`. Hanya presentasi berubah; seluruh
  metrics, statuses, selection dan run artefacts tetap sama.
- Raw foto, checkpoints, probability maps dan private environment tetap lokal.
  Publikasi hanya kode, konfigurasi, tabel numerik, metrik dan grafik agregat.

**Langkah berikutnya:** jangan mengulang seluruh matriks. Gunakan hasil ini
untuk memilih kontrol terbatas bila ingin mengisolasi kebijakan training atau
objective, lalu konfirmasi kandidat dengan beberapa seed/protokol terkunci.
Pengendalian empty FP dan evaluasi berlabel yang belum terekspos tetap diperlukan.
Tidak ada eksperimen baru di luar tiga arm ini yang dijalankan otomatis.

[Laporan utama dan grafik](cp2-cp1-recipe.md) ·
[Metrik lengkap](cp2-cp1-recipe-results.json) ·
[Analisis berpasangan dan historical provenance](cp2-cp1-recipe-analysis.json)
