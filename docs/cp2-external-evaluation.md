# CP2 — frozen-checkpoint external comparison

Pertanyaan: apakah peringkat checkpoint CP1/CP2 berubah saat menghadapi sumber
foto dan jenis luka yang berbeda? Ini **direct external inference**, tanpa
retraining, fine-tuning, threshold tuning, atau pembaruan BatchNorm.

## Data dan interpretasi

Target pertama adalah CO2Wounds-V2 **Mendeley version 1**, DOI
[10.17632/s2w7rjwz49.1](https://data.mendeley.com/datasets/s2w7rjwz49/1).
Arsip resmi 115.125.340 byte lolos pemeriksaan ZIP/CRC, checksum SHA256
`250aa3eaca580b5a2d4952e1c68642eb402a725477665ad18b9eebbdfc862342`.

Tersedia 607 pasangan berlabel (485 provider-train dan 122 provider-val),
serta 157 provider-test tanpa mask. Seluruh pasangan berlabel berperan sebagai
evaluasi eksternal kita; nama split provider tidak berarti kita melatih model
pada foto tersebut. Mask biner menggunakan **0/1**, dan foto mempunyai ekstensi
campuran JPG/JPEG/PNG. Mask tidak dibagi 255 lalu di-threshold secara keliru.

Audit sebelum model inference menemukan 23 kelompok RGB identik dengan mask
berbeda, mencakup 49 pasangan. Kelompok tersebut dikarantina seluruhnya,
tanpa memilih mask berdasarkan hasil model atau menggabungkan anotasi.
Primary cohort adalah **558 gambar** (447 provider-train +111 provider-val).
Perbedaan mask belum ditetapkan sebagai kesalahan anotasi medis; instance/area
anotasi berbeda mungkin memerlukan klarifikasi provider.

Tidak ditemukan exact RGB overlap atau kandidat near-duplicate terhadap 1.109
gambar Chronic Wound sumber. Screening near-duplicate memakai RGB-only nonblack
content crop, pHash Hamming <=4 dan thumbnail RGB MAE <=3. Ini bukan bukti
ketiadaan semua hubungan crop/pasien. Rilis menyebut 96 pasien untuk total dataset,
tetapi metadata publik yang tersedia tidak memberikan patient ID terverifikasi.
Tidak ada klaim patient-independent confidence interval.

Seluruh mask pada 558 gambar berisi luka: **GT-empty external evaluation tidak
tersedia**. Precision/FP pixels pada latar foto luka tetap dapat dihitung,
sedangkan kemampuan menolak foto tanpa luka tidak dapat dinilai dari cohort ini.

Mendeley menampilkan CC BY 4.0, sedangkan README penulis dan COCO JSON di dalam
arsip menyebut CC BY-NC-ND 4.0. Perbedaan ini didokumentasikan; data digunakan
untuk riset nonkomersial secara lokal. Foto, mask, dan cache probabilitas tidak
dipublikasikan. Atribusi: Sanchez et al., *CO2Wounds-V2: Extended Chronic Wounds
Dataset From Leprosy Patients*, ICIP 2024, pp.69–75.

[DFUC2022](https://dfu-challenge.github.io/dfuc2022.html) masih menyediakan
jalur permohonan akses; data berlabel belum tersedia lokal. Tidak ada permohonan
atau pesan ke provider yang dikirim. Chronic Wound → FUSeg tidak digunakan
sebagai pengganti validasi antar-sumber karena hubungan kedua dataset tersebut.

## Checkpoint yang dibekukan

Registry [external.json](../configs/cp2/external.json) memuat **26 checkpoint
unik**, konfigurasi, SHA256 dan epoch terpilih:

- CP1 original baseline.
- CP2 seed42: B0/B1, L1–L4, A1/A2/A3/A5/A6, H1/H2.
- CP1-recipe pada split CP2: R0/RH2/RA6.
- CP1-matched: C0/CH2/CA6.
- Replikasi CP2 seed43/44: B1/H2/A6.

L0/A0/H0 adalah alias B1 dan tidak dihitung ulang. Seluruh model, termasuk yang
gagal guard validation internal, dilaporkan. Semua seed ditampilkan, bukan hanya
seed terbaik. Mean/SD tiga seed dilaporkan khusus B1/H2/A6 pada protokol CP2.

Kontras B0–B1 adalah pembanding paling terkontrol untuk koreksi input focal.
CP1 original versus CP2 juga berbeda cohort training, checkpoint/scheduler dan
RNG. Perbandingan itu tidak dapat mengisolasi satu penyebab. Checkpoint CP1
adalah reproduksi lokal kita, bukan bobot asli penulis paper.

## Input, output dan evaluator

Semua model menerima RGB **224×224**, aspect-preserving resize dan black RGB
letterbox. Interpolasi downsample AREA, upsample LINEAR; normalisasi ImageNet
persis fungsi source, tanpa fitting statistik target. Tidak ada GT crop.

Probability map 224 dibuang paddingnya, di-resize bilinear ke koordinat gambar
asli, lalu threshold **>=0,5**. GT tetap pada koordinat native dan tidak di-resize.
Checkpoint sigmoid-output tidak disigmoid ulang; checkpoint logit-output
mendapat sigmoid satu kali. Auxiliary head hanya dimuat untuk verifikasi ketat;
prediksi menggunakan mask utama dari base network yang sama.

Metrik: ordinary TP/TN/FP/FN, Image/Data Dice dan IoU, precision/recall,
Boundary IoU (inner band 1% diagonal native), PFOM dengan empty rule CP1,
ukuran luka menurut fraksi kanvas native, dan total foreground misses.
Definisi small <=1% kanvas bukan ukuran fisik; distribusi native target berbeda
dari source yang berpadded. Semua model memakai gambar dan evaluator yang sama.

## Eksekusi

Environment `.venv-training` dipertahankan, FP32, batch1, eval/no-grad, seluruh
parameter dibekukan. GPU sibuk dengan job lain sehingga digunakan CPU:
**maksimal empat worker ×empat thread**, total 16 dari 256 logical cores.
Perubahan resource dari rancangan sequential awal dicatat sebelum target
prediction; checkpoint, preprocessing dan metrik tidak berubah. Timing CPU
shared adalah catatan biaya, bukan perbandingan latency eksklusif/paper.

```bash
.venv-training/bin/python scripts/register_cp2_external.py --run outputs/cp2/external/NEW_ID
python scripts/prepare_co2wounds_external.py --run outputs/cp2/external/NEW_ID
.venv-training/bin/python scripts/run_cp2_external.py --run outputs/cp2/external/NEW_ID --stage prepare
.venv-training/bin/python scripts/run_cp2_external.py --run outputs/cp2/external/NEW_ID --stage preflight
.venv-training/bin/python -u scripts/run_cp2_external.py --run outputs/cp2/external/NEW_ID --stage infer
.venv-xfuseg/bin/python scripts/report_cp2_external.py --run outputs/cp2/external/NEW_ID
```

Dataset cache yang intact tidak diunduh ulang. Registry menolak checkpoint
yang hilang/berubah. Source/config/input-geometry hashes dibekukan sebelum
inference; setiap model memverifikasi input dan strict-load seluruh state.
Preflight mewajibkan finite probabilities, deterministic repeat CPU, parameter
frozen dan BatchNorm buffers tidak berubah. Completed model stages dapat dimuat
dari cache dengan memverifikasi checksum; partial/failed output perlu diperiksa
secara eksplisit, tidak ditimpa atau dianggap berhasil.

Per-model probability maps 224, geometry, per-image CSV, metrics JSON, log,
status dan provenance disimpan dalam run append-only. Cache memungkinkan
penghitungan ulang seluruh native metrics tanpa model inference baru.
Hasil utama disajikan di [laporan eksternal](../reports/cp2-external-evaluation.md).

Langkah report memverifikasi seluruh checksum prediksi dan menghitung ulang
native region metrics, mencocokkan seluruh tensor dengan RGB mentah, serta
memeriksa file historis sebelum membuka perbandingan. Setelah pemeriksaan visual
ketiga figure dan pencatatan `visual_review.json` pada run, artefak agregat dapat
diterbitkan lokal menggunakan `scripts/publish_cp2_external.py --run ...`.
Publisher tidak mengunggah apa pun dan menolak menimpa laporan yang sudah ada.
Actual validated run: `outputs/cp2/external/20261118T040007Z`.

Baseline Image Dice eksternal: CP1 original 42,63%, B0 seed42 43,49%, B1
seed42 39,33%. Mean tiga seed B1/H2/CBAM adalah 40,95%/41,09%/41,80%.
Semua 26 checkpoint, paired effects, ukuran luka dan batasan interpretasi
tersedia pada laporan; angka di sini bukan pemilihan model berdasarkan target.

Evaluasi ini tidak membuktikan bahwa skor paper disebabkan kesalahan konfigurasi.
Peringkat baru adalah hasil transfer pada satu sumber/jenis luka yang diuji.
Pemilihan metode atau modifikasi setelah melihat hasil target menjadi eksploratif
dan memerlukan target independen lain untuk pengujian berikutnya.
