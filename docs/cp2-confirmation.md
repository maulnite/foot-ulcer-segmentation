# CP2 — audit false positive dan konfirmasi terbatas B1/H2/A6

## Tujuan dan preregistration

Kembali ke protokol CP2 B1/L0, **bukan** CP1-recipe atau CP1-matched.
Pertanyaan: apakah sinyal manfaat H2 (boundary supervision) dan A6 (CBAM) stabil
pada randomness training berbeda, dan bagaimana trade-off dengan GT-empty FP?

Kandidat dipilih dari screening yang sudah dilihat. Ini **exploratory internal
seed replication**, bukan konfirmasi pada data independen. H2/A6 seed42 tetap
gagal guard empty-FP historis; keputusan lama tidak diedit.

Manifest tetap **671 train / 118 validation / 261 source-test**, fingerprint
`fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`.
Group RGB identik tidak menyeberang split; tanpa patient IDs, tidak ada klaim
patient-level independence. Tidak mengubah anotasi, preprocessing, precision,
threshold, empty rules, split atau historical artifacts.

## Matriks, hipotesis dan kontrol

| Model | Faktor terhadap B1 | Seed42 | Seed43 | Seed44 |
|---|---|---|---|---|
| B1/L0 | FUSegNet asli, corrected focal | Cached P | Fresh | Fresh |
| H2 | Auxiliary boundary head stage3, weight0.1 | Cached H | Fresh | Fresh |
| A6 | CBAM stage2–3, output gain3 | Cached A | Fresh | Fresh |

Tepat **enam full trainings baru**, sequential satu job GPU. Tidak mengulang
seed42 atau CP1. H2 diharapkan membantu batas/luka kecil; A6 diharapkan membantu
pemilihan fitur. Keduanya boleh gagal atau meningkatkan false positive.

Semua numerical settings fase P tetap: EfficientNet-B7 ImageNet, logits,
source Dice(sigmoid(z)) + corrected focal(z), Adam1e-4/WD1e-5, batch2/workers4,
max200, FP32 tanpa AMP dan TF32 warisan. Checkpoint earliest strictly highest
validation Image Dice; scheduler Plateau atas metrik yang sama, factor0.1,
patience10/minLR1e-5; early stopping30.

Split seed tetap42. Training seeds **42,43,44** dipilih sebelum run baru.
Untuk seed43/44, satu source state ImageNet+decoder baru disimpan dan dimuat
identik oleh B1/H2/A6. Head seed = training_seed+30003, sehingga formula sama
dengan seed42=30045. A6 memakai constructor lama seed+20200+stage. Inisialisasi
ImageNet tidak diunduh ulang. Seluruh module source lama byte-identical.

Epoch RNG memakai training_seed+10007×zero_based_epoch; dedicated loader
generators, worker seeds dan augmentasi frozen dipakai ulang. Kesamaan source
tensors **dan actual sample order/augmented input bytes** diperiksa pada common
epochs. Seed sama saja tidak dianggap bukti pairing atau bitwise GPU replay.

## Audit FP sebelum training

Periksa ulang 354 saved seed42 validation probability/GT pairs, checksums,
raw masks, threshold≥0.5 dan metrik. Tampilkan **semua tujuh GT-empty validation
cases**, jumlah pixel FP, connected components8, komponen terbesar, centroid
normalisasi, pixel di border3 dan overlap kasus error antarmodel.

Ini deskripsi geometri; tidak menetapkan sebab anatomis, kesalahan label, atau
diagnosis sehat. Tidak melakukan filtering komponen, tuning threshold atau
relabeling. Bila diperlukan adjudikasi klinis di masa depan, gunakan prosedur
terpisah yang independen dari ranking model dan simpan dataset versi lama.

## Aturan seleksi baru sebelum seed43/44 dilatih

Unit ringkasan adalah **seed**, bobot sama. Laporkan mean±sample SD, semua skor
per seed dan paired delta terhadap B1, tanpa memilih seed terbaik. Ini tidak
menggantikan uncertainty akibat pasien, foto, split, atau validation reuse.

Hitung ulang seluruh guard historis **per seed**, dan guard pada **mean tiga
seed**: overall Image Dice harus naik; small Dice/Boundary IoU tidak turun;
total foreground misses dan empty-FP image count tidak naik; active training
parameter ratio≤1.01. Kandidat eligible bila semua guard mean lulus **dan**
setidaknya dua dari tiga seed lulus seluruh guard individual. Seed42 yang gagal
tetap terlihat; hasil baru tidak meloloskan ulang screening historis.

Ranking: highest mean validation Image Dice di antara eligible; tie H2 sebelum
A6; otherwise retain B1. Kunci selection.json **sebelum** inference source-test
baru. Seluruh sembilan model×seed tetap dilaporkan, termasuk yang gagal.
Source-test sudah terekspos; skor test deskriptif, tidak memilih arsitektur,
threshold, seed, epoch, atau jalur penelitian berikutnya.

Routing **untuk review**:

1. Kedua faktor eligible: rekomendasikan preregistration kombinasi terbatas.
2. Mean empty-FP excess berulang: rekomendasikan jalur N, presence supervision.
3. Selain itu: hold/diagnosis; tidak memaksakan matriks atau peningkatan.

Training kombinasi/N, H3, L3, external evaluation dan relabeling **tidak berjalan
otomatis**. Pengguna meminta trainer ditinggal di background setelah mulai;
assistant berhenti bekerja lalu memeriksa/meninjau hasil ketika diminta.

## Eksekusi dan artefak

```bash
.venv-training/bin/python scripts/run_cp2_confirmation.py --root <new-run> --stage prepare
.venv-training/bin/python scripts/run_cp2_confirmation.py --root <new-run> --stage audit
.venv-training/bin/python scripts/run_cp2_confirmation.py --root <new-run> --stage preflight --device cuda:0
# Hanya sesudah gate passed dan source hashes identik:
.venv-training/bin/python scripts/run_cp2_confirmation.py --root <new-run> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_confirmation.py --root <run> --stage status
```

Prepare memerlukan protected_before.json yang dicatat sebelum perubahan.
Preregistration memuat config/source hashes, timestamp, commit dan snapshot.
Tests existing+baru, dependency checks, paired initialization profiles dan tiga
real-data smoke tests harus lulus sebelum full training. Smoke mencakup empty
dan positive masks, forward/backward/optimizer, finite gradients, encoder/new
component updates, validation dan checkpoint save/reload.

Tiap worker memakai subprocess baru, satu GPU sequential. Logs persisten,
active_process.json, train_workflow.json dan status setiap epoch mendukung
monitoring. Last checkpoint menyimpan optimizer/scheduler/RNG; resume perlu
`--resume` eksplisit sesudah inspeksi kegagalan dan verifikasi kompatibilitas.
Completed stages tidak diulang. Tidak mengubah environment atau proses lain.

Setelah semua enam training selesai, runner mengevaluasi validation, menyimpan
keputusan, mengevaluasi test, memverifikasi semua probability/GT pairs serta
checkpoint policies, lalu membuat laporan dan grafik CPU. Semua historical
files diperiksa identik. Artefak penelitian append-only pada
`outputs/cp2/confirmation/<timestamp>/`; checkpoints dan clinical media lokal.
Public output hanya tabel numerik/grafik agregat. Direct visual review tetap
harus dilakukan sebelum menyatakan final delivery terverifikasi visual.
