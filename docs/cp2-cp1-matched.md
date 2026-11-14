# CP2 dengan protokol CP1 asli: hanya faktor metode yang berubah

## Koreksi cakupan

Pengguna menegaskan bahwa “konfigurasi CP1” mencakup **seluruh protokol dan
split CP1**, bukan sekadar resep training. Karena itu eksperimen ini terpisah
dari run terdahulu `cp1_recipe/20261113T082016Z`, yang memakai split CP2
671/118/261. Hasil terdahulu tetap dipertahankan dan tidak diganti namanya.

Kontrol di sini memakai Chronic Wound **707 train / 124 validation / 278 test**,
dengan **membership dan urutan persis manifest CP1**. Tidak melakukan shuffle
split baru, filtering, deduplikasi, koreksi label, atau perubahan raw data.

Manifest fingerprint:
`7937d6b039db4eedf736dd0482a6379917fd9640d39d03c2c5fdac5047166925`.

## Matriks yang dibekukan sebelum training penuh

| ID | Faktor yang diuji | Semua kontrol lain |
|---|---|---|
| C0 | FUSegNet/P-scSE sumber, fresh control | CP1 asli |
| CH2 | Auxiliary boundary head pada decoder stage 3 | CP1 asli; main loss tidak berubah |
| CA6 | CBAM menggantikan P-scSE pada decoder stage 2–3 | CP1 asli |

H2 dan CBAM tetap eksperimen terpisah, tanpa kombinasi atau pencarian konfigurasi
tambahan. Kandidat berasal dari screening sebelumnya dan hasilnya eksploratif.
C0 baru diperlukan untuk memverifikasi inisialisasi serta input training yang
dipasangkan; historical CP1 checkpoint dan skor tetap menjadi referensi lama.

## Konfigurasi yang harus identik

Seluruh key/value dari `outputs/runs/cp1-baseline-01/config.json` disalin tanpa
perubahan. Metadata `matched_experiment` hanya mengidentifikasi faktor/run.

- **Data:** file, order dan split 707/124/278 asli, gambar 224×224.
- **Model dasar:** EfficientNet-B7 pretrained ImageNet, encoder trainable,
  decoder sumber dan output **sigmoid asli**. Tidak mengubahnya menjadi logits.
- **Loss utama:** original `DiceLoss() + FocalLoss()` termasuk probability-input
  mismatch sumber. Alpha, gamma, epsilon, reduction dan agregasi tidak berubah.
- **Processing:** Dataset, ImageNet preprocessing dan augmentasi sumber yang sama.
- **Training:** original `TrainEpoch` dan `ValidEpoch`, Adam LR 1e-4,
  weight decay 1e-5, batch 2, workers 4, max 200 epochs, seed 42 sekali.
- **Precision:** environment CP1, FP32, AMP off, TF32 flags dan cuDNN settings
  sama. Tidak memasang/mengganti dependency atau menaikkan batch size.
- **Scheduler:** source ReduceLROnPlateau main validation loss, mode min,
  factor 0,1, patience 10, min LR 1e-5 beserta defaults CP1.
- **Checkpoint/stop:** source `if loss improves ... elif IoU improves`, termasuk
  stale IoU tracker; stop 30 epoch tanpa source-policy save. Tidak mengganti
  kebijakan menjadi best Image Dice.
- **Evaluasi:** source offline evaluator ≥0,5 dan empty rules yang sama;
  inline-legacy tetap dilaporkan terpisah. Training batch-IoU tetap >0,5.

Implementasi mengimpor engine sumber tanpa mengedit berkasnya. C0 menjalankan
bytecode `run_training` yang sama. Pada CH2/CA6, hanya **konstanta assertion jumlah
parameter** pada AST yang disesuaikan dengan arsitektur baru; seluruh statement
training lain sama. Langkah matematis optimizer, scheduler, loss utama, loaders
dan checkpoint bukan implementasi ulang. Unit tests memeriksa batas patch ini.

## Definisi tepat perubahan metode

**CH2:** head `Conv2d(32,1,1)` membaca fitur decoder stage 3 `[B,32,112,112]`.
Head menghasilkan probabilitas boundary pada 224×224. Target ialah morphological
gradient 3×3 dari GT sesudah augmentasi geometrik, exterior zero. Training
menggunakan `main source loss + 0,1 × auxiliary source loss`; auxiliary juga
mempertahankan primitives/mismatch sumber agar main objective tidak ikut berubah.
Validation dan scheduler/checkpoint tetap memakai **main source loss**. Head
tidak dijalankan pada inference; output mask tetap dari jaringan sumber tanpa
GT sebagai input. Tambahan supervisi merupakan faktor fase H yang memang diuji.

**CA6:** memakai definisi CBAM/gain 3 dan stage 2–3 yang sudah diuji pada fase A.
Final decoder stage tidak berubah. Seluruh tensor sumber tetap terdaftar;
modul yang tidak aktif tidak dihapus. Perubahan attention mencakup perubahan
forward/gain dari pembanding tersebut; ini bukan klaim kapasitas atau distribusi
aktivasi identik dengan P-scSE.

## Inisialisasi dan observasi tanpa perubahan RNG

Setiap arm dibangun dalam **fresh Python worker**. Original reference constructor
dipanggil sebelum import lazy helper kandidat. Hash tensor sumber harus cocok
dengan rekonstruksi CP1 historis:
`fd3fe8b007a897b5c3a977b8bbafec662fed4bdf6489d8a51a18376a4bb7e031`.
Tensor pasangan juga dibandingkan langsung dengan satu snapshot source initial
weights. Head/CBAM tidak boleh menimpa tensor sumber atau mengubah state RNG
Python, NumPy, CPU/CUDA sebelum training. Ini fresh ImageNet initialization,
bukan melanjutkan trained CP1 checkpoint.

Observer membaca indices dari sampler asli dan tensor yang benar-benar di-load;
tidak mengganti sampler, epoch seed, atau algoritma shuffle. Order dan augmented
bytes dibandingkan pada seluruh common epochs. Smoke CPU reference vs observed C0
memeriksa original engine step/state/logs secara exact, kemudian tiga smoke GPU
memeriksa eksekusi nyata. Perbedaan bitwise GPU tidak diselesaikan dengan
mengubah precision atau determinism flags. Kesamaan konfigurasi
tidak menjamin seluruh GPU training mengulang angka historis bit-for-bit;
determinism/precision flags tidak diubah untuk memaksakan hasil tersebut.

## Seleksi dan batas ilmiah

Epoch dipilih oleh **aturan CP1 asli**. Sesudah seluruh training selesai,
perbandingan antar-model memakai validation saja, dengan guard riset sebelumnya
terhadap fresh C0: overall Image Dice lebih tinggi; small Dice dan Boundary IoU
tidak lebih rendah; tidak ada tambahan foreground miss atau GT-empty FP images;
active training parameters ≤1,01×C0. Eligible tertinggi dipilih, tie CH2 lalu CA6;
bila tidak ada, pertahankan C0. Guard ini bukan standar klinis.

Subgroup small adalah GT >0 sampai 1% full padded canvas; metrik tambahan ini
tidak mengubah evaluator primer atau checkpoint CP1. Test 278 gambar baru
diproses setelah `selection.json` tersimpan; test lama sudah pernah dianalisis.

Audit original CP1 menemukan **30 duplicate-RGB groups, 18 cross-split groups,
dan 28 conflicting-mask groups** pada membership train/validation/test asli.
Semua dipertahankan untuk perbandingan yang diminta. Hasil tidak boleh disebut
clean split, patient-independent validation, atau bukti generalisasi klinis.
Source split paper dan evaluator headline tetap belum diketahui. Satu seed dan
pengembangan sebelumnya juga membatasi klaim peningkatan.

## Menjalankan dan memantau

```bash
.venv-training/bin/python scripts/run_cp1_matched.py \
  --root outputs/cp2/cp1_matched/<new-run> --stage prepare
.venv-training/bin/python scripts/run_cp1_matched.py \
  --root outputs/cp2/cp1_matched/<new-run> --stage preflight --device cuda:0
.venv-training/bin/python scripts/run_cp1_matched.py \
  --root outputs/cp2/cp1_matched/<new-run> --stage train --device cuda:0
```

Sebelum `prepare`, simpan inventori `protected_before.json` berisi SHA256 semua
artefak/user work yang perlu dilindungi. Root tiap run append-only terpisah.
Preflight menjalankan full tests, package/source checks, profile arsitektur,
source-step control dan smoke real-data semua arm. Training memakai satu child
GPU worker berurutan, C0 → CH2 → CA6. `active_process.json` menyimpan PID/log
child; setiap model memiliki `status.json`, `history.json` dan input hashes.

`--resume` hanya untuk run terinterupsi dengan source/config/environment yang
sama. Completed stages tidak dilatih ulang. Setelah training/evaluation lengkap,
validator memeriksa checkpoint, membership, source policy dan prediction exports;
report renderer membuat grafik serta metrik. Raw data, NPZ, checkpoints dan
environment tetap lokal. Tidak ada notebook lama yang diedit/dieksekusi ulang.
