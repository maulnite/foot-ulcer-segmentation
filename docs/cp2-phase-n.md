# CP2 Fase N — presence supervision mandiri

Preregistrasi sebelum full training. Branch `feat/cp2-phase-n`, dari commit
`9847e22be2fc7c7f876d5fe692ea4bb0a223fc90`. Seluruh hasil lama dipertahankan.
[Config](../configs/cp2/phase_n.json).

## Pertanyaan dan bukti awal

Apakah auxiliary foreground-presence supervision mengurangi prediksi luka
pada anotasi kosong, tanpa menambah luka yang terlewat? Konfirmasi tiga seed
sebelumnya mempertahankan B1: H2/A6 belum lolos seluruh guard. Pada seed42,
B1 mempunyai FP pada 4/7 validation GT-empty, H2/A6 5/7. Ukuran sampel kosong
kecil; selisih ini belum membuktikan sebab anatomis atau signifikansi klinis.

Hipotesis N1: tugas global mendeteksi keberadaan foreground dapat memperbaiki
representasi bersama untuk segmentasi. Hipotesis negatif yang juga sah: head
hanya belajar menebak mayoritas positif, tidak membantu mask, atau mengganggu
recall/luka kecil. Semua hasil dipertahankan.

CO2Wounds sudah dilihat. Dataset itu tidak dipakai memilih bobot loss, tap
feature, threshold, epoch, atau penerimaan kandidat baru. N1 merupakan screening
eksploratif source-validation; generalisasi konfirmatori perlu data independen.

## Matriks yang dikunci

| Kondisi | Arsitektur dan loss | Training baru |
| :--- | :--- | :--- |
| N0/B1 | Source FUSegNet/P-scSE, source Dice + corrected-logit focal | Tidak; cached phase P seed42 |
| N1 | N0 + global-average-pooling fitur decoder stage3 + Linear(32,1), B1 + 0,1 BCEWithLogits(mean) | Tepat satu full run seed42 |

Tap read-only stage3: `[B,32,112,112] → [B,32] → [B,1]`. Tambahan 33 parameter.
Tidak ada dropout/BN/attention tambahan. Head diinisialisasi dengan RNG CPU
terisolasi seed42042, tanpa mengonsumsi RNG global. Seluruh parameter dan buffer
base dimuat dari **shared initialization fase P**, bukan best checkpoint B1.
Kesetaraan diperiksa tensor-per-tensor dan primary forward awal harus bit-exact.

Presence target adalah `any(actual_augmented_mask == 1)` per gambar. Luka satu
piksel tetap positif. Mask yang terpotong seluruhnya oleh augmentasi menjadi
presence negatif sesuai input aktual. BCE tanpa pos_weight, class balancing,
label smoothing atau pencarian weight; ketidakseimbangan kelas dilaporkan.
Presence berarti foreground anotasi, bukan diagnosis medis.

Saat inference, mask tetap `sigmoid(main_logits) >= 0.5`. Presence probability
hanya diagnostik: **tidak ada hard gate atau soft modulation** pada tahap ini.
Head dapat dilepas tanpa mengubah primary prediction.

## Kontrol, split dan training

Manifest CP2 tetap **671 train /118 validation /261 source-test**, fingerprint
`fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`.
Kelompok decoded-RGB tidak menyeberang split; patient ID tidak tersedia sehingga
tidak disebut patient-level split. Raw images/masks tidak diubah.

Seed42, inisialisasi base, preprocessing, augmentasi, dedicated loader generators,
epoch RNG `42 +10007 ×zero_based_epoch`, batch2/workers4, Adam1e-4/WD1e-5,
FP32 tanpa AMP dan TF32 warisan identik dengan fase P/L. Early stopping30,
max200 epoch; scheduler Plateau atas validation Image Dice, factor0,1,
patience10/minLR1e-5. Earliest strictly highest validation Image Dice memilih
checkpoint; auxiliary loss/accuracy tidak menentukan checkpoint. Evaluator,
empty rules, threshold≥0,5, Boundary IoU dan kategori ukuran tetap.

Actual sample order dan augmented input bytes diverifikasi terhadap setiap
common epoch B1, tidak sekadar mengandalkan seed. Jumlah epoch/update aktual
dapat berbeda karena validation dan early stopping, sesuai baseline.

## Pemeriksaan tanpa smoke training terpisah

**Instruksi pengguna terbaru:** langsung satu full training seed42, tanpa run
smoke training tambahan. Unit tests dan structural forward CPU tetap dilakukan
sebelum training; input sintetis pada unit/shape checks diberi label engineering,
bukan data eksperimen atau skor keberhasilan metode.

First real full-training optimizer step memeriksa gradien finite dan pembaruan
encoder/presence weight/bias. Wrapper Adam melakukan update yang identik dengan
Adam biasa; kesetaraan diuji. Setiap langkah tetap menolak loss/gradient nonfinite.
Validation dan penyimpanan checkpoint berlangsung dalam training sebenarnya,
lalu checkpoint main/presence diuji reload bit-exact. Tidak ada fallback batch,
precision, loss, atau seed diam-diam saat kegagalan.

## Aturan seleksi sebelum source-test

N1 dibanding N0 pada **source validation**, checkpoint masing-masing terpilih
dengan aturan tetap. Kandidat eligible hanya bila seluruh kondisi terpenuhi:

1. Overall Image Dice meningkat.
2. Small-wound Dice dan Boundary IoU tidak turun.
3. Positive-pixel recall tidak turun; foreground misses tidak bertambah.
4. Jumlah GT-empty images dengan FP dan total GT-empty FP pixels tidak bertambah.
5. Sedikitnya salah satu dari FP image count atau FP pixel count berkurang.
6. Active training parameters ≤1,01× baseline.

Guard historis tetap; recall dan FP pixel guards ditambahkan sebelum training
karena sasaran eksperimen adalah kontrol FP tanpa mengorbankan foreground.
Hanya tujuh GT-empty validation: hasil satu seed adalah sinyal awal, bukan
konfirmasi klinis. Jika gagal, baseline tetap B1 dan lanjut diagnosis.

Persist `selection.json` sebelum prediksi source-test baru. Source-test sudah
terekspos, sehingga hasilnya deskriptif dan tidak dipakai meloloskan ulang
kandidat. Tidak ada pemilihan berdasarkan CO2Wounds atau seed terbaik.

Jika eligible, hasilnya dibawa ke review konfirmasi sebelum kombinasi. **Tidak
ada training N2, H2/A, H3, L3, multiseed atau eksternal otomatis.** Soft modulation
memerlukan studi berikutnya serta pembanding perubahan threshold yang dipilih
hanya dari source validation.

## Eksekusi dan artefak

Environment proyek `.venv-training`; tidak ada dependency baru atau perubahan
shared environment. GPU shared; satu proses training, minimal6GiB free saat
launch, monitor penggunaan keseluruhan/peak allocated dan reserved. Tidak ada
proses pengguna lain yang dihentikan atau diubah.

```bash
.venv-training/bin/python scripts/run_cp2_phase_n.py --root <new-run> --stage prepare
.venv-training/bin/python scripts/run_cp2_phase_n.py --root <new-run> --stage preflight
.venv-training/bin/python scripts/run_cp2_phase_n.py --root <new-run> --stage register
.venv-training/bin/python -u scripts/run_cp2_phase_n.py --root <new-run> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_n.py --root <run> --stage status
```

Simpan baseline inventory `protected_before.json` sebelum mengedit. Registration
membekukan hash kode/config dan menyimpan source snapshot sebelum full training.
Completed stages tidak dilatih ulang; resume memerlukan inspeksi dan `--resume`
eksplisit serta code/config/input/environment/checkpoint yang cocok. Initial
partial run yang belum mempunyai checkpoint tidak ditimpa.

Run baru append-only `outputs/cp2/phase_n/<timestamp>/`. Simpan best/last checkpoint,
optimizer/scheduler/RNG, log persisten, status epoch, input pairing, GPU monitor,
prediksi validation/test, presence diagnostics dan QA. Setelah training selesai,
runner mengekspor metrik serta laporan/grafik agregat secara otomatis; **visual
review manual tetap pending** sampai grafik diperiksa langsung.

Training ditinggal di background setelah launch sesuai preferensi pengguna.
Tidak ada checkpoint/model/result lama yang ditimpa. Data klinis dan probabilities
tetap lokal; laporan publik hanya kode, numerical tables dan grafik agregat.
