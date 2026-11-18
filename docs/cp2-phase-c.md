# CP2 Fase C — interaksi A1 dan H2, screening terbatas

Preregistrasi sebelum full training, branch `feat/cp2-phase-c`, dari fase N
commit `18eb364`. Seluruh hasil P/L/A/H/konfirmasi/N/CP1 tetap utuh.
[Config](../configs/cp2/phase_c.json).

## Pertanyaan dan alasan

Apakah local attention A1 dan auxiliary boundary supervision H2 saling melengkapi
ketika dilatih bersama? A1/H2 mempunyai sinyal manfaat pada source validation,
tetapi masing-masing gagal guard GT-empty FP. Konfirmasi H2 tiga seed juga belum
meloloskan kandidat. N1 gagal mengurangi FP dan tidak dimasukkan.

Pengguna mengizinkan fase C sebagai **hipotesis interaksi eksploratif baru**.
Kelolosan individual tidak diasumsikan, dan keputusan screening lama tidak
diubah. A1 dipilih sebagai cabang lokal paling sederhana dengan fixed fusion,
bukan karena ranking CO2Wounds. H2 adalah faktor supervisi batas yang dapat
diimplementasikan mandiri. Loss tetap B1; belum ada L3 atau full matrix2³.

Hipotesis: representasi lokal dan target batas dapat saling melengkapi. Hasil
negatif yang juga sah: keduanya redundan, mengganggu optimisasi, atau bersama-sama
meningkatkan FP. Interaksi yang positif pada satu metrik tidak otomatis berarti
overall Dice terbaik, guard lolos, atau peningkatan yang konsisten.

## Empat kondisi, satu training baru

| Kondisi | Local A1 | Boundary H2 | Eksekusi |
| :--- | :---: | :---: | :--- |
| B1 | — | — | Cached fase P seed42 |
| A1 | ✓ | — | Cached fase A seed42 |
| H2 | — | ✓ | Cached fase H seed42 |
| A1H2 | ✓ | ✓ | Tepat satu full training seed42 |

Nama baru `A1H2` berada di direktori fase C tersendiri; tidak menggunakan atau
menimpa `C0/CH2/CA6` dari CP1-matched. Model baru mulai dari shared initial state
fase P, **bukan** dari checkpoint terlatih salah satu kontrol.

A1 memakai implementasi historis utuh: DWConv3×3 → PWConv1×1 → sigmoid,
satu spatial map dikalikan fitur, fixed fusion dan gain3 di decoder stage2–3.
Faktor A1 mencakup cabang lokal beserta konvensi normalisasi fusion tersebut;
selisih terhadap P-scSE tidak mengisolasi local map dari seluruh efek skala fitur.
Decoder terakhir tetap. H2 mengambil output stage3 `[B,32,112,112]`,
Conv2d(32,1,1), upsample logit bilinear ke224×224 dengan align_corners=False.
Targetnya morphological gradient dilation−erosion3×3, exterior zero,
dihitung dari mask aktual sesudah augmentasi.

Objective: **B1(main,mask) +0,1×B1(aux,boundary)**. B1 berarti source Dice
pada sigmoid logits ditambah corrected-logit focal, tanpa perubahan epsilon,
alpha/gamma/reduction. H2 tidak memberi input ke local map; **ini bukan H3**.
Auxiliary boundary head dapat dilepas saat mask-only inference.

## Kontrol yang dipasangkan

Manifest CP2 tetap **671 train /118 validation /261 source-test**, fingerprint
`fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`.
Split berdasarkan grup decoded-RGB; tidak ada patient IDs dan tidak disebut
patient-level split. Semua pasangan foto/mask diperiksa dengan hash manifest.

Source tensors dimuat persis shared initialization, local tensors/prior persis
initial A1, dan auxiliary head persis initial H2. Fingerprint tensor serta
kesetaraan named tensors diperiksa. Primary forward awal B1=H2 dan A1=A1H2
harus bit-exact pada structural CPU fixture. Fixture diberi label sintetis,
bukan skor ilmiah atau training data tambahan.

Seed42, batch2/workers4, ImageNet preprocessing, augmentasi source, dedicated
loader generators dan epoch RNG `42+10007×zero_based_epoch` tetap. Actual order
dan augmented input bytes dibandingkan pada **semua common epochs** terhadap
ketiga kontrol, tidak sekadar seed. Adam1e-4/WD1e-5, FP32 tanpa AMP, TF32
warisan, max200, Plateau validation-Dice factor0,1/patience10/minLR1e-5,
early stopping30 dan earliest strictly highest validation Image Dice identik
dengan fase P/L. Jumlah update aktual boleh berbeda akibat aturan adaptif sama.

## Kontras dan seleksi yang dikunci

Laporkan A1H2−B1, A1H2−A1 dan A1H2−H2, serta interaksi
**A1H2−A1−H2+B1**. Hitung untuk Image/Data Dice, GT-positive Dice,
small Dice, Boundary IoU, positive-pixel recall, total foreground miss,
GT-empty FP image count dan pixel count. Rasio/delta dilabeli satuan dengan benar.
Interaksi ini selisih efek recipe pada satu seed, bukan bukti sinergi kausal
atau uji signifikansi; optimization budget/checkpoint berbeda dan metrik nonlinear.

Seleksi menggunakan **source validation saja**, dengan original CP2 guards:

1. Overall Image Dice >B1.
2. Small Image Dice dan Boundary IoU ≥B1.
3. Total foreground misses dan GT-empty FP images ≤B1.
4. Active training parameters ≤1,01×B1.

Untuk menunjukkan nilai tambahan kombinasi, overall validation Image Dice juga
harus **lebih tinggi daripada A1 dan H2**. Bila tidak seluruhnya lulus,
baseline tetap B1. Recall, GT-positive Dice dan FP pixel area selalu dilaporkan
untuk meninjau trade-off; jangan mengabaikannya hanya karena overall membaik.
Exact numerical ties tidak dianggap peningkatan.

Persist selection dan validation interaction sebelum semua new source-test
probabilities. Source-test sudah terekspos, hanya deskriptif; tidak dipakai untuk
menentukan arsitektur, epoch, seed atau perluasan. CO2Wounds sudah dilihat dan
tidak dipakai memilih konfigurasi C. Tidak ada evaluasi eksternal baru pada run
ini. Pengembangan setelah melihat target memerlukan evaluasi independen lain.

Jika eligible, review hasil lalu pertimbangkan replikasi sebelum klaim peningkatan.
Tidak ada N, CBAM, H3, L3, multiseed atau kombinasi lain yang berjalan otomatis.

## Engineering dan eksekusi

Tidak ada smoke training terpisah, mengikuti preferensi pengguna. Unit tests,
integrity dan structural forward CPU diperiksa sebelum full training. First
actual optimizer step memeriksa finite/nonzero gradient dan update encoder,
dua local branches, serta boundary head. `CheckedAdam` melakukan update yang
identik dengan Adam biasa; kesetaraan diuji. Semua steps menolak nonfinite,
dan checkpoint main/aux reload bit-exact diperiksa setelah training selesai.

Environment `.venv-training` dipertahankan, tanpa dependency/precision/batch
fallback. Satu job GPU shared; tidak mengubah atau menghentikan job orang lain.
Monitor pemakaian keseluruhan dan peak allocated/reserved; free VRAM minimal6GiB
saat launch. CPU/GPU timing bersama bukan latency benchmark paper.

```bash
.venv-training/bin/python scripts/run_cp2_phase_c.py --root <new-run> --stage prepare
.venv-training/bin/python scripts/run_cp2_phase_c.py --root <new-run> --stage preflight
.venv-training/bin/python scripts/run_cp2_phase_c.py --root <new-run> --stage register
.venv-training/bin/python -u scripts/run_cp2_phase_c.py --root <new-run> --stage train --device cuda:0
.venv-training/bin/python scripts/run_cp2_phase_c.py --root <run> --stage status
```

Simpan `protected_before.json` sebelum perubahan. Registry menyimpan hash kode,
config, cached checkpoints, initialization dan manifest sebelum training. New
run append-only `outputs/cp2/phase_c/<timestamp>/`; status/log/checkpoint/RNG
persisten. Resume eksplisit hanya sesudah pemeriksaan kompatibilitas code,
config/input/environment/checkpoint; completed stages tidak dilatih ulang.

Setelah training, worker menjalankan validation → selection → exposed source-test
→ independent QA semua empat kondisi → laporan dan grafik agregat. Manual visual
review tetap pending sampai figure diperiksa langsung. Data klinis, masks,
probability maps dan checkpoint tetap lokal. Training ditinggal di background
setelah launch dan pemeriksaan liveness, sesuai preferensi pengguna.
