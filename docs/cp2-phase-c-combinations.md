# CP2 Fase C — B1+A6+H2 dan L3+A6+H2

Preregistrasi sebelum training. Ini lanjutan eksploratif Fase C, **bukan pengganti** hasil A1+H2 yang sudah completed. User secara eksplisit meminta dua kombinasi ini. Seluruh historical checkpoints, metrics, manifest, notebook dan run tetap utuh.

## Dua kondisi baru, satu perbedaan utama

| Run | Main mask objective | Attention | Auxiliary objective | Training baru |
| :--- | :--- | :--- | :--- | :--- |
| B1A6H2 | B1 source Dice + corrected-logit focal | CBAM A6 persis | 0.1 × B1 pada H2 morphological boundary | Satu full seed42 |
| L3A6H2 | L3 = B1 +0.01 signed-distance boundary loss pada main mask | Sama A6 | Sama H2 dan bobot0.1 | Satu full seed42 |

**L3 sudah mencakup B1; B1 tidak dijumlahkan dua kali.** L3 diterapkan hanya pada primary mask. H2 tetap B1 pada target dilation−erosion3×3 dari mask aktual setelah augmentasi; tidak diberi signed-distance loss tambahan. Dua bentuk boundary supervision ini berbeda, dan dapat saling membantu atau redundan.

A6 mengikuti definisi historis: CBAM channel avg+max MLP dan spatial Conv7×7+BN, penggantian attention pada decoder stage2–3, gain3 yang sama. H2 adalah Conv1×1(32,1) di output decoderstage3 [B,32,112,112], bilinear logits ke224 (align_corners=False). Auxiliary logits tidak mengarahkan attention; **bukan H3**. Stage terakhir tetap.

Kedua model mulai dari initializer CP2 yang sama, **bukan menggabungkan trained checkpoints**. Seluruh original tensors diperiksa persis; CBAM tambahan harus cocok historical A6 dan head cocok historical H2. Whole initial model state kedua run identik, main/aux forward CPU bit-exact. Per-run config menambahkan `active_combination` untuk memastikan identity checkpoint berbeda meskipun arsitektur sama.

## Data dan protokol tetap

Manifest671train/118validation/261test, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Duplicate-RGB grouped split, **bukan patient-level**; tidak ada patient IDs. Raw data dan membership tidak diubah.

Seed42, source ImageNet preprocessing/augmentasi224, batch2/workers4, Adam1e-4/WD1e-5, FP32 tanpaAMP, TF32 warisan, dedicated loader generators dan epoch RNG42+10007×index tetap. Earliest strictly highest validation Image Dice memilih checkpoint. Max200epoch, Plateau max-Dice/factor0.1/patience10/minLR1e-5, early stopping30. Evaluator confusion counts biasa, threshold≥0.5, empty rules dan ukuran luka CP2 tetap.

Actual order dan augmented input bytes dipasangkan terhadap B1/A6/H2/L3 serta antara dua run baru pada seluruh common epochs. Model dilatih dengan scheduler/early stopping adaptif yang sama aturannya; jumlah epoch/update aktual dapat berbeda. Jangan menyebut ini eksperimen fixed-update-budget atau pembuktian kausal.

B1/A6/H2/L3 adalah cached completed standalone controls. Metrik/guard/konfirmasi tiga seed terdahulu tidak diubah; A6/H2 bukan kandidat yang sudah lolos semua guard. Pemilihan dua kombinasi kini adalah hipotesis baru yang eksploratif, berdasarkan instruksi user dan source-validation screening, bukan ranking CO2.

Snapshot `phase_a`, `phase_h`, dan `phase_l` dalam config merekam definisi komponen historis utuh; **bukan daftar perintah training baru**. Hanya `combinations.order` dieksekusi, persis dua run di atas.

## Seleksi dan interpretasi

Kedua training dan seluruh fresh validation harus selesai sebelum mengunci keputusan. Seluruh fresh test predictions dilakukan **sesudah selection.json**. Tiap kandidat harus memenuhi guards original terhadap B1: overall Image Dice meningkat, small Dice/Boundary IoU tidak turun, foreground misses dan GT-empty FP images tidak bertambah, active training params≤1.01×B1. Selain itu overall validation harus lebih tinggi daripada standalone relevan: A6/H2 untuk B1A6H2; A6/H2/L3 untuk L3A6H2.

Ranking hanya kandidat eligible berdasarkan validation Image Dice; exact tie memilih B1A6H2 karena main loss lebih sederhana. Jika tidak ada eligible, baseline tetap B1. Perbedaan kedua run baru tetap dilaporkan meskipun tidak lolos semua guard. GT+ recall/Dice, FP pixel counts, ukuran luka, Boundary IoU/PFOM dan costs selalu dilaporkan; tidak mengubah threshold/empty rules setelah melihat hasil.

Kontras: B1A6H2 vs B1/A6/H2; descriptive A6×H2 difference-of-effects; L3A6H2 vs B1A6H2 (tambahan main boundary loss pada paired architecture); L3A6H2 vs L3/A6/H2/B1. **Bukan full2³:** L3+A6 dan L3+H2 tidak dijalankan. Tidak mengestimasi interaksi tiga-arah atau mengklaim sinergi yang teridentifikasi.

Source-test telah terekspos CP1/CP2; hasilnya deskriptif. CO2Wounds juga sudah dilihat sebelumnya dan tidak dipakai memilih konfigurasi/epoch/threshold sekarang. **Tidak ada evaluasi eksternal baru, multiseed, presence, H3 atau ekspansi otomatis** dalam task ini. Kandidat yang menjanjikan masih perlu review/replikasi; tidak ada klaim clinical/patient generalization atau novelty global.

## Eksekusi dan artefak

Persiapan dan structural/unit tests CPU dilakukan sekarang. Worker menunggu **workflow cascade selesai** agar tidak menjalankan dua training lokal bersamaan. Jika workflow cascade failed/hilang, queue gagal jelas dan tidak menyentuh proses lain; inspeksi sebelum explicit resume. Sebelum CUDA training minimal6GiB VRAM tersedia; tidak mengubah batch, precision, environment atau pekerjaan orang lain.

Satu proses menjalankan B1A6H2 lalu L3A6H2 secara berurutan. Tidak ada separate smoke training; preferensi user tetap. Unit fixtures diberi label sintetis. First actual full-training step memeriksa finite/nonzero gradients dan update encoder, kedua channel/spatial CBAM gates, serta head H2. Seluruh step menolak loss/gradient nonfinite. Generic screening loop fase C dipakai unchanged, termasuk optimizer, RNG, strict resume dan checkpoint reload main/aux bit-exact. Auxiliary head dapat dilepas tanpa mengubah primary output.

```bash
.venv-training/bin/python -u scripts/run_cp2_combinations.py \
  --root outputs/cp2/phase_c_combinations/NEW_RUN_ID --stage prepare
.venv-training/bin/python -u scripts/run_cp2_combinations.py \
  --root outputs/cp2/phase_c_combinations/NEW_RUN_ID --stage queue --device cuda:0
```

Sebelum prepare, simpan `protected_before.json` inventory; run baru append-only. Protocol/config/input/code hashes dan source snapshot dipreregister sebelum training. `flock` mencegah worker ganda. Completed stages tidak diulang; interrupted checkpoint hanya `--resume` setelah verifikasi code/config/input/environment. Semua checkpoints/optimizer/RNG/status/log/GPUsamples/prediksi tersimpan lokal.

Setelah dua training: paired input QA→validation-only selection→test predictions→independent verification2274 cached+fresh prediction records→aggregate reports/tables/figures. Report `reports/cp2-phase-c-combinations.md` tidak dibuat sebagai hasil completed sebelum eksekusi ini selesai. **Visual review actual PNG masih wajib setelah hasil tersedia.** Raw media, masks, probability arrays dan model weights tetap lokal. Tidak ada automatic merge atau push main.
