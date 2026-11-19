# Cascade Ganta pada split bersih Orie — preregistrasi

Pertanyaan: apakah zoom/refinement memperbaiki segmentasi dibanding coarse yang sama pada split CP2 yang dibersihkan, dan bagaimana transfer keduanya ke CO2Wounds?

Ini **adaptasi desain cascade Ganta ke CP2**, bukan replay checkpoint historis partner. Checkpoint partner memakai split historis dengan overlap, sehingga tidak dipakai. Seluruh sumber, media, checkpoint partner dan artefak CP1/CP2 lama tetap tidak diubah. Tidak ada klaim patient-level split atau novelty global.

## Kondisi dan unit evaluasi

- Coarse: **B1 CP2 seed42 completed**, checkpoint epoch25 yang dipilih dari validation. Seluruh bobot dan BatchNorm dibekukan. Tidak dilatih ulang.
- Refiner: FUSegNet/EfficientNet-B7/P-scSE dari definisi Ganta yang diaudit. Seluruh active initial tensors dipetakan dari **initializer asli bersama CP2**, bukan dari trained B1 atau trained Ganta. Branch additive memakai source attention2, max memakai attention1; conv1 sumber dan head dipetakan ketat. Forward awal actual training inputs dibandingkan dengan implementasi sumber, bit-exact pada CPU.
- Tepat **satu full training seed42** untuk refiner. Tidak ada separate smoke training, perubahan loss, augmentation search, AMP, batch scaling, atau multiseed.
- Manifest CP2 identik: **671train / 118validation / 261test**, fingerprint `fa9f4914a55806248d25c4a4092361db2658bfc63a0031122b63028202ef8ed5`. Tidak membentuk split baru. Konflik mask dan duplikasi RGB mengikuti manifest lama; bukan patient-level.
- Setiap prediksi merepresentasikan satu foto utuh. Crop tidak dihitung sebagai foto/pasien independen.

## Desain crop dan proposal tetap

Desain Ganta dipertahankan: augmentasi training sebelum crop; bbox GT **training saja**, margin acak0.3–0.8, peluang random crop0.2, sisi minimum0.25 sisi gambar, resize224. GT kosong mendapat random crop sesuai algoritma asli. GT tidak pernah dipakai menentukan crop validation/test/CO2.

Saat inference: coarse probability≥0.3, komponen8-connected dengan≥4piksel, maksimal5 komponen terbesar, margin0.5. Probability refiner diresize ke kotak asal; maksimum antaroverlap, menggantikan coarse pada area tertutup crop; di luar tetap coarse. Tanpa proposal: coarse dikembalikan. Seluruh probabilitas dicek finite dan final threshold≥0.5.

## Training dan pemilihan checkpoint

Loss **B1**: source Dice(sigmoid(logits),mask)+corrected FocalWithLogits(logits,mask), parameter sama faseP. Batch2, workers4, AdamLR1e-4/WD1e-5, FP32 tanpaAMP, TF32 sama proyek. EpochRNG42+10007×index, generator loader terpisah. Augmentasi sumber tidak diubah. Cropping mengubah input refiner sehingga augmented bytes **tidak** diklaim berpasangan dengan training full-image B1; membership dan initializer yang dipasangkan.

**Perubahan penting dari historical Ganta:** checkpoint, scheduler dan early stopping memonitor **mean Image Dice cascade pada 118 foto validation utuh**, bukan crop bbox GT validation. Proposal validation hanya dari coarse yang dibekukan. Coarse validation dihitung sekali pada CPU batch1 dan disimpan immutable; refiner validation saat training memakai device training. Max200epoch, earliest strictly best Image Dice, ReduceLROnPlateau(max,factor0.1,patience10,minLR1e-5), early stopping30. Tidak memilih threshold/proposal config dari test atau CO2.

Checkpoint lengkap menyimpan optimizer/scheduler/RNG/identity. First actual optimizer step memeriksa gradien finite/nonzero dan update encoder serta segmentation head; seluruh step memeriksa gradien finite. Setelah training: strict reload, repeat prediction dan seluruh validation score pada device training harus cocok. Tidak mengklaim GPU bitwise determinism antarmesin/versi.

## Seleksi klinis dan evaluasi akhir

Setelah best checkpoint dipilih, coarse/cascade validation diulang di CPU dengan jalur inference final. Angka GPU training dan CPU replay disimpan bila berbeda. Epoch tidak dipilih ulang dari replay CPU.

Cascade diterima untuk investigasi lanjutan hanya jika seluruh validation guards lulus:

1. Overall Image Dice lebih tinggi dari coarse.
2. Small-wound Image Dice dan Boundary IoU tidak turun.
3. Foreground misses dan jumlah GT-empty images denganFP tidak bertambah.

Semua metrik (termasuk GT+ recall, GT-empty FPpixels dan kategori ukuran) tetap dilaporkan. Kapasitas1% untuk attention tunggal **tidak diterapkan** pada eksperimen dua-model ini; biaya tambahan adalah bagian eksplisit hipotesis. Coarse/refiner masing-masing64,898,248 active parameters; total129,796,496, inference1+jumlahproposal forwards. Source model juga mendaftarkan tensor yang tidak dipanggil, sehingga jumlah registered berbeda.

Keputusan dikunci sebelum source test/CO2 inference. Kedua pipeline tetap dievaluasi sebagai diagnostic paired comparison jika cascade gagal clinical guards; kandidat gagal tidak disebut model terpilih. Source test sudah terpapar pengembangan CP1/CP2; CO2 telah dipakai untuk analisis transfer terdahulu. Ini eksploratif, bukan confirmatory test baru.

## CO2Wounds

Gunakan **558 eligible gambar identik** dengan evaluasi eksternal sebelumnya (Mendeleyv1);49 konflik dari607 berlabel dikarantina;157 tanpa label dikecualikan. Fingerprint `63dfbab34137922985d5ff48fda36ba4264197fa1d3cee20d8bd5a5b2016b68f`. Input canonical letterbox224+ImageNet yang sama dan frozen; unpad dan resize probabilities bilinear ke native mask sebelum threshold. Refiner memakai proposal pada input letterbox, bukan bbox target.

Coarse dan cascade memakai population/grid/evaluator yang sama. Tidak ada fitting, konfigurasi, threshold, early stopping atau pemilihan checkpoint dari CO2. Dataset sebelumnya diaudit tanpa exact/strict near overlap terhadap sumber. Seluruh GT eligible eksternal positif: tidak mengukur eksternal false-positive GT kosong. Perbedaan domain/label/grid dan ketidakjelasan lisensi yang terdokumentasi tetap berlaku; semua media/probabilities/checkpoint tetap lokal.

## Resource dan eksekusi

GPU sedang digunakan PhaseC dan pekerjaan orang lain. Persiapan, cache validation coarse dan tests memakai CPU terbatas. Worker queue **menunggu PhaseC workflow completed**; jika failed/hilang, hentikan dengan error dan jangan launch training diam-diam. Sebelum mulai minimal8GiB VRAM tersedia. Tidak menghentikan job lain atau mengubah environment. GPU shared dicatat; timing bukan latency eksklusif.

```bash
.venv-training/bin/python -u scripts/run_cp2_cascade.py \
  --root outputs/cp2/cascade/NEW_RUN_ID --stage prepare

.venv-training/bin/python -u scripts/run_cp2_cascade.py \
  --root outputs/cp2/cascade/NEW_RUN_ID --stage queue --device cuda:0
```

Run baru harus mempunyai `protected_before.json` inventory yang dibuat sebelum perubahan. Worker memakai `flock`, status/log persisten dan sumber/input yang dipreregister. Partial stages menolak overwrite; inspected `--resume` hanya untuk interrupted checkpoint dengan identity identik. Completed training tidak diulang. Partial evaluation membutuhkan inspeksi/isolasi output sebelum pemulihan; jangan menimpa completed artifacts.

Setelah training: source-validation replay→selection.json→source-test→CO2→independent probability/GT/threshold/CSV validation→aggregate reports/figures. **Visual review PNG masih perlu dilakukan saat hasil sudah tersedia**, tidak diklaim selesai otomatis. Tidak ada GitHub push atau publikasi media otomatis.
