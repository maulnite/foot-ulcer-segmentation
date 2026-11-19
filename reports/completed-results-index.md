# Hasil penelitian yang sudah selesai

Snapshot publikasi: **2026-11-19T15:25:49.196236+00:00**, branch `feat/cp2-phase-c`. Halaman ini menghubungkan hasil yang sudah tersedia tanpa menimpa hasil historis atau mengubah training aktif.

## Eksperimen selesai dan tervalidasi

| Kelompok | Cakupan dan status | Laporan |
| :--- | :--- | :--- |
| CP2 fase P | B0/B1 dan protokol split bersih; completed | [Fase P](cp2-phase-p.md) |
| CP2 fase L | Screening loss; completed | [Fase L](cp2-phase-l.md) |
| CP2 fase A | Screening attention; completed | [Fase A](cp2-phase-a.md) |
| CP2 fase H | Auxiliary heads; completed | [Fase H](cp2-phase-h.md) |
| Resep CP1 pada split CP2 | R0/RH2/RA6; completed | [Resep CP1](cp2-cp1-recipe.md) |
| Resep **dan split CP1** | C0/CH2/CA6 pada707/124/278; ketiga training/evaluasi completed | [CP1 matched](cp1-matched.md) |
| Confirmation | B1/H2/A6, seed42/43/44; completed | [Confirmation](cp2-confirmation.md) |
| CP2 fase N | Presence supervision, tidak meloloskan guard; completed | [Fase N](cp2-phase-n.md) |
| Screening awal fase C | A1+H2; completed, B1 dipertahankan | [A1+H2](cp2-phase-c.md) |
| Transfer eksternal | **26 checkpoint ×558 CO2Wounds =14.508 prediksi**, tanpa retraining; completed | [Evaluasi eksternal](cp2-external-evaluation.md) |
| Cascade **asli Ganta** | Replay dua checkpoint asli pada278 test dan CPU timing; completed | [Replay Ganta](ganta-cascade-replay.md) |
| GT kosong | Ganta asli17 test negatives dan diagnostik sementara CP2 pada7 validation negatives; kedua inference completed | [Audit GT kosong](cascade-empty-gt-audit.md) |

Perbandingan C0/CH2/CA6 memakai **707/124/278** dengan overlap historis tetap dipertahankan. Kelompok CP2 memakai **671/118/261**, berbasis RGB duplikat, bukan patient-level split. Hasil dari dua cohort tidak disajikan sebagai satu perbandingan terkontrol. Seluruh evaluasi eksternal memakai cohort558 yang sama, tetapi tidak memiliki GT kosong.

Metrik per gambar ikut dibagikan: [C0/CH2/CA6](tables/), [26 model eksternal](tables/cp2-external-per-image/), dan [replay cascade asli](tables/ganta-cascade-replay-cascade-per-image.csv). Training histories CP1-matched ada di [training/](training/).

Review langsung tiga figure CP1-matched telah selesai pada publikasi ini: [bukti review](cp1-matched-presentation-review.json). Field PENDING pada JSON/report snapshot lama dipertahankan karena file tersebut dikunci inventory training aktif; catatan review tambahan ini melengkapi pemeriksaan visualnya.

## Tahapan yang belum selesai pada snapshot

- **B1+A6+H2:** training selesai67 epoch, checkpoint37, validation Image Dice84,2436%; strict checkpoint replay lulus. Perbandingan lengkap dengan L3, seleksi bersama dan evaluasi test masih menunggu workflow berikutnya. Ini bukan hasil akhir seluruh eksperimen kombinasi.
- **L3+A6+H2:** training masih berjalan pada snapshot; belum ada hasil test yang dipublikasikan.
- **Adaptasi cascade Ganta ke Orie CP2:** training masih berjalan. Audit7 GT kosong memakai snapshot epoch27; bukan checkpoint final dan bukan hasil test/CO2.

[Snapshot machine-readable tahapan aktif](cp2-active-experiments-snapshot.json). Angka progres hanya berlaku pada waktu snapshot; laporan completed baru diterbitkan setelah training, seleksi, evaluasi dan QA selesai. Kode/config/preregistrasi cascade ikut dilacak tanpa mengklaim training telah selesai.

## Reproduksi, integritas dan batas publikasi

Gunakan environment proyek yang sudah tersedia. [Panduan eksternal](../docs/cp2-external-evaluation.md), [replay Ganta](../docs/ganta-cascade-replay.md), dan [preregistrasi adaptasi cascade](../docs/cp2-cascade-clean.md) menjelaskan input serta perbedaan protokol. Source Ganta dipindah ke `notebook/scripts/` dengan SHA identik; perintah replay mencantumkan override path, tanpa mengedit source partner atau inventory historis.

**224 tests passed**, 29 deprecation warnings lama, CPU tanpa GPU. Pengujian dependency dua environment dan pemeriksaan publikasi dicatat pada [manifest QA](completed-results-publication.json). Ruff belum tersedia; tidak ada instalasi/upgrade environment untuk publikasi ini.

Checkpoint, dataset mentah, standalone mask/probability arrays dan log mesin tetap lokal. Contoh yang dipublikasikan berupa panel analisis terpilih dengan atribusi sumber; tidak ada full dataset mirror. Notebook CP1 yang sedang dimodifikasi pengguna dan dokumen roadmap lokal tidak disentuh oleh commit publikasi ini.
