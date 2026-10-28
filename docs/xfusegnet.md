# CP1 · Reproduksi x-FUSegNet

Eksperimen `cp1-xfusegnet-01` melengkapi reproduksi dengan lima model FUSegNet pada FUSeg Challenge 2021 beresolusi **512×512**. Rata-rata probabilitas kelima model kemudian di-threshold `>=0.5`, sesuai paper §3.2.4 dan `xfusegnet_test.py`. Ini bagian reproduksi, belum merupakan novelty CP2.

## Kode dan parameter yang diikuti

Snapshot penulis tetap utuh pada revisi **`60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc`**. Definisi Dataset, augmentasi, preprocessing, dan helper penyimpanan dimuat langsung dari AST `xfusegnet_train.py`. Seluruh loop training lima fold juga dieksekusi dari AST file itu; hook tambahan hanya menangani pencatatan, pemeriksaan, dan resume. Uji menghapus hook dari AST hasil instrumentasi dan memastikan seluruh pernyataan numerik sama dengan loop asli.

Augmentasi x-FUSegNet berbeda dari script FUSegNet biasa: pada cabang rotate, training x-FUSegNet menyertakan `shift_limit=0.1`. Karena itu runner baseline biasa tidak dipakai sebagai pengganti loop ini. `xfusegnet_all.py` memiliki definisi training yang sama. Fungsi training augmentation dalam file test berbeda, tetapi tidak dipanggil pada inference; transform validation/preprocessing yang dipakai inference sama.

| Komponen | Protokol sumber yang digunakan |
| :--- | :--- |
| Arsitektur | EfficientNet-B7 ImageNet, decoder P-scSE lengkap termasuk shorted stage terakhir |
| Parameter terdaftar | 70,968,939 per model |
| Split | Python seed-42 shuffle, lalu `KFold(n_splits=5, shuffle=False)` |
| Input | Native 512×512, tanpa resize, AMP, atau perubahan batch size |
| Batch / workers | 2 / 4 untuk training dan validation |
| Loss | Dice + focal sumber; probability-input focal dipertahankan |
| Optimizer | Adam, LR `1e-4`, weight decay `1e-5` |
| Scheduler | ReduceLROnPlateau, factor 0.1, patience 10, min LR `1e-5` |
| Batas training | Maksimum 200 epoch per fold, early stopping 30 |
| Checkpoint | `if loss improves ... elif IoU improves`, termasuk perilaku tracker sumber |
| Inference | Batch 1; lima raw sigmoid maps float32, `np.mean(axis=0)`, lalu threshold 0.5 |

Helper `save` sumber menyimpan field `epoch` satu lebih tinggi daripada jumlah epoch aktual akibat penambahan di pemanggil dan helper. Perilaku file sumber ini dipertahankan. Status, kurva, dan laporan menggunakan jumlah epoch aktual; pemilihan bobot tidak diubah. Lima `best_model_N.pth` dan companion `.pickle` kompatibel dengan struktur artefak penulis. Checkpoint `last.pth` tambahan menyimpan optimizer, scheduler, RNG, dan riwayat untuk resume.

## Data dan lima fold

Dataset publik menyediakan **810 train + 200 validation berlabel**, yang digabung menjadi pool 1.010 gambar sebagaimana jumlah training dalam paper. Penyatuan menggunakan symlink dalam folder staging lokal; file raw tidak diubah. Nama pool berlabel dan test tidak tumpang tindih. Setiap fold memakai **808 train / 202 validation**, dan setiap gambar menjadi validation tepat satu kali. **200 gambar test tidak dimasukkan ke training/fold.**

Manifest lokal: `data/splits/fuseg_2021_xfusegnet/manifest.json`.

```text
93aac316e596dcc34b8d55174ea1b10a564cb07f7398e4616c78409552076e2a
```

Urutan direktori awal lokal dicatat, lalu di-shuffle sebagaimana kode sumber. Urutan file dan fold asli penulis tidak tersedia; kesamaan persis assignment fold tidak dapat dibuktikan. Global torch/numpy seed **42** adalah tambahan reproducibility karena nilainya tidak dipublikasikan. Seed ditetapkan sekali sebelum training; RNG **berlanjut antar-fold**, mengikuti struktur loop sumber. Resume memulihkan state RNG, model, optimizer, scheduler, tracker, dan riwayat, termasuk batas antar-fold.

Duplikasi RGB sumber dipertahankan. Gambar validation yang identik RGB dengan training fold berjumlah **12 / 10 / 9 / 4 / 10** untuk fold 1–5. Karena itu tidak ada klaim bebas leakage piksel maupun independensi pasien. Filename tiap gambar tetap hanya berada di satu sisi fold.

Mask `0233.png` memiliki 84 piksel bernilai 32. **Training tetap memakai grayscale/255**, sehingga target fractional asli tidak dibinarisasi. Untuk evaluasi biner OOF dan PFOM saja, GT dibinarisasi pada `mask/255 >=0.5`; perbedaan kebijakan ini dicatat eksplisit.

## Evaluasi yang tersedia

- **OOF internal:** 1.010 prediksi, masing-masing dibuat hanya oleh model fold yang menahan gambar tersebut. Checkpoint dipilih pada validation fold yang sama. Dice/IoU/precision/recall dan PFOM adalah hasil validasi internal, belum merupakan independent test estimate.
- **Test ensemble:** 200 gambar diprediksi oleh kelima model; mean probabilitas dan PNG diverifikasi ulang pada CPU. Galeri hanya berisi gambar, mask prediksi, dan overlay batas.
- **Skor resmi challenge:** tidak tersedia secara lokal karena mask test privat tidak ada pada unduhan. Angka paper **89.23%** tetap referensi penulis, bukan hasil yang telah diverifikasi oleh run ini.

Tidak dilakukan evaluasi ensemble lima model pada pool berlabel: setiap gambar pada pool itu telah masuk training empat model. Skor semacam itu tidak akan menjadi validasi OOF. Skor OOF 512×512 juga tidak diperlakukan sebagai peningkatan langsung terhadap test Chronic Wound 224×224 yang berbeda protokol.

PFOM mengikuti [parameter lokal yang terdokumentasi](pfom.md), dengan empty-edge 1/0 dan proxy sentinel 2 terpisah. Pilihan Canny asli paper tetap belum diketahui. Lima contoh OOF dan lima contoh test ditetapkan sebelum training, tanpa pemilihan ulang berdasarkan hasil.

## Environment dan verifikasi

`.venv-training` dan `.venv-analysis` tidak diubah. `.venv-xfuseg` menggunakan Python yang sama dan mewarisi paket frozen training melalui `.pth`, lalu menambahkan dependency plotting asli ke environment baru. Semua versi paket training baseline diperiksa tetap identik. Tambahan dipin di [requirements-xfusegnet.lock.txt](../requirements-xfusegnet.lock.txt).

Untuk menyiapkan environment setelah mengikuti [training guide](training.md):

```bash
.venv-training/bin/python -m venv --without-pip .venv-xfuseg
.venv-xfuseg/bin/python - <<'PY'
from pathlib import Path
import sysconfig
target = Path(sysconfig.get_path('purelib')) / 'frozen-training.pth'
training = Path('.venv-training/lib/python3.10/site-packages').resolve()
target.write_text('import site; site.addsitedir(' + repr(str(training)) + ')\n')
PY
.venv-xfuseg/bin/python -m pip install -r requirements-xfusegnet.lock.txt
```

Unit tests melindungi kesetaraan AST, augmentasi x-FUSegNet, mean sebelum threshold, kebijakan mask fractional, pemulihan source tracker/RNG/optimizer, dan metadata checkpoint sumber. Smoke GPU native 512×512 menjalankan lima fold dengan subset diagnostik, memeriksa gradien finite, update encoder, dan prediksi identik setelah checkpoint diubah lalu dimuat ulang. Bobot smoke sementara dihapus; metrik smoke bukan hasil reproduksi.

**37 tests lulus** sebelum peluncuran, beserta smoke lima fold dan uji notebook executed. [Bukti verifikasi publik](../reports/tables/cp1-xfusegnet-verification.json) merekam hash kode/config/split, checksum loop AST asli, dan hasil pemeriksaan.

Uji analisis terpisah menggunakan fixture sintetis bertanda **DIAGNOSTIC** di folder lain untuk memeriksa notebook executed dan layout sebelum training panjang. Fixture tidak dipublikasikan sebagai hasil eksperimen.

## Menjalankan dan memantau

```bash
.venv-xfuseg/bin/python scripts/run_xfusegnet.py prepare
.venv-xfuseg/bin/python -m pytest -q
.venv-xfuseg/bin/python scripts/run_xfusegnet_batch.py --device cuda:0
```

Supervisor dijalankan detached pada server, satu proses training untuk kelima fold secara berurutan. GPU harus kosong sebelum proses training/inference dimulai. Proses pengguna lain tidak dihentikan. Batas 200 epoch berlaku **per fold**, sehingga durasi penuh jauh lebih panjang daripada satu baseline 224×224; tidak ada pengurangan setting untuk mempercepat run.

Cek status tanpa memulai training:

```bash
python scripts/xfusegnet_status.py
```

Status dan log berada di `outputs/xfusegnet/cp1-xfusegnet-01/`. Snapshot pemantauan dicatat setiap 30 menit selama proses aktif; tidak mengirim pesan otomatis ke chat. Bobot dan status per fold berada di `outputs/runs/cp1-xfusegnet-01/`.

Setelah training selesai, supervisor menjalankan OOF dan test ensemble, memverifikasi probabilitas/mask/metrics, menghitung PFOM, mengeksekusi seluruh notebook, lalu menyiapkan:

```text
notebooks/03_cp1_xfusegnet.ipynb
reports/cp1-xfusegnet-01.md
reports/tables/cp1-xfusegnet-*.csv / metadata.json
README.md (status, hasil, dan tautan notebook/laporan)
```

Commit dan push normal memakai `gh-orie` credential helper. Publikasi otomatis mensyaratkan working tree bersih, branch `main`, README masih sama dengan saat workflow dimulai, dan tidak menimpa file hasil berbeda. Jika syarat itu tidak terpenuhi, hasil tetap tersedia di `outputs/xfusegnet/cp1-xfusegnet-01/published/` beserta alasan penundaan dalam `workflow.json`.

Jika workflow terputus, periksa status/error terlebih dahulu. `scripts/run_xfusegnet_batch.py --resume` melanjutkan checkpoint dan fold yang belum selesai. Jangan menjalankan supervisor kedua saat yang pertama masih hidup.

Sumber: [paper](https://arxiv.org/abs/2305.02961v2), [kode penulis](https://github.com/mrinal054/FUSegNet), [dataset](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge).
