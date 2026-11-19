# Replay checkpoint asli Ganta

Replay memakai **dua checkpoint terlatih Ganta**, source/config Ganta, dan test historis278 foto. Ini terpisah dari adaptasi cascade Orie CP2 yang coarse-nya B1 dan refinernya dilatih ulang pada split bersih671/118/261.

Source partner telah dipindah dari `notebook/13_cascade_crop_refine_chronic_wound.py` ke `notebook/scripts/13_cascade_crop_refine_chronic_wound.py`. Isinya identik: SHA256 `c476b328f47e241e6114eebb310ddf5ada9f2c714d2fa8fadbc2491b7091c371`. Catatan runtime lama mempertahankan path historis. Gunakan override `--source` berikut tanpa mengedit file partner atau runner historis:

```bash
.venv-training/bin/python -u scripts/test_partner_cascade.py \
  --partner-root ../ganta \
  --source notebook/scripts/13_cascade_crop_refine_chronic_wound.py \
  --output outputs/partner-cascade/NEW_RUN_ID \
  --threads 4 --timing-images 32 --timing-repeats 3 --warmup 3
```

Gunakan output baru; completed run tidak ditimpa. Checkpoint lokal yang diperlukan: `output_chronic_wound/checkpoints/cascade_chronic_wound_efficientnet-b7_2026-11-14_00-59-20/{stage1_coarse.pth,stage2_refiner.pth}`. Keduanya strict-loaded tanpa download pretrained weights. Shared CPU digunakan; ini bukan benchmark GPU eksklusif. Tidak ada training atau konfigurasi Orie CP2 pada replay ini.

Hasil tersedia dalam [laporan replay278](../reports/ganta-cascade-replay.md), [pemeriksaan17 GT kosong](../reports/cascade-empty-gt-audit.md), serta tabel numeric per gambar di `reports/tables/ganta-cascade-replay-*.csv`. Dataset/media mentah, full mask caches dan checkpoint tetap lokal.
