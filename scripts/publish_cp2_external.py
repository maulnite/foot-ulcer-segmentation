"""Publish aggregate research outputs after the complete external validation."""

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    def read(name):
        return json.loads((run / name).read_text())
    result, manifest, protocol = read("comparison.json"), read("dataset_manifest.json"), read("preregistration.json")
    if result["validation"]["status"] != "passed" or len(result["models"]) != 26:
        raise ValueError("Publish only the fully validated registered comparison")
    qa, visual_review = read("engineering_qa.json"), read("visual_review.json")
    if visual_review["status"] != "passed":
        raise ValueError("Visual review must pass before publishing")
    validation = {**result["validation"], "engineering": qa, "visual_review": visual_review}
    prefix = "cp2-external-evaluation"
    report = ROOT / "reports" / (prefix + ".md")
    public_json = report.with_suffix(".json")
    if report.exists() or public_json.exists():
        raise FileExistsError("Preserve completed public report")
    figure_dir = ROOT / "reports/figures" / prefix
    figure_dir.mkdir(parents=True, exist_ok=False)
    for file in (run / "figures").glob("*.png"):
        shutil.copyfile(file, figure_dir / file.name)
    for source, destination in [("comparison.csv", "metrics.csv"), ("paired_effects.csv", "paired-effects.csv"),
                                ("three_seed_summary.csv", "three-seed.csv")]:
        target = ROOT / "reports/tables" / (prefix + "-" + destination)
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(run / source, target)
    write_json(public_json, {"status": "COMPLETED — FROZEN EXTERNAL INFERENCE; NO TRAINING",
               "run": str(run.relative_to(ROOT)), "dataset": {key: value for key, value in manifest.items()
                   if key not in ("eligible", "excluded")}, "eligible_images": len(manifest["eligible"]),
               "excluded_images": len(manifest["excluded"]), "models": result["models"],
               "paired_effects": result["paired_effects"], "replication": result["replication"],
               "validation": validation, "protocol": protocol,
               "execution_policy": read("execution_policy.json"),
               "runtime_source_hashes": read("execution_sources.json"),
               "runtime_input_hashes": read("execution_inputs.json")})
    write_json(ROOT / "reports" / (prefix + "-validation.json"), validation)
    table = result["models"]
    by_id = {row["model"]: row for row in table}
    best = max(table, key=lambda row: row["external_image_dice_percent"])
    focal = next(row for row in result["paired_effects"] if row["reference"] == "B0_s42" and row["candidate"] == "B1_s42")
    cp1, b0, b1 = [by_id[name] for name in ("CP1_original", "B0_s42", "B1_s42")]
    table_lines = ["| Model | Source val Image Dice | Source test Image Dice | External Image Dice | External Data Dice | External Boundary IoU | Miss |",
                   "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        table_lines.append(f"| {row['model']} | {row['source_validation_image_dice_percent']:.2f}% | "
                           f"{row['source_test_image_dice_percent']:.2f}% | {row['external_image_dice_percent']:.2f}% | "
                           f"{row['external_data_dice_percent']:.2f}% | {row['external_boundary_iou_percent']:.2f}% | "
                           f"{row['total_foreground_misses']} |")
    seed_lines = ["| Model | Seeds | External Image Dice mean ± SD | External Data Dice mean ± SD | External Boundary IoU mean ± SD |",
                  "| :--- | :--- | ---: | ---: | ---: |"]
    for row in result["replication"]:
        key = "external_image_dice_percent"
        data = "external_data_dice_percent"
        boundary = "external_boundary_iou_percent"
        seed_lines.append(f"| {row['model']} | 42, 43, 44 | {row[key+'_mean']:.2f} ± {row[key+'_sd']:.2f}% | "
                          f"{row[data+'_mean']:.2f} ± {row[data+'_sd']:.2f}% | "
                          f"{row[boundary+'_mean']:.2f} ± {row[boundary+'_sd']:.2f}% |")
    seed_effects = []
    for model in ("H2", "A6"):
        deltas = [by_id[f"{model}_s{seed}"]["external_image_dice_percent"] -
                  by_id[f"B1_s{seed}"]["external_image_dice_percent"] for seed in (42, 43, 44)]
        seed_effects.append(f"- {model} − B1 pada seed 42/43/44: " +
                           ", ".join(f"{delta:+.2f}" for delta in deltas) +
                           f" poin; membaik pada {sum(delta > 0 for delta in deltas)}/3 seed.")
    sign = "lebih tinggi" if focal["mean_image_dice_delta_pp"] > 0 else "lebih rendah" if focal["mean_image_dice_delta_pp"] < 0 else "sama"
    text = f'''# CP2 — perbandingan eksternal seluruh checkpoint CP1/CP2

**COMPLETED — 26 frozen checkpoints, {len(manifest['eligible'])} gambar per model, {result['validation']['predictions_verified']:,} prediksi diverifikasi. Tidak ada training/fine-tuning.**

Run: `{run.relative_to(ROOT)}`. Protokol, roster checkpoint dan SHA256 disimpan sebelum target prediction. [Panduan/protokol](../docs/cp2-external-evaluation.md) · [Registry](../configs/cp2/external.json) · [Seluruh metrik CSV](tables/{prefix}-metrics.csv) · [Hasil JSON]({prefix}.json).

## Apa yang diuji?

Apakah model yang memperoleh skor tinggi pada Chronic Wound tetap bekerja baik pada foto dari sumber dan jenis luka lain? Semua bobot tetap; inference membaca foto, bukan GT. **Loss tidak dihitung untuk mengubah prediksi saat inference**: yang dibandingkan adalah bobot hasil training sebelumnya.

Target adalah [CO2Wounds-V2 Mendeley version 1](https://data.mendeley.com/datasets/s2w7rjwz49/1), DOI 10.17632/s2w7rjwz49.1: foto luka kronis pasien kusta di Kolombia, dikumpulkan dengan kamera smartphone. Perubahan sumber sekaligus patologi ini bukan pengujian khusus DFU lintas rumah sakit saja. Atribusi: Sanchez et al., ICIP 2024, pp.69–75.

## Audit data sebelum membuka skor

- Provider menyediakan 607 pasangan berlabel (485 train/122 val), dan 157 test tanpa mask. Provider-train/val menjadi evaluasi eksternal kita, tanpa fitting pada keduanya.
- Ditemukan 23 kelompok RGB identik dengan mask berbeda, berjumlah 49 gambar. Seluruh kelompok dikarantina sesuai aturan sebelum inference, tanpa memilih/menggabungkan mask. Perbedaan anotasi tidak otomatis ditafsirkan sebagai kesalahan klinis.
- Primary cohort: **558 gambar**, 447 provider-train dan 111 provider-val, sama untuk seluruh 26 model.
- Tidak ditemukan exact RGB overlap atau kandidat near-duplicate pada audit terhadap 1.109 gambar sumber. Screening sempit tidak menjamin ketiadaan semua hubungan crop/pasien.
- Mask memakai 0/1, bukan 0/255. Foto JPG/JPEG/PNG dipasangkan berdasarkan nama terverifikasi. Seluruh pasangan decode dan cocok dimensi/identitas COCO.
- Seluruh 558 GT berisi luka; **GT-empty external evaluation tidak tersedia**. Pixel precision masih menilai FP pada latar foto luka, tetapi tidak membuktikan kemampuan menolak foto tanpa luka.
- Rilis total menyebut 96 pasien, tetapi patient ID tidak tersedia dalam metadata yang diuji. Tidak ada patient-clustered confidence interval atau klaim patient independence.

Mendeley menampilkan CC BY 4.0, sedangkan README penulis dan license COCO di arsip menyebut CC BY-NC-ND 4.0. Data/derived media/cache tetap lokal untuk riset nonkomersial. Yang dibagikan di repo adalah kode, metrik numerik dan grafik agregat. Raw bytes tidak diubah.

## Proses inference yang sama

RGB → resize dengan aspect ratio tetap → black letterbox 224×224 → normalisasi ImageNet source → model eval/no-grad → probability map → buang padding → bilinear kembali ke koordinat native → threshold ≥0,5 → evaluasi pada GT native.

Tidak ada GT crop, augmentation target, update parameter/BatchNorm, pseudo-label training, statistik normalisasi target, atau threshold search. Model probability-output tidak disigmoid ulang; model logit-output disigmoid sekali. Seluruh state checkpoint dimuat `strict=True`, termasuk auxiliary head; inference H1/H2/CH2/RH2 menggunakan mask utama dari base network saja.

Metrik memakai ordinary confusion counts, mean Image Dice dan pooled Data Dice/IoU/precision/recall, Boundary IoU inner-band 1% diagonal native, PFOM CP1, ukuran luka menurut persentase kanvas native, dan foreground miss (GT positif dengan TP=0). Small ≤1% kanvas bukan ukuran fisik; target native berbeda dari source yang berpadded. PFOM native juga dipengaruhi resolusi sehingga tidak dibandingkan langsung lintas grid sebagai satu skor identik.

## Seluruh hasil

Source columns adalah hasil historis, tidak dihitung ulang. **CP1 original/matched menggunakan split 707/124/278; kelompok CP2 dan CP1-recipe memakai 671/118/261.** Karena itu source columns lintas keluarga tidak menjadi perbandingan terkontrol. Kolom external memakai cohort dan preprocessing yang sama.

Besarnya selisih source–target juga dipengaruhi perbedaan populasi dan grid evaluasi: source historis dihitung pada grid input, sedangkan target ini pada koordinat native. Karena itu angka penurunan absolut tidak mengisolasi domain shift saja. Perbandingan antar-model pada target memakai evaluator identik.

{chr(10).join(table_lines)}

`_s42/_s43/_s44` adalah seed training historis. B0/B1: source versus corrected focal; L1/L2: Tversky/Focal Tversky; L3/L4: tambahan Boundary Loss; A1/A2/A3: local attention dengan fixed/learned-constant/input-dependent fusion; A5: kontrol kapasitas; A6: CBAM. H1/H2: auxiliary segmentation/boundary. R0/RH2/RA6: resep CP1 pada split CP2; C0/CH2/CA6: resep serta split CP1 dengan perubahan model saja. L0/A0/H0 adalah alias B1 dan tidak dihitung ulang.

![Seluruh model](figures/{prefix}/all_models.png)

**Yang terlihat:** seluruh 26 checkpoint ditampilkan dalam urutan registry, termasuk hasil yang lebih buruk. Angka tertinggi Image Dice pada tabel ini adalah **{best['model']} = {best['external_image_dice_percent']:.2f}%**. **Artinya:** ini ranking deskriptif pada target dan preprocessing yang diuji. **Perlu diingat:** pemenang satu checkpoint dari banyak percobaan bukan bukti superioritas antar-seed/pasien atau dasar meloloskan ulang guard internal.

## Kontras utama: koreksi focal B0 versus B1

Pada source validation, B0 **{b0['source_validation_image_dice_percent']:.2f}%** dan B1 **{b1['source_validation_image_dice_percent']:.2f}%**. Pada target yang sama, B0 **{b0['external_image_dice_percent']:.2f}%** dan B1 **{b1['external_image_dice_percent']:.2f}%**.

External Image Dice B1 {sign} daripada B0 dengan mean delta **{focal['mean_image_dice_delta_pp']:+.3f} poin persentase**. Dari 558 pasangan, B1 membaik pada **{focal['improved_images']}**, memburuk pada **{focal['worse_images']}**, dan sama pada **{focal['tied_images']}** gambar. Pembagian ini tidak memperlakukan piksel sebagai sampel independen dan bukan uji signifikansi pasien.

B0/B1 menggunakan source split, arsitektur, inisialisasi, augmentasi dan aturan seleksi yang dipasangkan. Jumlah epoch/checkpoint aktual berbeda sebagai konsekuensi validation. Kontras ini menguji efek objective dalam recipe tersebut, bukan isolasi fixed-update-budget atau bukti langsung penyebab skor paper.

Pada target ini, pixel precision B0/B1 adalah {b0['external_pixel_precision_percent']:.2f}%/{b1['external_pixel_precision_percent']:.2f}%, sedangkan pixel recall {b0['external_pixel_recall_percent']:.2f}%/{b1['external_pixel_recall_percent']:.2f}%. B1 menghasilkan lebih sedikit false positive relatif terhadap prediksi positif, tetapi melewatkan lebih banyak piksel luka. Foreground miss meningkat dari {b0['total_foreground_misses']} menjadi {b1['total_foreground_misses']} gambar. Ini trade-off pada checkpoint dan threshold tetap yang diuji, bukan jaminan bahwa mismatch selalu menguntungkan.

![Efek berpasangan focal](figures/{prefix}/focal_paired_effects.png)

**Yang terlihat:** sumbu vertikal positif berarti koreksi focal membantu gambar tersebut; negatif berarti sebaliknya. Seluruh gambar dimasukkan. **Artinya:** selisih mean dapat menyembunyikan kasus yang membaik dan memburuk. **Perlu diingat:** ukuran GT berasal dari native canvas dan tidak memberi ukuran luka fisik atau identitas pasien.

## Kestabilan B1/H2/CBAM pada tiga seed

{chr(10).join(seed_lines)}

SD adalah variasi tiga training pada source split yang sama, **bukan confidence interval klinis**. Ketiga seed selalu ditampilkan; tidak dipilih seed terbaik berdasarkan target.

{chr(10).join(seed_effects)}

Rerata H2/CBAM belum menunjukkan peningkatan yang konsisten pada semua seed. Tiga seed dari satu sumber target tidak cukup untuk menyatakan superioritas atau signifikansi klinis.

![Tiga seed](figures/{prefix}/three_seeds.png)

**Yang terlihat:** titik menunjukkan setiap seed; bar dan error bar menunjukkan mean/SD. **Artinya:** manfaat yang hanya muncul pada satu seed tidak cukup untuk menyatakan peningkatan konsisten. **Perlu diingat:** satu target, metadata pasien tidak tersedia, dan cohort tidak memiliki GT kosong.

## Menjawab dugaan tentang konfigurasi paper

CP1 original memperoleh external Image Dice **{cp1['external_image_dice_percent']:.2f}%** dan Data Dice **{cp1['external_data_dice_percent']:.2f}%**. Ini checkpoint reproduksi lokal, **bukan bobot asli penulis paper**.

Perbedaan ranking source–target, jika ada, mendeskripsikan ketergantungan pada data dan protokol. Itu belum membuktikan bahwa skor paper tinggi karena salah konfigurasi. Perbandingan CP1–CP2 sekaligus mengubah cohort training, loss, checkpoint/scheduler dan RNG. Perubahan patologi, kamera, skala luka dan kualitas anotasi target juga terjadi bersamaan.

Audit sebelumnya memang membuktikan focal input mismatch pada kode sumber serta perbedaan evaluator inline/offline. Namun evaluator dan cohort tepat yang menghasilkan headline paper belum dipastikan. [Audit sumber](../docs/reference-audit.md) tetap menjadi bukti terpisah; hasil eksternal ini tidak menggantikan audit tersebut atau mengubah skor historis.

Guard validation internal tetap berlaku. Tidak ada checkpoint baru, perubahan threshold, kombinasi architecture, training presence supervision, atau reselection berdasarkan target. Pengembangan setelah melihat CO2Wounds menjadi eksploratif dan memerlukan dataset independen lain. **DFUC2022: NOT RUN — ACCESS REQUIRED**; [sumber resmi](https://dfu-challenge.github.io/dfuc2022.html) masih menautkan permohonan akses dan tidak ada pesan/provider request dikirim.

## Eksekusi, QA dan reproduksi

- Torch 1.11.0+cu113 di CPU, float32, batch1, empat worker ×empat thread; GPU sibuk dengan job lain dan tidak digunakan. Tidak ada dependency/precision yang diubah.
- 26/26 real-input smoke tests lulus: strict checkpoint load, finite probability, deterministic repeat CPU, frozen parameter dan buffer BatchNorm tidak berubah.
- Seluruh **{result['validation']['raw_rgb_to_input_tensors_verified']} input tensors** direkonstruksi ulang dari RGB mentah dan source preprocessing, termasuk koordinat letterbox, tanpa membaca GT untuk menentukan input.
- **{result['validation']['predictions_verified']:,} probability caches** diverifikasi checksum dan native region metrics dihitung ulang. **{result['validation']['boundary_pfom_checks']}** native boundary/PFOM cases diperiksa ulang deterministik; seluruh agregasi boundary/PFOM CSV/JSON juga cocok.
- **{result['validation']['protected_files']} file historis/user** yang diinventaris sebelum perubahan tetap cocok SHA256. Seluruh checkpoint juga cocok dengan identitas completion/status historis. Tidak ada CP1/CP2 canonical result yang ditimpa.
- Dataset download 115,1 MB dan ekstraksi 121,6 MB, dalam batas download256MiB/extraction1GiB. ZIP CRC dan SHA256 lokal dicatat; provider checksum terpisah tidak tersedia.
- Dua kegagalan persiapan dicatat: adapter awal mengasumsikan JPG dan launcher pertama me-resolve symlink virtualenv ke Python dasar. Keduanya diperbaiki sebelum full inference tanpa mengganti environment atau scientific configuration.
- Suite pengujian: **{qa['tests_passed']} passed**, {qa['warnings']} warning deprecation yang sudah ada; compile checks dan dependency checks dua environment lulus. Ruff **NOT RUN — belum terpasang**; environment tidak diubah untuk memasangnya.
- Ketiga figure diperiksa secara visual: judul, label/satuan, urutan model, skala 0–100 atau delta simetris, legend, marker seed dan keterbacaan angka lulus. Grafik agregat tidak memuat foto atau mask pasien.

Timing dalam CSV adalah biaya pada shared CPU, termasuk kondisi proses paralel; tidak menjadi benchmark latency eksklusif atau perbandingan dengan RTX2080Ti paper. Tidak ada klaim patient-level inference uncertainty, novelty global, clinical benefit, atau SOTA dari run ini.

Reproduksi mengikuti [panduan](../docs/cp2-external-evaluation.md). Raw/local artifacts: `models/<id>/probabilities/`, per-image CSV, prediction manifests, config/checkpoint hashes, `input_cache.json`, native geometry, logs dan validation. Public artifacts: [metrics CSV](tables/{prefix}-metrics.csv), [paired effects](tables/{prefix}-paired-effects.csv), [three seeds](tables/{prefix}-three-seed.csv), [validation]({prefix}-validation.json).
'''
    with report.open("x") as stream:
        stream.write(text)
    write_json(ROOT / "outputs/cp2/latest_external.json", {"root": str(run.relative_to(ROOT)), "status": "completed"})
    print(json.dumps({"report": str(report.relative_to(ROOT)), "highest_descriptive_image_dice": best["model"],
                      "focal_delta_pp": focal["mean_image_dice_delta_pp"]}, indent=2))


if __name__ == "__main__":
    main()
