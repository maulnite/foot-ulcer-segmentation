"""Render verified recipe-control results without models or GPU use."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402

NAMES = ('R0', 'RH2', 'RA6')
COLORS = {'R0': '#64748b', 'RH2': '#7c3aed', 'RA6': '#0f766e'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--figures-only', action='store_true',
                        help='Refresh presentation figures after preserving originals; numeric fields must match')
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / 'final_validation.json')['status'] != 'passed':
        raise ValueError('Require independent verification of completed real runs')
    output_root = ROOT
    publication = ROOT / 'reports/cp2-cp1-recipe-results.json'
    cached = None
    if publication.exists() and not args.figures_only:
        cached = read(publication)
        if cached['run_root'] != str(root.relative_to(ROOT)):
            # A reproduction run must not replace the historical public report.
            output_root = root / 'presentation'
            publication = output_root / 'reports/cp2-cp1-recipe-results.json'
            cached = read(publication) if publication.exists() else None
    config = read(root / 'protocol.json')
    manifest = read(root / 'data/manifest.json')
    selection = read(root / 'selection.json')
    metrics = {n: {s: read(root / n / f'evaluation-{s}.json') for s in ('validation', 'test')} for n in NAMES}
    statuses = {n: read(root / n / 'status.json') for n in NAMES}
    if cached is not None:
        if cached['metrics'] != metrics or cached['statuses'] != statuses:
            raise ValueError('Existing completed report differs; preserve and inspect')
        print('CACHED verified completed report; no figures or numerical results overwritten')
        return
    if args.figures_only:
        previous = read(publication)
        if previous['metrics'] != metrics or previous['statuses'] != statuses:
            raise ValueError('Presentation refresh cannot change experiment results')
        if not (root / 'presentation_v1/cp2-cp1-recipe-results.json').exists():
            raise ValueError('Preserve original report and figures before presentation refresh')
    figures = output_root / 'reports/figures/cp2-cp1-recipe'
    figures.mkdir(parents=True, exist_ok=args.figures_only)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    files = []

    def save(fig, name):
        for suffix in ('png', 'svg'):
            path = figures / f'{name}.{suffix}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            files.append({'file': str(path.relative_to(ROOT)), 'sha256': digest_file(path)})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for name in NAMES:
        history = read(root / name / 'history.json')
        epochs = [e['epoch'] for e in history]
        for ax, field, label in [(axes[0, 0], 'regional', 'Main validation loss'),
                                  (axes[0, 1], 'source_iou', 'Source batch-mean validation IoU')]:
            ax.plot(epochs, [e['validation'][field] for e in history], color=COLORS[name], label=name)
            ax.set(title=label, xlabel='Epoch', ylabel='Loss' if field == 'regional' else 'IoU [0–1]')
        axes[1, 0].plot(epochs, [e['validation_image_dice'] for e in history], color=COLORS[name], label=name)
        chosen = statuses[name]['selected_epoch']
        axes[1, 0].scatter(chosen, history[chosen - 1]['validation_image_dice'], color=COLORS[name], s=65)
        axes[1, 1].step(epochs, [e['learning_rate'] for e in history], where='post', color=COLORS[name], label=name)
    axes[1, 0].set(title='Image Dice; dots = CP1-policy checkpoint', xlabel='Epoch', ylabel='Validation Image Dice [0–1]')
    axes[1, 1].set(title='Learning rate after scheduler step', xlabel='Epoch', ylabel='Learning rate', yscale='log')
    for ax in axes.flat:
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle('REAL / FRESH — CP1 training recipe on CP2 Chronic Wound cohort, seed 42', fontsize=15)
    save(fig, 'training_curves')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(NAMES))
    for ax, split in zip(axes, ('validation', 'test')):
        for offset, (label, getter) in zip((-.25, 0, .25), [
            ('Overall', lambda m: m['all']['image_based']['dice']),
            ('Small ≤1% canvas', lambda m: m['subgroups']['small']['image_based']['dice']),
            ('Boundary IoU', lambda m: m['all']['image_based']['boundary_iou'])]):
            ax.bar(positions + offset, [getter(metrics[n][split]) for n in NAMES], .24, label=label)
        ax.set(xticks=positions, xticklabels=NAMES, ylim=(0, 1), ylabel='Score [0–1]',
               title=f"{split.title()} — n={len(manifest['splits'][split])} images")
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3, fontsize=10)
    fig.suptitle('REAL / FRESH — segmentation quality; source-test exploratory', fontsize=15)
    save(fig, 'region_boundary_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, split in zip(axes, ('validation', 'test')):
        ax.bar(NAMES, [metrics[n][split]['gt_empty']['images_with_prediction'] for n in NAMES],
               color=[COLORS[n] for n in NAMES])
        count = metrics['R0'][split]['gt_empty']['images']
        ax.set(ylim=(0, count + .7), ylabel='GT-empty images with any predicted wound [images]',
               title=f'{split.title()} — {count} GT-empty images')
        ax.set_yticks(range(count + 1))
    fig.suptitle('REAL / FRESH — false positives on empty masks', fontsize=15)
    save(fig, 'empty_false_positives')

    if args.figures_only:
        previous['figures'] = files
        previous['presentation_updated_utc'] = datetime.now(timezone.utc).isoformat()
        previous['visual_review'] = 'PENDING repeated direct inspection after legend layout correction'
        write_json(publication, previous)
        print('Updated figures only; all model metrics, statuses and selection preserved')
        return

    rows = []
    for name in NAMES:
        for split in ('validation', 'test'):
            m = metrics[name][split]
            rows.append({'model': name, 'split': split, 'image_dice': m['all']['image_based']['dice'],
                         'data_dice': m['all']['data_based']['dice'],
                         'small_image_dice': m['subgroups']['small']['image_based']['dice'],
                         'boundary_iou': m['all']['image_based']['boundary_iou'],
                         'foreground_misses': m['gt_positive']['total_foreground_misses'],
                         'empty_fp_images': m['gt_empty']['images_with_prediction'],
                         'selected_epoch': statuses[name]['selected_epoch'],
                         'completed_epochs': statuses[name]['completed_epochs']})
    table = output_root / 'reports/tables/cp2-cp1-recipe-metrics.csv'
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'status': 'COMPLETED — TARGETED CP1 TRAINING-RECIPE EXPLORATORY COMPARISON',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'run_root': str(root.relative_to(ROOT)),
              'protocol': config, 'manifest_fingerprint': manifest['fingerprint'],
              'selection': selection, 'metrics': metrics, 'statuses': statuses, 'figures': files,
              'validation': read(root / 'final_validation.json'),
              'visual_review': 'PENDING direct figure inspection; numeric validation passed'}
    write_json(publication, result)
    lines = ['# CP2 — Targeted CP1 training-recipe comparison', '',
             f"Run: `{root.relative_to(ROOT)}`. Seed 42; split 671/118/261 tetap. Ini eksperimen eksploratif satu seed, bukan reproduksi CP1 identik.", '',
             '## Pertanyaan dan protokol', '',
             'Apakah H2 atau A6 memberi manfaat ketika checkpoint if-loss/elif-IoU, scheduler main validation loss, early stopping sumber dan evolving RNG CP1 digunakan? Sesuai pilihan pengguna, semua arm memakai B0 sumber termasuk focal input mismatch, preprocessing/augmentasi sumber, Adam 1e-4, WD 1e-5, batch 2, FP32 tanpa AMP. Bobot awal sumber dibuktikan identik; bukan warm-start checkpoint CP1.', '',
             'R0 = P-scSE asli; RH2 = auxiliary boundary head 33 parameter, auxiliary B0 berbobot 0,1; RA6 = CBAM stage 2–3 dengan gain historis 3. Pada H2, scheduler/checkpoint memakai loss mask utama saja, bukan auxiliary loss. Source IoU memakai threshold >0,5 dan batch averaging; pelaporan akhir tetap threshold ≥0,5 dan evaluator CP2.', '',
             '## Hasil yang benar-benar dijalankan', '',
             '| Model | Split | Image Dice | Data Dice | Small Dice | Boundary IoU | Miss | Empty FP | Epoch dipilih/selesai |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['split']} | {r['image_dice']:.4f} | {r['data_dice']:.4f} | {r['small_image_dice']:.4f} | {r['boundary_iou']:.4f} | {r['foreground_misses']} | {r['empty_fp_images']} | {r['selected_epoch']}/{r['completed_epochs']} |")
    lines += ['', f"Seleksi validation dengan guard historis terhadap R0 memilih **{selection['selected_candidate']}**. Keputusan tersimpan sebelum semua prediksi test baru.", '',
              '![Kurva training](figures/cp2-cp1-recipe/training_curves.png)', '',
              '**Yang terlihat:** checkpoint sumber dapat berbeda dari puncak Image Dice; titik menunjukkan epoch yang benar-benar dipilih.',
              '**Artinya:** prosedur training/checkpoint merupakan bagian dari metode yang sedang diuji.',
              '**Perlu diingat:** paket kebijakan berubah bersamaan; hasil ini tidak mengisolasi scheduler, RNG atau checkpoint secara individual.', '',
              '![Area dan boundary](figures/cp2-cp1-recipe/region_boundary_comparison.png)', '',
              '**Yang terlihat:** overall, luka kecil dan boundary diukur terpisah pada populasi yang sama untuk ketiga arm.',
              '**Artinya:** perubahan overall harus dibaca bersama perubahan subgroup dan bentuk batas.',
              '**Perlu diingat:** mean per gambar dan pooled Data Dice bukan metrik yang sama; source-test sudah pernah dianalisis.', '',
              '![GT kosong](figures/cp2-cp1-recipe/empty_false_positives.png)', '',
              '**Yang terlihat:** grafik menghitung gambar GT kosong dengan setidaknya satu pixel prediksi luka.',
              '**Artinya:** peningkatan Dice pada GT positif dapat disertai pertukaran false positive.',
              '**Perlu diingat:** hanya tujuh GT kosong validation; tidak menetapkan toleransi klinis dari jumlah kecil ini.', '',
              '## Biaya dan keterbatasan', '']
    for n, s in statuses.items():
        lines.append(f"- {n}: {s['seconds']/60:.1f} menit; active parameters {s['active_forward_parameters']:,}; registered {s['registered_parameters']:,}; peak allocated {s['peak_cuda_allocated_bytes']/2**30:.2f} GiB, reserved {s['peak_cuda_reserved_bytes']/2**30:.2f} GiB.")
    lines += ['', 'Validation sudah dipakai berulang untuk pengembangan. Pemilihan H2/A6 berdasarkan validation sebelumnya adalah adaptif dan eksploratif; tidak membuktikan generalisasi atau signifikansi. Tidak ada patient ID, sehingga ini duplicate-RGB-group split, bukan patient-level split. Tidak ada data eksternal, perubahan label, pemilihan seed terbaik, atau pengembalian data konflik. Guard tidak dilonggarkan setelah hasil. Run lama tetap baseline historis, bukan dicampur sebagai arm protokol baru.', '',
              '## Reproduksi', '', '```bash',
              '.venv-training/bin/python scripts/run_cp2_cp1_recipe.py --root outputs/cp2/cp1_recipe/<new-run> --stage prepare',
              '.venv-training/bin/python scripts/run_cp2_cp1_recipe.py --root outputs/cp2/cp1_recipe/<new-run> --stage preflight --device cuda:0',
              '.venv-training/bin/python scripts/run_cp2_cp1_recipe.py --root outputs/cp2/cp1_recipe/<new-run> --stage train --device cuda:0',
              '```', '', 'Gunakan `--resume` setelah pemeriksaan run terinterupsi; completed stages diverifikasi dan tidak dilatih ulang. Checkpoint, raw data dan probability maps tetap lokal.']
    (output_root / 'reports/cp2-cp1-recipe.md').write_text('\n'.join(lines) + '\n')
    print(f'Generated {len(files)} real-result figure files and completed report')


if __name__ == '__main__':
    main()
