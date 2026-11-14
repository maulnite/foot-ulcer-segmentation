"""Render completed original-CP1-cohort model-only experiments on CPU."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402

MODELS = ('C0', 'CH2', 'CA6')
COLORS = {'C0': '#64748b', 'CH2': '#7c3aed', 'CA6': '#0f766e'}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / 'final_validation.json').exists():
        subprocess.run([str(ROOT / '.venv-training/bin/python'), 'scripts/validate_cp1_matched.py',
                        '--root', str(root)], cwd=ROOT, check=True)
    qa = read(root / 'final_validation.json')
    if qa['status'] != 'passed':
        raise ValueError('Require independent verification of real completed runs')
    out = ROOT if not (ROOT / 'reports/cp1-matched-results.json').exists() else root / 'presentation'
    publication = out / 'reports/cp1-matched-results.json'
    if publication.exists():
        raise FileExistsError('Preserve previous publication')
    metrics = {n: {s: read(root / n / f'analysis-{s}.json') for s in ('validation', 'test')} for n in MODELS}
    statuses = {n: read(root / n / 'status.json') for n in MODELS}
    selection, audit = read(root / 'selection.json'), read(root / 'original_split_audit.json')
    figures = out / 'reports/figures/cp1-matched'
    figures.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    artifacts = []

    def save(fig, name):
        for suffix in ('png', 'svg'):
            path = figures / f'{name}.{suffix}'
            fig.savefig(path, dpi=160, bbox_inches='tight')
            if suffix == 'svg':
                path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines()) + '\n')
            artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': digest_file(path)})
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for n in MODELS:
        history = read(root / n / 'history.json')
        epochs = [r['epoch'] for r in history]
        loss_key = sorted(history[0]['validation'])[0]
        axes[0, 0].plot(epochs, [r['train'][loss_key] for r in history], color=COLORS[n], label=n)
        axes[0, 1].plot(epochs, [r['validation'][loss_key] for r in history], color=COLORS[n], label=n)
        axes[1, 0].plot(epochs, [r['validation']['iou_score'] for r in history], color=COLORS[n], label=n)
        chosen = statuses[n]['selected_epoch']
        axes[1, 0].scatter(chosen, history[chosen - 1]['validation']['iou_score'], color=COLORS[n], s=60)
        axes[1, 1].step(epochs, [r['learning_rate'] for r in history], where='post', color=COLORS[n], label=n)
    axes[0, 0].set(title='Training objective (CH2 includes auxiliary term)', ylabel='Loss')
    axes[0, 1].set(title='Original main-task validation loss', ylabel='Loss')
    axes[1, 0].set(title='Source batch-IoU; dots = selected checkpoints', ylabel='IoU [0–1]')
    axes[1, 1].set(title='Learning rate after source scheduler', ylabel='Learning rate', yscale='log')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    fig.suptitle('REAL / FRESH — original CP1 protocol, Chronic Wound 707/124/278, seed 42', fontsize=15)
    save(fig, 'source_training_curves')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(3)
    for ax, split in zip(axes, ('validation', 'test')):
        for offset, label, getter in [(-.25, 'Overall Image Dice', lambda m: m['all']['image_based']['dice']),
                                      (0, 'Small ≤1% canvas', lambda m: m['subgroups']['small']['image_based']['dice']),
                                      (.25, 'Boundary IoU', lambda m: m['all']['image_based']['boundary_iou'])]:
            ax.bar(x + offset, [getter(metrics[n][split]) for n in MODELS], .24, label=label)
        ax.set(xticks=x, xticklabels=MODELS, ylim=(0, 1), ylabel='Score [0–1]',
               title=f"{split.title()} — n={audit['counts'][split]} images")
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, fontsize=9, frameon=False)
    fig.suptitle('REAL / FRESH — original CP1 cohort; source-test exploratory', fontsize=15)
    save(fig, 'region_boundary')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, split in zip(axes, ('validation', 'test')):
        count = metrics['C0'][split]['gt_empty']['images']
        ax.bar(MODELS, [metrics[n][split]['gt_empty']['images_with_prediction'] for n in MODELS],
               color=[COLORS[n] for n in MODELS])
        ax.set(ylim=(0, count + .7), ylabel='GT-empty images with any predicted wound [images]',
               title=f'{split.title()} — n={count} empty GT images')
    fig.suptitle('REAL / FRESH — false-positive images, original CP1 cohort', fontsize=15)
    save(fig, 'empty_false_positives')

    lines = ['# Original CP1 protocol — H2 and CBAM model-only comparison', '',
             '**COMPLETED — REAL TRAINING AND EVALUATION.**', '',
             f"Run: `{root.relative_to(ROOT)}`. Original split **707/124/278**, fingerprint `{audit['split_fingerprint']}`.", '',
             'Semua original config keys CP1 identik, termasuk sigmoid output, focal input mismatch, source TrainEpoch/ValidEpoch, augmentation, Adam 1e-4, WD 1e-5, batch 2, 4 workers, FP32/TF32, scheduler, checkpoint if-loss/elif-IoU, dan early stopping. C0 adalah kontrol fresh; CH2 hanya menambah boundary supervision; CA6 hanya mengganti attention stage 2–3 dengan CBAM gain historis 3.', '',
             'Pada CH2, training menambahkan 0,1×source loss pada target batas dari mask sesudah augmentasi. Validation/checkpoint/scheduler tetap memantau main source loss. Source training statements identik; satu konstanta assertion jumlah parameter disesuaikan untuk arsitektur yang memang berubah. Tidak mengedit engine sumber.', '',
             '| Model | Split | Image Dice | Data Dice | Small Dice | Boundary IoU | Miss | Empty FP | Epoch selected/completed |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    rows = []
    for n in MODELS:
        for s in ('validation', 'test'):
            m = metrics[n][s]
            row = {'model': n, 'split': s, 'image_dice': m['all']['image_based']['dice'],
                   'data_dice': m['all']['data_based']['dice'], 'small_dice': m['subgroups']['small']['image_based']['dice'],
                   'boundary_iou': m['all']['image_based']['boundary_iou'], 'misses': m['gt_positive']['total_foreground_misses'],
                   'empty_fp_images': m['gt_empty']['images_with_prediction']}
            rows.append(row)
            lines.append(f"| {n} | {s} | {row['image_dice']:.4f} | {row['data_dice']:.4f} | {row['small_dice']:.4f} | {row['boundary_iou']:.4f} | {row['misses']} | {row['empty_fp_images']} | {statuses[n]['selected_epoch']}/{statuses[n]['completed_epochs']} |")
    lines += ['', f"Seleksi validation sebelum test memilih **{selection['selected_candidate']}**. Source-test sudah pernah dianalisis; hasil bersifat eksploratif.", '',
              '![Training](figures/cp1-matched/source_training_curves.png)', '',
              '**Yang terlihat:** training loss CH2 mencakup auxiliary term, sedangkan validation loss semua arm adalah main task. Titik menunjukkan checkpoint sumber yang dipilih.',
              '**Artinya:** prosedur optimisasi CP1 dikendalikan; besar training loss antar-objective tidak menjadi ranking kualitas.',
              '**Perlu diingat:** source policy bukan pemilihan puncak Image Dice; anggaran epoch aktual dapat berbeda.', '',
              '![Region and boundary](figures/cp1-matched/region_boundary.png)', '',
              '**Yang terlihat:** overall, luka kecil dan boundary diukur pada gambar yang sama untuk ketiga model.',
              '**Artinya:** manfaat harus dibaca bersama tradeoff antar-subgroup dan FP/FN.',
              '**Perlu diingat:** small di sini berarti ≤1% full padded canvas, bukan kategori klinis atau ukuran fisik.', '',
              '![Empty false positives](figures/cp1-matched/empty_false_positives.png)', '',
              '**Yang terlihat:** jumlah gambar GT kosong dengan sedikitnya satu pixel prediksi luka.',
              '**Artinya:** peningkatan segmentasi pada GT positif dapat disertai false positive.',
              '**Perlu diingat:** jumlah gambar kosong kecil dan tidak menentukan toleransi klinis.', '',
              f"Audit mempertahankan {audit['duplicate_RGB_groups']} duplicate RGB groups, {audit['cross_split_RGB_groups']} cross-split groups, dan {audit['conflicting_mask_RGB_groups']} conflicting-mask groups. Tidak ada gambar dihapus/dipindahkan. Ini sengaja mengikuti CP1, bukan bukti clean/patient-independent generalization.", '',
              'Seluruh bobot sumber dipasangkan dan cocok dengan rekonstruksi CP1 historis. Initial training RNG, actual input order dan augmented bytes diperiksa. Satu langkah kontrol CPU yang diberi observasi juga menghasilkan state/logs identik dengan engine CP1 asli; ketiga smoke GPU dijalankan terpisah. Ini tidak menjanjikan seluruh training GPU mengulang skor historis bit-for-bit.', '',
              'Laporan/metrics ini berbeda dari eksperimen sebelumnya pada split CP2 671/118/261. Semua artefak lama dipertahankan. Tidak menggabungkan skor dua cohort sebagai perbandingan terkontrol. Semua ini satu seed dengan validation/test yang sudah terekspos, tanpa klaim novelty global atau superioritas konfirmatori.', '',
              '## Resource use', '']
    for n in MODELS:
        s = statuses[n]
        lines.append(f"- {n}: {s['seconds']/60:.1f} minutes; {s['completed_epochs']} epochs; peak allocated {s['peak_cuda_allocated_bytes']/2**30:.2f} GiB; reserved {s['peak_cuda_reserved_bytes']/2**30:.2f} GiB.")
    lines += ['', 'Raw data, checkpoints dan NPZ tetap lokal. Grafik agregat memerlukan direct visual review sebelum delivery akhir.', '',
              'Reproduce: `scripts/run_cp1_matched.py --root outputs/cp2/cp1_matched/<new-run> --stage prepare`, lalu `--stage preflight --device cuda:0` dan `--stage train --device cuda:0`. Gunakan interpreter `.venv-training/bin/python` tanpa mengubah environment.']
    tables = out / 'reports/tables'
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / 'cp1-matched-metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(publication, {'status': 'COMPLETED — ORIGINAL CP1 PROTOCOL MODEL-ONLY SCREEN',
                            'run_root': str(root.relative_to(ROOT)), 'protocol': read(root / 'protocol.json'),
                            'metrics': metrics, 'statuses': statuses, 'selection': selection, 'validation': qa,
                            'figures': artifacts, 'visual_review': 'PENDING direct inspection'})
    (out / 'reports/cp1-matched.md').write_text('\n'.join(lines) + '\n')
    print('Generated verified CP1-matched results and figures')


if __name__ == '__main__':
    main()
