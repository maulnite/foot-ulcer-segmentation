"""CPU-only presentation from verified caches; no training or model selection."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'B1': '#687582', 'H2': '#367bb6', 'A6': '#7659a6'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Preserve completed presentation: {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def save_figure(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        destination = directory / f'{name}.{extension}'
        if destination.exists():
            raise FileExistsError('Use new presentation output instead of overwriting')
        figure.savefig(destination, dpi=170, bbox_inches='tight')
    plt.close(figure)


def audit_report(root):
    audit = read(root / 'false_positive_audit.json')
    target = ROOT / 'reports/cp2-confirmation-fp-audit.md'
    base = ROOT / 'reports'
    if target.exists():
        base = root / 'presentation-audit'
    figure_dir = base / 'figures/cp2-confirmation'
    cases = sorted({r['case'] for r in audit['cases']})
    matrix = np.array([[next(r['fp_pixels'] for r in audit['cases']
                            if r['model'] == model and r['case'] == case)
                        for case in cases] for model in COLORS])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})
    picture = axes[0].imshow(np.log1p(matrix), cmap='Blues', aspect='auto')
    axes[0].set(xticks=range(len(cases)), xticklabels=cases,
                yticks=range(3), yticklabels=list(COLORS), xlabel='GT-empty validation case (n=7)',
                ylabel='Model • cached seed 42', title='Di gambar mana false positive muncul?')
    for i in range(3):
        for j in range(len(cases)):
            axes[0].text(j, i, str(matrix[i, j]), ha='center', va='center',
                         color='white' if np.log1p(matrix[i, j]) > np.log1p(matrix.max()) * .62 else '#17212b')
    figure.colorbar(picture, ax=axes[0], label='log(1 + FP pixels); angka = piksel asli', shrink=.8)
    counts = [audit['summaries'][name]['fp_images'] for name in COLORS]
    axes[1].bar(list(COLORS), counts, color=list(COLORS.values()), width=.55)
    axes[1].set(ylim=(0, 7), yticks=range(8), ylabel='Gambar dengan ≥1 FP pixel / 7 gambar',
                xlabel='Model • threshold ≥0.5', title='Any-FP tetap dihitung sebagai error')
    for i, value in enumerate(counts):
        axes[1].text(i, value + .12, f'{value}/7', ha='center')
    figure.suptitle('Chronic Wound • CP2 validation • REAL CACHED predictions, FRESH geometry audit', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    save_figure(figure, figure_dir, 'cached_empty_fp')
    rows = ['# CP2 — audit false positive sebelum konfirmasi seed', '',
            '**EXECUTED — cached real seed42 validation; tidak ada training/inference baru.**', '',
            'Split tetap 671/118/261, duplicate-RGB-group split, bukan patient-level. '
            'Seluruh 354 probability/GT pairs validation diperiksa ulang beserta checksum, raw mask, threshold dan metrik.', '',
            '| Model | GT-empty dengan FP | Total FP pixels | Largest connected FP component |',
            '|---|---:|---:|---:|']
    for model, summary in audit['summaries'].items():
        rows.append(f'| {model} | {summary["fp_images"]}/7 | {summary["fp_pixels"]} | {summary["largest_component_pixels"]} |')
    rows.extend(['', '![Cached empty FP](figures/cp2-confirmation/cached_empty_fp.png)', '',
                 '**Yang terlihat:** angka per kotak menunjukkan luas prediksi luka pada setiap GT kosong. '
                 'Skala warna logaritmik ditulis eksplisit; angka anotasi memakai jumlah piksel asli.', '',
                 '**Artinya:** jumlah gambar error dan luas error memberi informasi berbeda. '
                 'Satu piksel dan ratusan piksel sama-sama membuat Image Dice gambar kosong menjadi 0. '
                 'Data Dice tetap peka pada jumlah pikselnya.', '',
                 '**Perlu diingat:** hanya tujuh contoh kosong, validation dipakai berulang, dan geometry mask '
                 'tidak membuktikan penyebab medis atau label salah. Tidak ada adjudikasi klinis, relabeling, '
                 'component filtering, atau penyesuaian threshold. Semua kasus ditampilkan tanpa cherry-picking.', '',
                 'Matriks ini menjadi diagnosis awal; kandidat belum diterima. Konfirmasi menggunakan '
                 'seed 42 cached dan seed 43/44 fresh, dengan aturan seleksi tersendiri yang dipraspesifikasikan.'])
    write(base / 'cp2-confirmation-fp-audit.md', '\n'.join(rows) + '\n')
    write(base / 'cp2-confirmation-fp-audit.json', json.dumps(audit, indent=2) + '\n')


def final_report(root):
    results = read(root / 'results.json')
    if results['validation']['status'] != 'passed':
        raise ValueError('Verified results required')
    base = ROOT / 'reports'
    if (base / 'cp2-confirmation.md').exists():
        base = root / 'presentation-final'
    directory = base / 'figures/cp2-confirmation'
    metrics = results['metrics']
    seeds = results['protocol']['seeds']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axes, ('validation', 'test')):
        for name, color in COLORS.items():
            values = [metrics[str(seed)][name][split]['image_dice'] * 100 for seed in seeds]
            ax.plot(seeds, values, marker='o', color=color, label=name)
        ax.set(xlabel='Training seed • fixed CP2 split', ylabel='Image Dice (%)', xticks=seeds,
               title=f'Chronic Wound • {split} • n={118 if split == "validation" else 261} images')
        ax.legend(frameon=False)
        ax.grid(alpha=.18)
    figure.suptitle('B1 vs H2 vs A6 • seed42 CACHED; seed43/44 FRESH training', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    save_figure(figure, directory, 'three_seed_image_dice')
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, seed in enumerate(results['protocol']['fresh_seeds']):
        for j, (name, color) in enumerate(COLORS.items()):
            history = read(root / f'seed-{seed}' / name / 'history.json')
            epochs = [h['epoch'] for h in history]
            ax = axes[i, j]
            ax.plot(epochs, [h['train_loss'] for h in history], color=color, label='Training total loss')
            ax.plot(epochs, [h['validation_loss'] for h in history], color='#ab4e53', label='Validation total loss', linestyle='--')
            chosen = read(root / f'seed-{seed}' / name / 'status.json')['selected_epoch']
            ax.axvline(chosen, color='#65717b', linewidth=1, linestyle=':')
            ax.set(title=f'{name} • seed {seed} • checkpoint {chosen}', xlabel='Epoch', ylabel='Loss (unitless)')
            ax.legend(fontsize=8, frameon=False)
    figure.suptitle('FRESH CP2 training • checkpoint dipilih dari validation Image Dice, bukan minimum loss', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    save_figure(figure, directory, 'training_curves')
    records = []
    lines = ['# CP2 — konfirmasi B1/H2/A6 pada tiga seed', '',
             '**COMPLETED — six new real training runs; three historical seed42 runs preserved.**', '',
             'Same duplicate-group split 671/118/261, corrected focal B1, Image-Dice checkpoint, FP32/TF32, '
             'batch2, Adam, scheduler dan early stopping CP2. Seed mengubah initialization/training randomness, '
             'bukan membership. Semua model memakai bobot awal source yang dipasangkan dalam seed yang sama.', '',
             '| Model | Split | Image Dice mean ± SD | Data Dice mean ± SD | Small Dice mean | Empty-FP images mean |',
             '|---|---|---:|---:|---:|---:|']
    for name in COLORS:
        for split in ('validation', 'test'):
            values = [metrics[str(seed)][name][split] for seed in seeds]
            mean = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            sd = {k: float(np.std([v[k] for v in values], ddof=1)) for k in values[0]}
            lines.append(f'| {name} | {split} | {mean["image_dice"]:.4f} ± {sd["image_dice"]:.4f} | '
                         f'{mean["data_dice"]:.4f} ± {sd["data_dice"]:.4f} | {mean["small_dice"]:.4f} | {mean["empty_fp_images"]:.2f} |')
            for seed, value in zip(seeds, values):
                records.append({'model': name, 'seed': seed, 'split': split, **value})
    decision = results['selection']
    lines.extend(['', f'**Validation-only selection:** {decision["selected_candidate"]}. '
                  f'**Next route for review:** `{decision["next_research_route"]}`. '
                  'Tidak ada training kombinasi/presence otomatis; kegagalan screening seed42 tetap dicatat.', '',
                  '![Three seeds](figures/cp2-confirmation/three_seed_image_dice.png)', '',
                  '**Yang terlihat:** semua seed ditampilkan, termasuk hasil yang lebih buruk. '
                  'Panel test bersifat deskriptif dan dibuka setelah selection tersimpan.', '',
                  '**Artinya:** selisih berpasangan dalam seed adalah pembanding utama; seed terbaik tidak dipilih.', '',
                  '**Perlu diingat:** tiga seed pada satu split tidak membuktikan patient independence. '
                  'Validation digunakan berulang dan kandidat dipilih dari screening sebelumnya; ini konfirmasi '
                  'kestabilan internal yang eksploratif, bukan bukti konfirmatori independen atau manfaat klinis.', '',
                  '![Training](figures/cp2-confirmation/training_curves.png)', '',
                  '**Yang terlihat:** garis vertikal menandai checkpoint validation-Dice, dan H2 total loss '
                  'mencakup auxiliary term 0.1. **Artinya:** nilai loss lintas model tidak menjadi ranking performa. '
                  '**Perlu diingat:** adaptive scheduler/early stopping menghasilkan anggaran update berbeda.', '',
                  f'Artifact QA: **{results["validation"]["status"]}**; run `{results["run_root"]}`. '
                  'Raw clinical media, probabilities dan checkpoints tetap lokal. Direct visual review masih pending.'])
    write(base / 'cp2-confirmation.md', '\n'.join(lines) + '\n')
    write(base / 'cp2-confirmation-results.json', json.dumps(results, indent=2) + '\n')
    path = base / 'tables/cp2-confirmation-metrics.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    if args.audit_only:
        audit_report(args.root)
    else:
        final_report(args.root)


if __name__ == '__main__':
    main()
