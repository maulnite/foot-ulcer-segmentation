"""CPU-only paired analysis of completed recipe-control CSVs; no reselection."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if read(root / 'train_workflow.json')['status'] != 'completed' or read(root / 'final_validation.json')['status'] != 'passed':
        raise ValueError('Require completed independently validated real-data experiment')
    destination = ROOT / 'reports/cp2-cp1-recipe-analysis.json'
    if destination.exists():
        raise FileExistsError('Preserve completed analysis')
    contrasts, inputs, pairwise = {}, {}, []
    for split in ('validation', 'test'):
        rows = {}
        for name in ('R0', 'RH2', 'RA6'):
            path = root / name / f'evaluation-{split}.csv'
            with path.open(newline='') as stream:
                values = list(csv.DictReader(stream))
            rows[name] = sorted(values, key=lambda r: r['name'])
            assert len({r['name'] for r in values}) == len(values)
            assert len({r['rgb_group'] for r in values}) == len(values)
            report = read(root / name / f'evaluation-{split}.json')
            assert abs(np.mean([float(r['dice']) for r in values]) - report['all']['image_based']['dice']) < 1e-12
            inputs[str(path.relative_to(ROOT))] = digest_file(path)
            target = ROOT / f'reports/tables/cp2-cp1-recipe-{name.lower()}-{split}.csv'
            if target.exists():
                raise FileExistsError('Preserve published per-image table')
            shutil.copyfile(path, target)
        for name in ('RH2', 'RA6'):
            assert [(r['name'], r['rgb_group']) for r in rows[name]] == [(r['name'], r['rgb_group']) for r in rows['R0']]
            delta = np.array([float(a['dice']) - float(b['dice']) for a, b in zip(rows[name], rows['R0'])])
            indices = np.random.default_rng(42).integers(0, len(delta), (2000, len(delta)))
            samples = delta[indices].mean(axis=1)
            contrasts[f'{name}_minus_R0_{split}'] = {
                'delta_image_dice': float(delta.mean()),
                'percentile_95_interval': np.quantile(samples, [.025, .975]).tolist(),
                'improved_images': int((delta > 0).sum()), 'worse_images': int((delta < 0).sum()),
                'tied_images': int((delta == 0).sum()), 'images': len(delta),
            }
    histories = {n: read(root / n / 'history.json') for n in ('R0', 'RH2', 'RA6')}
    for a, b in (('R0', 'RH2'), ('R0', 'RA6'), ('RH2', 'RA6')):
        common = list(zip(histories[a], histories[b]))
        assert all(all(x[k] == y[k] for k in ('training_order_sha256', 'augmented_input_sha256')) for x, y in common)
        pairwise.append({'first': a, 'second': b, 'common_epochs': len(common), 'actual_inputs_equal': True})
    historical = {}
    for name, folder in {
        'B0': 'phase_p/20261107T170015Z/B0', 'B1': 'phase_p/20261107T170015Z/B1',
        'H2': 'phase_h/20261112T105427Z/H2', 'A6': 'phase_a/20261108T185142Z/A6',
    }.items():
        directory = ROOT / 'outputs/cp2' / folder
        assert read(directory / 'split-manifest.json')['fingerprint'] == read(root / 'data/manifest.json')['fingerprint']
        historical[name] = {s: read(directory / f'evaluation-{s}.json') for s in ('validation', 'test')}
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'run_root': str(root.relative_to(ROOT)), 'paired_contrasts': contrasts,
              'input_csv_sha256': inputs, 'all_pairwise_common_epoch_inputs': pairwise,
              'historical_same_cohort_metrics': historical,
              'bootstrap': {'seed': 42, 'replicates': 2000, 'unit': 'paired exact-RGB group; one retained image per group',
                            'interpretation': 'Descriptive intervals only. No patient IDs; excludes seed, model/checkpoint selection and validation reuse uncertainty. Not confirmatory significance.'},
              'selection_sha256': digest_file(root / 'selection.json'),
              'training_status_test_count_scope': 'test_images_evaluated=0 in training status describes training-stage evaluation only; subsequent independently verified exports contain 261 test predictions per model'}
    write_json(destination, result)
    write_json(root / 'posthoc_analysis.json', result)
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
