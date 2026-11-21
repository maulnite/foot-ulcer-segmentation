"""Verify source-evaluator exports and add descriptive subgroups without refitting."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def audit_original_split(root):
    manifest = read(root / 'split-manifest.json')
    rows, grouped = {}, defaultdict(list)
    for split, members in manifest['splits'].items():
        rows[split] = []
        for row in members:
            with Image.open(ROOT / row['image_path']) as im:
                rgb = np.asarray(im.convert('RGB'))
            with Image.open(ROOT / row['mask_path']) as im:
                mask = np.asarray(im.convert('L'))
            group = hashlib.sha256(rgb.tobytes()).hexdigest()
            item = {**row, 'sample_id': row['name'], 'rgb_group': group,
                    'gt_pixels': int(np.count_nonzero(mask)),
                    'mask_pixels_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
            rows[split].append(item)
            grouped[group].append({**item, 'split': split})
    duplicates = {k: v for k, v in grouped.items() if len(v) > 1}
    cross = {k: v for k, v in duplicates.items() if len({r['split'] for r in v}) > 1}
    conflicts = {k: v for k, v in duplicates.items() if len({r['mask_pixels_sha256'] for r in v}) > 1}
    audit = {'split_fingerprint': manifest['fingerprint'], 'counts': {s: len(v) for s, v in rows.items()},
             'duplicate_RGB_groups': len(duplicates), 'cross_split_RGB_groups': len(cross),
             'conflicting_mask_RGB_groups': len(conflicts), 'cross_split_groups': cross,
             'data_changed_or_removed': 0,
             'interpretation': 'Original CP1 cohort intentionally retained. Exact-RGB overlap and annotation conflicts limit generalization; not patient-independent.'}
    write_json(root / 'analysis_rows.json', rows)
    write_json(root / 'original_split_audit.json', audit)
    return audit


def evaluation_settings():
    # Same final threshold/ordinary counts as CP1; extra metrics do not select epochs.
    return read(ROOT / 'configs/cp2/phase_p.json')['evaluation']


def summarize_exports(root, variant, split):
    run = root / variant
    original = read(run / ('validation.json' if split == 'validation' else 'evaluation-test.json'))
    by_name = {r['name']: r for r in read(root / 'analysis_rows.json')[split]}
    export = run / 'predictions' / split
    index = read(export / 'manifest.json')
    if len(index) != len(by_name) or {r['name'] for r in index} != by_name.keys():
        raise ValueError('Original CP1 prediction population/count differs')
    evaluator = CP2Evaluator(evaluation_settings(), boundaries=True)
    for item in index:
        path = export / item['file']
        if digest_file(path) != item['sha256']:
            raise ValueError('Prediction checksum changed')
        row = by_name[item['name']]
        with Image.open(ROOT / row['mask_path']) as im:
            truth = np.asarray(im.convert('L')) / 255
        with np.load(path, allow_pickle=False) as values:
            np.testing.assert_array_equal(values['truth'], truth)
            np.testing.assert_array_equal(values['prediction'], values['probability'] >= .5)
            if values['probability'].dtype != np.float32:
                raise ValueError('Source FP32 probability dtype changed')
            evaluator.add(row, truth, values['probability'])
    primary = evaluator.region.summary()
    for metric in ('dice', 'iou', 'precision', 'recall'):
        for kind in ('data_based', 'image_based'):
            np.testing.assert_allclose(primary['offline_hard_line'][kind][metric],
                                       original['offline_hard_line'][kind][metric], rtol=0, atol=1e-12)
    if primary['offline_hard_line']['counts'] != original['offline_hard_line']['counts']:
        raise ValueError('Source evaluator counts changed')
    result = {**evaluator.summary(), 'split': split, 'variant': variant,
              'checkpoint_sha256': digest_file(run / 'best.pth'),
              'original_CP1_primary_evaluator_agrees': True}
    destination = run / f'analysis-{split}.json'
    if destination.exists():
        if read(destination) != result:
            raise ValueError('Completed analysis changed')
    else:
        write_json(destination, result)
        with destination.with_suffix('.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evaluator.rows[0]))
            writer.writeheader()
            writer.writerows(evaluator.rows)
    return result


def select(root):
    from foot_ulcer_segmentation.cp2_phase_h import select_candidate
    reports = {n: read(root / n / 'analysis-validation.json') for n in ('C0', 'CH2', 'CA6')}
    profiles = read(root / 'architecture_preflight.json')['profiles']
    decision = select_candidate({'H1': reports['CH2'], 'H2': reports['CA6']}, reports['C0'],
                                {'H0': {'active_forward_parameters': profiles['C0']['active_training_parameters']},
                                 'H1': {'active_forward_parameters': profiles['CH2']['active_training_parameters']},
                                 'H2': {'active_forward_parameters': profiles['CA6']['active_training_parameters']}})
    names = {'H0': 'C0', 'H1': 'CH2', 'H2': 'CA6'}
    decision['selected_candidate'] = names[decision['selected_candidate']]
    decision['decisions'] = {names[k]: v for k, v in decision['decisions'].items()}
    decision['unexecuted'] = []
    decision['caveat'] = 'Exploratory original-CP1-cohort model-only screen with known exact-RGB overlap; not patient-independent'
    return decision
