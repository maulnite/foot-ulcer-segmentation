"""Independent replay of frozen exports, source policy, selection and provenance."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from foot_ulcer_segmentation.checkpoint import AuthorCheckpointPolicy  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_recipe import EXPERIMENTS, read, select_recipe, sources, verify_paired_inputs  # noqa: E402
from validate_cp2_phase_a import validate_exports  # noqa: E402
from validate_cp2_phase_l import compare, compare_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'final_validation.json'
    if path.exists():
        raise FileExistsError('Preserve completed validation evidence')
    config = read(root / 'protocol.json')
    manifest = verify_cp2_manifest(read(root / 'data/manifest.json'))
    assert read(root / 'pretraining_checks.json')['training_sources'] == sources()
    compare(read(root / 'train_paired_inputs.json'), verify_paired_inputs(root))
    selected = read(root / 'selection.json')
    recomputed = select_recipe(root)
    compare({k: v for k, v in selected.items() if k != 'created_utc'},
            {k: v for k, v in recomputed.items() if k != 'created_utc'})
    selected_at = datetime.fromisoformat(selected['created_utc']).timestamp()
    preregistered_at = datetime.fromisoformat(read(root / 'preregistration.json')['created_utc'])
    proofs = []
    for name in EXPERIMENTS:
        run = root / name
        status = read(run / 'status.json')
        assert status['status'] == 'completed' and not status['diagnostic_smoke_only']
        assert status['training_sources'] == sources()
        assert status['checkpoint_reload_passed']
        assert read(run / 'split-manifest.json') == manifest
        assert datetime.fromisoformat(read(run / 'environment.json')['created_at_utc']) > preregistered_at
        initial = read(run / 'initialization_verified.json')
        assert initial['all_original_named_tensors_equal']
        assert initial['source_tensor_sha256'] == config['recipe']['control_initial_tensor_sha256']
        policy, chosen_epoch = AuthorCheckpointPolicy(), None
        history = read(run / 'history.json')
        for entry in history:
            decision = policy.update(entry['validation']['regional'], entry['validation']['source_iou'])
            assert decision == entry['selected_checkpoint']
            compare(policy.state_dict(), entry['policy'])
            if decision:
                chosen_epoch = entry['epoch']
        assert chosen_epoch == status['selected_epoch']
        assert len(history) == status['completed_epochs']
        assert len(history) == config['training']['max_epochs'] or policy.patience_count >= config['training']['early_stopping_patience']
        for checkpoint, expected_epoch in [('best.pth', chosen_epoch), ('last.pth', len(history))]:
            state = torch.load(run / checkpoint, map_location='cpu')
            assert state['identity'] == status['identity']
            assert state['completed_epochs'] == expected_epoch
            del state
        assert digest_file(run / 'best.pth') == status['best_checkpoint_sha256']
        for split in ('validation', 'test'):
            report = read(run / f'evaluation-{split}.json')
            assert report['checkpoint_sha256'] == status['best_checkpoint_sha256']
            evaluator = validate_exports(run / 'predictions' / split, manifest['splits'][split], report, config['evaluation'])
            with (run / f'evaluation-{split}.csv').open(newline='') as stream:
                compare_csv(list(csv.DictReader(stream)), evaluator.rows)
            if split == 'test':
                for item in read(run / 'predictions/test/manifest.json'):
                    assert (run / 'predictions/test' / item['file']).stat().st_mtime >= selected_at
            else:
                assert abs(report['all']['image_based']['dice'] - history[chosen_epoch - 1]['validation_image_dice']) < 1e-10
            proofs.append({'model': name, 'split': split, 'predictions': len(evaluator.rows),
                           'checksums_raw_GT_threshold_metrics_CSV': 'passed'})
    write_json(path, {'status': 'passed', 'source_policy_replayed': True, 'validation_selection_replayed': True,
                      'prediction_exports': proofs, 'no_patient_independence_claim': True,
                      'manifest_fingerprint': manifest['fingerprint']})
    print('PASS: independent recipe checkpoint policy, selection, exports and metrics')


if __name__ == '__main__':
    main()
