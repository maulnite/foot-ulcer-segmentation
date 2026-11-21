"""Independent source-policy/checkpoint/export validation for CP1-matched runs."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from foot_ulcer_segmentation.checkpoint import AuthorCheckpointPolicy  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp1_matched import MODELS, read, runtime_sources, validate_protocol  # noqa: E402
from cp1_matched_analysis import select, summarize_exports  # noqa: E402
from run_cp1_matched import check_inputs, protected  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / 'final_validation.json'
    if target.exists():
        raise FileExistsError('Preserve completed validation')
    protocol = read(root / 'protocol.json')
    manifest = validate_protocol(protocol)
    if read(root / 'pretraining_checks.json')['sources'] != runtime_sources():
        raise ValueError('Runtime changed after successful preflight')
    actual = select(root)
    saved = read(root / 'selection.json')
    assert {k: v for k, v in actual.items() if k != 'created_utc'} == {k: v for k, v in saved.items() if k != 'created_utc'}
    selected_at = datetime.fromisoformat(saved['created_utc']).timestamp()
    proofs, initial_rng = [], set()
    for name in MODELS:
        run = root / name
        status = read(run / 'status.json')
        assert status['status'] == 'completed' and not status['smoke_only']
        config = read(run / 'config.json')
        assert all(config[k] == v for k, v in protocol['baseline'].items())
        assert read(run / 'split-manifest.json') == manifest
        proof = read(run / 'model_verified.json')
        assert proof['all_pairable_source_tensors_equal']
        assert proof['source_initial_state_sha256'] == protocol['expected_initial_state_sha256']
        initial_rng.add(proof['shared_post_constructor_rng_sha256'])
        runtime = read(run / 'runtime.json')
        assert runtime['sources'] == runtime_sources()
        assert runtime['every_original_CP1_config_key_equal']
        assert runtime['source_engine_adaptation']['changed_AST_nodes'] == (0 if name == 'C0' else 1)
        history = read(run / 'history.json')
        policy, chosen = AuthorCheckpointPolicy(), None
        for row in history:
            loss_key = sorted(row['validation'])[0]
            decision = policy.update(row['validation'][loss_key], row['validation']['iou_score'])
            assert decision == row['selected_checkpoint']
            if decision:
                chosen = row['epoch']
        assert chosen == status['selected_epoch']
        assert len(history) == status['completed_epochs']
        assert len(history) == 200 or policy.patience_count == 30
        inputs = read(run / 'training_inputs.json')
        assert len(inputs) == len(history) and all(r['images'] == 707 for r in inputs)
        for filename, epoch in [('best.pth', chosen), ('last.pth', len(history))]:
            state = torch.load(run / filename, map_location='cpu')
            assert state['completed_epochs'] == epoch
            assert state['config_fingerprint'] == digest_json(config)
            assert state['split_fingerprint'] == protocol['split_fingerprint']
            del state
        assert digest_file(run / 'best.pth') == read(run / 'completion.json')['checkpoint_sha256']
        for split in ('validation', 'test'):
            report = summarize_exports(root, name, split)
            assert report['all']['images'] == len(manifest['splits'][split])
            if split == 'test':
                for item in read(run / 'predictions/test/manifest.json'):
                    assert (run / 'predictions/test' / item['file']).stat().st_mtime >= selected_at
            proofs.append({'variant': name, 'split': split, 'predictions': report['all']['images'],
                           'checksums_GT_threshold_primary_subgroup_metrics': 'passed'})
    assert len(initial_rng) == 1
    paired = check_inputs(root)
    assert paired == read(root / 'paired_training_inputs.json')
    preservation = protected(root)
    write_json(target, {'status': 'passed', 'source_checkpoint_policy_replayed': True,
                       'every_original_CP1_config_key_and_split_order_equal': True,
                       'constructor_tensors_and_RNG_paired': True, 'paired_inputs': paired,
                       'prediction_exports': proofs, 'preservation': preservation,
                       'known_RGB_overlap_retained_and_disclosed': True,
                       'selection_before_test_verified': True})
    print('PASS: original CP1 protocol, checkpoint policy, paired inputs and all exports')


if __name__ == '__main__':
    main()
