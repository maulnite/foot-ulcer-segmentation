"""CPU-only recovery of the JSON-key comparison in the frozen CP2 validator.

No training/inference/resume, source edits or replacement of historical records.
The original validator's code object is reused with only its selection result
converted to the same JSON representation as the persisted decision.
"""

import argparse
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_confirmation import (  # noqa: E402
    MODELS, compact, model_directory, preservation, read, sources, utc,
)


def json_value(value):
    """Use JSON's documented string-key representation; keep all values exact."""
    return json.loads(json.dumps(value, allow_nan=False))


def json_safe_validator(original):
    selector = original.__globals__['select']

    def serialized_selection(*args, **kwargs):
        return json_value(selector(*args, **kwargs))

    namespace = {**original.__globals__, 'select': serialized_selection}
    return types.FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / '.confirmation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    workflow = root / 'postprocess_workflow.json'
    if workflow.exists():
        raise FileExistsError('Preserve existing recovery evidence; inspect it before proceeding')
    if read(root / 'preregistration.json')['training_sources'] != sources():
        raise ValueError('Frozen training and reporting sources changed')
    failed = read(root / 'train_workflow.json')
    if failed['status'] != 'failed' or 'Persisted validation-only decision differs' not in failed.get('traceback', ''):
        raise ValueError('This recovery only handles the documented JSON-key failure')
    for filename in ('final_validation.json', 'results.json'):
        if (root / filename).exists():
            raise FileExistsError('Do not overwrite completed validation/results')
    # Inventory every existing run file, including all training outputs and the
    # original failed workflow. New recovery files are excluded by construction.
    inventory = {str(p.relative_to(root)): digest_file(p)
                 for p in root.rglob('*') if p.is_file()}
    write_json(root / 'before_postprocess_sha256.json', inventory)
    write_json(workflow, {'status': 'running', 'started_utc': utc(), 'CPU_only': True})
    try:
        original = runpy.run_path(str(ROOT / 'scripts/run_cp2_confirmation.py'))
        protocol = read(root / 'protocol.json')
        manifest = read(root / 'data/manifest.json')
        validation = {s: {n: read(model_directory(root, protocol, s, n) / 'evaluation-validation.json')
                          for n in MODELS} for s in protocol['seeds']}
        profiles = read(root / 'architecture_preflight.json')['per_seed_profiles']['43']
        decision = original['select'](validation, profiles, protocol)
        saved = read(root / 'selection.json')
        decision['created_utc'] = saved['created_utc']
        if json_value(decision) != saved or decision['means'] != saved['means']:
            raise ValueError('A genuine metric/decision discrepancy must NOT be normalized away')
        proof = {'cause': 'JSON converts integer dictionary keys to strings',
                 'raw_python_decision_equal': decision == saved,
                 'json_roundtrip_decision_equal': json_value(decision) == saved,
                 'all_scientific_values_exactly_equal': decision['means'] == saved['means'],
                 'selected_candidate_unchanged': saved['selected_candidate'],
                 'training_or_inference_executed': False,
                 'frozen_validator_code_object_unchanged': True,
                 'repair_source_sha256': digest_file(Path(__file__)),
                 'original_failed_workflow_preserved': True}
        write_json(root / 'postprocess_serialization_proof.json', proof)
        validator = json_safe_validator(original['full_validation'])
        assert validator.__code__ is original['full_validation'].__code__
        qa = validator(root, protocol, manifest)
        write_json(root / 'final_validation.json', qa)
        metrics = {str(s): {n: {split: compact(read(model_directory(root, protocol, s, n) /
                   f'evaluation-{split}.json')) for split in ('validation', 'test')} for n in MODELS}
                   for s in protocol['seeds']}
        write_json(root / 'results.json', {
            'status': 'COMPLETED — THREE-SEED CP2 REPLICATION; POSTPROCESS RECOVERED',
            'run_root': str(root.relative_to(ROOT)), 'protocol': protocol,
            'metrics': metrics, 'selection': saved, 'validation': qa,
            'postprocess_recovery': proof, 'visual_review': 'PENDING direct inspection'})
        subprocess.run([str(ROOT / '.venv-analysis/bin/python'),
                        str(ROOT / 'scripts/report_cp2_confirmation.py'), '--root', str(root)], check=True)
        altered = [name for name, expected in inventory.items()
                   if digest_file(root / name) != expected]
        if altered:
            raise ValueError(f'Existing run files changed: {altered}')
        write_json(root / 'final_preservation.json', {
            **preservation(root), 'existing_run_files_unchanged': len(inventory)})
        write_json(workflow, {'status': 'completed', 'finished_utc': utc(), 'CPU_only': True,
                   'original_training_workflow_status': 'failed; retained as historical evidence',
                   'final_validation': 'passed', 'scientific_decision_unchanged': True,
                   'training_or_inference_executed': False})
        print('COMPLETED — CPU postprocessing recovered; all training artifacts unchanged', flush=True)
    except BaseException:
        write_json(workflow, {'status': 'failed', 'utc': utc(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
