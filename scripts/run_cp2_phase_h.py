"""Sequential standalone H1/H2 supervision screen; historical runs remain untouched."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import xml.etree.ElementTree as ET

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_heads import build_head_model  # noqa: E402
from foot_ulcer_segmentation.cp2_phase_h import (  # noqa: E402
    EXPERIMENTS, evaluate_candidate, expansion_decision, phase_h_sources, prepare, read,
    run_experiment, select_candidate, verify_inputs,
)
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def preserve(root):
    path = root / 'protected_prior_sha256.json'
    if not path.exists():
        names = set(read(ROOT / 'outputs/cp2/phase_a/20261108T185142Z/protected_prior_sha256.json'))
        names.update(subprocess.check_output(['git', 'ls-files'], text=True).splitlines())
        names.add('Pengembangan_Novelty_dan_Eksperimen_CP2_FUSegNet.md')
        for directory in ('phase_p', 'phase_l', 'phase_a'):
            names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'outputs/cp2' / directory).rglob('*') if p.is_file())
        names = {n for n in names if not any(t in n for t in ('cp2_phase_h', 'cp2-phase-h', 'cp2_heads', 'cp2/phase_h.json'))}
        with ThreadPoolExecutor(max_workers=6) as pool:
            inventory = dict(pool.map(lambda n: (n, digest_file(ROOT / n)), sorted(names)))
        write_json(path, inventory)
    inventory = read(path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        changed = [n for n, same in pool.map(lambda n: (n, digest_file(ROOT / n) == inventory[n]), inventory) if not same]
    if changed:
        raise ValueError(f'Historical/user assets changed: {changed}')
    return len(inventory)


class MaskOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images, auxiliary=False).main_logits


def architecture_preflight(root, config):
    path = root / 'architecture_preflight.json'
    if path.exists():
        value = read(path)
        if value['training_sources'] != phase_h_sources():
            raise ValueError('Preserve prior diagnostic profile; changed source requires a new run')
        return value
    setup(config['seed'], config['cpu_threads'])
    initial = torch.load(root / 'initial_weights.pth', map_location='cpu')
    inputs = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    profiles, inference, proofs, outputs = {}, {}, {}, {}
    for name in ('H0', *EXPERIMENTS):
        model, proofs[name] = build_head_model(config, name, initial)
        profile = model_profile(model, inputs)
        profile.pop('decoder_calls')  # frozen counter uses unwrapped source names
        profiles[name] = profile
        inference[name] = model_profile(MaskOnly(model), inputs)
        inference[name].pop('decoder_calls')
        model.eval()
        with torch.no_grad():
            values = model(inputs)
            bare = model(inputs, auxiliary=False)
            assert torch.equal(values.main_logits, bare.main_logits)
            assert not model.base.decoder.blocks[3]._forward_hooks
        outputs[name] = hashlib.sha256(values.main_logits.numpy().tobytes()).hexdigest()
        if name != 'H0':
            assert list(values.auxiliary_logits.shape) == [1, 1, 224, 224]
        del model, values, bare
    assert len(set(outputs.values())) == 1
    assert proofs['H1']['head_tensor_sha256'] == proofs['H2']['head_tensor_sha256']
    assert profiles['H1']['active_forward_parameters'] == profiles['H0']['active_forward_parameters'] + 33
    assert profiles['H1'] == profiles['H2']
    assert inference['H1']['active_forward_parameters'] == profiles['H0']['active_forward_parameters']
    result = {'status': 'passed', 'profiles': profiles, 'mask_only_inference_profiles': inference,
              'initialization': proofs, 'initial_main_output_sha256': outputs,
              'initial_main_predictions_H0_H1_H2_equal': True, 'paired_auxiliary_head_tensors': True,
              'head_removal_preserves_main_forward': True, 'training_sources': phase_h_sources(),
              'fixture': 'SYNTHETIC linspace for structural engineering only, not scientific scores'}
    write_json(path, result)
    return result


def preflight(root, config, manifest, device):
    gate = root / 'pretraining_checks.json'
    if gate.exists():
        value = read(gate)
        if value['training_sources'] != phase_h_sources() or value['protocol_fingerprint'] != digest_json(config):
            raise ValueError('Preflight source/config changed')
        return value
    protected = preserve(root)
    architecture_preflight(root, config)
    with (root / 'pretraining_tests.log').open('x') as stream:
        subprocess.run([str(ROOT / '.venv-xfuseg/bin/python'), '-m', 'pytest', '-o', 'addopts=', '-q',
                        f'--junitxml={root / "pretraining_tests.xml"}'], stdout=stream, stderr=subprocess.STDOUT, check=True)
    for environment in ('.venv-training', '.venv-analysis'):
        subprocess.run([str(ROOT / environment / 'bin/python'), '-m', 'pip', 'check'], check=True)
    for name in EXPERIMENTS:
        run_experiment(config, manifest, root, name, device, smoke=True)
    paired = verify_inputs(config, root, list(EXPERIMENTS), smoke=True)
    write_json(root / 'smoke_paired_controls.json', paired)
    suites = list(ET.parse(root / 'pretraining_tests.xml').getroot().iter('testsuite'))
    tests = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'errors', 'failures', 'skipped')}
    if tests['errors'] or tests['failures']:
        raise ValueError('Engineering tests failed')
    value = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'historical_files_unchanged': protected, 'tests': tests, 'real_data_smoke_models': list(EXPERIMENTS),
             'training_sources': phase_h_sources(), 'protocol_fingerprint': digest_json(config),
             'manifest_fingerprint': manifest['fingerprint']}
    write_json(gate, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/cp2/phase_h.json'))
    parser.add_argument('--stage', choices=('prepare', 'preflight', 'train', 'status'), default='prepare')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    root, config = args.root.resolve(), read(args.config)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == 'status':
        print(json.dumps({n: read(root / n / 'status.json') if (root / n / 'status.json').exists()
                          else {'status': 'not_started'} for n in EXPERIMENTS}, indent=2))
        return
    lock = (root / '.phase_h.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(config, root)
    if args.stage == 'prepare':
        preserve(root)
        print('Prepared H0 cached B1 control and protected prior assets')
        return
    status_path = root / f'{args.stage}_workflow.json'
    if status_path.exists() and read(status_path)['status'] == 'completed':
        if read(root / 'pretraining_checks.json')['training_sources'] != phase_h_sources():
            raise ValueError('Completed workflow source mismatch')
        print('CACHED completed workflow')
        return
    stop = threading.Event()

    def monitor():
        with (root / f'{args.stage}_gpu_samples.jsonl').open('a', buffering=1) as stream:
            while not stop.is_set():
                sample = gpu_snapshot()
                shm = os.statvfs('/dev/shm')
                sample['shm_free_bytes'] = shm.f_bavail * shm.f_frsize
                stream.write(json.dumps(sample) + '\n')
                stop.wait(5)

    write_json(status_path, {'status': 'running', 'stage': args.stage, 'pid': os.getpid(),
                             'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        if args.device.startswith('cuda'):
            free, _ = torch.cuda.mem_get_info()
            if free < 6 * 1024**3:
                raise RuntimeError('Less than 6 GiB free; no batch/precision fallback')
            threading.Thread(target=monitor, daemon=True).start()
        if args.stage == 'preflight':
            preflight(root, config, manifest, args.device)
        else:
            proof = read(root / 'pretraining_checks.json')
            if proof['status'] != 'passed' or proof['training_sources'] != phase_h_sources() or proof['protocol_fingerprint'] != digest_json(config):
                raise ValueError('Frozen pretraining gate incompatible')
            preregistration = read(root / 'preregistration.json')
            if preregistration['training_sources'] != phase_h_sources() or preregistration['protocol_fingerprint'] != digest_json(config):
                raise ValueError('Preregister executed code/config before full training')
            snapshot = root / 'training_code_snapshot'
            if not snapshot.exists():
                for name in phase_h_sources():
                    path = snapshot / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((ROOT / name).read_bytes())
            run_experiment(config, manifest, root, 'H1', args.device, resume=args.resume)
            first = evaluate_candidate(config, manifest, root, 'H1', 'validation', args.device)
            baseline = read(root / 'control_reference.json')['validation']
            decision = expansion_decision(first, baseline, config)
            decision_path = root / 'expansion_decision.json'
            if decision_path.exists():
                previous = read(decision_path)
                if previous != {**decision, 'created_utc': previous['created_utc']}:
                    raise ValueError('Persisted H1 expansion changed')
            else:
                write_json(root / 'H1_validation_diagnosis.json', {
                    'status': 'H1 full training analyzed before H2', 'validation_only': True,
                    'H1': first, 'H0': baseline, 'decision': decision, 'new_test_access': False})
                write_json(decision_path, decision)
            print('H1 VALIDATION DIAGNOSIS: ' + json.dumps(decision), flush=True)
            names = ['H1', *decision['remaining_models']]
            for name in names[1:]:
                run_experiment(config, manifest, root, name, args.device, resume=args.resume)
            validation = {name: evaluate_candidate(config, manifest, root, name, 'validation', args.device) for name in names}
            write_json(root / 'train_paired_controls.json', verify_inputs(config, root, names))
            selection = select_candidate(validation, baseline, read(root / 'architecture_preflight.json')['profiles'])
            selected_path = root / 'selection.json'
            if selected_path.exists():
                previous = read(selected_path)
                if previous != {**selection, 'created_utc': previous['created_utc']}:
                    raise ValueError('Persisted selection changed')
            else:
                write_json(selected_path, selection)
            for name in names:
                evaluate_candidate(config, manifest, root, name, 'test', args.device)
                write_json(root / name / 'final_evaluation.json', {s: read(root / name / f'evaluation-{s}.json') for s in ('validation', 'test')})
        write_json(status_path, {'status': 'completed', 'stage': args.stage, 'pid': os.getpid(),
                                 'finished_utc': datetime.now(timezone.utc).isoformat()})
    except Exception:
        failure = {'status': 'failed', 'stage': args.stage, 'pid': os.getpid(),
                   'utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc()}
        write_json(status_path, failure)
        print(json.dumps(failure), flush=True)
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
