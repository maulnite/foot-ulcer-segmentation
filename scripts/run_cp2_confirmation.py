"""Sequential six-run CP2 seed replication; cached seed42 and all old assets immutable."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_attention import model_profile  # noqa: E402
from foot_ulcer_segmentation.cp2_confirmation import (  # noqa: E402
    MODELS, audit_false_positives, build, check_pairing, compact, evaluate, model_directory,
    prepare, preservation, read, select, sources, train, utc, verify_exports,
)
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402


def child(args, stage, seed, variant):
    root = args.root
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(root),
               '--config', str(args.config.resolve()), '--worker', stage,
               '--seed', str(seed), '--variant', variant, '--device', args.device]
    if args.resume:
        command.append('--resume')
    label = f'{stage}-{seed}-{variant}'
    log = root / (label + '.log')
    with log.open('a', buffering=1) as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / 'active_process.json', {'stage': stage, 'seed': seed, 'variant': variant,
                   'pid': process.pid, 'log': log.name, 'started_utc': utc()})
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; inspect {log.name}; no automatic config changes')


def profile(root, protocol):
    output = root / 'architecture_preflight.json'
    if output.exists():
        return read(output)
    all_profiles, all_proofs, values = {}, {}, {}
    inputs = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    for seed in protocol['fresh_seeds']:
        config = read(root / f'seed-{seed}/config.json')
        setup(seed, config['cpu_threads'])
        initial = torch.load(root / f'seed-{seed}/initial_weights.pth', map_location='cpu')
        proofs, profiles, outputs = {}, {}, {}
        for name in MODELS:
            model, _, proofs[name] = build(config, name, initial)
            profiles[name] = model_profile(model, inputs)
            model.eval()
            with torch.no_grad():
                outputs[name] = model(inputs).sigmoid().cpu()
            if name == 'H2':
                with torch.no_grad():
                    assert torch.equal(outputs[name], model(inputs, auxiliary=False).sigmoid().cpu())
            del model
        if not torch.equal(outputs['B1'], outputs['H2']):
            raise ValueError('Auxiliary head changed primary initialization')
        if len({p['source_tensor_sha256'] for p in proofs.values()}) != 1:
            raise ValueError('Unpaired initial source tensors')
        all_profiles[str(seed)] = profiles
        all_proofs[str(seed)] = proofs
        values[str(seed)] = {'B1_H2_main_forward_equal': True}
    result = {'status': 'passed', 'per_seed_profiles': all_profiles, 'initialization': all_proofs,
              'checks': values, 'sources': sources(),
              'fixture': 'SYNTHETIC linspace input for engineering checks only; no scientific scores'}
    write_json(output, result)
    return result


def cached_reference(root, protocol):
    records = {}
    for name in MODELS:
        directory = model_directory(root, protocol, 42, name)
        records[name] = {'source': str(directory.relative_to(ROOT)), 'mode': 'CACHED real seed42; never retrained',
                         'status': read(directory / 'status.json'),
                         'status_sha256': digest_file(directory / 'status.json'),
                         'history_sha256': digest_file(directory / 'history.json'),
                         'checkpoint_sha256': digest_file(directory / 'best.pth')}
    write_json(root / 'cached_seed42.json', records)
    write_json(root / 'cached_pairing.json', check_pairing(root, protocol, 42))


def full_validation(root, protocol, manifest):
    # Selection is recomputed BEFORE reading any test scores here.
    validation = {seed: {n: read(model_directory(root, protocol, seed, n) / 'evaluation-validation.json')
                         for n in MODELS} for seed in protocol['seeds']}
    profiles = read(root / 'architecture_preflight.json')['per_seed_profiles']['43']
    decision = select(validation, profiles, protocol)
    saved = read(root / 'selection.json')
    if decision != {**saved, 'created_utc': decision['created_utc']}:
        raise ValueError('Persisted validation-only decision differs')
    checks = []
    for seed in protocol['seeds']:
        check_pairing(root, protocol, seed)
        for name in MODELS:
            directory = model_directory(root, protocol, seed, name)
            history = read(directory / 'history.json')
            status = read(directory / 'status.json')
            chosen = max(history, key=lambda e: e['validation_image_dice'])
            if chosen['epoch'] != status['selected_epoch']:
                raise ValueError('Not the earliest best validation Dice checkpoint')
            checkpoint = torch.load(directory / 'best.pth', map_location='cpu')
            if checkpoint['completed_epochs'] != status['selected_epoch']:
                raise ValueError('Checkpoint epoch differs')
            del checkpoint
            if seed != 42:
                if status['training_sources'] != sources():
                    raise ValueError('Training sources changed')
                if status['diagnostic_smoke_only'] or not status['checkpoint_replay_exact']:
                    raise ValueError('Actual training/reload proof required')
                prereg = read(root / 'preregistration.json')['registered_before_full_training_utc']
                start = read(directory / 'environment.json')['created_at_utc']
                if datetime.fromisoformat(start) <= datetime.fromisoformat(prereg):
                    raise ValueError('Preregistration did not precede full training')
            for split in ('validation', 'test'):
                file = directory / f'evaluation-{split}.json'
                if seed != 42 and split == 'test' and file.stat().st_mtime < datetime.fromisoformat(saved['created_utc']).timestamp():
                    raise ValueError('New test opened before selection')
                checks.append({'seed': seed, 'model': name, **verify_exports(directory, split, manifest,
                               read(ROOT / protocol['base_config'])['evaluation'])})
    return {'status': 'passed', 'created_utc': utc(), 'exports': checks,
            'new_prediction_count': sum(c['predictions'] for c in checks if c['seed'] != 42),
            'cached_prediction_count': sum(c['predictions'] for c in checks if c['seed'] == 42),
            'selection_before_new_test': True, 'earliest_best_checkpoint_replayed': True,
            'preservation': preservation(root), 'patient_independence_claimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/cp2/confirmation.json'))
    parser.add_argument('--stage', choices=('prepare', 'audit', 'preflight', 'train', 'status'), default='prepare')
    parser.add_argument('--worker', choices=('smoke', 'train', 'validation', 'test'))
    parser.add_argument('--seed', type=int, choices=(43, 44))
    parser.add_argument('--variant', choices=MODELS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    root = args.root = args.root.resolve()
    protocol = read(args.config)
    if args.worker:
        if args.seed is None or args.variant is None:
            raise ValueError('Worker requires seed and variant')
        if read(root / 'preregistration.json')['training_sources'] != sources():
            raise ValueError('Worker source differs from registration')
        if args.worker in ('smoke', 'train'):
            train(root, protocol, args.seed, args.variant, args.device, args.worker == 'smoke', args.resume)
        else:
            evaluate(root, protocol, args.seed, args.variant, args.worker, args.device)
        return
    if args.stage == 'status':
        print(json.dumps({f'{s}-{n}': read(root / f'seed-{s}' / n / 'status.json')
                          if (root / f'seed-{s}' / n / 'status.json').exists() else {'status': 'queued'}
                          for s in protocol['fresh_seeds'] for n in MODELS}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.confirmation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'prepare':
        prepare(root, protocol)
        cached_reference(root, protocol)
        write_json(root / 'preservation_prepare.json', preservation(root))
        print('PREPARED original CP2 split and paired seed43/44 initializations')
        return
    if read(root / 'protocol.json') != protocol:
        raise ValueError('Prepared configuration changed')
    registered = root / 'preregistration.json'
    if registered.exists() and read(registered)['training_sources'] != sources():
        raise ValueError('Registered source changed; use a new run root')
    if not registered.exists():
        write_json(registered, {'registered_before_full_training_utc': utc(), 'training_sources': sources(),
                   'protocol_fingerprint': digest_json(protocol),
                   'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'historical_validation_already_observed': True, 'scope': protocol['scope']})
        for name in sources():
            target = root / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    workflow = root / (args.stage + '_workflow.json')
    if workflow.exists() and read(workflow)['status'] == 'completed':
        print('CACHED', args.stage)
        return
    stop = threading.Event()
    def monitor():
        with (root / (args.stage + '_gpu_samples.jsonl')).open('a', buffering=1) as stream:
            while not stop.is_set():
                sample = gpu_snapshot()
                shm = os.statvfs('/dev/shm')
                sample['shm_free_bytes'] = shm.f_bavail * shm.f_frsize
                stream.write(json.dumps(sample) + '\n')
                stop.wait(5)
    write_json(workflow, {'status': 'running', 'pid': os.getpid(), 'started_utc': utc()})
    try:
        manifest = read(root / 'data/manifest.json')
        if args.stage == 'audit':
            audit_false_positives(root, protocol)
            subprocess.run([str(ROOT / '.venv-analysis/bin/python'), 'scripts/report_cp2_confirmation.py',
                            '--root', str(root), '--audit-only'], check=True)
        elif args.stage == 'preflight':
            if not (root / 'false_positive_audit.json').exists():
                raise ValueError('Finish cached validation FP audit first')
            with (root / 'tests.log').open('x') as stream:
                subprocess.run([str(ROOT / '.venv-xfuseg/bin/python'), '-m', 'pytest', '-o', 'addopts=', '-q',
                                f'--junitxml={root / "tests.xml"}'], stdout=stream, stderr=subprocess.STDOUT, check=True)
            for environment in ('.venv-training', '.venv-analysis'):
                subprocess.run([str(ROOT / environment / 'bin/python'), '-m', 'pip', 'check'], check=True)
            profile(root, protocol)
            if args.device.startswith('cuda'):
                if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
                    raise RuntimeError('Less than 6 GiB free; do not change batch or precision')
                threading.Thread(target=monitor, daemon=True).start()
            for name in MODELS:
                child(args, 'smoke', 43, name)
            paired = check_pairing(root, protocol, 43, smoke=True)
            write_json(root / 'pretraining_checks.json', {'status': 'passed', 'created_utc': utc(),
                       'sources': sources(), 'protocol_fingerprint': digest_json(protocol),
                       'smoke_paired_inputs': paired, 'real_data_smokes': list(MODELS),
                       'preservation': preservation(root)})
        else:
            gate = read(root / 'pretraining_checks.json')
            if gate['status'] != 'passed' or gate['sources'] != sources() or gate['protocol_fingerprint'] != digest_json(protocol):
                raise ValueError('Successful unchanged preflight required')
            if args.device.startswith('cuda'):
                if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
                    raise RuntimeError('Less than 6 GiB free; do not change batch or precision')
                threading.Thread(target=monitor, daemon=True).start()
            for seed in protocol['fresh_seeds']:
                for name in MODELS:
                    child(args, 'train', seed, name)
                    child(args, 'validation', seed, name)
                write_json(root / f'paired_inputs-{seed}.json', check_pairing(root, protocol, seed))
            validation = {seed: {n: read(model_directory(root, protocol, seed, n) / 'evaluation-validation.json')
                                 for n in MODELS} for seed in protocol['seeds']}
            profiles = read(root / 'architecture_preflight.json')['per_seed_profiles']['43']
            decision = select(validation, profiles, protocol)
            if (root / 'selection.json').exists():
                previous = read(root / 'selection.json')
                if previous != {**decision, 'created_utc': previous['created_utc']}:
                    raise ValueError('Stored decision changed')
            else:
                write_json(root / 'selection.json', decision)
            for seed in protocol['fresh_seeds']:
                for name in MODELS:
                    child(args, 'test', seed, name)
            qa = full_validation(root, protocol, manifest)
            write_json(root / 'final_validation.json', qa)
            metrics = {str(seed): {n: {split: compact(read(model_directory(root, protocol, seed, n) /
                       f'evaluation-{split}.json')) for split in ('validation', 'test')} for n in MODELS}
                       for seed in protocol['seeds']}
            write_json(root / 'results.json', {'status': 'COMPLETED — THREE-SEED CP2 REPLICATION',
                       'run_root': str(root.relative_to(ROOT)), 'protocol': protocol,
                       'metrics': metrics, 'selection': read(root / 'selection.json'), 'validation': qa,
                       'visual_review': 'PENDING direct inspection'})
            subprocess.run([str(ROOT / '.venv-analysis/bin/python'), 'scripts/report_cp2_confirmation.py',
                            '--root', str(root)], check=True)
            write_json(root / 'final_preservation.json', preservation(root))
        write_json(workflow, {'status': 'completed', 'finished_utc': utc()})
        print('COMPLETED', args.stage, flush=True)
    except BaseException:
        write_json(workflow, {'status': 'failed', 'utc': utc(), 'traceback': traceback.format_exc()})
        raise
    finally:
        stop.set()


if __name__ == '__main__':
    main()
