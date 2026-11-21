"""Exact CP1 configuration/cohort with only H2 or CBAM changed, in fresh workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foot_ulcer_segmentation import engine, reference  # noqa: E402
from foot_ulcer_segmentation.common import digest_file, digest_json, write_json  # noqa: E402
from foot_ulcer_segmentation.cp1_matched import (  # noqa: E402
    MODELS, PARAMETERS, instrumented_namespace, read, run_config, runtime_sources,
    source_training_function, state_digest, validate_protocol,
)


def protected(root):
    inventory = read(root / 'protected_before.json')
    with ThreadPoolExecutor(max_workers=6) as pool:
        changed = [n for n, ok in pool.map(lambda n: (n, digest_file(ROOT / n) == inventory[n]), inventory) if not ok]
    if changed:
        raise ValueError(f'Protected historical/user work changed: {changed}')
    return {'status': 'passed', 'unchanged_files': len(inventory)}


def check_inputs(root, prefix=''):
    histories = {n: read(root / (prefix + n) / 'training_inputs.json') for n in MODELS}
    proofs = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            common = list(zip(histories[a], histories[b]))
            if not common or any(x != y for x, y in common):
                raise ValueError(f'Actual training order/augmented bytes differ: {a}/{b}')
            proofs.append({'first': a, 'second': b, 'common_epochs': len(common), 'actual_inputs_equal': True})
    return {'status': 'passed', 'comparisons': proofs}


def worker(args):
    """Import lifecycle matches CP1: reference is cold until original construction."""
    import torch
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    if torch.get_default_dtype() != torch.float32 or flags != (True, True, False):
        raise ValueError('Preserve the actual CP1 FP32/TF32 backend defaults')
    protocol = read(args.config)
    validate_protocol(protocol)
    root = args.root.resolve()
    if args.worker == 'source-smoke':
        run = root / 'source-smoke'
        capture = {}

        def build(config, pretrained=True):
            capture['model'] = reference.build_model(config, pretrained)
            return capture['model']

        function, proof = source_training_function({'build_model': build}, PARAMETERS['C0'])
        function(protocol['baseline'], run, args.device, smoke=True)
        write_json(run / 'original_engine_proof.json', {**proof, 'final_state_sha256': state_digest(capture['model'])})
        if args.device == 'cpu':
            from foot_ulcer_segmentation.checkpoint import save_checkpoint
            save_checkpoint(run / 'step_state.pth', capture['model'].state_dict())
        return
    variant = args.variant
    diagnostic = args.worker in ('smoke', 'cpu-control')
    prefix = ('probe-' if args.worker == 'probe' else 'smoke-' if args.worker == 'smoke'
              else 'cpu-control-' if args.worker == 'cpu-control' else '')
    run = root / (prefix + variant)
    if args.worker == 'probe':
        run.mkdir(exist_ok=False)
        captures = {}
        config, overrides = instrumented_namespace(protocol, variant, run, 'validation', captures)
        engine.setup(config['seed'], config['cpu_threads'])
        torch.hub.set_dir(str(ROOT / 'outputs/cache/torch/hub'))
        model = overrides['build_model'](config, pretrained=True)
        rng = engine.random_state()
        # Structural fixture only, never a scientific accuracy result.
        from foot_ulcer_segmentation.cp2_attention import model_profile
        inputs = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        profile = model_profile(model, inputs)
        engine.restore_random_state(rng)
        profile.pop('decoder_calls')
        profile['active_training_parameters'] = profile['active_forward_parameters'] + (33 if variant == 'CH2' else 0)
        with torch.no_grad():
            output = model(inputs)
        write_json(run / 'profile.json', {**profile, 'initial_primary_output_sha256': digest_json(output.numpy().tolist()),
                                        'fixture': 'SYNTHETIC linspace for structural verification only'})
        return
    if args.worker == 'evaluate' and not (root / 'selection.json').exists():
        raise ValueError('Persist validation-only selection before test')
    if args.worker in ('train', 'smoke', 'cpu-control') and (run / 'status.json').exists():
        old = read(run / 'status.json')
        if old['status'] in ('passed', 'completed'):
            if diagnostic:
                if read(run / 'runtime.json')['sources'] != runtime_sources():
                    raise ValueError('Completed smoke source changed')
            elif digest_file(run / 'best.pth') != read(run / 'completion.json')['checkpoint_sha256']:
                raise ValueError('Completed model changed')
            print('CACHED', variant, args.worker, flush=True)
            return
        if not args.resume:
            raise FileExistsError('Interrupted run; inspect and explicitly resume')
    captures = {}
    split = 'test' if args.worker == 'evaluate' else 'validation'
    config, overrides = instrumented_namespace(protocol, variant, run, split, captures)
    function, engine_proof = source_training_function(overrides, PARAMETERS[variant])
    if args.worker == 'evaluate':
        target = run / 'evaluation-test.json'
        if target.exists():
            if digest_file(run / 'best.pth') != read(run / 'completion.json')['checkpoint_sha256']:
                raise ValueError('Cached test checkpoint changed')
            print('CACHED test', variant, flush=True)
            return
        namespace = function.__globals__
        evaluate = types.FunctionType(engine.evaluate_run.__code__, namespace, argdefs=engine.evaluate_run.__defaults__)
        evaluate(run, split='test', device_name=args.device)
    else:
        resume_run = args.resume and (run / 'last.pth').exists()
        outcome, _ = function(config, run, args.device, smoke=diagnostic, resume=resume_run)
        write_json(run / 'runtime.json', {'sources': runtime_sources(), 'source_engine_adaptation': engine_proof,
                                         'every_original_CP1_config_key_equal': all(config[k] == v for k, v in protocol['baseline'].items())})
        model = captures['model']
        if variant == 'CH2':
            changed = any(not torch.equal(v.detach().cpu(), captures['initial_head'][n])
                          for n, v in model.auxiliary_head.named_parameters()) if 'initial_head' in captures else None
            if diagnostic and not changed:
                raise ValueError('Auxiliary head did not learn in actual smoke step')
        if diagnostic:
            write_json(run / 'smoke_proof.json', {'checkpoint_reload_passed': outcome['checkpoint_reload_passed'],
                                                'final_state_sha256': state_digest(model),
                                                'auxiliary_parameters_updated': changed if variant == 'CH2' else None})
            if args.worker == 'cpu-control':
                from foot_ulcer_segmentation.checkpoint import save_checkpoint
                save_checkpoint(run / 'step_state.pth', model.state_dict())
        else:
            write_json(run / 'completion.json', {'checkpoint_sha256': digest_file(run / 'best.pth'),
                                                'selected_epoch': outcome['selected_epoch'],
                                                'training_sources': runtime_sources(),
                                                'completed_utc': datetime.now(timezone.utc).isoformat()})
    evaluator = captures['evaluator']
    write_json(evaluator.export / 'manifest.json', evaluator.index)


def child(args, mode, variant=None, device=None):
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_cp1_matched.py'), '--root', str(args.root),
               '--config', str(args.config), '--device', device or args.device, '--worker', mode]
    if variant:
        command += ['--variant', variant]
    if args.resume and mode == 'train':
        command += ['--resume']
    label = mode + ('-' + variant if variant else '')
    path = args.root / (label + '.log')
    with path.open('a' if args.resume else 'x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.root / 'active_process.json', {'stage': mode, 'variant': variant, 'pid': process.pid,
                                                     'log': path.name, 'utc': datetime.now(timezone.utc).isoformat()})
        code = process.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; see {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/cp2/cp1_matched.json'))
    parser.add_argument('--stage', choices=('prepare', 'preflight', 'train', 'status'), default='prepare')
    parser.add_argument('--worker', choices=('source-smoke', 'cpu-control', 'probe', 'smoke', 'train', 'evaluate'))
    parser.add_argument('--variant', choices=MODELS)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.root = args.root.resolve()
    if args.worker:
        worker(args)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    root = args.root
    if args.stage == 'status':
        print(json.dumps({n: read(root / n / 'status.json') if (root / n / 'status.json').exists()
                          else {'status': 'queued'} for n in MODELS}, indent=2))
        return
    lock = (root / '.matched.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(args.config)
    manifest = validate_protocol(protocol)
    preservation = protected(root)
    registered = {'protocol': protocol, 'runtime_sources': runtime_sources(),
                  'original_split_fingerprint': manifest['fingerprint']}
    if (root / 'preregistration.json').exists():
        old = read(root / 'preregistration.json')
        if any(old[k] != v for k, v in registered.items()):
            raise ValueError('Registered source/config changed; use a new run')
    else:
        write_json(root / 'preregistration.json', {**registered, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                                 'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()})
        write_json(root / 'protocol.json', protocol)
        write_json(root / 'split-manifest.json', manifest)
        for name in runtime_sources():
            target = root / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    record = root / (args.stage + '_workflow.json')
    if record.exists() and read(record)['status'] == 'completed':
        print('CACHED completed', args.stage)
        return
    write_json(record, {'status': 'running', 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()})
    try:
        if args.stage == 'prepare':
            from cp1_matched_analysis import audit_original_split
            audit_original_split(root)
        elif args.stage == 'preflight':
            with (root / 'tests.log').open('x') as log:
                subprocess.run([str(ROOT / '.venv-xfuseg/bin/python'), '-m', 'pytest', '-o', 'addopts=', '-q',
                                f'--junitxml={root / "tests.xml"}'], stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
            profiles = {}
            for name in MODELS:
                child(args, 'probe', name)
                profiles[name] = read(root / ('probe-' + name) / 'profile.json')
            proofs = {n: read(root / ('probe-' + n) / 'model_verified.json') for n in MODELS}
            if len({p['source_initial_state_sha256'] for p in proofs.values()}) != 1 or len({p['shared_post_constructor_rng_sha256'] for p in proofs.values()}) != 1:
                raise ValueError('Source tensors/RNG differ across cold constructors')
            if profiles['C0']['initial_primary_output_sha256'] != profiles['CH2']['initial_primary_output_sha256']:
                raise ValueError('Training-only head changed initial main inference')
            write_json(root / 'architecture_preflight.json', {'profiles': profiles, 'proofs': proofs})
            child(args, 'source-smoke', device='cpu')
            child(args, 'cpu-control', 'C0', device='cpu')
            for name in MODELS:
                child(args, 'smoke', name)
            if read(root / 'source-smoke/original_engine_proof.json')['final_state_sha256'] != read(root / 'cpu-control-C0/smoke_proof.json')['final_state_sha256']:
                raise ValueError('Observed baseline update differs from original source step')
            source_history = read(root / 'source-smoke/history.json')[0]
            observed_history = read(root / 'cpu-control-C0/history.json')[0]
            for key in ('train', 'validation', 'selected_checkpoint', 'learning_rate'):
                if source_history[key] != observed_history[key]:
                    raise ValueError('Observed baseline logs differ from original source')
            write_json(root / 'pretraining_checks.json', {'status': 'passed', 'sources': runtime_sources(),
                       'original_source_CPU_step_and_logs_exactly_equal': True,
                       'GPU_bitwise_determinism_claimed': False, 'actual_smoke_inputs': check_inputs(root, 'smoke-'),
                       'original_config_and_split_exact': True, 'preservation': protected(root)})
        else:
            if read(root / 'pretraining_checks.json')['sources'] != runtime_sources():
                raise ValueError('Successful unchanged preflight required')
            from cp1_matched_analysis import select, summarize_exports
            for name in MODELS:
                child(args, 'train', name)
                summarize_exports(root, name, 'validation')
            write_json(root / 'paired_training_inputs.json', check_inputs(root))
            if not (root / 'selection.json').exists():
                write_json(root / 'selection.json', select(root))
            for name in MODELS:
                child(args, 'evaluate', name)
                summarize_exports(root, name, 'test')
            subprocess.run([str(ROOT / '.venv-analysis/bin/python'), 'scripts/report_cp1_matched.py', '--root', str(root)], check=True)
            write_json(root / 'final_preservation.json', protected(root))
        write_json(record, {'status': 'completed', 'utc': datetime.now(timezone.utc).isoformat()})
        print('COMPLETED', args.stage, flush=True)
    except BaseException:
        write_json(record, {'status': 'failed', 'utc': datetime.now(timezone.utc).isoformat(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
