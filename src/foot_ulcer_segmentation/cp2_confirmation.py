"""Paired-seed replication of frozen B1/H2/A6 on the original CP2 manifest.

Historical trainers remain byte-identical. The new orchestration loop reuses
their actual train/validation steps, architecture constructors and evaluators.
"""

import copy
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import link_best, random_state, restore_random_state, save_checkpoint
from .common import ROOT, digest_file, digest_json, write_json
from .cp2_attention import build_attention_model
from .cp2_data import verify_cp2_manifest
from .cp2_heads import HeadObjective, build_head_model
from .cp2_phase_l import _verify_precision, phase_l_sources, train_epoch, validate_epoch
from .cp2_protocol import BaselineObjective, CP2Evaluator, ImageDiceCheckpoint
from .cp2_train import export_evaluation, make_cp2_loader, shared_initialization, tensor_fingerprint
from .engine import environment_record, setup
from .reference import build_model, verify_reference

MODELS = ('B1', 'H2', 'A6')


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def sources():
    result = phase_l_sources()
    names = ('src/foot_ulcer_segmentation/cp2_confirmation.py',
             'src/foot_ulcer_segmentation/cp2_attention.py',
             'src/foot_ulcer_segmentation/cp2_heads.py',
             'scripts/run_cp2_confirmation.py', 'scripts/report_cp2_confirmation.py',
             'configs/cp2/confirmation.json', 'configs/cp2/phase_p.json',
             'configs/cp2/phase_a.json', 'configs/cp2/phase_h.json')
    result.update({name: digest_file(ROOT / name) for name in names})
    return result


def validate_protocol(protocol):
    from importlib.metadata import version
    if protocol['seeds'] != [42, 43, 44] or protocol['fresh_seeds'] != [43, 44]:
        raise ValueError('Only the registered three-seed replication')
    if tuple(protocol['models']) != MODELS or protocol['cached_seed'] != 42:
        raise ValueError('Do not change candidate set or cached seed')
    original = read(ROOT / protocol['base_root'] / 'protocol.json')
    if original != read(ROOT / protocol['base_config']):
        raise ValueError('Frozen phase-P configuration changed')
    manifest = verify_cp2_manifest(read(ROOT / protocol['base_root'] / 'data/manifest.json'))
    if manifest['fingerprint'] != protocol['manifest_fingerprint']:
        raise ValueError('Do not change CP2 membership')
    environment = read(ROOT / protocol['base_root'] / 'B1/environment.json')
    for name, expected in environment['packages'].items():
        if version(name) != expected:
            raise ValueError(f'Frozen training environment changed: {name}')
    initial = read(ROOT / protocol['base_root'] / 'initialization.json')
    if digest_file(ROOT / protocol['base_root'] / 'initial_weights.pth') != initial['file_sha256']:
        raise ValueError('Historical initial state changed')
    for name, expected in initial['pretrained_weights'].items():
        if digest_file(ROOT / 'outputs/cache/torch/hub/checkpoints' / name) != expected:
            raise ValueError('Use the same verified pretrained weights')
    for model in MODELS:
        root = ROOT / protocol['cached_roots'][model]
        status = read(root / model / 'status.json')
        config = read(root / 'protocol.json')
        for key in ('reference_revision', 'seed', 'split_seed', 'cpu_threads', 'data', 'model', 'loss', 'training'):
            if config[key] != original[key]:
                raise ValueError(f'Cached seed42 {model} differs in {key}')
        for key, value in original['evaluation'].items():
            if key != 'test_policy' and config['evaluation'][key] != value:
                raise ValueError('Cached evaluation differs')
        if read(root / 'data/manifest.json') != manifest:
            raise ValueError('Cached manifest differs')
        if status['status'] != 'completed' or status['diagnostic_smoke_only']:
            raise ValueError('Only completed real training can be reused')
        for name, expected in status['training_sources'].items():
            if digest_file(ROOT / name) != expected:
                raise ValueError(f'Historical source changed: {name}')
        if digest_file(root / model / 'best.pth') != status['best_checkpoint_sha256']:
            raise ValueError('Historical checkpoint changed')
    return manifest


def seed_config(protocol, seed, directory):
    config = copy.deepcopy(read(ROOT / protocol['base_config']))
    config['seed'] = seed
    # Retain the original numerical RNG formula, substituting the training seed.
    config['training']['epoch_rng_seed_rule'] = f'{seed} + 10007 * zero_based_epoch; dedicated loader generators'
    config['evaluation']['test_policy'] = protocol['selection']['test']
    initial = read(Path(directory) / 'initialization.json')
    config['phase_h'] = copy.deepcopy(read(ROOT / protocol['architecture_configs']['H2'])['phase_h'])
    config['phase_a'] = copy.deepcopy(read(ROOT / protocol['architecture_configs']['A6'])['phase_a'])
    for key in ('phase_h', 'phase_a'):
        config[key]['control_initial_tensor_sha256'] = initial['tensor_fingerprint']
    config['phase_h']['head_seed'] = seed + 30003
    config['confirmation'] = {'seed': seed, 'split_seed': 42, 'models': list(MODELS)}
    return config


class RegionalObjective(torch.nn.Module):
    """Dictionary interface to the SAME numerical B1 loss primitives."""
    def __init__(self):
        super().__init__()
        self.baseline = BaselineObjective('B1')

    def components(self, logits, truth):
        dice, focal = self.baseline.components(logits, truth)
        return {'dice': dice, 'focal': focal, 'total': dice + focal}

    def forward(self, logits, truth):
        return self.components(logits, truth)['total']


def build(config, variant, initial):
    if variant == 'H2':
        model, proof = build_head_model(config, 'H2', initial)
        objective = HeadObjective('H2', .1)
    elif variant == 'A6':
        model, proof = build_attention_model(config, 'A6', initial)
        objective = RegionalObjective()
    elif variant == 'B1':
        model = build_model(config, pretrained=False)
        model.load_state_dict(initial['state_dict'])
        proof = {'source_tensor_sha256': tensor_fingerprint(model.state_dict()),
                 'complete_initial_tensor_sha256': tensor_fingerprint(model.state_dict()),
                 'all_original_named_tensors_equal': True}
        objective = RegionalObjective()
    else:
        raise ValueError('Only B1, H2 and A6')
    if proof['source_tensor_sha256'] != initial['tensor_fingerprint']:
        raise ValueError('Unpaired source initialization')
    return model, objective, proof


def model_directory(root, protocol, seed, variant):
    if seed == 42:
        return ROOT / protocol['cached_roots'][variant] / variant
    return Path(root) / f'seed-{seed}' / variant


def prepare(root, protocol):
    manifest = validate_protocol(protocol)
    destination = Path(root) / 'protocol.json'
    if destination.exists() and read(destination) != protocol:
        raise ValueError('Use a new run after a protocol change')
    write_json(destination, protocol)
    write_json(Path(root) / 'data/manifest.json', manifest)
    for seed in protocol['fresh_seeds']:
        directory = Path(root) / f'seed-{seed}'
        directory.mkdir(exist_ok=True)
        config = copy.deepcopy(read(ROOT / protocol['base_config']))
        config['seed'] = seed
        shared_initialization(config, directory)
        config = seed_config(protocol, seed, directory)
        path = directory / 'config.json'
        if path.exists() and read(path) != config:
            raise ValueError('Seed configuration changed')
        write_json(path, config)
    return manifest


def train(root, protocol, seed, variant, device_name, smoke=False, resume=False):
    seed_root = Path(root) / f'seed-{seed}'
    config = read(seed_root / 'config.json')
    manifest = read(Path(root) / 'data/manifest.json')
    run = seed_root / (('smoke-' if smoke else '') + variant)
    identity = digest_json({'config': config, 'variant': variant, 'sources': sources(),
                            'manifest': manifest['fingerprint'],
                            'initial_file_sha256': digest_file(seed_root / 'initial_weights.pth')})
    if (run / 'status.json').exists():
        old = read(run / 'status.json')
        if old['status'] in ('completed', 'passed'):
            if old['identity'] != identity or digest_file(run / 'best.pth') != old['best_checkpoint_sha256']:
                raise ValueError('Completed stage identity/checkpoint changed')
            return old
        if not resume:
            raise FileExistsError('Inspect interrupted run before explicit --resume')
    if run.exists() and not (resume and (run / 'last.pth').exists()):
        raise FileExistsError('Partial initialization has no compatible resume checkpoint')
    if not run.exists():
        resume = False
        run.mkdir()
    started = time.monotonic()
    device = torch.device(device_name)
    verify_cp2_manifest(manifest)
    verify_reference()
    setup(seed, config['cpu_threads'])
    _verify_precision(config)
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
    initial = torch.load(seed_root / 'initial_weights.pth', map_location='cpu')
    model, objective, proof = build(config, variant, initial)
    del initial
    environment = environment_record(config, manifest, device)
    if resume:
        if read(run / 'identity.json')['fingerprint'] != identity:
            raise ValueError('Cannot resume changed config/code/data')
        if read(run / 'environment.json')['packages'] != environment['packages']:
            raise ValueError('Cannot resume changed environment')
    else:
        write_json(run / 'identity.json', {'fingerprint': identity})
        write_json(run / 'config.json', {'variant': variant, **config})
        write_json(run / 'environment.json', environment)
        write_json(run / 'initialization_verified.json', proof)
    model.to(device)
    settings = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **{
        k: v for k, v in settings['scheduler'].items() if k not in ('type', 'monitor')})
    policy, history, first_epoch = ImageDiceCheckpoint(), [], 0
    if resume:
        state = torch.load(run / 'last.pth', map_location=device)
        if state['identity'] != identity:
            raise ValueError('Resume checkpoint identity mismatch')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        policy = ImageDiceCheckpoint(**state['policy'])
        history, first_epoch = state['history'], state['completed_epochs']
        restore_random_state(state['rng'])
        del state
    new_before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if 'auxiliary_head.' in n or '.cbam.' in n} if smoke else {}
    encoder_before = next(model.parameters()).detach().clone() if smoke else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    write_json(run / 'status.json', {'status': 'running', 'seed': seed, 'variant': variant,
                                     'pid': os.getpid(), 'started_utc': utc(), 'completed_epochs': first_epoch})
    finite_steps = 0
    for epoch in range(first_epoch, 1 if smoke else settings['max_epochs']):
        if policy.bad_epochs >= settings['early_stopping_patience']:
            break
        setup(seed + 10007 * epoch, config['cpu_threads'])
        loader, rows = make_cp2_loader(config, manifest, 'train', epoch, smoke)
        valid, valid_rows = make_cp2_loader(config, manifest, 'validation', epoch, smoke)
        epoch_start = time.monotonic()
        used_lr = optimizer.param_groups[0]['lr']
        training = train_epoch(model, objective, optimizer, loader, device)
        finite_steps += training['steps']
        evaluator, parts = validate_epoch(model, objective, valid, valid_rows, config, device)
        metric = evaluator.summary()['all']['image_based']['dice']
        chosen = policy.update(metric, epoch + 1)
        scheduler.step(metric)
        entry = {'epoch': epoch + 1, 'train_loss': training['components']['total'],
                 'validation_loss': parts['total'], 'train_components': training['components'],
                 'validation_components': parts, 'validation_image_dice': metric,
                 'learning_rate_used': used_lr, 'learning_rate_next': optimizer.param_groups[0]['lr'],
                 'selected': chosen, 'bad_epochs': policy.bad_epochs,
                 'training_order_sha256': digest_json([r['sample_id'] for r in rows]),
                 'augmented_input_sha256': training['augmented_input_sha256'],
                 'steps': training['steps'], 'step_seconds_mean': training['step_seconds_mean'],
                 'epoch_compute_seconds': time.monotonic() - epoch_start}
        history.append(entry)
        save_checkpoint(run / 'last.pth', {'identity': identity, 'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'rng': random_state(), 'policy': policy.state_dict(), 'history': history,
                        'completed_epochs': epoch + 1})
        if chosen:
            link_best(run / 'last.pth', run / 'best.pth')
        write_json(run / 'history.json', history)
        write_json(run / 'status.json', {'status': 'running', 'seed': seed, 'variant': variant,
                   'pid': os.getpid(), 'completed_epochs': epoch + 1, 'selected_epoch': policy.epoch,
                   'latest': entry, 'elapsed_seconds': time.monotonic() - started})
        print(json.dumps({'seed': seed, 'variant': variant, **entry}), flush=True)
    changed = [n for n, p in model.named_parameters()
               if n in new_before and not torch.equal(p.detach(), new_before[n])]
    if smoke and (torch.equal(encoder_before, next(model.parameters()).detach()) or (new_before and not changed)):
        raise ValueError('Smoke did not update encoder/new component')
    state = torch.load(run / 'best.pth', map_location='cpu')
    if state['identity'] != identity:
        raise ValueError('Selected checkpoint identity mismatch')
    model.load_state_dict(state['state_dict'])
    loader, _ = make_cp2_loader(config, manifest, 'validation', smoke=smoke)
    images, _ = next(iter(loader))
    model.eval()
    with torch.no_grad():
        expected = model(images.to(device)).sigmoid().cpu()
        next(model.parameters()).add_(1)
    model.load_state_dict(state['state_dict'])
    with torch.no_grad():
        actual = model(images.to(device)).sigmoid().cpu()
    if not torch.equal(expected, actual):
        raise ValueError('Checkpoint reload predictions differ')
    result = {'status': 'passed' if smoke else 'completed', 'diagnostic_smoke_only': smoke,
              'seed': seed, 'variant': variant, 'pid': os.getpid(), 'finished_utc': utc(),
              'completed_epochs': len(history), 'selected_epoch': state['completed_epochs'],
              'selected_validation_image_dice': policy.best,
              'seconds_this_invocation': time.monotonic() - started,
              'finite_gradient_steps_this_invocation': finite_steps, 'checkpoint_replay_exact': True,
              'encoder_update_verified_in_smoke': smoke, 'new_parameters_updated_in_smoke': changed,
              'best_checkpoint_sha256': digest_file(run / 'best.pth'), 'identity': identity,
              'training_sources': sources(), 'split_fingerprint': manifest['fingerprint'],
              'source_initial_tensor_sha256': proof['source_tensor_sha256'],
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0}
    write_json(run / 'status.json', result)
    return result


def evaluate(root, protocol, seed, variant, split, device_name):
    if split not in ('validation', 'test'):
        raise ValueError('Only registered evaluation splits')
    if split == 'test' and not (Path(root) / 'selection.json').exists():
        raise ValueError('Lock validation selection before test')
    run = model_directory(root, protocol, seed, variant)
    status = read(run / 'status.json')
    destination = run / f'evaluation-{split}.json'
    if destination.exists():
        if read(destination)['checkpoint_sha256'] != digest_file(run / 'best.pth'):
            raise ValueError('Cached prediction checkpoint mismatch')
        return read(destination)
    if seed == 42:
        raise FileNotFoundError('Never write into historical run')
    if status['status'] != 'completed' or status['diagnostic_smoke_only']:
        raise ValueError('Completed actual training required')
    config = read(Path(root) / f'seed-{seed}' / 'config.json')
    manifest = read(Path(root) / 'data/manifest.json')
    setup(seed, config['cpu_threads'])
    initial = torch.load(Path(root) / f'seed-{seed}/initial_weights.pth', map_location='cpu')
    model, objective, _ = build(config, variant, initial)
    del initial
    state = torch.load(run / 'best.pth', map_location='cpu')
    if state['identity'] != status['identity']:
        raise ValueError('Checkpoint identity mismatch')
    model.load_state_dict(state['state_dict'])
    del state
    device = torch.device(device_name)
    model.to(device)
    loader, rows = make_cp2_loader(config, manifest, split)
    export = run / 'predictions' / split
    if export.exists():
        archive = run / 'incomplete_exports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.parent.mkdir(exist_ok=True)
        export.rename(archive)
    evaluator, parts = validate_epoch(model, objective, loader, rows, config, device, export)
    return export_evaluation(destination, evaluator, {
        'seed': seed, 'variant': variant, 'split': split, 'components': parts,
        'checkpoint_sha256': digest_file(run / 'best.pth'), 'selected_epoch': status['selected_epoch'],
        'split_fingerprint': manifest['fingerprint'], 'diagnostic_smoke_only': False,
        'test_caveat': 'Previously exposed source-test; descriptive only'})


def check_pairing(root, protocol, seed, smoke=False):
    directories = {name: (model_directory(root, protocol, seed, name) if not smoke else
                         Path(root) / f'seed-{seed}' / ('smoke-' + name)) for name in MODELS}
    histories = {n: read(p / 'history.json') for n, p in directories.items()}
    proofs = {n: read(p / 'initialization_verified.json') for n, p in directories.items()}
    hashes = {p.get('source_tensor_sha256', p.get('shared_initialization_tensor_sha256')) for p in proofs.values()}
    if len(hashes) != 1:
        raise ValueError('Source tensors not paired within seed')
    comparisons = []
    for i, first in enumerate(MODELS):
        for second in MODELS[i + 1:]:
            pairs = list(zip(histories[first], histories[second]))
            if not pairs:
                raise ValueError('No actual training steps')
            for a, b in pairs:
                if any(a[k] != b[k] for k in ('training_order_sha256', 'augmented_input_sha256')):
                    raise ValueError('Actual training order/augmented bytes differ')
            comparisons.append({'first': first, 'second': second, 'common_epochs': len(pairs)})
    return {'status': 'passed', 'seed': seed, 'source_tensor_sha256': hashes.pop(), 'comparisons': comparisons}


def guard_checks(candidate, baseline, ratio):
    return {'higher_overall_dice': candidate['image_dice'] > baseline['image_dice'],
            'small_not_lower': candidate['small_dice'] >= baseline['small_dice'],
            'boundary_not_lower': candidate['boundary_iou'] >= baseline['boundary_iou'],
            'no_extra_total_miss': candidate['misses'] <= baseline['misses'],
            'no_extra_empty_fp': candidate['empty_fp_images'] <= baseline['empty_fp_images'],
            'capacity_budget': ratio <= 1.01}


def compact(report):
    if report['split'] not in ('validation', 'test'):
        raise ValueError('An evaluation split is required')
    return {'image_dice': report['all']['image_based']['dice'],
            'data_dice': report['all']['data_based']['dice'],
            'small_dice': report['subgroups']['small']['image_based']['dice'],
            'boundary_iou': report['all']['image_based']['boundary_iou'],
            'pfom': report['all']['image_based']['pfom'],
            'misses': report['gt_positive']['total_foreground_misses'],
            'empty_fp_images': report['gt_empty']['images_with_prediction'],
            'empty_fp_pixels': report['gt_empty']['counts']['fp']}


def select(validation, profiles, protocol):
    if set(validation) != set(protocol['seeds']):
        raise ValueError('All registered seeds required; never select a favorable subset')
    if any(r['split'] != 'validation' for models in validation.values() for r in models.values()):
        raise ValueError('Model selection must never read test metrics')
    values = {seed: {n: compact(validation[seed][n]) for n in MODELS} for seed in validation}
    means = {n: {k: float(np.mean([values[s][n][k] for s in values]))
                 for k in values[42][n]} for n in MODELS}
    decisions = {}
    for name in ('H2', 'A6'):
        ratio = profiles[name]['active_forward_parameters'] / profiles['B1']['active_forward_parameters']
        checks = guard_checks(means[name], means['B1'], ratio)
        individual = {s: guard_checks(values[s][name], values[s]['B1'], ratio) for s in values}
        passes = sum(all(c.values()) for c in individual.values())
        decisions[name] = {'eligible': all(checks.values()) and passes >= protocol['selection']['min_full_guard_seed_passes'],
                           'mean_guards': checks, 'per_seed_guards': individual, 'full_guard_seed_passes': passes,
                           'paired_image_dice_deltas': {s: values[s][name]['image_dice'] - values[s]['B1']['image_dice'] for s in values}}
    eligible = [n for n in ('H2', 'A6') if decisions[n]['eligible']]
    selected = max(eligible, key=lambda n: means[n]['image_dice']) if eligible else 'B1'
    fp_excess = {n: means[n]['empty_fp_images'] - means['B1']['empty_fp_images'] for n in ('H2', 'A6')}
    route = ('REVIEW_BOUNDED_COMBINATION' if len(eligible) == 2 else
             'REVIEW_PRESENCE_SUPERVISION' if any(v > 0 for v in fp_excess.values()) else 'HOLD_AND_DIAGNOSE')
    return {'created_utc': utc(), 'validation_only': True, 'selected_candidate': selected,
            'means': means, 'decisions': decisions, 'next_research_route': route,
            'automatic_new_architecture_training': False,
            'historical_seed42_decisions_unchanged': True,
            'caveat': 'Exploratory seed replication on reused validation; not independent confirmatory evidence'}


def verify_exports(run, split, manifest, settings):
    """Recompute ALL metrics from checksummed probability/GT pairs and raw masks."""
    import cv2
    report = read(run / f'evaluation-{split}.json')
    if report['split_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Prediction split mismatch')
    rows = manifest['splits'][split]
    entries = read(run / 'predictions' / split / 'manifest.json')
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in entries]:
        raise ValueError('Missing, reordered or duplicate predictions')
    evaluator = CP2Evaluator(settings, boundaries=True)
    for row, entry in zip(rows, entries):
        path = run / 'predictions' / split / entry['file']
        if digest_file(path) != entry['sha256']:
            raise ValueError('Prediction checksum mismatch')
        with np.load(path) as value:
            truth, probability, prediction = value['truth'], value['probability'], value['prediction']
            raw = cv2.imread(str(ROOT / row['mask_path']), cv2.IMREAD_GRAYSCALE)
            if raw is None or not np.array_equal(truth, (raw / 255).astype(np.uint8)):
                raise ValueError('Prediction GT differs from frozen raw mask')
            if not np.array_equal(prediction, (probability >= .5).astype(np.uint8)):
                raise ValueError('Wrong threshold or exported mask')
            evaluator.add(row, truth, probability)
    if evaluator.summary() != {k: report[k] for k in evaluator.summary()}:
        raise ValueError('Recomputed metrics differ')
    if digest_file(run / 'best.pth') != report['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed')
    return {'status': 'passed', 'split': split, 'predictions': len(rows)}


def fp_geometry(prediction, probability):
    """Geometry only; no anatomical or clinical attribution from a binary mask."""
    from scipy.ndimage import label
    prediction = np.asarray(prediction, dtype=bool)
    probability = np.asarray(probability)
    if prediction.ndim != 2 or probability.shape != prediction.shape or not np.isfinite(probability).all():
        raise ValueError('Matched finite 2D arrays required')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be in [0,1]')
    if not np.array_equal(prediction, probability >= .5):
        raise ValueError('Keep the preregistered threshold')
    labels, number = label(prediction, np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    y, x = np.nonzero(prediction)
    border = np.zeros_like(prediction)
    border[:3] = border[-3:] = True
    border[:, :3] = border[:, -3:] = True
    return {'fp_pixels': int(prediction.sum()), 'fp_fraction': float(prediction.mean()),
            'components_8_connected': int(number), 'largest_component_pixels': int(sizes.max()) if number else 0,
            'border_3px_fp_pixels': int((prediction & border).sum()),
            'centroid_x_fraction': float(x.mean() / max(1, prediction.shape[1] - 1)) if x.size else None,
            'centroid_y_fraction': float(y.mean() / max(1, prediction.shape[0] - 1)) if y.size else None,
            'maximum_probability': float(probability.max())}


def audit_false_positives(root, protocol):
    """Recompute validation caches, then describe all seven GT-empty examples."""
    settings = read(ROOT / protocol['base_config'])['evaluation']
    manifest = read(Path(root) / 'data/manifest.json')
    validation = manifest['splits']['validation']
    empty = [r for r in validation if r['gt_pixels'] == 0]
    index = {r['sample_id']: i + 1 for i, r in enumerate(empty)}
    cases, summaries, verification = [], {}, {}
    for name in MODELS:
        run = model_directory(root, protocol, 42, name)
        verification[name] = verify_exports(run, 'validation', manifest, settings)
        entries = {r['sample_id']: r for r in read(run / 'predictions/validation/manifest.json')}
        group = []
        for row in empty:
            entry = entries[row['sample_id']]
            with np.load(run / 'predictions/validation' / entry['file']) as value:
                if value['truth'].any():
                    raise ValueError('Expected an actual GT-empty example')
                case = {'model': name, 'case': f'V{index[row["sample_id"]]:02}',
                        'sample_id': row['sample_id'], 'rgb_group': row['rgb_group'],
                        'source': 'CACHED REAL seed42 validation probability maps; freshly recomputed geometry',
                        **fp_geometry(value['prediction'], value['probability'])}
                cases.append(case)
                group.append(case)
        summaries[name] = {'gt_empty_images': len(group),
                           'fp_images': sum(c['fp_pixels'] > 0 for c in group),
                           'fp_pixels': sum(c['fp_pixels'] for c in group),
                           'largest_component_pixels': max(c['largest_component_pixels'] for c in group)}
    baseline = {c['sample_id'] for c in cases if c['model'] == 'B1' and c['fp_pixels']}
    comparisons = {}
    for name in ('H2', 'A6'):
        predicted = {c['sample_id'] for c in cases if c['model'] == name and c['fp_pixels']}
        comparisons[name] = {'new_fp_cases_vs_B1': sorted(predicted - baseline),
                             'resolved_fp_cases_vs_B1': sorted(baseline - predicted),
                             'shared_fp_cases_with_B1': sorted(predicted & baseline)}
    result = {'status': 'passed', 'created_utc': utc(), 'split': 'validation', 'seed': 42,
              'no_training_or_new_inference': True, 'prediction_validation': verification,
              'summaries': summaries, 'comparisons': comparisons, 'cases': cases,
              'interpretation': 'Same 0.5 threshold and empty rules; any FP gives empty-image Dice 0. Geometry does not establish cause or annotation correctness. No relabeling, component filtering or threshold tuning.',
              'limitations': 'Only seven empty validation examples, reused validation, no clinical adjudication or patient independence.'}
    write_json(Path(root) / 'false_positive_audit.json', result)
    return result


def preservation(root):
    from concurrent.futures import ThreadPoolExecutor
    inventory = read(Path(root) / 'protected_before.json')
    def changed(item):
        name, expected = item
        return name if not (ROOT / name).is_file() or digest_file(ROOT / name) != expected else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        bad = [name for name in pool.map(changed, inventory.items()) if name]
    if bad:
        raise ValueError(f'Protected historical/user files changed: {bad[:10]}')
    return {'status': 'passed', 'unchanged_files': len(inventory)}
