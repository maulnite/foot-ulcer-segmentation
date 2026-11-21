"""Model-only interventions around the original CP1 engine, split and runners.

No CP1/CP2 source file is edited. The one hard-coded architecture-size assertion
in the frozen engine is adjusted for the declared added parameters; every other
training statement remains the original AST. Observers consume no random draws.
Keep candidate helper imports lazy: source model construction must happen first.
"""

import ast
import copy
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import types
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from . import engine, reference
from .checkpoint import random_state, restore_random_state
from .common import ROOT, digest_file, digest_json, write_json
from .data import verify_manifest

MODELS = ('C0', 'CH2', 'CA6')
PARAMETERS = {'C0': 70_968_939, 'CH2': 70_968_972, 'CA6': 70_975_382}


def read(path):
    return json.loads(Path(path).read_text())


def state_digest(model):
    """Same named-tensor serialization used by the historical CP1 ablation audit."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        for part in (name.encode(), str(array.dtype).encode(), str(array.shape).encode(), array.tobytes()):
            digest.update(part)
    return digest.hexdigest()


def rng_digest(state):
    return digest_json({'python': repr(state['python']), 'numpy': repr(state['numpy']),
                        'numpy_array_sha256': hashlib.sha256(state['numpy'][1].tobytes()).hexdigest(),
                        'torch': hashlib.sha256(state['torch'].numpy().tobytes()).hexdigest(),
                        'cuda': [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()
                                 for t in (state['cuda'] or [])]})


def runtime_sources():
    paths = ['src/foot_ulcer_segmentation/cp1_matched.py', 'scripts/run_cp1_matched.py',
             'scripts/cp1_matched_analysis.py',
             'configs/cp2/cp1_matched.json', 'src/foot_ulcer_segmentation/cp2_attention.py',
             'src/foot_ulcer_segmentation/cp2_heads.py']
    return {p: digest_file(ROOT / p) for p in paths}


def validate_protocol(protocol):
    original = read(ROOT / protocol['baseline_run'] / 'config.json')
    if protocol['baseline'] != original or tuple(protocol['models']) != MODELS:
        raise ValueError('Original CP1 configuration and bounded matrix are immutable')
    manifest = verify_manifest(read(ROOT / protocol['baseline_run'] / 'split-manifest.json'))
    current = verify_manifest(read(ROOT / original['data']['split_manifest']))
    if current != manifest or manifest['fingerprint'] != protocol['split_fingerprint']:
        raise ValueError('Use the exact CP1 manifest, including membership and order')
    if {s: len(v) for s, v in manifest['splits'].items()} != {'train': 707, 'validation': 124, 'test': 278}:
        raise ValueError('Original CP1 population changed')
    environment = read(ROOT / protocol['baseline_run'] / 'environment.json')
    for name, expected in environment['packages'].items():
        if version(name) != expected:
            raise ValueError(f'Frozen training dependency changed: {name}')
    for name, expected in environment['pipeline_sha256'].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError(f'Original CP1 source changed: {name}')
    reference.verify_reference()
    weights = ROOT / 'outputs/cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth'
    if digest_file(weights) != protocol['pretrained_sha256']:
        raise ValueError('Use the same cached ImageNet weights as CP1')
    expected_precision = {'dtype': 'float32', 'amp': False,
                          'matmul_allow_tf32': True, 'cudnn_allow_tf32': True, 'cudnn_benchmark': False}
    if protocol['precision'] != expected_precision:
        raise ValueError('Do not change CP1 precision')
    return manifest


def run_config(protocol, variant):
    if variant not in MODELS:
        raise ValueError('Exactly C0/CH2/CA6 are registered')
    return {**copy.deepcopy(protocol['baseline']),
            'matched_experiment': {'id': protocol['id'], 'variant': variant,
                                   'factor': copy.deepcopy(protocol['variants'][variant]),
                                   'split_fingerprint': protocol['split_fingerprint']}}


class TrainingProbabilities(NamedTuple):
    main: torch.Tensor
    auxiliary: torch.Tensor


class BoundarySupervisedModel(nn.Module):
    """Original sigmoid network; a training-only head taps decoder stage 3."""
    def __init__(self, base, seed):
        super().__init__()
        self.base = base
        with torch.random.fork_rng(devices=[]):
            # Seed only the CPU generator; never reseed CUDA as a side effect.
            torch.random.default_generator.manual_seed(seed)
            self.auxiliary_head = nn.Conv2d(32, 1, 1)

    def forward(self, images):
        if not self.training:
            return self.base(images)
        features = []
        handle = self.base.decoder.blocks[3].register_forward_hook(lambda m, i, o: features.append(o))
        try:
            probabilities = self.base(images)
        finally:
            handle.remove()
        if len(features) != 1 or features[0].shape[1:] != (32, 112, 112):
            raise ValueError('Expected original decoder stage-3 [B,32,112,112]')
        auxiliary = nn.functional.interpolate(self.auxiliary_head(features[0]), size=probabilities.shape[-2:],
                                               mode='bilinear', align_corners=False).sigmoid()
        return TrainingProbabilities(probabilities, auxiliary)


class AuxiliarySourceLoss(nn.Module):
    def __init__(self, source_loss, boundary, weight):
        super().__init__()
        self.source_loss, self.boundary, self.weight = source_loss, boundary, weight

    @property
    def __name__(self):
        return self.source_loss.__name__

    def forward(self, prediction, truth):
        if isinstance(prediction, TrainingProbabilities):
            return (self.source_loss(prediction.main, truth)
                    + self.weight * self.source_loss(prediction.auxiliary, self.boundary(truth)))
        # Preserve the CP1 main-task validation monitor, key and source primitives.
        return self.source_loss(prediction, truth)


class MainMetric(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.__name__ = metric.__name__

    def forward(self, prediction, truth):
        return self.metric(prediction.main if isinstance(prediction, TrainingProbabilities) else prediction, truth)


class ObservedSampler:
    def __init__(self, original):
        self.original, self.indices = original, []

    def __iter__(self):
        self.indices = []
        for index in self.original:
            self.indices.append(index)
            yield index

    def __len__(self):
        return len(self.original)


class ObservedLoader:
    """Read actual tensors and indices; keep the original DataLoader and RNG."""
    def __init__(self, loader, rows, path):
        self.loader, self.rows, self.path = loader, rows, Path(path)
        self.sampler = ObservedSampler(loader.batch_sampler.sampler)
        loader.batch_sampler.sampler = self.sampler

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        digest = hashlib.sha256()
        for images, truth in self.loader:
            digest.update(images.numpy().tobytes())
            digest.update(truth.numpy().tobytes())
            yield images, truth
        order = [self.rows[i]['name'] for i in self.sampler.indices]
        if len(order) != len(self.rows) or len(set(order)) != len(order):
            raise ValueError('Training membership/count changed')
        records = read(self.path) if self.path.exists() else []
        records.append({'epoch': len(records) + 1, 'images': len(order),
                        'training_order_sha256': digest_json(order),
                        'augmented_input_sha256': digest.hexdigest()})
        write_json(self.path, records)


def source_training_function(overrides, expected_parameters):
    """Reuse every original statement, permitting only the parameter-count guard."""
    namespace = dict(vars(engine))
    namespace.update(overrides)
    # evaluate_model's own globals must refer to the observed evaluator as well.
    namespace['evaluate_model'] = types.FunctionType(engine.evaluate_model.__code__, namespace,
                                                     argdefs=engine.evaluate_model.__defaults__)
    if expected_parameters == PARAMETERS['C0']:
        function = types.FunctionType(engine.run_training.__code__, namespace,
                                      argdefs=engine.run_training.__defaults__)
        return function, {'original_bytecode_unchanged': True, 'changed_AST_nodes': 0}
    tree = ast.parse(inspect.getsource(engine.run_training))
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == PARAMETERS['C0']]
    if len(candidates) != 1 or expected_parameters not in (PARAMETERS['CH2'], PARAMETERS['CA6']):
        raise ValueError('Only the registered source parameter-count assertion may change')
    candidates[0].value = expected_parameters
    exec(compile(tree, str(ROOT / 'src/foot_ulcer_segmentation/engine.py'), 'exec'), namespace)
    return namespace['run_training'], {'original_bytecode_unchanged': False, 'changed_AST_nodes': 1,
                                      'only_change': 'architecture-size assertion constant',
                                      'from': PARAMETERS['C0'], 'to': expected_parameters}


def instrumented_namespace(protocol, variant, run, split, captures):
    """Explicit model/loss changes plus read-only provenance and export observers."""
    config = run_config(protocol, variant)

    def build(cfg, pretrained=True):
        base = reference.build_model(cfg, pretrained=pretrained)
        before = state_digest(base)
        if pretrained and before != protocol['expected_initial_state_sha256']:
            raise ValueError('Source constructor differs from the verified CP1 reconstruction')
        rng = random_state()
        if variant == 'CH2':
            model = BoundarySupervisedModel(base, protocol['variants']['CH2']['head_seed'])
            captures['initial_head'] = {n: p.detach().cpu().clone() for n, p in model.auxiliary_head.named_parameters()}
        elif variant == 'CA6':
            # Import only AFTER original source construction and preserve all RNGs.
            from .cp2_attention import attach_attention
            architecture = read(ROOT / protocol['variants']['CA6']['constructor_config'])
            model = attach_attention(base, 'A6', architecture)
        else:
            model = base
        restore_random_state(rng)
        current_base = model.base if variant == 'CH2' else model
        source_names = set(base.state_dict()) if variant == 'CH2' else None
        # For A6 the original source tensors remain registered, alongside CBAM.
        proof = {'variant': variant, 'source_initial_state_sha256': before,
                 'expected_CP1_reconstruction_sha256': protocol['expected_initial_state_sha256'],
                 'shared_post_constructor_rng_sha256': rng_digest(rng),
                 'registered_parameters': sum(p.numel() for p in model.parameters()),
                 'original_sigmoid_output_preserved': True, 'pretrained_initialization': pretrained}
        if proof['registered_parameters'] != PARAMETERS[variant]:
            raise ValueError('Unexpected registered parameter count')
        if pretrained:
            # A persistent tensor snapshot allows direct pairable-tensor verification.
            shared = run.parent / 'initial_source_weights.pth'
            if not shared.exists():
                if variant != 'C0':
                    raise ValueError('C0 initialization snapshot must be prepared first')
                from .checkpoint import save_checkpoint
                save_checkpoint(shared, base.state_dict())
            initial = torch.load(shared, map_location='cpu')
            state = current_base.state_dict()
            if not all(n in state and torch.equal(state[n], value) for n, value in initial.items()):
                raise ValueError('Pairable source tensors changed while adding the intervention')
            if source_names is not None and set(initial) != source_names:
                raise ValueError('Original model tensor schema changed')
            proof['all_pairable_source_tensors_equal'] = True
            write_json(run / 'model_verified.json', proof)
        captures['model'] = model
        return model

    def loss_and_metrics():
        loss, metrics = reference.build_loss_and_metrics()
        if variant == 'CH2':
            rng = random_state()
            from .cp2_heads import morphological_boundary
            restore_random_state(rng)
            loss = AuxiliarySourceLoss(loss, morphological_boundary, protocol['variants']['CH2']['auxiliary_weight'])
            metrics = [MainMetric(metric) for metric in metrics]
        return loss, metrics

    def loaders(cfg, manifest, smoke=False):
        train, validation, rows = engine.make_loaders(cfg, manifest, smoke)
        train_rows = manifest['splits']['train'][:cfg['training']['batch_size']] if smoke else manifest['splits']['train']
        return ObservedLoader(train, train_rows, run / 'training_inputs.json'), validation, rows

    def environment(cfg, manifest, device):
        record = engine.environment_record(cfg, manifest, device)
        record['pipeline_sha256'].update(runtime_sources())
        record['matched_source_protocol'] = {'variant': variant, 'original_config_keys_equal':
            all(cfg[k] == v for k, v in protocol['baseline'].items()),
            'original_manifest_fingerprint': manifest['fingerprint']}
        return record

    class CachedEvaluator(engine.SegmentationEvaluator):
        def __init__(self, threshold=.5):
            super().__init__(threshold)
            self.index = []
            self.export = run / 'predictions' / split
            self.export.mkdir(parents=True, exist_ok=False)
            captures['evaluator'] = self

        def add(self, name, truth, probability):
            row = super().add(name, truth, probability)
            path = self.export / (Path(name).stem + '.npz')
            temporary = path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, truth=truth, probability=probability,
                                    prediction=(np.asarray(probability) >= self.threshold).astype(np.uint8))
            temporary.replace(path)
            self.index.append({'name': name, 'file': path.name, 'sha256': digest_file(path)})
            return row

    return config, {'build_model': build, 'build_loss_and_metrics': loss_and_metrics,
                    'make_loaders': loaders, 'environment_record': environment,
                    'SegmentationEvaluator': CachedEvaluator}
