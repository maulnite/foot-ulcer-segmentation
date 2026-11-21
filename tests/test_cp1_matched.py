"""Engineering fixtures for exact-source config, observation and head isolation."""

import ast
import copy
import inspect
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from foot_ulcer_segmentation import engine
from foot_ulcer_segmentation.checkpoint import random_state, restore_random_state
from foot_ulcer_segmentation.common import ROOT
from foot_ulcer_segmentation.cp1_matched import (
    AuxiliarySourceLoss, BoundarySupervisedModel, MainMetric, ObservedLoader, PARAMETERS,
    TrainingProbabilities, run_config, source_training_function,
)


def test_all_original_config_fields_and_only_variant_metadata_change():
    protocol = json.loads((ROOT / 'configs/cp2/cp1_matched.json').read_text())
    original = json.loads((ROOT / 'outputs/runs/cp1-baseline-01/config.json').read_text())
    for name in PARAMETERS:
        config = run_config(protocol, name)
        assert {k: config[k] for k in original} == original
        assert set(config) - set(original) == {'matched_experiment'}
    with pytest.raises(ValueError):
        run_config(protocol, 'H3')


def test_source_function_baseline_bytecode_and_only_architecture_assertion_change():
    baseline, proof = source_training_function({}, PARAMETERS['C0'])
    assert baseline.__code__ is engine.run_training.__code__
    assert proof == {'original_bytecode_unchanged': True, 'changed_AST_nodes': 0}
    for name in ['CH2', 'CA6']:
        changed, proof = source_training_function({}, PARAMETERS[name])
        assert proof['changed_AST_nodes'] == 1
        expected = ast.parse(inspect.getsource(engine.run_training))
        node = next(n for n in ast.walk(expected) if isinstance(n, ast.Constant) and n.value == PARAMETERS['C0'])
        node.value = PARAMETERS[name]
        compiled = compile(expected, str(ROOT / 'src/foot_ulcer_segmentation/engine.py'), 'exec')
        original_constants = next(c for c in compiled.co_consts if inspect.iscode(c))
        assert changed.__code__.co_code == original_constants.co_code
        assert changed.__code__.co_consts == original_constants.co_consts
    with pytest.raises(ValueError):
        source_training_function({}, 123)


def test_observation_preserves_original_loader_rng_order_and_bytes(tmp_path):
    images = torch.arange(17 * 3 * 4 * 4).reshape(17, 3, 4, 4).float()
    truth = torch.zeros(17, 1, 4, 4)
    rows = [{'name': str(i)} for i in range(17)]
    dataset = TensorDataset(images, truth)
    torch.manual_seed(42)
    rng = random_state()
    source = DataLoader(dataset, batch_size=2, shuffle=True)
    expected = [[x.clone() for x, y in source] for _ in range(2)]
    after = torch.get_rng_state()
    restore_random_state(rng)
    observer = ObservedLoader(DataLoader(dataset, batch_size=2, shuffle=True), rows, tmp_path / 'inputs.json')
    for epoch in range(2):
        actual = [x for x, y in observer]
        assert all(torch.equal(a, b) for a, b in zip(actual, expected[epoch]))
    assert torch.equal(torch.get_rng_state(), after)
    records = json.loads((tmp_path / 'inputs.json').read_text())
    assert len(records) == 2 and all(r['images'] == 17 for r in records)


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images):
        features = self.encoder(images)
        for block in self.decoder.blocks:
            features = block(features)
        return nn.functional.interpolate(self.head(features), size=images.shape[-2:]).sigmoid()


def test_boundary_head_does_not_change_initial_source_tensors_rng_or_main_inference():
    torch.set_num_threads(2)
    base = ToyBase()
    original = copy.deepcopy(base).eval()
    rng = torch.get_rng_state()
    model = BoundarySupervisedModel(base, seed=30045)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.base.state_dict()[n]) for n, v in original.state_dict().items())
    inputs = torch.linspace(-1, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    model.eval()
    assert torch.equal(model(inputs), original(inputs))
    model.train()
    prediction = model(inputs)
    assert isinstance(prediction, TrainingProbabilities)
    assert prediction.main.shape == prediction.auxiliary.shape == (2, 1, 224, 224)
    assert not base.decoder.blocks[3]._forward_hooks


@pytest.mark.parametrize('empty', [True, False])
def test_auxiliary_source_loss_preserves_mismatch_and_main_validation_monitor(empty):
    from foot_ulcer_segmentation.reference import build_loss_and_metrics
    from foot_ulcer_segmentation.cp2_heads import morphological_boundary
    original, metrics = build_loss_and_metrics()
    model = BoundarySupervisedModel(ToyBase(), 30045)
    truth = torch.zeros(2, 1, 224, 224)
    if not empty:
        truth[:, :, 80:100, 50:80] = 1
    inputs = torch.linspace(-1, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    criterion = AuxiliarySourceLoss(original, morphological_boundary, .1)
    prediction = model(inputs)
    expected = original(prediction.main, truth) + .1 * original(prediction.auxiliary, morphological_boundary(truth))
    assert torch.equal(criterion(prediction, truth), expected)
    assert criterion.__name__ == original.__name__
    assert torch.equal(MainMetric(metrics[0])(prediction, truth), metrics[0](prediction.main, truth))
    criterion(prediction, truth).backward()
    for parameter in [model.base.encoder.weight, model.auxiliary_head.weight, model.auxiliary_head.bias]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        probability = model(inputs)
        assert torch.equal(criterion(probability, truth), original(probability, truth))
