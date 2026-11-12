"""Synthetic software fixtures for auxiliary-artifact QA, not research metrics."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from foot_ulcer_segmentation.common import digest_file
from foot_ulcer_segmentation.cp2_protocol import CP2Evaluator


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', ['H1', 'H2'])
def test_auxiliary_qa_checks_target_semantics_even_after_checksum_is_updated(tmp_path, variant):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/cp2/phase_h.json').read_text())
    mask = np.zeros((7, 9), np.uint8)
    mask[2:5, 3:6] = 1
    path = tmp_path / 'synthetic_mask.png'
    Image.fromarray(mask * 255).save(path)
    row = {'sample_id': 'synthetic-fixture', 'rgb_group': 'synthetic-rgb-group', 'mask_path': str(path)}
    target = mask
    if variant == 'H2':
        kernel = np.ones((3,3), np.uint8)
        target = cv2.dilate(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) - cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    probability = target.astype(np.float32)
    export = tmp_path / 'synthetic.npz'
    np.savez_compressed(export, probability=probability, prediction=target, truth=target)
    evaluator = CP2Evaluator(config['evaluation'], boundaries=True)
    evaluator.add(row, target, probability)
    expected = evaluator.summary()

    def index():
        (tmp_path / 'manifest.json').write_text(json.dumps([
            {'sample_id': row['sample_id'], 'file': export.name, 'sha256': digest_file(export)}]))

    index()
    verify = script('validate_cp2_phase_h').auxiliary_exports
    assert len(verify(tmp_path, [row], expected, config['evaluation'], variant).rows) == 1
    incorrect = target.copy()
    incorrect[0, 0] = 1 - incorrect[0, 0]
    np.savez_compressed(export, probability=probability, prediction=target, truth=incorrect)
    index()  # A valid hash is insufficient if the target was changed.
    with pytest.raises(AssertionError):
        verify(tmp_path, [row], expected, config['evaluation'], variant)


def test_head_gallery_uses_all_models_symmetrically_and_includes_failures():
    names = ('H0', 'H1', 'H2')
    rows = {str(i): {n: {'gt_pixels': '0' if i == 0 else str(i), 'dice': str(i/10),
                         'tp': '0' if i == 1 else '1', 'fp': '1' if i == 0 else '0',
                         'gt_components_8_connected': '2' if i == 2 else '1'} for n in names} for i in range(4)}
    choose = script('cp2_phase_h_cases').select_cases
    result = choose(rows, names)
    assert choose(dict(reversed(list(rows.items()))), names) == result
    assert result[2][1] == '1' and result[3][1] == '0' and result[4][1] == '2'
