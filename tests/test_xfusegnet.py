"""Protect source-loop parity, probability averaging, mask policy, and resume."""

import ast
import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from xfuseg_train import Runtime, instrumented_loop, original_loop, load_definitions, restore_training_state
from xfuseg_inference import TestImages as InferenceImages, mean_probability, evaluate_oof
from foot_ulcer_segmentation.checkpoint import random_state
from foot_ulcer_segmentation.evaluation import SegmentationEvaluator


def test_numeric_training_loop_is_original_ast():
    class StripObservers(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == "_xf_skip_fold":
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id.startswith("_xf_"):
                return None
            return self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "epoch":
                node.iter = ast.parse("range(EPOCHS)", mode="eval").body
            return self.generic_visit(node)

    restored = StripObservers().visit(instrumented_loop())
    assert ast.dump(restored) == ast.dump(original_loop())


def test_original_xfuseg_augmentation_and_fractional_targets():
    _, _, functions = load_definitions()
    augmentation = functions["get_training_augmentation"]()
    assert augmentation.p == .9
    # This rotation includes shift .1 in the author's x-FUSegNet training file.
    rotation = augmentation.transforms[1].transforms[1]
    assert tuple(rotation.shift_limit_x) == (-.1, .1)
    assert tuple(rotation.rotate_limit) == (-30, 30)
    tensor = functions["to_tensor"](np.array([[[32 / 255.]]]))
    assert tensor.dtype == np.float32 and tensor.item() == np.float32(32 / 255.)


def test_ensemble_averages_probabilities_before_threshold():
    values = np.array([.49, .49, .49, .99, .99], dtype=np.float32).reshape(5, 1, 1)
    actual = mean_probability(values)
    assert np.array_equal(actual, np.mean(np.array(list(values)), axis=0))
    assert actual.item() >= .5
    assert np.mean(values >= .5) < .5


@pytest.mark.parametrize("maps", [np.ones((4, 2, 2), np.float32), np.ones((5, 2, 2), np.float64),
                                    np.full((5, 2, 2), np.nan, np.float32)])
def test_ensemble_rejects_invalid_inputs(maps):
    with pytest.raises(ValueError):
        mean_probability(maps)


def test_oof_fractional_mask_binarization_is_explicit():
    evaluator = SegmentationEvaluator(.5)
    row = evaluate_oof(evaluator, 'fractional.png', np.array([0, 32/255., 1]), np.array([.1, .1, .9]))
    assert (row['tp'], row['tn'], row['fp'], row['fn']) == (1, 2, 0, 0)


def test_resume_restores_source_policy_histories_and_random_streams():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    model(torch.ones(2, 2)).sum().backward(); optimizer.step()
    logs = {'dice_loss + focal_loss': .4, 'iou_score': .6, 'fscore': .7}
    variables = {'best_vloss': .4, 'best_viou': .55, 'cnt_patience': 2, 'best_model_epoch': 3,
                 'train_logs': logs, 'valid_logs': logs}
    checkpoint = copy.deepcopy({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(),
                               'scheduler':scheduler.state_dict(), 'source_variables':variables,
                               'random_state':random_state(), 'history':[{'train':logs,'validation':logs}]})
    expected = (random.random(), np.random.rand(), torch.rand(3))
    other = torch.nn.Linear(2, 1)
    other_optimizer = torch.optim.Adam(other.parameters(), lr=.9)
    ns={'model':other, 'optimizer':other_optimizer,
        'scheduler':torch.optim.lr_scheduler.ReduceLROnPlateau(other_optimizer)}
    restore_training_state(ns, checkpoint)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert all(torch.equal(v, other.state_dict()[k]) for k,v in model.state_dict().items())
    assert ns['best_viou'] == .55 and ns['cnt_patience'] == 2
    assert ns['store_val_loss'] == [.4] and ns['store_train_dice'] == [.7]
    assert other_optimizer.param_groups[0]['lr'] == .001


def test_original_checkpoint_epoch_field_is_retained(tmp_path):
    _, _, functions = load_definitions()
    path = tmp_path / 'author.pth'
    functions['save'](str(path), 1, {'weight':torch.ones(1)}, {'test':True})
    assert torch.load(path)['epoch'] == 2


def test_resume_keeps_previous_training_duration(tmp_path):
    import json
    fold=tmp_path/'fold-1';fold.mkdir()
    started='2026-10-28T00:00:00+00:00'
    (fold/'status.json').write_text(json.dumps({'started_at_utc':started,'seconds':12.5}))
    runtime=Runtime({},tmp_path,{}, {'DEVICE':torch.device('cpu')},False,True)
    runtime.fold=fold;runtime.index=0
    runtime.begin_fold()
    status=json.loads((fold/'status.json').read_text())
    assert status['started_at_utc']==started and status['seconds']>=12.5


def test_missing_gt_inference_keeps_original_image_preprocessing(tmp_path):
    from PIL import Image
    smp, _, functions=load_definitions()
    images=tmp_path/'images';labels=tmp_path/'labels';images.mkdir();labels.mkdir()
    image=np.zeros((512,512,3),dtype=np.uint8);image[:]=[20,80,160]
    mask=np.zeros((512,512),dtype=np.uint8);mask[0,0]=32
    Image.fromarray(image).save(images/'sample.png');Image.fromarray(mask).save(labels/'sample.png')
    preprocessing=smp.encoders.get_preprocessing_fn('efficientnet-b7','imagenet')
    original=functions['Dataset'](['sample.png'],str(images),str(labels),
               augmentation=functions['get_validation_augmentation'](),preprocessing=functions['get_preprocessing'](preprocessing))
    without_gt=InferenceImages([{'image_path':str(images/'sample.png'),'mask_path':None}],functions,preprocessing)
    original_image,target=original[0]
    assert original_image.shape==(3,512,512) and np.array_equal(original_image,without_gt[0])
    assert target[0,0,0]==np.float32(32/255.)
