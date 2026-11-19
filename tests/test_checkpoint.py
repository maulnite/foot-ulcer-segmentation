import ast
from contextlib import redirect_stdout
import io
import random

import numpy as np
import torch

from foot_ulcer_segmentation.checkpoint import (AuthorCheckpointPolicy, link_best,
                                               random_state, restore_random_state, save_checkpoint)
from foot_ulcer_segmentation.reference import REFERENCE


def test_checkpoint_policy_matches_the_original_conditional():
    tree = ast.parse((REFERENCE / "fusegnet_train.py").read_text())
    loop = next(node for node in tree.body if isinstance(node, ast.For))
    node = next(node for node in loop.body if isinstance(node, ast.If)
                and "best_vloss" in ast.unparse(node.test))
    code = compile(ast.Module(body=[node], type_ignores=[]), "reference_checkpoint", "exec")
    original = {"best_vloss": 1_000_000.0, "best_viou": 0.0,
                "cnt_patience": 0, "val_loss_key": "loss"}
    policy = AuthorCheckpointPolicy()
    with redirect_stdout(io.StringIO()):
        for epoch, (loss, iou) in enumerate([(1.0, 0.9), (1.1, 0.8), (0.8, 0.95),
                                            (0.9, 0.9), (1.0, 0.7), (1.0, 0.7)]):
            original.update(epoch=epoch, save_model=False, valid_logs={"loss": loss, "iou_score": iou})
            exec(code, original)
            assert policy.update(loss, iou) == original["save_model"]
            assert policy.best_loss == original["best_vloss"]
            assert policy.best_iou == original["best_viou"]
            assert policy.patience_count == original["cnt_patience"]


def test_resume_reproduces_the_next_optimizer_step(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    def model():
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.4), torch.nn.Linear(8, 1))

    def step(network, optimizer):
        inputs = torch.randn(3, 4) + random.random() + np.random.random()
        optimizer.zero_grad()
        loss = network(inputs).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    first = model()
    optimizer = torch.optim.Adam(first.parameters(), lr=1e-4)
    step(first, optimizer)
    save_checkpoint(tmp_path / "last.pth", {"model": first.state_dict(),
                                           "optimizer": optimizer.state_dict(), "rng": random_state()})
    link_best(tmp_path / "last.pth", tmp_path / "best.pth")
    expected_loss = step(first, optimizer)
    expected_parameters = [p.detach().clone() for p in first.parameters()]
    restored = model()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    state = torch.load(tmp_path / "best.pth", map_location="cpu")
    restored.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    restore_random_state(state["rng"])
    actual_loss = step(restored, restored_optimizer)
    assert torch.equal(actual_loss, expected_loss)
    assert all(torch.equal(a, b) for a, b in zip(restored.parameters(), expected_parameters))
    # Updating last atomically must not alter the selected best checkpoint.
    previous_best = (tmp_path / "best.pth").read_bytes()
    save_checkpoint(tmp_path / "last.pth", {"completed_epochs": 2})
    assert (tmp_path / "best.pth").read_bytes() == previous_best
