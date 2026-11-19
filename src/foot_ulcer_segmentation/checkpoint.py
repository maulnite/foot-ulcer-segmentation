"""Atomic state persistence and the audited source checkpoint selection rule."""

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class AuthorCheckpointPolicy:
    best_loss: float = 1_000_000.0
    best_iou: float = 0.0
    patience_count: int = 0

    def update(self, loss, iou):
        if not math.isfinite(loss) or not math.isfinite(iou):
            raise ValueError("Nonfinite validation metrics")
        if self.best_loss > loss:
            self.best_loss = loss
            self.patience_count = 0
            return True
        elif self.best_iou < iou:
            self.best_iou = iou
            self.patience_count = 0
            return True
        self.patience_count += 1
        return False

    def state_dict(self):
        return asdict(self)


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def link_best(last_path, best_path):
    """Preserve the selected epoch without serializing a second large checkpoint."""
    last_path, best_path = Path(last_path), Path(best_path)
    temporary = best_path.with_name(best_path.name + ".tmp")
    try:
        os.link(last_path, temporary)
        temporary.replace(best_path)
    finally:
        if temporary.exists():
            temporary.unlink()
