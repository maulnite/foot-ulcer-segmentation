"""Execute the pinned author's x-FUSegNet training loop with recording hooks."""

import ast
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import random
import time

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as BaseDataset
from sklearn.model_selection import KFold

from foot_ulcer_segmentation import engine, reference
from foot_ulcer_segmentation.checkpoint import random_state, restore_random_state, save_checkpoint
from foot_ulcer_segmentation.common import ROOT, digest_file, digest_json, project_path, write_json
from xfuseg_data import prepare

SOURCE = reference.REFERENCE / "xfusegnet_train.py"
DEFINITIONS = {"Dataset", "get_training_augmentation", "get_validation_augmentation", "to_tensor", "get_preprocessing", "save"}
RUNTIME = ["scripts/xfuseg_data.py", "scripts/xfuseg_train.py", "scripts/run_xfusegnet.py", "configs/xfusegnet.json"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def runtime_hashes():
    return {name: digest_file(ROOT / name) for name in RUNTIME}


def verify_source(config):
    snapshot = reference.verify_reference()
    constants = {}
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value
    expected = {"ENCODER": "efficientnet-b7", "ENCODER_WEIGHTS": "imagenet", "ACTIVATION": "sigmoid",
                "BATCH_SIZE": 2, "n_classes": 1, "LR": .0001, "EPOCHS": 200, "WEIGHT_DECAY": .00001,
                "PATIENCE": 30, "n_split": 5, "EARLY_STOP": True, "SAVE_BEST_MODEL": True,
                "SAVE_LAST_MODEL": False, "PERIOD": None}
    if any(constants.get(k) != v for k, v in expected.items()):
        raise ValueError("Source hyperparameters differ from the audited x-FUSegNet protocol")
    if config["model"] != {"encoder_name": expected["ENCODER"], "encoder_weights": expected["ENCODER_WEIGHTS"],
                            "classes": 1, "activation": "sigmoid", "decoder_attention_type": "pscse"}:
        raise ValueError("Model configuration differs from source")
    training = config["training"]
    for key, value in {"batch_size": 2, "num_workers": 4, "max_epochs": 200, "learning_rate": .0001,
                       "weight_decay": .00001, "early_stopping_patience": 30,
                       "scheduler": {"factor": .1, "patience": 10, "min_lr": .00001},
                       "checkpoint_policy": "author_if_loss_elif_iou",
                       "loss": "author_dice_plus_focal_with_probability_input"}.items():
        if training[key] != value:
            raise ValueError(f"Training configuration differs from source: {key}")
    if config["reference_revision"] != reference.REVISION or config["data"]["folds"] != 5:
        raise ValueError("Wrong source revision or fold count")
    if (config['seed'] != 42 or config['data']['image_size'] != [512,512]
            or config['data']['source_counts'] != {'train':810,'validation':200,'test':200}
            or config['evaluation']['prediction_threshold'] != .5):
        raise ValueError('Unexpected data/seed/threshold protocol')
    frozen = json.loads((ROOT / "outputs/runs/cp1-baseline-01/environment.json").read_text())
    import importlib.metadata as metadata
    if {name: metadata.version(name) for name in frozen["packages"]} != frozen["packages"]:
        raise ValueError("Use .venv-xfuseg with the unchanged frozen training dependencies")
    for name, expected_hash in frozen["pipeline_sha256"].items():
        if digest_file(ROOT / name) != expected_hash:
            raise ValueError("Frozen baseline package source changed")
    weight=ROOT/'outputs/cache/torch/hub/checkpoints/efficientnet-b7-dcc49843.pth'
    expected_weight=json.loads((ROOT/'outputs/runs/cp1-baseline-01/status.json').read_text())['pretrained_weight_sha256'][weight.name]
    if digest_file(weight)!=expected_weight:
        raise ValueError('Pretrained ImageNet weights changed')
    return snapshot, constants


def load_definitions():
    smp, utils, _ = reference.load_reference()
    namespace = {"__name__": __name__, "A": A, "cv2": cv2, "np": np, "os": os,
                 "BaseDataset": BaseDataset, "torch": torch, "deepcopy": copy.deepcopy}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS]
    if {node.name for node in nodes} != DEFINITIONS:
        raise ValueError("Missing original training definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return smp, utils, namespace


def original_loop():
    return next(node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple) and node.target.elts[0].id == "current_split")


def instrumented_loop():
    """Insert observers/resume support; leave every original numeric statement intact."""
    loop = copy.deepcopy(original_loop())
    loop.body[:0] = ast.parse("if _xf_skip_fold(current_split):\n    continue\n_xf_begin_fold()\n").body
    inner = next(node for node in loop.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "epoch")
    inner.iter = ast.parse("_xf_epoch_indices()", mode="eval").body
    inner.body[:0] = ast.parse("_xf_begin_epoch()\n").body
    stop = next(i for i, node in enumerate(inner.body) if isinstance(node, ast.If)
                and any(isinstance(n, ast.Name) and n.id == "EARLY_STOP" for n in ast.walk(node.test)))
    inner.body[stop:stop] = ast.parse("_xf_end_epoch()\n").body
    loop.body.extend(ast.parse("_xf_end_fold()\n").body)
    return ast.fix_missing_locations(loop)


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def restore_training_state(namespace, checkpoint):
    namespace["model"].load_state_dict(checkpoint["state_dict"])
    namespace["optimizer"].load_state_dict(checkpoint["optimizer"])
    namespace["scheduler"].load_state_dict(checkpoint["scheduler"])
    namespace.update(checkpoint["source_variables"])
    for prefix, family in [("train", "train"), ("val", "validation")]:
        for suffix, metric in [("loss", "dice_loss + focal_loss"), ("iou", "iou_score"), ("dice", "fscore")]:
            namespace[f"store_{prefix}_{suffix}"] = [r[family][metric] for r in checkpoint["history"]]
    restore_random_state(checkpoint["random_state"])


class Runtime:
    def __init__(self, config, run, manifest, namespace, smoke, resume):
        self.config, self.run, self.manifest, self.ns = config, run, manifest, namespace
        self.smoke, self.resume = smoke, resume
        self.history, self.selected_epoch = [], None
        self.started = time.monotonic()
        self.start_utc = utc()
        self.hashes = runtime_hashes()

    def skip_fold(self, index):
        self.fold = self.run / f"fold-{index+1}"
        self.index = index
        status = self.fold / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") in {"completed", "passed"}:
            if not self.resume:
                raise FileExistsError("Fold already completed; inspect before resuming")
            # Capture RNG after the original fold's plotting, preserving the chain.
            restore_random_state(torch.load(self.fold / "end-rng.pth", map_location="cpu"))
            return True
        return False

    def begin_fold(self):
        self.fold.mkdir(parents=True, exist_ok=True)
        self.history, self.selected_epoch = [], None
        self.fold_started = time.monotonic()
        self.fold_utc = utc()
        self.elapsed_offset = 0
        previous=self.fold/'status.json'
        if self.resume and previous.exists():
            record=json.loads(previous.read_text())
            self.fold_utc=record['started_at_utc']
            self.elapsed_offset=record.get('seconds',0)
        if self.ns["DEVICE"].type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.ns["DEVICE"])
        self.save_status("initializing")

    def epoch_indices(self):
        ns = self.ns
        expected = self.manifest["folds"][self.index]
        if ns["list_IDs_train"] != expected["train"] or ns["list_IDs_val"] != expected["validation"]:
            raise ValueError("Original KFold assignments differ from persisted manifest")
        if sum(p.numel() for p in ns["model"].parameters()) != 70968939:
            raise ValueError("Unexpected original P-scSE architecture")
        start = 0
        if self.resume and (self.fold / "last.pth").exists():
            checkpoint = torch.load(self.fold / "last.pth", map_location=ns["DEVICE"])
            if (checkpoint["runtime_sha256"] != self.hashes or checkpoint["config_fingerprint"] != digest_json(self.config)
                    or checkpoint["split_fingerprint"] != self.manifest["fingerprint"]):
                raise ValueError("Resume runtime/configuration/split changed")
            restore_training_state(ns, checkpoint)
            self.history, self.selected_epoch = checkpoint["history"], checkpoint["selected_epoch"]
            start = checkpoint["completed_epochs"]
            del checkpoint
        else:
            write_json(self.fold / "initialization.json", {"state_sha256": state_hash(ns["model"]),
                       "rng_policy": self.config["implementation"]["rng_policy"],
                       "registered_parameters": 70968939, "source_model_unmodified": True})
        return range(start, ns["EPOCHS"]) if ns["cnt_patience"] < ns["PATIENCE"] else range(0)

    def begin_epoch(self):
        self.epoch_started = time.monotonic()
        self.selected = False
        if self.smoke:
            self.encoder_before = next(self.ns["model"].parameters()).detach().clone()

    def save(self, *args, **kwargs):
        self.author_save(*args, **kwargs)
        self.selected = True

    def end_epoch(self):
        ns = self.ns
        train = {k: float(v) for k, v in ns["train_logs"].items()}
        valid = {k: float(v) for k, v in ns["valid_logs"].items()}
        if not all(np.isfinite(v) for v in [*train.values(), *valid.values()]):
            raise ValueError("Nonfinite training/validation metric")
        if self.smoke:
            gradients = [p.grad for p in ns["model"].parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("Nonfinite/missing smoke gradients")
            if torch.equal(self.encoder_before, next(ns["model"].parameters()).detach()):
                raise ValueError("Encoder did not update")
        epoch = ns["epoch"] + 1
        if self.selected:
            self.selected_epoch = epoch
        self.history.append({"epoch": epoch, "train": train, "validation": valid,
                             "selected_checkpoint": self.selected, "learning_rate": ns["optimizer"].param_groups[0]["lr"],
                             "seconds": time.monotonic() - self.epoch_started})
        variables = {k: ns[k] for k in ["best_vloss", "best_viou", "cnt_patience", "best_model_epoch", "train_logs", "valid_logs"]}
        state = {"state_dict": ns["model"].state_dict(), "optimizer": ns["optimizer"].state_dict(),
                 "scheduler": ns["scheduler"].state_dict(), "random_state": random_state(), "source_variables": variables,
                 "history": self.history, "selected_epoch": self.selected_epoch, "completed_epochs": epoch,
                 "config_fingerprint": digest_json(self.config), "split_fingerprint": self.manifest["fingerprint"],
                 "runtime_sha256": self.hashes}
        save_checkpoint(self.fold / "last.pth", state)
        write_json(self.fold / "history.json", self.history)
        self.save_status("running")
        print(json.dumps({"fold": self.index + 1, "epoch": epoch, "selected": self.selected,
                          "validation_loss": valid["dice_loss + focal_loss"], "patience": ns["cnt_patience"]}), flush=True)

    def save_status(self, status):
        ns = self.ns
        record = {"status": status, "fold": self.index + 1, "smoke_only": self.smoke,
                  "started_at_utc": self.fold_utc, "updated_at_utc": utc(), "completed_epochs": len(self.history),
                  "max_epochs": 1 if self.smoke else 200, "selected_epoch": self.selected_epoch,
                  "seconds": self.elapsed_offset + time.monotonic() - self.fold_started}
        if self.history:
            record.update(latest_validation=self.history[-1]["validation"], early_stopping_count=ns["cnt_patience"])
        if ns["DEVICE"].type == "cuda":
            record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(ns["DEVICE"]))
        write_json(self.fold / "status.json", record)
        write_json(self.run / "status.json", {"status": "running", "current_fold": self.index + 1,
                   "smoke_only": self.smoke, "updated_at_utc": utc(), "fold_status": record})

    def end_fold(self):
        plt.close("all")
        save_checkpoint(self.fold / "end-rng.pth", random_state())
        self.save_status("passed" if self.smoke else "completed")


def train(config, run, device="cuda:0", smoke=False, resume=False):
    _, constants = verify_source(config)
    manifest = prepare(config)
    smp, utils, ns = load_definitions()
    engine.setup(config["seed"], config["cpu_threads"])
    torch.hub.set_dir(str(ROOT / "outputs/cache/torch/hub"))
    if resume:
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Resume configuration differs")
        environment = json.loads((run / "environment.json").read_text())
        if environment["xfuseg_runtime_sha256"] != runtime_hashes():
            raise ValueError("Resume runtime changed")
        current = engine.environment_record(config, {"fingerprint": manifest["fingerprint"]}, torch.device(device))
        if current["packages"] != environment["packages"]:
            raise ValueError("Resume packages changed")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "config.json", config)
        write_json(run / "split-manifest.json", manifest)
        environment = engine.environment_record(config, {"fingerprint": manifest["fingerprint"]}, torch.device(device))
        environment["xfuseg_runtime_sha256"] = runtime_hashes()
        environment["source_training_file_sha256"] = digest_file(SOURCE)
        write_json(run / "environment.json", environment)
    checkpoint_loc = run / "author-checkpoints"
    checkpoint_loc.mkdir(exist_ok=True)
    plots = run / "plots"
    plots.mkdir(exist_ok=True)
    names = manifest["initial_directory_order"].copy()
    random.seed(42); random.shuffle(names)
    if names != manifest["shuffled_order"]:
        raise ValueError("Python shuffle changed")
    staging = project_path(config["data"]["staging"])
    ns.update(constants)
    ns.update(smp=smp, plt=plt, DataLoader=DataLoader, KFold=KFold, np=np,
              DEVICE=torch.device(device), EPOCHS=1 if smoke else 200,
              list_IDs_kfold=names, list_IDs_test=[r["name"] for r in manifest["test"]],
              x_train_dir=str(staging / "train/images"), x_valid_dir=str(staging / "train/images"),
              y_train_dir=str(staging / "train/labels"), y_valid_dir=str(staging / "train/labels"),
              model_name=config["name"], checkpoint_loc=str(checkpoint_loc),
              total_loss=utils.base.SumOfLosses(utils.losses.DiceLoss(), utils.losses.FocalLoss()),
              metrics=[utils.metrics.IoU(threshold=.5), utils.metrics.Fscore(threshold=.5)],
              kf=KFold(n_splits=5, shuffle=False), store_kfold_best_log=[])
    if smoke:
        author_dataset = ns["Dataset"]
        def small_dataset(ids, *args, **kwargs):
            return author_dataset(ids[:2], *args, **kwargs)
        ns["Dataset"] = small_dataset
    runtime = Runtime(config, run, manifest, ns, smoke, resume)
    runtime.author_save = ns["save"]
    ns.update(save=runtime.save, _xf_skip_fold=runtime.skip_fold, _xf_begin_fold=runtime.begin_fold,
              _xf_epoch_indices=runtime.epoch_indices, _xf_begin_epoch=runtime.begin_epoch,
              _xf_end_epoch=runtime.end_epoch, _xf_end_fold=runtime.end_fold)
    # Redirect only plot output; the original plot contents are retained verbatim.
    original_savefig = plt.Figure.savefig
    def savefig(figure, filename, *args, **kwargs):
        return original_savefig(figure, plots / Path(filename).name, *args, **kwargs)
    plt.Figure.savefig = savefig
    try:
        exec(compile(ast.Module(body=[instrumented_loop()], type_ignores=[]), str(SOURCE), "exec"), ns)
    finally:
        plt.Figure.savefig = original_savefig
    # Retain the original companion artifact expected by xfusegnet_test.py.
    best_logs=[]
    for index in range(5):
        history=json.loads((run/f'fold-{index+1}/history.json').read_text())
        selected=json.loads((run/f'fold-{index+1}/status.json').read_text())['selected_epoch']
        best_logs.append([index,min(r['validation']['dice_loss + focal_loss'] for r in history),selected-1])
    with (checkpoint_loc/(config['name']+'.pickle')).open('wb') as stream:
        pickle.dump(best_logs,stream)
    outcome = {"status": "passed" if smoke else "completed", "smoke_only": smoke, "folds": 5,
               "started_at_utc": runtime.start_utc, "finished_at_utc": utc(),
               "seconds": time.monotonic() - runtime.started, "split_fingerprint": manifest["fingerprint"],
               "source_training_loop_executed": True, "official_test_metrics_available": False}
    if smoke:
        model = ns["model"]
        inputs, _ = next(iter(ns["valid_loader"]))
        model.eval()
        with torch.no_grad():
            expected = model(inputs.to(device)).cpu()
            next(model.parameters()).add_(1)
        saved = torch.load(checkpoint_loc / "best_model_5.pth", map_location=device)
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            actual = model(inputs.to(device)).cpu()
        if not torch.equal(expected, actual):
            raise ValueError("Smoke checkpoint predictions changed after reload")
        outcome.update(checkpoint_prediction_equality_passed=True, finite_gradients_passed=True, encoder_updated=True)
        del saved
        for path in [*checkpoint_loc.glob("*.pth"), *run.glob("fold-*/last.pth")]:
            path.unlink()
        outcome["temporary_checkpoints_removed"] = True
    write_json(run / "status.json", outcome)
    return outcome
