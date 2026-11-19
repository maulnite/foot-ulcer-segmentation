import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    config = json.loads(project_path(path).read_text())
    if config["reference_revision"] != "60e8cfe0c8e2dacffa55fe208b8abb6815f7f6cc":
        raise ValueError("This pipeline implements the audited reference revision only")
    training = config["training"]
    if training["loss"] != "author_dice_plus_focal_with_probability_input":
        raise ValueError("Unsupported loss; label modified objectives as separate implementations")
    if training["checkpoint_policy"] != "author_if_loss_elif_iou":
        raise ValueError("Unsupported checkpoint policy")
    if config["evaluation"]["evaluator"] != "offline_hard_line":
        raise ValueError("Primary evaluator must be explicitly offline_hard_line")
    return config
