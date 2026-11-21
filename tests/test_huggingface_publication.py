"""Publication checks never need network access, real credentials, or GPU."""

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/publish_huggingface_weights.py"
SPEC = importlib.util.spec_from_file_location("weight_publication", SCRIPT)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def test_inventory_excludes_venv_pretraining_and_rng(tmp_path):
    names = ["outputs/cp2/phase_p/run/B1/best.pth", "outputs/cp2/phase_p/run/B1/last.pth",
             "outputs/cache/torch/hub/pretrained.pth", "outputs/runs/run/end-rng.pth",
             ".venv/lib/site-packages/namespace.pth", "data/raw/mask.pt"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit-test-placeholder")
    found, excluded = publication.scan_checkpoints(tmp_path)
    assert {p.name for p in found} == {"best.pth", "last.pth"}
    assert len(excluded) == 2


@pytest.mark.parametrize(("owner", "source", "expected"), [
    ("orie", "outputs/cp2/phase_p/run/B1/best.pth", "orie/cp2/phase_p/run/B1/best.pth"),
    ("orie", "outputs/runs/cp1/fold-1/last.pth", "orie/cp1/cp1/fold-1/last.pth"),
    ("ganta", "output_chronic_wound/checkpoints/run/stage2_refiner.pth", "ganta/chronic-wound/run/stage2_refiner.pth"),
    ("ganta", "output_fuseg_2021/checkpoints/run/best_model.pth", "ganta/fuseg-2021/run/best_model.pth"),
])
def test_destination_retains_run_and_stage_identity(owner, source, expected):
    assert publication.destination(owner, Path(source)) == expected


def test_unclassified_weight_is_not_silently_published():
    with pytest.raises(ValueError, match="Unclassified"):
        publication.destination("orie", Path("unexpected/model.pth"))


def test_research_vs_smoke_resume_and_initializer_roles():
    assert publication.role_for(Path("run/smoke-A1/best.pth")) == "engineering_smoke_not_research_result"
    assert publication.role_for(Path("run/B1/best.pth")) == "historical_selected_checkpoint"
    assert publication.role_for(Path("run/B1/last.pth")) == "last_training_checkpoint_not_selected"
    assert publication.role_for(Path("run/initial_weights.pth")) == "initialization_only_not_trained"
    assert publication.role_for(Path("run/stage2_refiner.pth")) == "cascade_refiner_stage"


def test_checkpoint_inspection_preserves_original_bytes(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({"epoch": 8, "state_dict": {"weight": torch.ones(2, 3)}, "optimizer": {}}, path)
    before = publication.sha256(path)
    summary = publication.checkpoint_summary(path, torch)
    assert summary["tensor_count"] == 1
    assert summary["state_tensor_elements_including_buffers"] == 6
    assert summary["epoch_as_stored"] == 8
    assert summary["optimizer_included"]
    assert publication.sha256(path) == before


def test_rng_state_is_not_mislabeled_as_a_model(tmp_path):
    path = tmp_path / "rng.pth"
    torch.save({"python": (1, 2, 3)}, path)
    with pytest.raises(ValueError, match="does not contain model"):
        publication.checkpoint_summary(path, torch)


def test_sidecar_sanitization_retains_original(tmp_path):
    original = {"root": str(tmp_path / "orie/data"), "metrics": [0.84, 0.9]}
    safe = publication.sanitize(original, {"orie": tmp_path / "orie"})
    assert safe["root"] == "${ORIE_PROJECT_ROOT}/data"
    assert original["root"] == str(tmp_path / "orie/data")
    assert safe["metrics"] == original["metrics"]


def test_text_credential_scan_blocks_secret_publication(tmp_path):
    path = tmp_path / "config.txt"
    path.write_text("hf_" + "x" * 35)
    with pytest.raises(ValueError, match="Credential-like"):
        publication.checked_bytes(path)
