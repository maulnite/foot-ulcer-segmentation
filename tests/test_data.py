import pytest

from foot_ulcer_segmentation import common, data


def test_persisted_split_is_stable_and_detects_data_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(data, "ROOT", tmp_path)
    root = tmp_path / "data/raw/example"
    for split, count in [("train", 10), ("test", 2)]:
        for kind in ["images", "labels"]:
            directory = root / split / kind
            directory.mkdir(parents=True)
            for index in range(count):
                (directory / f"{index}.png").write_bytes(f"{split}-{kind}-{index}".encode())
    (root / "train/images/.ipynb_checkpoints").mkdir()
    config = {"data": {"root": "data/raw/example", "split_manifest": "data/splits/example.json",
                       "validation_fraction": 0.2, "shuffle_seed": 42,
                       "source_counts": {"train": 10, "test": 2}}}
    first = data.prepare_splits(config)
    assert {s: len(v) for s, v in first["splits"].items()} == {"train": 8, "validation": 2, "test": 2}
    train = {row["name"] for row in first["splits"]["train"]}
    validation = {row["name"] for row in first["splits"]["validation"]}
    assert not train & validation
    assert first["ignored_nonimage_directories"]["train"] == [".ipynb_checkpoints"]
    assert data.prepare_splits(config) == first
    (root / "train/images/0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Dataset file changed"):
        data.prepare_splits(config)


def test_manifest_edits_are_detected(tmp_path):
    manifest = {"splits": {}, "fingerprint": "incorrect"}
    with pytest.raises(ValueError, match="fingerprint"):
        data.verify_manifest(manifest)
