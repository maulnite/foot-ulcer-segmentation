"""Preservation must remain strict when source files have been relocated."""

import pytest

from foot_ulcer_segmentation.common import digest_file
from foot_ulcer_segmentation.publication_integrity import append_json, verify_preserved


def test_explicit_relocation_requires_same_content_without_mutating_inventory(tmp_path):
    current = tmp_path/"current.py"
    current.write_text("original research code\n")
    checksum = digest_file(current)
    inventory = {"old.py": checksum}
    result = verify_preserved(tmp_path, inventory, {"old.py": "current.py"})
    assert result["unchanged"] and result["files"] == 1
    assert result["verified_relocations"][0]["sha256"] == checksum
    assert inventory == {"old.py": checksum} and not (tmp_path/"old.py").exists()
    current.write_text("changed code\n")
    with pytest.raises(ValueError, match="content differs"):
        verify_preserved(tmp_path, inventory, {"old.py": "current.py"})


def test_changed_original_cannot_be_hidden_by_valid_relocation(tmp_path):
    current, old = tmp_path/"current.py", tmp_path/"old.py"
    current.write_text("original\n")
    old.write_text("modified\n")
    with pytest.raises(ValueError, match="content changed"):
        verify_preserved(tmp_path, {"old.py": digest_file(current)}, {"old.py": "current.py"})


def test_missing_file_without_explicit_mapping_fails(tmp_path):
    with pytest.raises(ValueError, match="file missing"):
        verify_preserved(tmp_path, {"absent": "checksum"})


def test_existing_json_can_only_be_reused_identically(tmp_path):
    path = tmp_path/"metrics.json"
    append_json(path, {"dice": .84})
    before = (digest_file(path), path.stat().st_mtime_ns)
    append_json(path, {"dice": .84})
    assert before == (digest_file(path), path.stat().st_mtime_ns)
    with pytest.raises(FileExistsError, match="existing artifact"):
        append_json(path, {"dice": .85})
