"""Explicit, checksum-bound relocation support for historical publication audits."""

from pathlib import Path

from .common import digest_file, write_json
from .cp2_phase_n import read


def verify_preserved(base, inventory, relocations=None):
    """Accept a missing path only at its declared destination with identical bytes.

    A changed file at the original path is always an error. This does not edit
    the original inventory, create replacement links, or search for substitutes.
    """
    relocations = relocations or {}
    verified_moves = []
    for name, expected in inventory.items():
        path = Path(base)/name
        if path.is_file():
            if digest_file(path) != expected:
                raise ValueError(f"Protected content changed: {name}")
            continue
        if name not in relocations:
            raise ValueError(f"Protected file missing: {name}")
        replacement = Path(base)/relocations[name]
        if not replacement.is_file() or digest_file(replacement) != expected:
            raise ValueError(f"Relocated content differs: {name}")
        verified_moves.append({"original": name, "current": relocations[name], "sha256": expected})
    return {"files": len(inventory), "unchanged": True,
            "meaning_of_unchanged": "all content hashes identical; explicit path relocation recorded",
            "verified_relocations": verified_moves}


def append_json(path, value):
    """Reuse an identical completed JSON without touching its bytes or timestamp."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise FileExistsError(f"Preserve existing artifact: {path.name}")
        return
    write_json(path, value)
