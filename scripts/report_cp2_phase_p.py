"""Render CP2 completed baseline results using the existing analysis environment."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foot_ulcer_segmentation.cp2_report import build_report  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build_report(parser.parse_args().root)
