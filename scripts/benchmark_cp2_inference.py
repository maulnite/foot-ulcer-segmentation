"""Post-training batch-1 inference timing on the same actual validation image."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foot_ulcer_segmentation.common import digest_file, write_json  # noqa: E402
from foot_ulcer_segmentation.cp2_data import verify_cp2_manifest  # noqa: E402
from foot_ulcer_segmentation.cp2_train import gpu_snapshot  # noqa: E402
from foot_ulcer_segmentation.data import make_dataset  # noqa: E402
from foot_ulcer_segmentation.engine import setup  # noqa: E402
from foot_ulcer_segmentation.reference import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    target = root / "batch1_inference.json"
    if target.exists():
        raise FileExistsError("Preserve completed inference timing")
    if json.loads((root / "train_workflow.json").read_text())["status"] != "completed":
        raise ValueError("Training and final evaluation must finish first")
    config = json.loads((root / "protocol.json").read_text())
    manifest = verify_cp2_manifest(json.loads((root / "data/manifest.json").read_text()))
    setup(config["seed"], config["cpu_threads"])
    dataset, rows = make_dataset(config, manifest, "validation")
    image, _ = dataset[0]
    results = {}
    for name in ("B0", "B1"):
        status = json.loads((root / name / "status.json").read_text())
        if digest_file(root / name / "best.pth") != status["best_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint checksum mismatch")
        model = build_model(config, pretrained=False)
        state = torch.load(root / name / "best.pth", map_location="cpu")
        model.load_state_dict(state["state_dict"])
        del state
        model = model.cuda().eval()
        inputs = torch.from_numpy(image[None]).cuda()
        times = []
        with torch.no_grad():
            for _ in range(5):
                model(inputs).sigmoid()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(30):
                start = time.perf_counter()
                probability = model(inputs).sigmoid()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        if not torch.isfinite(probability).all():
            raise ValueError("Nonfinite timed inference")
        results[name] = {"sample_id": rows[0]["sample_id"], "input_shape": list(inputs.shape),
                         "warmup_steps": 5, "timed_steps": 30,
                         "latency_median_ms": float(np.median(times) * 1000),
                         "latency_p95_ms": float(np.percentile(times, 95) * 1000),
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "gpu_overall": gpu_snapshot(),
                         "caveat": "Shared-GPU model forward+sigmoid only; excludes data loading and export; not exclusive latency"}
        del model, inputs, probability
        torch.cuda.empty_cache()
    write_json(target, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
