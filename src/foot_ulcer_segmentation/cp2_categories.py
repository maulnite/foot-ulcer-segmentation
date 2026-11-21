"""Posthoc author-membership analysis; no model, split or selection changes."""

from pathlib import PurePosixPath

import numpy as np

from .evaluation import scores


METRICS = ("dice", "iou", "precision", "recall", "boundary_iou", "pfom")


def author_membership(categories):
    """Use verified JSON identities, never infer cutoffs from ambiguous comments."""
    if set(categories) != {str(i) for i in range(10)}:
        raise ValueError("Expected author JSON keys 0 through 9")
    membership = {}
    for category in range(10):
        names = categories[str(category)]
        if not isinstance(names, list):
            raise ValueError("Category members must be filename lists")
        for name in names:
            if (not isinstance(name, str) or not name or "\\" in name
                    or PurePosixPath(name).name != name or name in (".", "..")):
                raise ValueError("Author identities must be plain filenames")
            if name in membership:
                raise ValueError("Duplicate author identity, including across categories")
            membership[name] = category
    return membership


def author_cohort(manifest, membership):
    """Account for every original test identity, including audited exclusions."""
    eligible = manifest["splits"]["test"]
    excluded = [row for row in manifest["excluded"] if row["source_split"] == "test"]
    rows = eligible + excluded
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(membership):
        raise ValueError("Original author test population is missing, repeated or unknown")
    if any(row["source_split"] != "test" for row in rows):
        raise ValueError("Author membership applies to source-test only")
    if any((row["gt_pixels"] == 0) != (membership[row["name"]] == 0) for row in rows):
        raise ValueError("Author C0 must match the actual empty GT identities")
    output = []
    for category in range(10):
        kept = [r for r in eligible if membership[r["name"]] == category]
        removed = [r for r in excluded if membership[r["name"]] == category]
        fractions = [r["gt_pixels"] / r["canvas_pixels"] for r in kept]
        output.append({
            "category": f"C{category}", "json_key": str(category),
            "paper_category": category + 1,
            "original_images": len(kept) + len(removed), "eligible_images": len(kept),
            "excluded_images": len(removed),
            "eligible_gt_fraction_min": min(fractions) if fractions else None,
            "eligible_gt_fraction_max": max(fractions) if fractions else None,
            "range_interpretation": "observed retained GT range, NOT reconstructed category cutoffs",
            "eligible_sample_ids": sorted(r["sample_id"] for r in kept),
            "excluded": [{"sample_id": r["sample_id"], "reason": r["reason"]}
                         for r in sorted(removed, key=lambda r: r["sample_id"])],
        })
    return output


def summarize_rows(rows):
    """Same frozen CP2 metrics; data metrics are recomputed from pooled counts."""
    if not rows:
        return {"images": 0, "image_based": None, "data_based": None}
    counts = {key: sum(int(r[key]) for r in rows) for key in ("tp", "tn", "fp", "fn")}
    if any(value < 0 for value in counts.values()):
        raise ValueError("Confusion counts cannot be negative")
    return {
        "images": len(rows), "counts": counts, "data_based": scores(counts),
        "image_based": {key: float(np.mean([r[key] for r in rows])) for key in METRICS},
        "total_foreground_misses": sum(bool(r["total_foreground_miss"]) for r in rows),
        "images_with_prediction": sum(r["prediction_pixels"] > 0 for r in rows),
    }


def grouped_metrics(rows, membership):
    """Return C0–C9 even if a category has zero eligible observations."""
    identities = [r["name"] for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated prediction identity")
    groups = {f"C{i}": [] for i in range(10)}
    for row in rows:
        identity = PurePosixPath(row["name"])
        if identity.parent.as_posix() != "test" or identity.name not in membership:
            raise ValueError("Unmapped identity or unsupported validation/train population")
        category = membership[identity.name]
        if (row["gt_pixels"] == 0) != (category == 0):
            raise ValueError("Category does not match actual GT emptiness")
        groups[f"C{category}"].append(row)
    result = {category: summarize_rows(values) for category, values in groups.items()}
    whole = summarize_rows(rows)
    if sum(v["images"] for v in result.values()) != len(rows):
        raise ValueError("Category partition lost predictions")
    if rows:
        for key in ("tp", "tn", "fp", "fn"):
            if sum(v.get("counts", {}).get(key, 0) for v in result.values()) != whole["counts"][key]:
                raise ValueError("Category partition lost confusion counts")
        for key in METRICS:
            weighted = sum(v["images"] * v["image_based"][key]
                           for v in result.values() if v["images"]) / len(rows)
            np.testing.assert_allclose(weighted, whole["image_based"][key], rtol=0, atol=1e-12)
    return result
