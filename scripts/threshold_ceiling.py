"""
scripts/threshold_ceiling.py

How much of each dataset is reachable by intensity at all.

All the analytical methods answer the same question - "which connected region
around the seed point is darker than the tissue around it" - and they differ
only in how they settle its boundary. Whether that answer *is* the annotated
target depends on the scan, not on the method: in one fish the bladder is a gas
space, in another a fluid-filled organ barely darker than the muscle around it.

This sweeps a single intensity threshold over the whole range, under the same
restriction to the body the methods use, and reports the best Dice any threshold
could reach. Comparing a method's Dice with that ceiling says whether the method
left anything on the table, which a raw Dice across species does not.

Usage:
    python scripts/threshold_ceiling.py              # every case
    python scripts/threshold_ceiling.py --case tuna
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_methods import dice_iou, load_case  # noqa: E402
from scripts.measure_contrast import CASES  # noqa: E402
from src.models import contrast as C  # noqa: E402

STEPS = 120


def ceiling(name: str) -> dict:
    volume, seed, reference = load_case(name)
    data = np.asarray(volume.data, dtype=np.float32)
    spacing = tuple(float(s) for s in volume.spacing_mm)
    smooth = C.smooth_mm(data, spacing, 0.8)
    estimate = C.estimate(data, spacing, seed, smooth=smooth)

    body = C.largest_component(smooth > threshold_otsu(smooth))
    hull = C.axis_hulls(body) if body.any() else np.ones(smooth.shape, dtype=bool)

    best_plain = best_region = 0.0
    best_level = float("nan")
    for level in np.percentile(smooth, np.linspace(0.5, 99.5, STEPS)):
        candidates = (smooth < level) & hull
        plain = dice_iou(candidates, reference)[0]
        region = 0.0
        if candidates[seed]:
            component = C.component_at(candidates, seed, spacing)
            if component is not None:
                region = dice_iou(component, reference)[0]
        if max(plain, region) > max(best_plain, best_region):
            best_plain, best_region, best_level = plain, region, float(level)

    candidates = (smooth < estimate.threshold) & hull
    component = C.component_at(candidates, seed, spacing) if candidates[seed] else None
    return {
        "case": name,
        "reference_voxels": int(reference.sum()),
        "separability": estimate.separability,
        "regime": estimate.regime,
        "ceiling_plain": best_plain,
        "ceiling_region": best_region,
        "ceiling_level": best_level,
        "at_method_threshold": dice_iou(component, reference)[0] if component is not None else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", choices=list(CASES))
    arguments = parser.parse_args()
    names = arguments.case or [name for name in CASES if name != "tuna_1mm"]

    print("| case | contrast | regime | best possible Dice | at the method's threshold |")
    print("|---|---|---|---|---|")
    for name in names:
        row = ceiling(name)
        print(
            f"| {row['case']} | {row['separability']:.3f} | {row['regime']} | "
            f"{max(row['ceiling_plain'], row['ceiling_region']):.3f} | {row['at_method_threshold']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
