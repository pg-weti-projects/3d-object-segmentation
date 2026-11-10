"""
scripts/compare_methods.py

Compares the segmentation methods inside the same box, on the same seed point,
against the reference mask. Used to report what the contrast-driven start
(``init_from: contrast``) changes for each method.

Usage:
    python scripts/compare_methods.py                        # every case, every method
    python scripts/compare_methods.py --case oreo
    python scripts/compare_methods.py --case gadus --method chan_vese
    python scripts/compare_methods.py --init seed_ball       # only the old start

The table is printed as Markdown, with Dice, IoU and the run time. The cases and
the cached volumes are the ones of ``scripts/measure_contrast.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_contrast import CASES, _box_around_mask, _volume_from_npy, reference_mask  # noqa: E402
from src.core.box import Box, crop_volume  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.models import box_region  # noqa: E402
from src.models.params import CHAN_VESE, GEODESIC_ACTIVE_CONTOUR, GROWCUT  # noqa: E402
from src.models.segment import segment_volume  # noqa: E402
from src.pipeline.runner import _load_reference, match_grid  # noqa: E402

ANALYTICAL = (CHAN_VESE, GEODESIC_ACTIVE_CONTOUR, GROWCUT)
METHODS = ("box_region", *ANALYTICAL)

# The box is the domain for these two, and their own resampling is turned off, so
# that every method sees exactly the same voxels.
WHOLE_BOX = {"domain_size_mm": None, "compute_spacing_mm": None}
BASE_PARAMETERS: dict[str, dict] = {
    CHAN_VESE: {},
    GEODESIC_ACTIVE_CONTOUR: dict(WHOLE_BOX),
    GROWCUT: dict(WHOLE_BOX),
}

REFERENCE_MASKS = {"oreo": "data/reference_masks/Oreo_fish/Segmentation.nii.gz"}


def dice_iou(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    prediction, reference = prediction != 0, reference != 0
    overlap = float(np.count_nonzero(prediction & reference))
    union = float(np.count_nonzero(prediction | reference))
    total = float(prediction.sum() + reference.sum())
    return (2 * overlap / total if total else 0.0), (overlap / union if union else 0.0)


def load_case(name: str):
    """The volume cut to the box, the seed inside the box and the reference mask
    on the same voxels."""
    case = CASES[name]
    if case["kind"] == "npy":
        volume = _volume_from_npy(Path(case["path"]), case["spacing_mm"])
    else:
        volume = load_volume(case["path"], case["kind"])

    if case.get("box_from_mask") is not None:
        full_reference = reference_mask(case, volume)
        box, seed_full = _box_around_mask(full_reference, volume.spacing_mm, case["box_from_mask"])
    else:
        box = Box(min_mm=tuple(case["box"][0]), max_mm=tuple(case["box"][1]))
        seed_full = tuple(case["seed"])
        reference, spacing = _load_reference(REFERENCE_MASKS[name])
        full_reference = match_grid(reference, spacing, volume.shape, volume.spacing_mm)

    cropped, where = crop_volume(volume, box)
    seed = tuple(int(s) - int(w.start or 0) for s, w in zip(seed_full, where))
    return cropped, seed, full_reference[where]


def run_method(name: str, volume, seed, init_from: str):
    parameters = dict(BASE_PARAMETERS.get(name, {}))
    if name == "box_region":
        if init_from != "contrast":  # box_region has no other way of starting
            return None
        started = time.perf_counter()
        mask = box_region.segment(volume, {"seed_zyx": list(seed)})
        return mask, time.perf_counter() - started, {}
    parameters["init_from"] = init_from
    started = time.perf_counter()
    result = segment_volume(volume, {"name": name, "parameters": parameters}, list(seed))
    elapsed = time.perf_counter() - started
    return result.mask, elapsed, {"status": result.status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=list(CASES))
    parser.add_argument("--method", action="append", choices=list(METHODS))
    parser.add_argument("--init", action="append", choices=["seed_ball", "contrast"])
    arguments = parser.parse_args()
    cases = arguments.case or [name for name in CASES if name != "tuna_1mm"]
    methods = arguments.method or list(METHODS)
    inits = arguments.init or ["seed_ball", "contrast"]

    print("| case | method | start | Dice | IoU | voxels | time [s] | status |")
    print("|---|---|---|---|---|---|---|---|")
    for case in cases:
        volume, seed, reference = load_case(case)
        for method in methods:
            for init_from in inits:
                try:
                    outcome = run_method(method, volume, seed, init_from)
                except Exception as error:  # noqa: BLE001 - a failing method is a result too
                    print(f"| {case} | {method} | {init_from} | - | - | - | - | {type(error).__name__}: {error} |")
                    continue
                if outcome is None:
                    continue
                mask, elapsed, extra = outcome
                dice, iou = dice_iou(mask, reference)
                print(
                    f"| {case} | {method} | {init_from} | {dice:.3f} | {iou:.3f} | "
                    f"{int(np.count_nonzero(mask))} | {elapsed:.1f} | {extra.get('status', '')} |"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
