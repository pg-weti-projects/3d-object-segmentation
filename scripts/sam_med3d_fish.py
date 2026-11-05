"""
scripts/sam_med3d_fish.py

The SAM-Med3D fish evaluation of the protocol (md-files/sam_med3d_protocol.md, outside the repo).

    python scripts/sam_med3d_fish.py plan    # clicks and boxes per fish, from the reference masks; no model

`plan` writes configs/experiments/sam_med3d_fish_plan.json. The protocol requires that file to be
committed before any model run, so that clicks and boxes cannot follow the results.

Per fish: the 9 clicks of the protocol's rule (`test_clicks`: the deepest voxel of the bladder, then one
voxel in each of 8 equal-count slabs along its longest axis, numpy seed 42), on the input's grid; and the
boxes of every setting:

    primary   the protocol's box (section 4: the mask's bounding box grown 25%)
    s3        the same box over the whole z range, as the web box draws it today
    s4_up, s4_down   the primary box shifted by 10% of its size along the diagonal
    s5        the bounding box grown 50% instead of 25%

A click that falls outside a box moves to the deepest bladder voxel inside it, as a user would click.
The model never sees the mask: it chooses where to click and draws the boxes, and is used to score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sam_med3d_control import imperfect_boxes, test_clicks  # noqa: E402
from src.core.box import Box  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.pipeline.metrics import dice  # noqa: E402
from src.pipeline.runner import _load_reference, match_grid  # noqa: E402

PLAN = Path("configs/experiments/sam_med3d_fish_plan.json")
CLICKS = 9

FISH = {
    "oreo": {
        "input_path": "data/raw/Oreo_fish/mass_density/mass_density.json",
        "format": None,
        "reference_mask_path": "data/reference_masks/Oreo_fish/Segmentation.nii.gz",
        "box": {"min_mm": [12.0, 129.0, 15.0], "max_mm": [87.0, 219.0, 48.0]},
        "fill_box_dice": 0.246,
        "extra_rows": {"sound_speed": "data/raw/Oreo_fish/sound_speed_compressional/sound_speed_compressional.json"},
    },
    "gadus": {
        "input_path": "data/raw/Gadus_Morhua/gadus_morhua_0.413mm.nii",
        "format": None,
        "reference_mask_path": "data/reference_masks/Gadus_Morhua/Segmentation.nii.gz",
        "box": {"min_mm": [64.85, 27.26, 45.85], "max_mm": [135.28, 47.09, 60.93]},
        "fill_box_dice": 0.177,
    },
    "tuna": {
        "input_path": "data/raw/DD2210_Blue_Tuna/02_CUERPO_CUERPO_7_0",
        "format": "dicom",
        "reference_mask_path": "data/reference_masks/DD2210_Bluefin_Tuna/Series_3_segmentation.nii.gz",
        "box": {"min_mm": [199.5, 124.17, 185.42], "max_mm": [567.0, 234.08, 307.07]},
        "fill_box_dice": 0.122,
        "extra_references": {"segmentation": "data/reference_masks/DD2210_Bluefin_Tuna/Segmentation.nii.gz"},
    },
}


def clicks_for(mask: np.ndarray, spacing) -> list[list[int]]:
    """`test_clicks` on a crop around the bladder, so that its distance transform does not cover the whole scan."""
    found = np.argwhere(mask)
    window = tuple(slice(max(int(a) - 1, 0), min(int(b) + 2, n)) for a, b, n in zip(found.min(0), found.max(0), mask.shape))
    return [[int(v) + w.start for v, w in zip(click, window)] for click in test_clicks(mask[window], spacing, CLICKS)]


def clicks_in_box(clicks, mask: np.ndarray, spacing, box: dict) -> list[list[int]]:
    where = Box.from_mapping(box).slices(mask.shape, spacing)
    deepest = None
    result = []
    for click in clicks:
        if all(w.start <= c < w.stop for c, w in zip(click, where)):
            result.append(list(click))
            continue
        if deepest is None:  # on the crop, as for the clicks
            inside = mask[where]
            deepest = [int(v) + w.start for v, w in zip(test_clicks(inside, spacing, 1)[0], where)]
        result.append(deepest)
    return result


def plan() -> int:
    output: dict = {"note": "clicks [z, y, x] on the input's grid; boxes in mm from the first voxel, [z, y, x]. Made without the model.", "fish": {}}
    for name, fish in FISH.items():
        volume = load_volume(fish["input_path"], fish["format"])
        reference, reference_spacing = _load_reference(fish["reference_mask_path"])
        mask = match_grid(reference, reference_spacing, volume.shape, volume.spacing_mm) != 0
        del reference
        spacing = volume.spacing_mm
        shape = volume.shape
        del volume
        clicks = clicks_for(mask, spacing)
        primary = fish["box"]
        variants = imperfect_boxes(mask, spacing, shape, primary)
        boxes = {
            "primary": primary,
            "s3": {"min_mm": [0.0, *primary["min_mm"][1:]], "max_mm": [shape[0] * spacing[0], *primary["max_mm"][1:]]},
            "s4_up": variants["shift_up_10"],
            "s4_down": variants["shift_down_10"],
            "s5": variants["grown_50"],
        }
        fill = np.zeros(shape, bool)
        fill[Box.from_mapping(primary).slices(shape, spacing)] = True
        fill_dice = dice(fill, mask)
        settings = {}
        for setting, box in boxes.items():
            where = Box.from_mapping(box).slices(shape, spacing)
            settings[setting] = {
                "box": box,
                "clicks": clicks_in_box(clicks, mask, spacing, box),
                "bladder_share_in_box": float(mask[where].sum() / mask.sum()),
            }
        extent = (np.argwhere(mask).max(0) - np.argwhere(mask).min(0) + 1) * np.asarray(spacing)
        output["fish"][name] = {
            **{key: value for key, value in fish.items() if key not in ("box", "fill_box_dice")},
            "grid_zyx": list(shape),
            "spacing_mm_zyx": list(spacing),
            "bladder_voxels": int(mask.sum()),
            "bladder_longest_extent_mm": float(extent.max()),
            "hd95_bar_mm": 0.1 * float(extent.max()),
            "clicks": clicks,
            "fill_box_dice": fill_dice,
            "fill_box_dice_in_protocol": fish["fill_box_dice"],
            "settings": settings,
        }
        moved = {s: sum(a != b for a, b in zip(v["clicks"], clicks)) for s, v in settings.items()}
        print(f"{name}: grid {shape} at {tuple(round(s, 4) for s in spacing)} mm, bladder {int(mask.sum())} voxels, "
              f"longest {extent.max():.1f} mm; fill-box Dice {fill_dice:.3f} (protocol {fish['fill_box_dice']}); "
              f"clicks moved per setting {moved}; bladder in box {[round(v['bladder_share_in_box'], 3) for v in settings.values()]}", flush=True)
        if abs(fill_dice - fish["fill_box_dice"]) > 0.0015:
            raise SystemExit(f"{name}: fill-box Dice {fill_dice:.4f} does not match the protocol's {fish['fill_box_dice']}: stopping")
    PLAN.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    print(f"written {PLAN}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("plan",))
    parser.parse_args()
    return plan()


if __name__ == "__main__":
    raise SystemExit(main())
