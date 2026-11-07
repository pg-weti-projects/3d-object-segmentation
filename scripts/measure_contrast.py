"""
scripts/measure_contrast.py

Reports how well the swim bladder stands out from the tissue inside the box, for
the scans of the project. These are the numbers behind the regime boundaries in
``src/models/contrast.py``.

Usage:
    python scripts/measure_contrast.py            # every case
    python scripts/measure_contrast.py oreo       # one case

The Gadus and tuna cases read volumes cached as .npy by the earlier analysis
(see CACHE below); they are skipped when the cache is not there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.box import Box, crop_volume  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.core.volume import Volume  # noqa: E402
from src.models import contrast as C  # noqa: E402

CACHE = Path.home() / "fishdata"

CASES: dict[str, dict] = {
    # The box and the seed of configs/experiments/check_oreo_box.yaml.
    "oreo": {
        "kind": "dicom",
        "path": "data/raw/Oreo_fish/density",
        "box": ([0.0, 147.0, 15.0], [165.0, 207.0, 48.0]),
        "seed": (17, 59, 10),
    },
    # 2x downsampled (the full grid does not fit in memory here); the box is the
    # bounding box of the reference mask, grown by 25%.
    "gadus": {
        "kind": "npy",
        "path": CACHE / "gadus_ds.npy",
        "mask_npy": CACHE / "gadus_ds_mask.npy",
        "spacing_mm": (0.413, 0.413, 0.413),
        "box_from_mask": 0.25,
    },
    # The same body at 1 mm (918 slices). The reference mask is on the 3.5 mm grid,
    # so it is resampled onto this one; both series cover the same z range
    # (596-1513 mm) from the same origin, which is what makes that valid.
    "tuna_1mm": {
        "kind": "dicom",
        "path": "data/raw/DD2210_Blue_Tuna/DD2210_dicom/04_CUERPO_CUERPO_2_0",
        "mask_nii": "data/reference_masks/DD2210_Bluefin_Tuna/Segmentation.nii.gz",
        "box_from_mask": 0.25,
    },
    # The body series of the tuna dump, split out by scripts/to_dicom_folders.py.
    # Its grid matches the reference mask exactly, so the box is the bounding box
    # of the mask, grown by 25%.
    "tuna": {
        "kind": "dicom",
        "path": "data/raw/DD2210_Blue_Tuna/DD2210_dicom/02_CUERPO_CUERPO_7_0",
        "mask_nii": "data/reference_masks/DD2210_Bluefin_Tuna/Segmentation.nii.gz",
        "box_from_mask": 0.25,
    },
}


def _volume_from_npy(path: Path, spacing_mm) -> Volume:
    data = np.asarray(np.load(path), dtype=np.float32)
    return Volume(data, tuple(float(s) for s in spacing_mm), "unknown", "npy", "0" * 64)


def reference_mask(case: dict, volume: Volume | None = None) -> np.ndarray:
    """The reference mask of a case, on the grid of its volume (z, y, x). A mask
    stored on a coarser grid (the tuna mask is on the 3.5 mm series) is resampled
    onto the grid of the volume."""
    if case.get("mask_npy") is not None:
        return np.asarray(np.load(case["mask_npy"]), dtype=bool)
    import nibabel as nib

    from src.pipeline.runner import match_grid

    image = nib.load(str(case["mask_nii"]))
    mask = np.asarray(image.dataobj) != 0
    mask = mask.transpose(2, 1, 0)  # NIfTI (x, y, z) -> (z, y, x)
    spacing = tuple(float(z) for z in image.header.get_zooms()[:3])[::-1]
    if volume is not None and mask.shape != tuple(volume.shape):
        mask = match_grid(mask, spacing, volume.shape, volume.spacing_mm)
    return mask


def _box_around_mask(mask: np.ndarray, spacing, grow: float) -> tuple[Box, tuple[int, int, int]]:
    """A box around the reference mask, grown by `grow` of its extent, and the
    centre of the mask as the seed - roughly what a user would draw by hand."""
    where = np.argwhere(mask)
    low, high = where.min(axis=0), where.max(axis=0) + 1
    margin = np.maximum(((high - low) * grow / 2).astype(int), 1)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, mask.shape)
    box = Box(
        min_mm=tuple(float(v * s) for v, s in zip(low, spacing)),
        max_mm=tuple(float(v * s) for v, s in zip(high, spacing)),
    )
    seed = tuple(int(v) for v in np.round(where.mean(axis=0)))
    return box, seed  # type: ignore[return-value]


def _case(name: str) -> dict | None:
    case = CASES[name]
    if case["kind"] == "npy":
        if not Path(case["path"]).exists():
            print(f"{name}: skipped, no cached volume at {case['path']}", file=sys.stderr)
            return None
        volume = _volume_from_npy(Path(case["path"]), case["spacing_mm"])
    else:
        volume = load_volume(case["path"], case["kind"])

    if case.get("box_from_mask") is not None:
        mask = reference_mask(case, volume)
        box, seed_full = _box_around_mask(mask, volume.spacing_mm, case["box_from_mask"])
    elif case.get("box") is not None:
        box = Box(min_mm=tuple(case["box"][0]), max_mm=tuple(case["box"][1]))
        seed_full = tuple(case["seed"])
    else:
        box, seed_full = None, tuple(n // 2 for n in volume.shape)

    if box is None:
        cropped, where = volume, (slice(0, None), slice(0, None), slice(0, None))
    else:
        cropped, where = crop_volume(volume, box)
    seed = tuple(int(s) - int(w.start or 0) for s, w in zip(seed_full, where))

    estimate = C.estimate(cropped.data, cropped.spacing_mm, seed)
    smooth = C.smooth_mm(cropped.data, cropped.spacing_mm, 0.8)
    found = C.region(smooth, cropped.spacing_mm, estimate)

    out = estimate.as_dict()
    out["case"] = name
    out["full_shape"] = list(volume.shape)
    out["box_shape"] = list(cropped.shape)
    out["spacing_mm"] = list(cropped.spacing_mm)
    out["seed_in_box"] = list(seed)
    out["region_voxels"] = int(found.sum())
    out["region_mm3"] = float(found.sum() * float(np.prod(cropped.spacing_mm)))
    return out


def main() -> int:
    names = sys.argv[1:] or list(CASES)
    results = []
    for name in names:
        row = _case(name)
        if row is not None:
            results.append(row)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
