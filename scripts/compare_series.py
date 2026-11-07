"""
scripts/compare_series.py

Which series of a DICOM dump is the best one to annotate on? For every series
produced by scripts/to_dicom_folders.py this reports the geometry, the
reconstruction kernel, the noise and - when a mask already exists - how well the
annotated structure stands out from the tissue around it.

Usage:
    python scripts/compare_series.py data/raw/DD2210_Blue_Tuna/DD2210_dicom \\
        --mask data/reference_masks/DD2210_Bluefin_Tuna/Segmentation.nii.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.loaders import load_volume  # noqa: E402
from src.pipeline.runner import match_grid  # noqa: E402

MIN_SLICES = 20


def header_of(folder: Path):
    first = sorted(folder.glob("*.dcm"))[0]
    return pydicom.dcmread(first, stop_before_pixels=True)


def noise_of(data: np.ndarray, mask: np.ndarray) -> float:
    """Robust spread of the tissue, as a stand-in for the noise."""
    values = data[mask]
    if values.size < 100:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--mask", type=Path)
    arguments = parser.parse_args()

    reference = None
    if arguments.mask:
        import nibabel as nib
        image = nib.load(str(arguments.mask))
        reference = (np.asarray(image.dataobj) != 0).transpose(2, 1, 0)
        zooms = tuple(float(z) for z in image.header.get_zooms()[:3])[::-1]

    print("| series | slices | voxel [mm] | kernel | noise | inside mask | tissue around | CNR |")
    print("|---|---|---|---|---|---|---|---|")
    for folder in sorted(p for p in arguments.folder.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.dcm"))
        header = header_of(folder)
        kernel = str(getattr(header, "ConvolutionKernel", "") or "?")
        if len(files) < MIN_SLICES:
            print(f"| {folder.name} | {len(files)} | - | {kernel} | too few slices | | | |")
            continue
        volume = load_volume(folder, "dicom")
        data = np.asarray(volume.data, np.float32)
        spacing = tuple(float(s) for s in volume.spacing_mm)
        voxel = "x".join(f"{s:g}" for s in spacing)

        inside = around = cnr = float("nan")
        noise = float("nan")
        if reference is not None and data.shape[1:] == reference.shape[1:]:
            mask = match_grid(reference, zooms, data.shape, spacing)
            if mask.any():
                shell = ndi.binary_dilation(mask, iterations=6) & ~ndi.binary_dilation(mask, iterations=2)
                tissue = shell & (data > -100) & (data < 150)
                inside_values = data[mask & (data > -100) & (data < 150)]
                inside = float(np.median(inside_values)) if inside_values.size else float("nan")
                around = float(np.median(data[tissue])) if tissue.any() else float("nan")
                noise = noise_of(data, tissue)
                cnr = abs(inside - around) / noise if noise and np.isfinite(noise) else float("nan")
        if not np.isfinite(noise):
            middle = np.zeros(data.shape, bool)
            z, y, x = (n // 2 for n in data.shape)
            middle[z-5:z+5, y-20:y+20, x-20:x+20] = True
            noise = noise_of(data, middle & (data > -100) & (data < 150))
        print(f"| {folder.name} | {len(files)} | {voxel} | {kernel} | {noise:.0f} HU | "
              f"{inside:.0f} HU | {around:.0f} HU | {cnr:.2f} |")
        del data, volume
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
