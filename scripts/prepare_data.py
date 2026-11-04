"""
scripts/prepare_data.py

Prepares the scans of the project from the original files, in the layout of the
Raw Data Preprocessing panel, so that the same inputs work in the experiment
runner and in the web application. Independent of any segmentation method.

    data/raw/DD2210_Blue_Tuna/<series>/*.dcm       one folder per kept series
    data/raw/Gadus_Morhua/gadus_morhua_<mm>mm.nii  averaged down from the TIFF stack
    data/raw/Oreo_fish/<array>.json                one file per voxel array
    data/raw/manifest.json                         how each output was made

Usage:
    python scripts/prepare_data.py                        # every scan, one after another
    python scripts/prepare_data.py --only gadus --gadus-factor 2
    python scripts/prepare_data.py --only tuna --tuna-series 2 4
    python scripts/prepare_data.py --only tuna --all-tuna-series
    python scripts/prepare_data.py --force                # rebuild what already exists

The originals are only read. An output appears under its final name once it is
complete, so re-running skips what exists. Orientation problems are reported in
the manifest and on the console; nothing here corrects them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import nibabel as nib
import numpy as np
import pydicom
import tifffile
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import to_dicom_folders  # noqa: E402
from src.core.loaders import _hash_files, _series_spacing, _slice_position, _spacing_from_units  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINALS = REPO.parent / "data-from-mm"
DEFAULT_RAW = REPO / "data" / "raw"
DEFAULT_MASKS = REPO / "data" / "reference_masks"
MANIFEST = "manifest.json"

TUNA_ARCHIVE = "DD2210.tar.gz"
TUNA_DATASET = "DD2210_Blue_Tuna"
# Both have the grid of the reference masks: 02 is the body kernel the configs use,
# 03 the bone kernel the masks were drawn on.
TUNA_SERIES = (2, 3)
TUNA_MASKS = ("DD2210_Bluefin_Tuna/Series_3_segmentation.nii.gz", "DD2210_Bluefin_Tuna/Segmentation.nii.gz")

GADUS_ARCHIVE = "morphosource_media-id-000063105_download-c8572f9e.zip"
GADUS_DATASET = "Gadus_Morhua"
GADUS_FACTOR = 4
GADUS_MASKS = ("Gadus_Morhua/Segmentation.nii.gz",)

OREO_JSON = "boe002.json"
OREO_DATASET = "Oreo_fish"
OREO_ARRAYS = ("mass_density", "sound_speed_compressional")
OREO_MASKS = ("Oreo_fish/Segmentation.nii.gz",)

UNKNOWN_ORIENTATION = "this format carries no patient orientation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"WARNING: {message}", file=sys.stderr)


def staging(raw: Path) -> tempfile.TemporaryDirectory:
    """A scratch folder on the same disk as the outputs, so they can be renamed into place."""
    raw.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=raw, prefix=".prepare_")


def install(built: Path, target: Path) -> None:
    """Move a finished output to its final name, replacing an older one."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    built.replace(target)


def record(raw: Path, output: Path, entry: dict[str, Any]) -> None:
    """Add one output to the manifest straight away, so that a stopped run keeps what it finished."""
    path = raw / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"outputs": {}}
    manifest["outputs"][output.relative_to(raw).as_posix()] = {
        **entry,
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    partial = path.with_name(f"{MANIFEST}.part")
    partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    partial.replace(path)


def source_entry(path: Path, **details: Any) -> dict[str, Any]:
    return {"file": path.name, "sha256": sha256_file(path), **details}


def mask_grid(masks_root: Path, relative: str) -> dict[str, Any] | None:
    """Grid and orientation of a reference mask, from its header only."""
    path = masks_root / relative
    if not path.exists():
        return None
    image = nib.load(str(path))
    return {
        "path": relative,
        "shape_xyz": [int(n) for n in image.shape[:3]],
        "spacing_mm_xyz": [round(float(z), 5) for z in image.header.get_zooms()[:3]],
        "axcodes": "".join(nib.aff2axcodes(image.affine)),
        "world_z_of_first_slice": float(image.affine[2, 3]),
        "world_z_of_last_slice": float((image.affine @ [0, 0, image.shape[2] - 1, 1])[2]),
    }


def check_mask_grids(
    masks: list[dict[str, Any]], shape_zyx: tuple[int, ...], spacing_zyx: tuple[float, ...], warnings: list[str]
) -> None:
    for mask in masks:
        same_shape = list(shape_zyx)[::-1] == mask["shape_xyz"]
        same_spacing = np.allclose(list(spacing_zyx)[::-1], mask["spacing_mm_xyz"], rtol=1e-3)
        if not (same_shape and same_spacing):
            warn(
                warnings,
                f"reference mask {mask['path']} is on another grid: {mask['shape_xyz']} voxels (x, y, z) at "
                f"{mask['spacing_mm_xyz']} mm, this output is {list(shape_zyx)[::-1]} at "
                f"{[round(s, 5) for s in list(spacing_zyx)[::-1]]} mm. It must be resampled before scoring.",
            )


# --- DICOM dump (tuna) ---------------------------------------------------------------------------


def unpack_dicom_dump(archive: Path, target: Path, wanted: Callable[[int], bool]) -> dict[str, int]:
    """Writes the DICOM files of the wanted series from a flat tar into `target`, under
    their base names, one file in memory at a time. macOS ``._*`` files and anything
    without the DICOM marker are left out and counted."""
    target.mkdir(parents=True, exist_ok=True)
    counts = {"written": 0, "other_series": 0, "macos_files": 0, "not_dicom": 0}
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name).name  # never a path taken from the archive
            if name.startswith("._"):
                counts["macos_files"] += 1
                continue
            content = stream.extractfile(member).read()
            magic = content[to_dicom_folders.MAGIC_OFFSET : to_dicom_folders.MAGIC_OFFSET + 4]
            if magic != to_dicom_folders.DICOM_MAGIC:
                counts["not_dicom"] += 1
                continue
            header = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
            if not wanted(to_dicom_folders.series_key(header)[0]):
                counts["other_series"] += 1
                continue
            destination = target / name
            if destination.exists():
                raise ValueError(f"{archive.name} holds two files named {name}")
            destination.write_bytes(content)
            counts["written"] += 1
    return counts


def slice_z(header: Any) -> float | None:
    position = getattr(header, "ImagePositionPatient", None)
    return None if position is None else float(position[2])


def dicom_orientation(headers: list[Any]) -> dict[str, Any]:
    """Where the slices are, in the order the loader puts them (along the slice normal).
    `headers` must be in that order."""
    first = headers[0]
    orientation = getattr(first, "ImageOrientationPatient", None)
    normal = None
    if orientation is not None and len(orientation) == 6:
        values = [float(v) for v in orientation]
        normal = [float(v) for v in np.cross(values[:3], values[3:])]
    by_instance = min(headers, key=lambda header: int(getattr(header, "InstanceNumber", 0) or 0))
    return {
        "image_orientation_patient": [float(v) for v in orientation] if orientation is not None else None,
        "slice_normal": normal,
        "patient_position": str(getattr(first, "PatientPosition", "") or ""),
        "loader_slice_order": "along the slice normal (row x column direction)",
        "z_of_first_loaded_slice": slice_z(first),
        "z_of_last_loaded_slice": slice_z(headers[-1]),
        "z_of_first_instance": slice_z(by_instance),
    }


def check_z_order(orientation: dict[str, Any], masks: list[dict[str, Any]], warnings: list[str]) -> None:
    """Reports a mask whose slices run the other way than the loaded series. Does not correct it."""
    first, last = orientation["z_of_first_loaded_slice"], orientation["z_of_last_loaded_slice"]
    for mask in masks:
        start = mask["world_z_of_first_slice"]
        mirrored = abs(start - last) < abs(start - first)
        mask["mirrored_in_z_against_loaded_series"] = bool(mirrored)
        if mirrored:
            warn(
                warnings,
                f"reference mask {mask['path']} is mirrored in z against this series as loaded: slice 0 of the "
                f"mask is at z={start:g} mm, slice 0 of the loaded series at z={first:g} mm. Not corrected here.",
            )


def dicom_series_entry(folder: Path, archive: Path, masks: list[dict[str, Any]], unpacked: dict[str, int]) -> dict[str, Any]:
    files = sorted(folder.glob("*.dcm"))
    pairs = [(pydicom.dcmread(path, stop_before_pixels=True), path) for path in files]
    pairs.sort(key=lambda pair: _slice_position(pair[0]))
    headers = [header for header, _ in pairs]
    first = headers[0]
    info = to_dicom_folders.describe([(path, header) for header, path in pairs])
    spacing = _series_spacing(headers)
    shape = (len(headers), int(first.Rows), int(first.Columns))
    warnings: list[str] = []
    masks = [dict(mask) for mask in masks]
    orientation = dicom_orientation(headers)
    if orientation["slice_normal"] is not None:
        check_mask_grids(masks, shape, spacing, warnings)
        check_z_order(orientation, masks, warnings)
    else:
        warn(warnings, f"{folder.name} has no slice geometry and is not a volume of the fish")
    return {
        "dataset": folder.parent.name,
        "kind": "dicom-series",
        "source": source_entry(archive, unpacked=unpacked),
        "series": {
            "number": to_dicom_folders.series_key(first)[0],
            "description": info["description"],
            "kernel": str(getattr(first, "ConvolutionKernel", "") or ""),
            "thickness_mm": info["thickness_mm"],
        },
        "shape_zyx": list(shape),
        "spacing_mm_zyx": [float(s) for s in spacing],
        "spacing_source": "PixelSpacing and the distance between ImagePositionPatient of the slices",
        "scaling": {
            "pixels": "unchanged; the files are copied as they are",
            "rescale_slope": float(getattr(first, "RescaleSlope", 1.0)),
            "rescale_intercept": float(getattr(first, "RescaleIntercept", 0.0)),
        },
        "averaging": None,
        "orientation": orientation,
        "reference_masks": masks,
        "warnings": warnings,
        "output_sha256": _hash_files([path for _, path in pairs]),  # as Volume.source_hash of the series
        "output_bytes": sum(path.stat().st_size for path in files),
    }


def existing_series(dataset: Path) -> dict[int, Path]:
    """The series folders of a dataset by series number; ``to_dicom_folders`` names them NN_description."""
    folders = dataset.iterdir() if dataset.is_dir() else []
    return {int(path.name.split("_")[0]): path for path in folders if re.match(r"\d+_", path.name)}


def prepare_tuna(
    originals: Path,
    raw: Path,
    masks_root: Path = DEFAULT_MASKS,
    series: Iterable[int] | None = TUNA_SERIES,
    force: bool = False,
) -> list[Path]:
    """`series` = None keeps every series of the dump."""
    archive = originals / TUNA_ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(f"original not found: {archive}")
    dataset = raw / TUNA_DATASET
    numbers = None if series is None else set(series)
    existing = existing_series(dataset)
    if not force and numbers is not None and numbers <= set(existing):
        for number in sorted(numbers):
            print(f"tuna: {existing[number].name} exists, skipped (--force rebuilds it)")
        return []

    def wanted(number: int) -> bool:
        return (numbers is None or number in numbers) and (force or number not in existing)

    masks = [grid for grid in (mask_grid(masks_root, relative) for relative in TUNA_MASKS) if grid]
    written = []
    with staging(raw) as scratch:
        dump, split = Path(scratch) / "dump", Path(scratch) / "split"
        print(f"tuna: reading {archive.name} ...")
        unpacked = unpack_dicom_dump(archive, dump, wanted)
        print(f"tuna: {unpacked}")
        if not unpacked["written"]:
            print("tuna: nothing new to write")
            return []
        if to_dicom_folders.main([str(dump), "--out", str(split), "--mode", "move"]) != 0:
            raise RuntimeError(f"could not split {archive.name} into series")
        for folder in sorted(path for path in split.iterdir() if path.is_dir()):
            entry = dicom_series_entry(folder, archive, masks, unpacked)
            target = dataset / folder.name
            install(folder, target)
            record(raw, target, entry)
            written.append(target)
            print(f"tuna: {target.relative_to(raw)}: {entry['shape_zyx']} at {entry['spacing_mm_zyx']} mm")
    for number in sorted(set() if numbers is None else numbers - set(existing_series(dataset))):
        print(f"WARNING: series {number} is not in {archive.name}", file=sys.stderr)
    return written


# --- TIFF stack in a zip within a zip (Gadus) ------------------------------------------------------


def manifest_spacing(archive: zipfile.ZipFile) -> tuple[tuple[float, float, float], str]:
    """Voxel size (x, y, z) in mm from the media manifest in the archive, and which file gave it."""
    names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not names:
        raise ValueError("the archive has no media manifest (.csv), and the TIFF stack has no z spacing of its own")
    with archive.open(names[0]) as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
    if len(rows) != 1:
        raise ValueError(f"{names[0]}: expected one media, found {len(rows)}")
    row = rows[0]
    if str(row.get("unit", "")).strip().lower() != "mm":
        raise ValueError(f"{names[0]}: voxel size is in {row.get('unit')!r}, expected mm")
    return (float(row["x_pixel_spacing"]), float(row["y_pixel_spacing"]), float(row["z_pixel_spacing"])), names[0]


def tiff_names(archive: zipfile.ZipFile) -> list[str]:
    """The TIFF slices of an archive in slice order, without macOS leftovers."""

    def number(name: str) -> tuple[int, str]:
        digits = re.findall(r"\d+", PurePosixPath(name).stem)
        return (int(digits[-1]) if digits else -1, name)

    names = [
        name
        for name in archive.namelist()
        if name.lower().endswith((".tif", ".tiff"))
        and not PurePosixPath(name).name.startswith("._")
        and "__MACOSX" not in name
    ]
    return sorted(names, key=number)


def tag_spacing_mm(page: Any) -> tuple[float, float] | None:
    """In-plane voxel size (x, y) from the resolution tags of a TIFF page, if they are in inches."""
    tags = page.tags
    if not {"XResolution", "YResolution", "ResolutionUnit"} <= {tag.name for tag in tags.values()}:
        return None
    if int(tags["ResolutionUnit"].value) != 2:
        return None
    return tuple(25.4 / float(Fraction(*tags[name].value)) for name in ("XResolution", "YResolution"))


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Averages `factor` x `factor` blocks of a slice. Rows and columns that do not fill a block are dropped."""
    rows, columns = (n // factor for n in image.shape)
    trimmed = np.asarray(image, dtype=np.float64)[: rows * factor, : columns * factor]
    return trimmed.reshape(rows, factor, columns, factor).mean(axis=(1, 3))


def averaged_slices(slices: Iterable[np.ndarray], factor: int) -> Iterator[np.ndarray]:
    """Averages blocks of `factor` voxels on all three axes, holding one output slice in memory.
    Slices left over at the end, too few to fill a block, are dropped."""
    total, count = None, 0
    for image in slices:
        reduced = block_mean(image, factor)
        total = reduced if total is None else total + reduced
        count += 1
        if count == factor:
            yield (total / factor).astype(np.float32)
            total, count = None, 0


def write_nifti(
    path: Path, slices: Iterable[np.ndarray], shape_zyx: tuple[int, int, int], spacing_xyz: tuple[float, float, float]
) -> None:
    """Writes one uncompressed float32 .nii slice by slice. Array axes are (x, y, z) with no flip,
    and the header codes say the orientation is unknown, because it is."""
    depth, rows, columns = shape_zyx
    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header.set_data_shape((columns, rows, depth))
    header.set_zooms(tuple(float(s) for s in spacing_xyz))
    header.set_xyzt_units("mm")
    header.set_qform(None, code=0)
    header.set_sform(None, code=0)
    header["vox_offset"] = 352
    written = 0
    with path.open("wb") as stream:
        header.write_to(stream)
        stream.write(b"\0" * (352 - stream.tell()))
        for image in slices:
            if image.shape != (rows, columns):
                raise ValueError(f"slice {written} has shape {image.shape}, expected {(rows, columns)}")
            stream.write(np.asarray(image, dtype="<f4").tobytes())  # C order of (y, x): x runs fastest
            written += 1
    if written != depth:
        raise ValueError(f"wrote {written} slices, expected {depth}")


def prepare_gadus(
    originals: Path, raw: Path, masks_root: Path = DEFAULT_MASKS, factor: int = GADUS_FACTOR, force: bool = False
) -> list[Path]:
    if factor < 1:
        raise ValueError("the averaging factor must be at least 1")
    archive = originals / GADUS_ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(f"original not found: {archive}")
    with zipfile.ZipFile(archive) as outer:
        spacing, manifest_name = manifest_spacing(outer)
        output_spacing = tuple(s * factor for s in spacing)
        target = raw / GADUS_DATASET / f"gadus_morhua_{output_spacing[0]:.3f}mm.nii"
        if target.exists() and not force:
            print(f"gadus: {target.name} exists, skipped (--force rebuilds it)")
            return []
        inners = [name for name in outer.namelist() if name.lower().endswith(".zip")]
        if len(inners) != 1:
            raise ValueError(f"{archive.name}: expected one inner zip, found {len(inners)}")
        with outer.open(inners[0]) as inner_stream, zipfile.ZipFile(inner_stream) as inner:
            names = tiff_names(inner)
            if not names:
                raise ValueError(f"{inners[0]} holds no TIFF slices")

            def read(name: str) -> np.ndarray:
                return tifffile.imread(io.BytesIO(inner.read(name)))

            first = read(names[0])
            with tifffile.TiffFile(io.BytesIO(inner.read(names[0]))) as tiff:
                tags = tag_spacing_mm(tiff.pages[0])
            warnings: list[str] = []
            if tags is not None and not np.allclose(tags, spacing[:2], rtol=0.01):
                warn(warnings, f"TIFF resolution tags give {tags} mm, the media manifest {spacing[:2]} mm")
            shape = (len(names) // factor, first.shape[0] // factor, first.shape[1] // factor)
            if 0 in shape:
                raise ValueError(f"factor {factor} is larger than the stack {(len(names), *first.shape)}")

            def slices() -> Iterator[np.ndarray]:
                for index, name in enumerate(names):
                    image = read(name)
                    if image.shape != first.shape:
                        raise ValueError(f"{name} has shape {image.shape}, expected {first.shape}")
                    if index % 200 == 0:
                        print(f"gadus: slice {index} of {len(names)}", flush=True)
                    yield image

            print(f"gadus: {len(names)} slices of {first.shape} {first.dtype}, averaging by {factor}")
            with staging(raw) as scratch:
                built = Path(scratch) / target.name
                write_nifti(built, averaged_slices(slices(), factor), shape, output_spacing)
                install(built, target)

    masks = [grid for grid in (mask_grid(masks_root, relative) for relative in GADUS_MASKS) if grid]
    check_mask_grids(masks, shape, output_spacing[::-1], warnings)
    warnings.append("load_volume labels every NIfTI as HU; these are uncalibrated TIFF values")
    record(
        raw,
        target,
        {
            "dataset": GADUS_DATASET,
            "kind": "nifti",
            "source": source_entry(archive, member=inners[0], slices=len(names), slice_shape=list(first.shape)),
            "series": None,
            "shape_zyx": list(shape),
            "spacing_mm_zyx": [float(s) for s in output_spacing[::-1]],
            "spacing_source": f"{manifest_name} (x/y/z_pixel_spacing) x averaging factor",
            "scaling": {"values": f"{first.dtype} TIFF values, averaged, stored as float32; not calibrated"},
            "averaging": {
                "factor": factor,
                "method": "mean of factor^3 blocks on all three axes",
                "original_spacing_mm_xyz": list(spacing),
                "dropped_trailing_voxels_zyx": [
                    len(names) - shape[0] * factor,
                    first.shape[0] - shape[1] * factor,
                    first.shape[1] - shape[2] * factor,
                ],
            },
            "orientation": {
                "known": False,
                "note": UNKNOWN_ORIENTATION,
                "array_axes": "z = TIFF slice index ascending, y = TIFF row, x = TIFF column; no flips",
            },
            "reference_masks": masks,
            "warnings": warnings,
            "output_sha256": sha256_file(target),
            "output_bytes": target.stat().st_size,
        },
    )
    print(f"gadus: {target.relative_to(raw)}: {list(shape)} at {[round(s, 5) for s in output_spacing[::-1]]} mm")
    return [target]


# --- Voxel JSON (Oreo) ------------------------------------------------------------------------------


def voxel_arrays(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The arrays of the single shape of a voxel JSON, in the form ``load_volume`` reads."""
    specimens = payload["specimens"]
    if len(specimens) != 1 or len(specimens[0]["shapes"]) != 1:
        raise ValueError("expected one specimen with one shape")
    shape = specimens[0]["shapes"][0]
    spacing = _spacing_from_units(shape["voxel_size"], shape.get("voxel_size_units", "mm"))
    return {
        name: {"data": shape[name], "spacing_mm": list(spacing), "units": str(shape.get(f"{name}_units", "unknown"))}
        for name in OREO_ARRAYS
    }


def compare_geometry(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    """Do two arrays of one specimen outline the same shapes? Compared as air (the lowest value)
    and as the two classes of an Otsu split; the values themselves have different units."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        return {"same_shape": False, "differs": True, "shapes": [list(first.shape), list(second.shape)]}

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        total = int(a.sum()) + int(b.sum())
        return 2.0 * int((a & b).sum()) / total if total else 1.0

    air_a, air_b = first <= first.min(), second <= second.min()
    dense_a, dense_b = first > threshold_otsu(first.ravel()), second > threshold_otsu(second.ravel())
    return {
        "same_shape": True,
        "differs": bool(np.any(air_a != air_b) or np.any(dense_a != dense_b)),
        "air_voxels": [int(air_a.sum()), int(air_b.sum())],
        "air_disagreement_voxels": int(np.sum(air_a != air_b)),
        "dice_of_non_air": dice(~air_a, ~air_b),
        "otsu_disagreement_voxels": int(np.sum(dense_a != dense_b)),
        "dice_of_otsu_dense_class": dice(dense_a, dense_b),
        "correlation": float(np.corrcoef(first.ravel(), second.ravel())[0, 1]),
        "voxels": int(first.size),
    }


def prepare_oreo(originals: Path, raw: Path, masks_root: Path = DEFAULT_MASKS, force: bool = False) -> list[Path]:
    source = originals / OREO_JSON
    if not source.is_file():
        raise FileNotFoundError(f"original not found: {source}")
    dataset = raw / OREO_DATASET
    targets = {name: dataset / f"{name}.json" for name in OREO_ARRAYS}
    if not force and all(path.exists() for path in targets.values()):
        print("oreo: both arrays exist, skipped (--force rebuilds them)")
        return []
    arrays = voxel_arrays(json.loads(source.read_text(encoding="utf-8")))
    geometry = compare_geometry(*(np.asarray(arrays[name]["data"]) for name in OREO_ARRAYS))
    verdict = "differs" if geometry["differs"] else "is the same"
    print(f"oreo: the geometry of {' and '.join(OREO_ARRAYS)} {verdict}: {geometry}")
    masks = [grid for grid in (mask_grid(masks_root, relative) for relative in OREO_MASKS) if grid]
    written = []
    with staging(raw) as scratch:
        for name, target in targets.items():
            if target.exists() and not force:
                print(f"oreo: {target.name} exists, skipped (--force rebuilds it)")
                continue
            array = arrays[name]
            built = Path(scratch) / target.name
            built.write_text(json.dumps(array), encoding="utf-8")
            install(built, target)
            shape = list(np.shape(array["data"]))
            warnings: list[str] = []
            check_mask_grids(masks, tuple(shape), tuple(array["spacing_mm"]), warnings)
            record(
                raw,
                target,
                {
                    "dataset": OREO_DATASET,
                    "kind": "voxels-json",
                    "source": source_entry(source, member=f"specimens[0].shapes[0].{name}"),
                    "series": name,
                    "shape_zyx": shape,
                    "spacing_mm_zyx": array["spacing_mm"],
                    "spacing_source": "specimens[0].shapes[0].voxel_size, converted to mm",
                    "scaling": {"values": f"unchanged, in {array['units']}"},
                    "averaging": None,
                    "orientation": {
                        "known": False,
                        "note": UNKNOWN_ORIENTATION,
                        "array_axes": "as stored in the JSON; load_volume reads them as (z, y, x); no flips",
                    },
                    "geometry_comparison": geometry,
                    "reference_masks": masks,
                    "warnings": warnings,
                    "output_sha256": sha256_file(target),
                    "output_bytes": target.stat().st_size,
                },
            )
            written.append(target)
            print(f"oreo: {target.relative_to(raw)}: {shape} at {array['spacing_mm']} mm")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--originals", type=Path, default=DEFAULT_ORIGINALS, help="folder with the original scans")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="where to write the prepared scans")
    parser.add_argument("--only", choices=("tuna", "gadus", "oreo"), action="append", help="prepare only these scans")
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    parser.add_argument("--gadus-factor", type=int, default=GADUS_FACTOR, help="averaging factor (default: %(default)s)")
    parser.add_argument("--tuna-series", type=int, nargs="+", default=list(TUNA_SERIES), help="series numbers to keep")
    parser.add_argument("--all-tuna-series", action="store_true", help="keep every series of the dump")
    arguments = parser.parse_args(argv)

    steps: dict[str, Callable[[], list[Path]]] = {
        "tuna": lambda: prepare_tuna(
            arguments.originals,
            arguments.raw,
            series=None if arguments.all_tuna_series else arguments.tuna_series,
            force=arguments.force,
        ),
        "gadus": lambda: prepare_gadus(
            arguments.originals, arguments.raw, factor=arguments.gadus_factor, force=arguments.force
        ),
        "oreo": lambda: prepare_oreo(arguments.originals, arguments.raw, force=arguments.force),
    }
    for name, step in steps.items():
        if arguments.only and name not in arguments.only:
            continue
        try:
            step()
        except (OSError, ValueError, RuntimeError) as error:
            print(f"error: {name}: {error}", file=sys.stderr)
            return 1
    print(f"manifest: {arguments.raw / MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
