"""
scripts/to_dicom_folders.py

Turns a flat DICOM dump (files without an extension, several series mixed
together, as in ``data/raw/DD2210_Blue_Tuna/DD2210``) into one folder per series
with ``.dcm`` files, which is what ``load_volume(path, "dicom")`` expects: it
reads ``*.dcm`` from a single folder and orders the slices by their position.

Nothing is converted in the sense of re-encoding - the files already are DICOM
(they carry the "DICM" marker). They only get a name the loader recognises, and
are split by series so that one folder is one volume.

Usage:
    python scripts/to_dicom_folders.py data/raw/DD2210_Blue_Tuna/DD2210
    python scripts/to_dicom_folders.py <source> --out <folder> --mode link
    python scripts/to_dicom_folders.py <source> --dry-run

Modes:
    copy  (default) leaves the dump untouched and writes copies
    link  hard-links instead of copying: no extra disk space, same bytes
    move  renames the files out of the dump (the dump ends up empty)

Raw data stays out of git: keep the result under ``data/raw/`` (ignored).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

DICOM_MAGIC = b"DICM"
MAGIC_OFFSET = 128


def is_dicom(path: Path) -> bool:
    """True when the file carries the DICOM marker at offset 128."""
    try:
        with path.open("rb") as handle:
            handle.seek(MAGIC_OFFSET)
            return handle.read(4) == DICOM_MAGIC
    except OSError:
        return False


def slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return cleaned[:60] or "series"


def series_key(dataset: Any) -> tuple[int, str]:
    number = getattr(dataset, "SeriesNumber", None)
    try:
        number = int(number)
    except (TypeError, ValueError):
        number = 0
    return number, str(getattr(dataset, "SeriesInstanceUID", "") or "unknown")


def folder_name(number: int, uid: str, description: str, duplicate: bool) -> str:
    name = f"{number:02d}_{slug(description)}"
    return f"{name}_{uid.rsplit('.', 1)[-1][:8]}" if duplicate else name


def scan(source: Path) -> tuple[dict[tuple[int, str], list[tuple[Path, Any]]], list[Path]]:
    """Group the files of the dump by series; also report what is not DICOM."""
    series: dict[tuple[int, str], list[tuple[Path, Any]]] = defaultdict(list)
    skipped: list[Path] = []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        if not is_dicom(path):
            skipped.append(path)
            continue
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as error:  # noqa: BLE001 - a broken file is a result too
            print(f"  cannot read {path.name}: {error}", file=sys.stderr)
            skipped.append(path)
            continue
        series[series_key(dataset)].append((path, dataset))
    return series, skipped


def describe(datasets: list[tuple[Path, Any]]) -> dict[str, Any]:
    _, first = datasets[0]
    thickness = getattr(first, "SliceThickness", None)
    positions = []
    for _path, dataset in datasets:
        position = getattr(dataset, "ImagePositionPatient", None)
        if position is not None and len(position) == 3:
            positions.append(float(position[2]))
    step = None
    if len(positions) > 1:
        ordered = sorted(positions)
        gaps = [round(b - a, 4) for a, b in zip(ordered, ordered[1:])]
        step = sorted(gaps)[len(gaps) // 2] if gaps else None  # median gap
    return {
        "description": str(getattr(first, "SeriesDescription", "") or ""),
        "modality": str(getattr(first, "Modality", "") or ""),
        "slices": len(datasets),
        "rows": int(getattr(first, "Rows", 0) or 0),
        "columns": int(getattr(first, "Columns", 0) or 0),
        "thickness_mm": None if thickness is None else float(thickness),
        "spacing_z_mm": step,
    }


def place(source_file: Path, target: Path, mode: str) -> None:
    if target.exists():
        return
    if mode == "copy":
        shutil.copy2(source_file, target)
    elif mode == "link":
        try:
            os.link(source_file, target)
        except OSError:  # different filesystem, or links not allowed
            shutil.copy2(source_file, target)
    elif mode == "move":
        shutil.move(str(source_file), str(target))
    else:  # pragma: no cover - argparse limits the choices
        raise ValueError(f"unknown mode {mode!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="folder with the DICOM dump")
    parser.add_argument("--out", type=Path, help="where to write the series (default: <source>_dicom)")
    parser.add_argument("--mode", choices=("copy", "link", "move"), default="copy")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be written")
    arguments = parser.parse_args(argv)

    source: Path = arguments.source
    if not source.is_dir():
        print(f"error: {source} is not a folder", file=sys.stderr)
        return 2
    out: Path = arguments.out or source.with_name(f"{source.name}_dicom")

    print(f"reading {source} ...")
    series, skipped = scan(source)
    if not series:
        print("error: no DICOM files found (no file carries the DICM marker)", file=sys.stderr)
        return 1

    names: dict[tuple[int, str], str] = {}
    counts: dict[str, int] = defaultdict(int)
    for (number, uid), datasets in series.items():
        counts[f"{number:02d}_{slug(describe(datasets)['description'])}"] += 1
    for key, datasets in sorted(series.items()):
        number, uid = key
        info = describe(datasets)
        base = f"{number:02d}_{slug(info['description'])}"
        names[key] = folder_name(number, uid, info["description"], counts[base] > 1)

    total = sum(len(datasets) for datasets in series.values())
    extra = " ({} file(s) that are not DICOM)".format(len(skipped)) if skipped else ""
    print("\n{} series, {} files{}\n".format(len(series), total, extra))
    print("| folder | description | slices | size | thickness | z step |")
    print("|---|---|---|---|---|---|")
    for key, datasets in sorted(series.items()):
        info = describe(datasets)
        thickness = "?" if info["thickness_mm"] is None else "{:g} mm".format(info["thickness_mm"])
        step = "?" if info["spacing_z_mm"] is None else "{:g} mm".format(info["spacing_z_mm"])
        print(
            "| {} | {} | {} | {}x{} | {} | {} |".format(
                names[key],
                info["description"],
                info["slices"],
                info["rows"],
                info["columns"],
                thickness,
                step,
            )
        )

    if arguments.dry_run:
        print(f"\ndry run: nothing written; would write to {out}")
        return 0

    print(f"\nwriting to {out} ({arguments.mode}) ...")
    written = 0
    for key, datasets in sorted(series.items()):
        folder = out / names[key]
        folder.mkdir(parents=True, exist_ok=True)
        for path, _dataset in datasets:
            place(path, folder / f"{path.stem}.dcm", arguments.mode)
            written += 1
    print(f"done: {written} files in {len(series)} folders")
    first = out / names[sorted(series)[0]]
    print('\nload one series with:\n    load_volume("{}", "dicom")'.format(first))
    for path in skipped[:5]:
        print(f"skipped (not DICOM): {path}")
    if len(skipped) > 5:
        print(f"... and {len(skipped) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
