"""
scripts/sam_med3d_control.py

The trust checks of the SAM-Med3D wrapper on upstream's human CT (amos_0013), one scan. No fish.

    python scripts/sam_med3d_control.py run           # control, negative controls, overlays; resumes
    python scripts/sam_med3d_control.py summary       # tables from what was saved
    python scripts/sam_med3d_control.py determinism   # the same run in fresh processes

Procedures, each organ, each click from `test_clicks`:

    box_method       the method: the scan cropped to the organ's box grown 25%, one click, the cube
                     scaled to the box
    mask_centred     cube centred on the organ at 1.5 mm, as upstream's validation does. It uses the
                     mask to place the cube, so it is a reference point for this control, not a track
    box_<variant>    the method with an imperfect box (`imperfect_boxes`): shifted by 10% or 20% of its
                     size along the diagonal, 10% tighter than the organ, or grown 50%. A click outside
                     such a box moves to the deepest voxel of the organ inside it, as a user would click

Results go to results/sam_med3d_checks/box_method/ as they are made. `test_clicks` implements the
protocol's click rule, and is meant to be used for the fish as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.measure_contrast import _box_around_mask  # noqa: E402
from src.core.box import Box, crop_volume, restore_mask  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.core.volume import Volume  # noqa: E402
from src.models.foundation.sam_med3d import assets, cube as geometry, network, segment  # noqa: E402
from src.models.foundation.sam_med3d.pipeline import prepare  # noqa: E402
from src.pipeline.metrics import compute_metrics  # noqa: E402
from src.pipeline.runner import share_in_view  # noqa: E402

try:
    from src.pipeline.alignment import check_alignment  # noqa: E402
except ImportError:  # not on every branch
    check_alignment = None

ORGANS = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "aorta", 9: "postcava", 10: "pancreas", 11: "right_adrenal", 12: "left_adrenal", 13: "duodenum",
    14: "bladder", 15: "prostate_uterus",
}
CLICKS_PER_ORGAN = 3
GROW = 0.25
OUT = Path("results/sam_med3d_checks/box_method")
PREPARE = {"clamp": None, "intensity_shift": 0.0, "padding": 0.0}
PROCEDURES = ("box_method", "mask_centred")
IMPERFECT = ("shift_up_10", "shift_down_10", "shift_up_20", "shift_down_20", "tight_10", "grown_50")


def test_clicks(mask: np.ndarray, spacing, count: int, seed: int = 42) -> list[tuple[int, int, int]]:
    """Clicks as a user would place them, inside the target: the deepest voxel of its largest part,
    then one voxel in each of `count - 1` equal slabs along its longest axis, all at least half
    as deep as the deepest. The mask only chooses where to click."""
    labels, parts = ndi.label(mask)
    largest = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:])) if parts > 1 else np.asarray(mask, bool)
    depth = ndi.distance_transform_edt(largest, sampling=spacing)
    deepest = tuple(int(v) for v in np.unravel_index(int(depth.argmax()), depth.shape))
    clicks = [deepest]
    interior = np.argwhere(depth >= 0.5 * depth.max())
    extent = (np.argwhere(largest).max(0) - np.argwhere(largest).min(0) + 1) * np.asarray(spacing)
    order = interior[np.argsort(interior[:, int(np.argmax(extent))], kind="stable")]
    rng = np.random.default_rng(seed)
    for slab in np.array_split(order, count - 1) if count > 1 else []:
        clicks.append(tuple(int(v) for v in slab[rng.integers(len(slab))]))
    return clicks


def scan_and_labels() -> tuple[Volume, np.ndarray]:
    source = assets.locate().source / "test_data" / "amos_val_toy_data"
    volume = load_volume(source / "imagesVa" / "amos_0013.nii.gz")
    labels = np.asarray(nib.load(str(source / "labelsVa" / "amos_0013.nii.gz")).dataobj).transpose(2, 1, 0)
    return volume, labels


def run_mask_centred(volume: Volume, organ_mask: np.ndarray, click) -> tuple[np.ndarray, geometry.Cube, tuple]:
    """Upstream's placement: 128 voxels of 1.5 mm centred on the bounding box of the mask. An organ longer
    than the cube can leave the click outside it; upstream's click always lies in the part of the label
    inside its cube, so here the click then moves to the deepest voxel of that part. Returns the click used."""
    found = np.argwhere(organ_mask)
    low, high = found.min(0) * np.asarray(volume.spacing_mm), (found.max(0) + 1) * np.asarray(volume.spacing_mm)
    centre = (low + high) / 2
    cube = geometry.Cube(tuple(float(c - geometry.SIZE * 1.5 / 2) for c in centre), 1.5)
    values, inside = geometry.sample(volume.data, volume.spacing_mm, cube)
    prepared, _ = prepare(values, inside, PREPARE)
    net = network.get_network()
    embedding = net.embed(geometry.to_network_axes(prepared))
    in_cube = geometry.paste_back(np.ones((geometry.SIZE,) * 3, bool), cube, volume.shape, volume.spacing_mm) != 0
    if not in_cube[tuple(click)]:
        click = test_clicks(organ_mask & in_cube, volume.spacing_mm, 1)[0]
    click_xyz = geometry.network_click(geometry.voxel_centre_mm(click, volume.spacing_mm), cube)
    probability = geometry.to_network_axes(net.predict(embedding, click_xyz))
    return geometry.paste_back(probability > 0.5, cube, volume.shape, volume.spacing_mm), cube, tuple(click)


class Region:
    """A box in mm, drawn on the overlays like a cube."""

    def __init__(self, low, high) -> None:
        self.low, self.high = tuple(low), tuple(high)

    def bounds_mm(self):
        return self.low, self.high


def box_slices(volume: Volume, box: dict) -> tuple[slice, slice, slice]:
    return Box.from_mapping(box).slices(volume.shape, volume.spacing_mm)


def run_box(volume: Volume, box: dict, click, **parameters) -> tuple[np.ndarray, dict, Region]:
    """The method as the runner runs it: crop to the box, click in the box's voxels, paste the mask back."""
    cropped, where = crop_volume(volume, Box.from_mapping(box))
    local = [int(c) - w.start for c, w in zip(click, where)]
    result = segment(cropped, {"seed_zyx": local, **parameters})
    mask = restore_mask(np.asarray(result), volume.shape, where) != 0
    low = [w.start * s for w, s in zip(where, volume.spacing_mm)]
    high = [w.stop * s for w, s in zip(where, volume.spacing_mm)]
    report = {**result.report, "view_mm": {"min_zyx": low, "max_zyx": high}}
    return mask, report, Region(low, high)


def box_of(mask: np.ndarray, spacing) -> dict:
    box, _ = _box_around_mask(mask, spacing, GROW)
    return box.to_mapping()


def imperfect_boxes(mask: np.ndarray, spacing, shape, box: dict) -> dict[str, dict]:
    """Boxes a user might draw instead of `box` (the bounding box grown 25%), clipped to the scan."""
    spacing, limit = np.asarray(spacing), np.asarray(shape) * np.asarray(spacing)
    found = np.argwhere(mask)
    low, high = found.min(0) * spacing, (found.max(0) + 1) * spacing
    extent = high - low
    standard_low, standard_high = np.asarray(box["min_mm"]), np.asarray(box["max_mm"])
    size = standard_high - standard_low

    def make(a, b):
        return {"min_mm": [float(v) for v in np.clip(a, 0, limit)], "max_mm": [float(v) for v in np.clip(b, 0, limit)]}

    boxes = {}
    for share in (10, 20):
        for sign, word in ((1, "up"), (-1, "down")):  # every axis at once, towards higher or lower voxel indices
            boxes[f"shift_{word}_{share}"] = make(standard_low + sign * share / 100 * size, standard_high + sign * share / 100 * size)
    boxes["tight_10"] = make(low + 0.05 * extent, high - 0.05 * extent)
    boxes["grown_50"] = make(low - 0.25 * extent, high + 0.25 * extent)
    return boxes


def score(prediction: np.ndarray, reference: np.ndarray, spacing) -> dict:
    """The metrics, on a crop holding both masks and one voxel more: every surface voxel is in it, so the
    distances are those of the whole scan, without distance transforms over all of it."""
    either = np.asarray(prediction, bool) | np.asarray(reference, bool)
    if either.any():
        found = np.argwhere(either)
        window = tuple(slice(max(int(a) - 1, 0), min(int(b) + 2, n)) for a, b, n in zip(found.min(0), found.max(0), either.shape))
        prediction, reference = np.asarray(prediction)[window], np.asarray(reference)[window]
    metrics = compute_metrics(prediction, reference, spacing)
    return {key: metrics[key] for key in ("dice", "iou", "vs", "hd95_mm", "assd_mm")}


def overlay(volume, prediction, reference, click, title, cube, folder: Path, stem: str) -> None:
    """Prediction (red) and reference (green) on the three planes through the click."""
    folder.mkdir(parents=True, exist_ok=True)
    spacing = volume.spacing_mm
    planes = {
        "axial": (0, click[0], (2, 1)),
        "coronal": (1, click[1], (2, 0)),
        "sagittal": (2, click[2], (1, 0)),
    }
    low, high = cube.bounds_mm() if cube is not None else (None, None)
    for name, (axis, index, (horizontal, vertical)) in planes.items():
        path = folder / f"{stem}_{name}.png"
        if path.exists():
            continue
        take = lambda array: np.take(array, index, axis=axis)  # noqa: E731, B023
        extent = (0, volume.shape[horizontal] * spacing[horizontal], volume.shape[vertical] * spacing[vertical], 0)
        figure, axes = plt.subplots(figsize=(4.2, 4.2), dpi=100)
        axes.imshow(take(volume.data), cmap="gray", vmin=-200, vmax=300, extent=extent, interpolation="nearest")
        for mask, colour in ((reference, "#00d060"), (prediction, "#ff3030")):
            plane = take(mask)
            if plane.any():
                centres_x = (np.arange(plane.shape[1]) + 0.5) * spacing[horizontal]
                centres_y = (np.arange(plane.shape[0]) + 0.5) * spacing[vertical]
                axes.contour(centres_x, centres_y, plane.astype(float), levels=[0.5], colors=colour, linewidths=1.2)
        axes.plot((click[horizontal] + 0.5) * spacing[horizontal], (click[vertical] + 0.5) * spacing[vertical], "+", color="yellow", markersize=10)
        if cube is not None:
            rectangle = plt.Rectangle(
                (low[horizontal], low[vertical]), high[horizontal] - low[horizontal], high[vertical] - low[vertical],
                fill=False, edgecolor="#40c0ff", linestyle="--", linewidth=0.8,
            )
            axes.add_patch(rectangle)
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
        axes.set_title(f"{title}\n{name}", fontsize=8)
        axes.tick_params(labelsize=6)
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)


def contact_sheets(out: Path) -> None:
    """One picture per group of organs, each with its three planes, to look through a procedure quickly."""
    for procedure in (*PROCEDURES, "negative_outside_near", "negative_outside_air", *(f"box_{v}" for v in IMPERFECT)):
        folder = out / "overlays" / procedure
        if not folder.is_dir():
            continue
        stems = sorted({path.name.rsplit("_", 1)[0] for path in folder.glob("*.png")})
        for part in range(0, len(stems), 5):
            group = stems[part : part + 5]
            sheet = Image.new("RGB", (3 * 420, len(group) * 420), "white")
            for row, stem in enumerate(group):
                for column, plane in enumerate(("axial", "coronal", "sagittal")):
                    sheet.paste(Image.open(folder / f"{stem}_{plane}.png"), (column * 420, row * 420))
            sheet.save(out / "overlays" / f"sheet_{procedure}_{part // 5 + 1}.png")


class Results:
    """One JSON line per run, written at once, so that a stopped run can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.done[row["key"]] = row

    def add(self, row: dict) -> None:
        self.done[row["key"]] = row
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")


def run(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    results = Results(out / "runs.jsonl")
    volume, labels = scan_and_labels()
    spacing = volume.spacing_mm
    canonical = Volume(np.ascontiguousarray(volume.data[:, :, ::-1]), spacing, volume.units, volume.source_format, volume.source_hash)
    any_organ = labels != 0
    body = ndi.binary_fill_holes(volume.data > -300)

    def predict(procedure, organ_mask, click, box):
        if procedure == "box_method":
            return run_box(volume, box, click)
        mask, cube, used = run_mask_centred(volume, organ_mask, click)
        report = {"view_mm": {"min_zyx": list(cube.bounds_mm()[0]), "max_zyx": list(cube.bounds_mm()[1])}, "spacing_mm": 1.5, "click_used": list(used)}
        return mask, report, cube

    for organ, name in ORGANS.items():
        organ_mask = labels == organ
        box = box_of(organ_mask, spacing)
        clicks = test_clicks(organ_mask, spacing, CLICKS_PER_ORGAN)
        fill = np.zeros(labels.shape, bool)
        fill[box_slices(volume, box)] = True  # the same voxels the method is given
        print(f"organ {organ} {name}: clicks {clicks}", flush=True)

        for procedure in PROCEDURES:
            for index, click in enumerate(clicks):
                key = f"{procedure}/{organ}/{index}"
                if key in results.done:
                    continue
                started = time.perf_counter()
                prediction, report, cube = predict(procedure, organ_mask, click, box)
                seconds = time.perf_counter() - started
                row = {
                    "key": key, "procedure": procedure, "organ": organ, "name": name, "click_index": index, "click_zyx": list(click),
                    "voxels": int(prediction.sum()), "spacing_mm": report["spacing_mm"], "seconds": seconds,
                    "reference_share_in_view": share_in_view(organ_mask, spacing, report["view_mm"]),
                    "padding_share_of_cube": report.get("padding_share_of_cube"),
                    "click_used_zyx": report.get("click_used"),
                    "box_faces_reached": report.get("box_faces_reached"),
                    "mask_sha": hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest()[:16],
                    **score(prediction, organ_mask, spacing),
                }
                if index == 0:
                    title = f"{name} | {procedure} | deepest click | Dice {row['dice']:.3f}"
                    shown = tuple(report.get("click_used") or click)
                    overlay(volume, prediction, organ_mask, shown, title, cube, out / "overlays" / procedure, f"{organ:02d}_{name}")
                    if procedure == "box_method":
                        row["mirrored"] = {
                            axis: score(prediction, np.flip(organ_mask, position), spacing)["dice"]
                            for position, axis in enumerate("zyx")
                        }
                        if check_alignment is not None:
                            row["alignment_as_given"] = check_alignment(volume.data, organ_mask, spacing)["status"]
                            row["alignment_z_mirrored"] = check_alignment(volume.data, np.flip(organ_mask, 0), spacing)["status"]
                        row["empty"] = score(np.zeros(labels.shape, bool), organ_mask, spacing)
                        row["fill_box"] = score(fill, organ_mask, spacing)
                results.add(row)
                print(f"  {key}: {row['dice']:.3f} in {seconds:.1f} s", flush=True)

        # upstream's canonical order: x flipped. Same placement and click; only the array is mirrored.
        key = f"mask_centred_canonical/{organ}/0"
        if key not in results.done:
            mirrored = np.ascontiguousarray(organ_mask[:, :, ::-1])
            click = clicks[0]
            prediction, *_ = run_mask_centred(canonical, mirrored, (click[0], click[1], organ_mask.shape[2] - 1 - click[2]))
            results.add({"key": key, "procedure": "mask_centred_canonical", "organ": organ, "name": name, "click_index": 0, **score(prediction, mirrored, spacing)})

        # clicks outside the target but inside the organ's box, so the cube is the same: other tissue, and air
        rng = np.random.default_rng(42 + organ)
        distance = ndi.distance_transform_edt(~organ_mask, sampling=spacing)
        in_box = np.zeros(labels.shape, bool)
        in_box[box_slices(volume, box)] = True
        candidates = {
            "negative_outside_near": in_box & (distance >= 5) & body & ~any_organ & (volume.data > -300),
            "negative_outside_air": in_box & (distance >= 5) & (volume.data < -500),
        }
        for procedure, where in candidates.items():
            key = f"{procedure}/{organ}/0"
            if key in results.done:
                continue
            found = np.argwhere(where)
            if not len(found):
                results.add({"key": key, "procedure": procedure, "organ": organ, "name": name, "click_index": 0, "dice": None, "note": "no such voxel"})
                continue
            click = tuple(int(v) for v in found[rng.integers(len(found))])
            prediction, report, cube = predict("box_method", organ_mask, click, box)
            row = {
                "key": key, "procedure": procedure, "organ": organ, "name": name, "click_index": 0, "click_zyx": list(click),
                "click_hu": float(volume.data[click]), "voxels": int(prediction.sum()), **score(prediction, organ_mask, spacing),
            }
            overlay(volume, prediction, organ_mask, click, f"{name} | {procedure} | Dice {row['dice']:.3f}", cube, out / "overlays" / procedure, f"{organ:02d}_{name}")
            results.add(row)
            print(f"  {key}: {row['dice']:.3f}", flush=True)
        # the method with imperfect boxes
        for variant, variant_box in imperfect_boxes(organ_mask, spacing, labels.shape, box).items():
            where = box_slices(volume, variant_box)
            inside = organ_mask[where]
            for index, click in enumerate(clicks):
                key = f"box_{variant}/{organ}/{index}"
                if key in results.done:
                    continue
                if all(w.start <= c < w.stop for c, w in zip(click, where)):
                    used = tuple(click)
                else:  # on the crop, so the distance transform does not cover the whole scan
                    used = tuple(int(v) + w.start for v, w in zip(test_clicks(inside, spacing, 1)[0], where))
                started = time.perf_counter()
                prediction, report, region = run_box(volume, variant_box, used)
                row = {
                    "key": key, "procedure": f"box_{variant}", "organ": organ, "name": name, "click_index": index,
                    "click_zyx": list(used), "click_moved": used != tuple(click), "box": variant_box,
                    "organ_share_in_box": float(inside.sum() / organ_mask.sum()), "spacing_mm": report["spacing_mm"],
                    "box_faces_reached": report["box_faces_reached"], "seconds": time.perf_counter() - started,
                    **score(prediction, organ_mask, spacing),
                }
                if index == 0:
                    overlay(volume, prediction, organ_mask, used, f"{name} | box {variant} | Dice {row['dice']:.3f}", region, out / "overlays" / f"box_{variant}", f"{organ:02d}_{name}")
                results.add(row)
                print(f"  {key}: {row['dice']:.3f}", flush=True)
    contact_sheets(out)
    return summary(out)


def summary(out: Path) -> int:
    rows = list(Results(out / "runs.jsonl").done.values())
    organs = list(ORGANS)
    report: dict = {"note": "one scan (amos_0013); says nothing about fish"}

    def median_of_organs(procedure: str, key: str = "dice") -> float | None:
        values = [
            np.median([row[key] for row in rows if row["procedure"] == procedure and row["organ"] == organ and row.get(key) is not None] or [np.nan])
            for organ in organs
        ]
        values = [value for value in values if not np.isnan(value)]
        return float(np.median(values)) if values else None

    print("procedure                 median Dice (of organ medians)   median HD95 mm   all clicks: min / median / max")
    for procedure in (*PROCEDURES, "mask_centred_canonical", "negative_outside_near", "negative_outside_air"):
        dice = [row["dice"] for row in rows if row["procedure"] == procedure and row.get("dice") is not None]
        if not dice:
            continue
        report[procedure] = {"median_of_organs": median_of_organs(procedure), "median_hd95": median_of_organs(procedure, "hd95_mm"), "min": min(dice), "median": float(np.median(dice)), "max": max(dice), "runs": len(dice)}
        print(f"{procedure:26s}{report[procedure]['median_of_organs']:8.3f}{'':24s}{report[procedure]['median_hd95'] or float('nan'):8.1f}{'':8s}{min(dice):.3f} / {np.median(dice):.3f} / {max(dice):.3f}  ({len(dice)} runs)")

    print("\norgan            box method, 3 clicks  mask-centred, 3 clicks   mm spacing")
    per_organ = {}
    for organ, name in ORGANS.items():
        cells = []
        for procedure in PROCEDURES:
            values = [row["dice"] for row in sorted(rows, key=lambda r: r["click_index"]) if row["procedure"] == procedure and row["organ"] == organ]
            cells.append(values)
        spacing = next((row["spacing_mm"] for row in rows if row["procedure"] == "box_method" and row["organ"] == organ), float("nan"))
        per_organ[name] = {procedure: cell for procedure, cell in zip(PROCEDURES, cells)}
        print(f"{name:16s} " + "   ".join(" ".join(f"{value:.2f}" for value in cell).ljust(19) for cell in cells) + f"   {spacing:6.2f}")
    report["per_organ"] = per_organ

    deepest = {row["organ"]: row for row in rows if row["procedure"] == "box_method" and row["click_index"] == 0 and "mirrored" in row}
    if deepest:
        negative = {
            "correct_box_method_deepest": [row["dice"] for row in deepest.values()],
            "mirrored_z": [row["mirrored"]["z"] for row in deepest.values()],
            "mirrored_y": [row["mirrored"]["y"] for row in deepest.values()],
            "mirrored_x": [row["mirrored"]["x"] for row in deepest.values()],
            "fill_box": [row["fill_box"]["dice"] for row in deepest.values()],
            "empty": [row["empty"]["dice"] for row in deepest.values()],
            "outside_near": [row["dice"] for row in rows if row["procedure"] == "negative_outside_near" and row.get("dice") is not None],
            "outside_air": [row["dice"] for row in rows if row["procedure"] == "negative_outside_air" and row.get("dice") is not None],
        }
        print("\nnegative controls, Dice against the organ    median   max   (15 organs)")
        for name, values in negative.items():
            print(f"  {name:32s} {np.median(values):6.3f} {max(values):6.3f}")
        report["negative_controls"] = {name: {"median": float(np.median(values)), "max": max(values), "values": values} for name, values in negative.items()}
        statuses = [row.get("alignment_as_given") for row in deepest.values()], [row.get("alignment_z_mirrored") for row in deepest.values()]
        print("  alignment check, mask as given:", {s: statuses[0].count(s) for s in set(statuses[0])}, "| z mirrored:", {s: statuses[1].count(s) for s in set(statuses[1])})
        report["alignment_as_given"], report["alignment_z_mirrored"] = statuses
    standard = [row for row in rows if row["procedure"] == "box_method"]
    if any(row["procedure"].startswith("box_") and row["procedure"] != "box_method" for row in rows):
        print("\nimperfect boxes   median of organ medians   mean    min / median / max    clicks moved   mask at box face   organ in box")
        imperfect = {}
        for procedure in ("box_method", *(f"box_{v}" for v in IMPERFECT)):
            chosen = [row for row in rows if row["procedure"] == procedure]
            if not chosen:
                continue
            dice = [row["dice"] for row in chosen]
            entry = {
                "median_of_organs": median_of_organs(procedure), "mean": float(np.mean(dice)), "min": min(dice), "median": float(np.median(dice)), "max": max(dice),
                "clicks_moved": sum(bool(row.get("click_moved")) for row in chosen), "at_box_face": sum(bool(row.get("box_faces_reached")) for row in chosen),
                "median_organ_share_in_box": float(np.median([row.get("organ_share_in_box", 1.0) for row in chosen])), "runs": len(chosen),
            }
            imperfect[procedure] = entry
            print(f"{procedure:20s}{entry['median_of_organs']:8.3f}{'':14s}{entry['mean']:6.3f}   {entry['min']:.2f} / {entry['median']:.2f} / {entry['max']:.2f}"
                  f"   {entry['clicks_moved']:5d}{'':10s}{entry['at_box_face']:5d}{'':13s}{entry['median_organ_share_in_box']:.2f}")
        report["imperfect_boxes"] = imperfect
    del standard
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def determinism_once(threads: int) -> int:
    volume, labels = scan_and_labels()
    organ_mask = labels == 6
    click = test_clicks(organ_mask, volume.spacing_mm, 1)[0]
    box = box_of(organ_mask, volume.spacing_mm)
    output = {}
    mask, _, _ = run_box(volume, box, click, threads=threads)
    output["box_method"] = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    output["box_method_voxels"] = int(mask.sum())
    print(json.dumps(output))
    return 0


def determinism() -> int:
    """Fresh processes, so nothing is reused between runs; then one with another thread count."""
    runs = []
    for threads in (4, 4, 4, 1):
        found = subprocess.run([sys.executable, __file__, "determinism-once", "--threads", str(threads)], capture_output=True, text=True, check=True)
        row = json.loads(found.stdout.strip().splitlines()[-1])
        runs.append({"threads": threads, **row})
        print(threads, "threads:", row["box_method"][:16], flush=True)
    print("identical across fresh processes, same threads:", len({run["box_method"] for run in runs[:3]}) == 1)
    print("identical with 1 thread:", runs[3]["box_method"] == runs[0]["box_method"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "determinism.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("run", "summary", "determinism", "determinism-once"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.step == "run":
        return run(arguments.out)
    if arguments.step == "summary":
        return summary(arguments.out)
    if arguments.step == "determinism":
        return determinism()
    return determinism_once(arguments.threads)


if __name__ == "__main__":
    raise SystemExit(main())
