"""
scripts/sam_med3d_explore.py

Exploratory runs of SAM-Med3D closer to how it was trained, after the first fish evaluation
(scripts/sam_med3d_fish.py, results/sam_med3d_fish, 2026-11-05) found it not usable on any fish.
They cannot change that verdict; they led to the defaults of 2026-11-06 (context "scan", shift "auto").

    python scripts/sam_med3d_explore.py check      # real weights: PADDED_NO_SHIFT gives the first evaluation's masks
    python scripts/sam_med3d_explore.py control    # human scan: E1 and E3 (median Dice >= 0.50 needed), and the
                                                   # shift check on the scan + 1024
    python scripts/sam_med3d_explore.py shifts     # the automatic shift per fish, written before any fish run
    python scripts/sam_med3d_explore.py fish       # E1-E3 on the fish, through the runner (as the CLI)
    python scripts/sam_med3d_explore.py summary    # the tables, from what was saved

Settings, on top of the first evaluation's primary setting (E0 = PADDED_NO_SHIFT):

    E1  context "scan"                                  the scan around the box fills the cube
    E2  E1 + intensity_shift "auto"                     air moved below 0, as in HU
    E3  E2 + spacing "trained"                          1.5 mm unless the box is longer than 192 mm

Same clicks and boxes as the first evaluation (configs/experiments/sam_med3d_fish_plan.json). Every run
is saved as it ends and is not run again. Repeats of the deepest click are left out (the first
evaluation found them identical on every fish). The saved E2 runs used an earlier trigger for "auto"
(median of Otsu's low class above 0); since 2026-11-06 it shifts only a box with under 1% of its voxels
below 0, which changes the tuna only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sam_med3d_control import ORGANS, box_of, box_slices, scan_and_labels, score, test_clicks  # noqa: E402
from scripts.sam_med3d_fish import CLICKS, PLAN  # noqa: E402
from src.core.box import Box  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.core.volume import Volume  # noqa: E402
from src.models.foundation.sam_med3d import segment  # noqa: E402
from src.models.foundation.sam_med3d.pipeline import PADDED_NO_SHIFT, auto_shift  # noqa: E402

OUT = Path("results/sam_med3d_explore")
E0 = Path("results/sam_med3d_fish")
SETTINGS = {
    "e1": {"context": "scan", "intensity_shift": 0.0},
    "e2": {"context": "scan", "intensity_shift": "auto"},
    "e3": {"context": "scan", "intensity_shift": "auto", "spacing": "trained"},
}
FISH_SETTINGS = {"oreo": ("e1", "e2", "e3"), "gadus": ("e1", "e2", "e3"), "tuna": ("e1", "e2")}  # tuna: E3 = E2 (368 mm box)
RECOVERY_ORGANS = (1, 3, 8)  # spleen, left kidney, aorta, 3 clicks each
RECOVERY_OFFSET = 1024.0


def _rows(path: Path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[row["key"]] = row
    return rows


def _add(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def check(only: str | None = None) -> int:
    """Click 0 of the first evaluation's primary setting, again with PADDED_NO_SHIFT: the mask must be identical."""
    from src.pipeline.runner import _load_reference, run_experiment

    OUT.mkdir(parents=True, exist_ok=True)
    for fish in [only] if only else ("oreo", "gadus", "tuna"):
        source = E0 / "configs" / f"{fish}__primary__click0.yaml"
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["experiment"].update({"id": f"check_defaults__{fish}", "repetitions": 1})
        config["algorithm"]["parameters"] = {**PADDED_NO_SHIFT, **config["algorithm"]["parameters"]}
        config["output"]["results_dir"] = OUT.as_posix() + "/"
        path = OUT / "configs" / f"check_defaults__{fish}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        new = run_experiment(path).mask != 0
        old = _load_reference(E0 / f"{fish}__primary__click0" / "mask.nii.gz")[0] != 0
        same = bool(np.array_equal(new, old))
        print(f"{fish}: PADDED_NO_SHIFT gives the first evaluation's mask: {same} ({int(new.sum())} vs {int(old.sum())} voxels)", flush=True)
        if not same:
            raise SystemExit("PADDED_NO_SHIFT no longer gives the first evaluation's mask: stopping")
    return 0


def control() -> int:
    """E1 and E3 on amos_0013 (15 organs x 3 clicks, as scripts/sam_med3d_control.py), then the shift's recovery
    check: the scan + 1024 under E2 against the HU scan under E1."""
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "control.jsonl"
    done = _rows(log)
    volume, labels = scan_and_labels()
    spacing = volume.spacing_mm
    offset = Volume((volume.data + RECOVERY_OFFSET).astype(np.float32), spacing, volume.units, volume.source_format, volume.source_hash)
    runs = [("e1", volume, organ) for organ in ORGANS] + [("e3", volume, organ) for organ in ORGANS]
    runs += [("e2_offset", offset, organ) for organ in RECOVERY_ORGANS]
    for setting, scan, organ in runs:
        organ_mask = labels == organ
        box = box_of(organ_mask, spacing)
        where = box_slices(volume, box)
        parameters = SETTINGS["e2" if setting == "e2_offset" else setting]
        for index, click in enumerate(test_clicks(organ_mask, spacing, 3)):
            key = f"{setting}/{organ}/{index}"
            if key in done:
                continue
            started = time.perf_counter()
            result = segment(scan, {"seed_zyx": list(click), "box_zyx": [[w.start, w.stop] for w in where], **parameters})
            prediction = np.asarray(result) != 0
            report = result.report
            row = {
                "key": key, "setting": setting, "organ": organ, "name": ORGANS[organ], "click_index": index,
                "click_zyx": list(click), "seconds": time.perf_counter() - started, "spacing_mm": report["spacing_mm"],
                "padding_share_of_cube": report["padding_share_of_cube"], "box_faces_reached": report["box_faces_reached"],
                "intensity_shift_rule": report["intensity_shift_rule"],
                "mask_sha": hashlib.sha256(np.packbits(prediction).tobytes()).hexdigest()[:16],
                **score(prediction, organ_mask, spacing),
            }
            _add(log, row)
            print(f"  {key} {ORGANS[organ]}: {row['dice']:.3f} in {row['seconds']:.1f} s", flush=True)
    return 0


def shifts() -> int:
    """R2's air, tissue and shift per fish, from the primary box's values; no model, no reference."""
    plan = json.loads(PLAN.read_text(encoding="utf-8"))["fish"]
    output = {}
    for name, fish in plan.items():
        volume = load_volume(fish["input_path"], fish["format"])
        where = Box.from_mapping(fish["settings"]["primary"]["box"]).slices(volume.shape, volume.spacing_mm)
        output[name] = auto_shift(volume.data[where])
        print(f"{name}: {output[name]}", flush=True)
        del volume
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shifts.json").write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


def fish(only: str | None = None) -> int:
    from src.pipeline.runner import run_experiment

    plan = json.loads(PLAN.read_text(encoding="utf-8"))["fish"]
    if not (OUT / "shifts.json").exists():
        raise SystemExit("run `shifts` first: R2's values are recorded before any fish run")
    log = OUT / "runs.jsonl"
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    jobs = [(name, setting, index, click) for name, settings in FISH_SETTINGS.items() if only in (None, name) for setting in settings
            for index, click in enumerate(plan[name]["settings"]["primary"]["clicks"])]
    for number, (name, setting, index, click) in enumerate(jobs, 1):
        fish_plan = plan[name]
        run_id = f"{name}__{setting}__click{index}"
        if (OUT / run_id / "metrics.json").exists():
            continue
        config = {
            "experiment": {"id": run_id, "repetitions": 1, "description": f"exploratory, closer to training, {setting}"},
            "data": {"input_path": fish_plan["input_path"], "format": fish_plan["format"],
                     "reference_mask_path": fish_plan["reference_mask_path"],
                     "box": fish_plan["settings"]["primary"]["box"], "seed_zyx": click},
            "algorithm": {"name": "sam_med3d", "parameters": SETTINGS[setting]},
            "mesh": {"smoothing_iterations": 5, "decimation_factor": 0.2, "output_format": "glb"},
            "output": {"results_dir": OUT.as_posix() + "/"},
        }
        path = OUT / "configs" / f"{run_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        started = time.perf_counter()
        status, error = "ok", None
        try:
            run_experiment(path)
        except Exception as caught:  # noqa: BLE001 - a crash is recorded, not a score
            status, error = "failed", f"{type(caught).__name__}: {caught}"
        seconds = time.perf_counter() - started
        _add(log, {"id": run_id, "status": status, "seconds": seconds, "error": error, "script_sha256": script_sha})
        print(f"[{number}/{len(jobs)}] {run_id}: {status} in {seconds:.0f} s" + (f" ({error[:150]})" if error else ""), flush=True)
    return 0


def _stats(values, hd95) -> dict:
    values = np.asarray(values, float)
    return {
        "runs": len(values), "dice": [float(v) for v in values], "dice_median": float(np.median(values)),
        "dice_iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
        "dice_min": float(values.min()), "dice_max": float(values.max()),
        "clicks_at_least_0.50": int((values >= 0.5).sum()),
        "hd95_median_mm": float(np.median([h if h is not None else np.inf for h in hd95])),
        "empty": int(sum(h is None for h in hd95)),
    }


def summary() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))["fish"]
    report: dict = {"control": {}, "recovery": {}, "fish": {}}
    rows = list(_rows(OUT / "control.jsonl").values())
    for setting in ("e1", "e3"):
        per_organ = {}
        for row in rows:
            if row["setting"] == setting:
                per_organ.setdefault(row["organ"], []).append(row["dice"])
        if per_organ:
            medians = [float(np.median(v)) for v in per_organ.values()]
            report["control"][setting] = {
                "organs": len(per_organ), "runs": sum(len(v) for v in per_organ.values()),
                "median_of_organs": float(np.median(medians)), "mean": float(np.mean([d for v in per_organ.values() for d in v])),
                "per_organ_median": {ORGANS[o]: float(np.median(v)) for o, v in per_organ.items()},
                "median_padding_share": float(np.median([r["padding_share_of_cube"] for r in rows if r["setting"] == setting])),
                "median_spacing_mm": float(np.median([r["spacing_mm"] for r in rows if r["setting"] == setting])),
                "g3_pass": float(np.median(medians)) >= 0.50,
            }
    offset = {r["key"].split("/", 1)[1]: r for r in rows if r["setting"] == "e2_offset"}
    plain = {r["key"].split("/", 1)[1]: r for r in rows if r["setting"] == "e1"}
    pairs = [(plain[k]["dice"], offset[k]["dice"]) for k in offset if k in plain]
    if pairs:
        hu, shifted = (np.array(v) for v in zip(*pairs))
        report["recovery"] = {
            "runs": len(pairs), "hu_e1_median": float(np.median(hu)), "offset_e2_median": float(np.median(shifted)),
            "difference": float(np.median(shifted) - np.median(hu)),
            "pass": abs(float(np.median(shifted) - np.median(hu))) <= 0.03,
            "shifts": sorted({round(offset[k]["intensity_shift_rule"]["shift"], 1) for k in offset}),
        }
    e0 = json.loads((E0 / "summary.json").read_text(encoding="utf-8"))["fish"]
    for name, settings in FISH_SETTINGS.items():
        entry = {"e0": {k: e0[name]["primary"][k] for k in ("dice", "dice_median", "dice_iqr", "dice_min", "dice_max", "clicks_at_least_0.50", "hd95_median_mm", "empty")}}
        bar = plan[name]["hd95_bar_mm"]
        for setting in settings:
            metrics = [(OUT / f"{name}__{setting}__click{i}" / "metrics.json") for i in range(CLICKS)]
            found = [json.loads(p.read_text(encoding="utf-8")) for p in metrics if p.exists()]
            if not found:
                continue
            stats = _stats([m["dice"] for m in found], [m["hd95_mm"] for m in found])
            method = found[0]["method_report"]
            stats["cube_click0"] = {
                "spacing_mm": method["spacing_mm"], "side_mm": method["side_mm"], "padding_share_of_cube": method["padding_share_of_cube"],
                "air_is_above_0": method["normalisation"]["air_is_above_0"],
                "share_of_scan_voxels_above_0": method["normalisation"]["share_of_scan_voxels_above_0"],
                "intensity_shift": method["normalisation"]["intensity_shift"],
            }
            stats["box_faces_reached"] = sum(bool(m["method_report"]["box_faces_reached"]) for m in found)
            stats["meets_criteria"] = (stats["runs"] == CLICKS and stats["dice_median"] >= 0.70 and stats["clicks_at_least_0.50"] >= 7
                                       and stats["hd95_median_mm"] <= bar)
            stats["runtime_s_median"] = float(np.median([m.get("runtime_s", np.nan) for m in found]))
            entry[setting] = stats
        entry["hd95_bar_mm"] = bar
        entry["best_analytical"] = e0[name]["verdict"]["best_analytical"]
        report["fish"][name] = entry
    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str)[:6000])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("check", "control", "shifts", "fish", "summary"))
    parser.add_argument("--fish", choices=tuple(FISH_SETTINGS), help="one fish only (one process each keeps memory low)")
    arguments = parser.parse_args()
    if arguments.step in ("check", "fish"):
        return {"check": check, "fish": fish}[arguments.step](arguments.fish)
    return {"control": control, "shifts": shifts, "summary": summary}[arguments.step]()


if __name__ == "__main__":
    raise SystemExit(main())
