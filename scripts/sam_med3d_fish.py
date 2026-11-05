"""
scripts/sam_med3d_fish.py

The SAM-Med3D fish evaluation of the protocol (md-files/sam_med3d_protocol.md, outside the repo).

    python scripts/sam_med3d_fish.py plan       # clicks and boxes per fish, from the reference masks; no model
    python scripts/sam_med3d_fish.py run        # every run of the plan, in the protocol's order; resumes
    python scripts/sam_med3d_fish.py summary    # the tables and verdicts, from what was saved

`plan` writes configs/experiments/sam_med3d_fish_plan.json. The protocol requires that file to be
committed before any model run, so that clicks and boxes cannot follow the results.

`run` writes one YAML config per run and runs it through `run_experiment`, the same code as the CLI,
into results/sam_med3d_fish/<run id>/. A run with a metrics.json is not run again. SAM-Med3D runs in
this process, so the weights load once; the analytical baselines run in a child process each, so that
one taking longer than 30 minutes can be stopped and reported as "not run".

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
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

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


OUT = Path("results/sam_med3d_fish")
BASELINE_TIMEOUT_S = 30 * 60
WHOLE_BOX = {"domain_size_mm": None, "compute_spacing_mm": None}  # as scripts/compare_methods.py
BASELINES = {
    "box_region": {},
    "chan_vese": {"init_from": "contrast"},
    "geodesic_active_contour": {**WHOLE_BOX, "init_from": "contrast"},
    "growcut": {**WHOLE_BOX, "init_from": "contrast"},
}


def jobs(plan_data: dict) -> list[dict]:
    """Every run, in the protocol's order: per fish, the primary setting, then the sensitivity settings,
    then the extra rows, then the baselines."""
    result = []
    for fish_name, fish in plan_data["fish"].items():
        settings = fish["settings"]

        def add(setting, algorithm, parameters, setting_key="primary", input_path=None, repetitions_on_deepest=1):
            for index, click in enumerate(settings[setting_key]["clicks"]):
                result.append({
                    "id": f"{fish_name}__{setting}__click{index}",
                    "fish": fish_name, "setting": setting, "click_index": index, "algorithm": algorithm,
                    "parameters": parameters, "input_path": input_path or fish["input_path"], "format": fish["format"],
                    "reference_mask_path": fish["reference_mask_path"], "box": settings[setting_key]["box"],
                    "seed_zyx": click, "repetitions": repetitions_on_deepest if index == 0 else 1,
                })

        add("primary", "sam_med3d", {}, repetitions_on_deepest=3)
        if fish_name in ("oreo", "gadus"):
            add("s1_pad_minimum", "sam_med3d", {"padding": "minimum"})
        if fish_name == "tuna":
            add("s2_clamp", "sam_med3d", {"clamp": [-1000.0, 1000.0]})
        add("s3_whole_z", "sam_med3d", {}, setting_key="s3")
        add("s4_up", "sam_med3d", {}, setting_key="s4_up")
        add("s4_down", "sam_med3d", {}, setting_key="s4_down")
        add("s5_grown_50", "sam_med3d", {}, setting_key="s5")
        if "sound_speed" in fish.get("extra_rows", {}):
            add("extra_sound_speed", "sam_med3d", {}, input_path=fish["extra_rows"]["sound_speed"])
        for method, parameters in BASELINES.items():
            add(f"baseline_{method}", method, parameters)
        if fish_name == "tuna":
            add("baseline_box_region_bright", "box_region", {"bladder_is_dark": False})
    return result


def config_for(job: dict) -> dict:
    return {
        "experiment": {"id": job["id"], "repetitions": job["repetitions"], "description": f"protocol run, {job['setting']}"},
        "data": {
            "input_path": job["input_path"], "format": job["format"], "reference_mask_path": job["reference_mask_path"],
            "box": job["box"], "seed_zyx": job["seed_zyx"],
        },
        "algorithm": {"name": job["algorithm"], "parameters": job["parameters"]},
        "mesh": {"smoothing_iterations": 5, "decimation_factor": 0.2, "output_format": "glb"},
        "output": {"results_dir": OUT.as_posix() + "/"},
    }


def register_baselines() -> None:
    """The analytical methods through the runner, called as scripts/compare_methods.py calls them."""
    from src.models import box_region
    from src.models.segment import segment_volume
    from src.pipeline.runner import register_algorithm

    register_algorithm("box_region", box_region.segment)
    for method in ("chan_vese", "geodesic_active_contour", "growcut"):
        def run_method(volume, parameters, method=method):
            parameters = dict(parameters)
            seed = parameters.pop("seed_zyx")
            return segment_volume(volume, {"name": method, "parameters": parameters}, list(seed)).mask
        register_algorithm(method, run_method)


def run_one(config_path: str) -> int:
    """One baseline run, in a child process."""
    from src.pipeline.runner import run_experiment

    register_baselines()
    run_experiment(config_path)
    return 0


def run() -> int:
    from src.pipeline.runner import run_experiment

    plan_data = json.loads(PLAN.read_text(encoding="utf-8"))
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    log = OUT / "runs.jsonl"
    all_jobs = jobs(plan_data)
    print(f"{len(all_jobs)} runs; script {script_sha[:12]}, plan {plan_sha[:12]}", flush=True)
    for number, job in enumerate(all_jobs, 1):
        if (OUT / job["id"] / "metrics.json").exists():
            continue
        path = OUT / "configs" / f"{job['id']}.yaml"
        path.write_text(yaml.safe_dump(config_for(job), sort_keys=False), encoding="utf-8")
        started = time.perf_counter()
        status, error = "ok", None
        try:
            if job["algorithm"] == "sam_med3d":
                run_experiment(path)
            else:
                completed = subprocess.run([sys.executable, __file__, "run-one", str(path)], timeout=BASELINE_TIMEOUT_S, capture_output=True, text=True)
                if completed.returncode != 0:
                    status, error = "failed", (completed.stderr or completed.stdout)[-2000:]
        except subprocess.TimeoutExpired:
            status, error = "not run", f"stopped after {BASELINE_TIMEOUT_S} s"
        except Exception as caught:  # noqa: BLE001 - a crash is recorded, not a score
            status, error = "failed", f"{type(caught).__name__}: {caught}"
        seconds = time.perf_counter() - started
        entry = {"id": job["id"], "status": status, "seconds": seconds, "error": error, "script_sha256": script_sha, "plan_sha256": plan_sha}
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"[{number}/{len(all_jobs)}] {job['id']}: {status} in {seconds:.0f} s" + (f" ({error.splitlines()[-1][:150]})" if error else ""), flush=True)
    return 0


def summary() -> int:
    """Per fish and setting: Dice over the 9 clicks, the protocol's three criteria and the verdict (primary
    setting only; the others cannot change it), the baselines, and the tuna against its second mask."""
    from src.pipeline.metrics import compute_metrics
    from src.pipeline.runner import _load_reference, match_grid

    plan_data = json.loads(PLAN.read_text(encoding="utf-8"))
    status = {}
    if (OUT / "runs.jsonl").exists():
        for line in (OUT / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            status[entry["id"]] = entry
    report: dict = {"fish": {}}
    for fish_name, fish in plan_data["fish"].items():
        groups: dict[str, list] = {}
        for job in jobs({"fish": {fish_name: fish}}):
            path = OUT / job["id"] / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            groups.setdefault(job["setting"], []).append((job, metrics))
        fish_report = {}
        print(f"\n{fish_name}: HD95 bar {fish['hd95_bar_mm']:.2f} mm, fill-box Dice {fish['fill_box_dice']:.3f}")
        print(f"  {'setting':28s} runs  Dice median [IQR]        min    max   >=0.50  HD95 median  empty  alignment")
        for setting, runs in groups.items():
            done = [(job, m) for job, m in runs if m is not None]
            missing = [job["id"] for job, m in runs if m is None]
            if not done:
                fish_report[setting] = {"runs": 0, "missing": missing, "status": [status.get(i, {}).get("status", "not run") for i in missing]}
                print(f"  {setting:28s} 0/{len(runs)}  {', '.join(sorted(set(fish_report[setting]['status'])))}")
                continue
            dice_values = np.array([m["dice"] for _, m in done])
            hd95 = [m["hd95_mm"] for _, m in done]
            hd95_for_median = [h if h is not None else np.inf for h in hd95]
            entry = {
                "runs": len(done), "missing": missing,
                "dice": [float(v) for v in dice_values],
                "dice_median": float(np.median(dice_values)),
                "dice_iqr": [float(np.percentile(dice_values, 25)), float(np.percentile(dice_values, 75))],
                "dice_min": float(dice_values.min()), "dice_max": float(dice_values.max()),
                "clicks_at_least_0.50": int((dice_values >= 0.5).sum()),
                "hd95_mm": hd95, "hd95_median_mm": float(np.median(hd95_for_median)),
                "empty": int(sum(h is None for h in hd95)),
                "alignment": sorted({m.get("alignment", {}).get("status", "?") for _, m in done}),
                "runtime_s_median": float(np.median([m.get("runtime_s", np.nan) for _, m in done])),
            }
            deepest = OUT / runs[0][0]["id"] / "run.log"
            if runs[0][0]["repetitions"] > 1 and deepest.exists():
                entry["deepest_click_repeated_identical"] = "repetitions gave different masks" not in deepest.read_text(encoding="utf-8")
            fish_report[setting] = entry
            print(f"  {setting:28s} {len(done)}/{len(runs)}  {entry['dice_median']:.3f} [{entry['dice_iqr'][0]:.3f}, {entry['dice_iqr'][1]:.3f}]"
                  f"   {entry['dice_min']:.3f}  {entry['dice_max']:.3f}   {entry['clicks_at_least_0.50']}/{len(done)}"
                  f"     {entry['hd95_median_mm']:8.2f}     {entry['empty']}    {','.join(entry['alignment'])}")

        primary = fish_report.get("primary", {})
        if primary.get("runs") == CLICKS and primary.get("alignment") == ["ok"]:
            criteria = {
                "overlap_median_dice_at_least_0.70": primary["dice_median"] >= 0.70,
                "robustness_7_of_9_at_least_0.50": primary["clicks_at_least_0.50"] >= 7,
                "boundary_median_hd95_at_most_bar": primary["hd95_median_mm"] <= fish["hd95_bar_mm"],
            }
            verdict = "usable" if all(criteria.values()) else "not usable"
        else:
            criteria, verdict = {}, "not assessed"
        baselines = {s: e["dice_median"] for s, e in fish_report.items() if s.startswith("baseline_") and s != "baseline_box_region_bright" and e.get("runs")}
        best = max(baselines.items(), key=lambda item: item[1]) if baselines else None
        fish_report["verdict"] = {
            "verdict": verdict, "criteria": criteria,
            "deepest_click_repeated_identical": primary.get("deepest_click_repeated_identical"),
            "best_analytical": best,
            "beats_best_analytical_by_0.05": (primary.get("dice_median", -1) - best[1] >= 0.05) if best and primary.get("runs") else None,
        }
        print(f"  verdict: {verdict} {criteria}")
        print(f"  deepest click identical on repeat: {fish_report['verdict']['deepest_click_repeated_identical']}; best analytical {best}; beats it by >= 0.05: {fish_report['verdict']['beats_best_analytical_by_0.05']}")

        second = fish.get("extra_references", {}).get("segmentation")
        if second and primary.get("runs"):
            reference, reference_spacing = _load_reference(second)
            grid = tuple(fish["grid_zyx"])
            on_grid = match_grid(reference, reference_spacing, grid, tuple(fish["spacing_mm_zyx"]))
            del reference
            values = []
            for job, metrics in groups["primary"]:
                if metrics is None:
                    continue
                mask, _ = _load_reference(OUT / job["id"] / "mask.nii.gz")
                values.append(compute_metrics(mask, on_grid, tuple(fish["spacing_mm_zyx"]))["dice"])
            fish_report["extra_second_reference"] = {"dice": values, "dice_median": float(np.median(values))}
            print(f"  against Segmentation.nii.gz (not part of the verdict): median Dice {np.median(values):.3f}, min {min(values):.3f}, max {max(values):.3f}")
        report["fish"][fish_name] = fish_report
    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"\nwritten {OUT / 'summary.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("plan", "run", "summary", "run-one"))
    parser.add_argument("config", nargs="?")
    arguments = parser.parse_args()
    if arguments.step == "plan":
        return plan()
    if arguments.step == "run":
        return run()
    if arguments.step == "run-one":
        return run_one(arguments.config)
    return summary()


if __name__ == "__main__":
    raise SystemExit(main())
