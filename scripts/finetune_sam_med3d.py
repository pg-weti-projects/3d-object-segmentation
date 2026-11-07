"""
scripts/finetune_sam_med3d.py

Fine-tunes SAM-Med3D's prompt encoder and mask decoder on fish scans and evaluates the result, as fixed in
md-files/sam_med3d_finetune_protocol.md. The input is the standard weights and a set of scans with reference
masks; the output is a named set of tuned weights in sam-med3d-assets/finetuned/<name>/, used with the
sam_med3d method's `weights` parameter. The standard weights are never written to.

    python scripts/finetune_sam_med3d.py encode --fish oreo --kind real --device xpu
    python scripts/finetune_sam_med3d.py train --name s1-oreo --fish oreo --device xpu
    python scripts/finetune_sam_med3d.py evaluate --weights s1-oreo --fish oreo gadus tuna
    python scripts/finetune_sam_med3d.py evaluate --weights standard --human

The fish, their test clicks and boxes come from configs/experiments/sam_med3d_fish_plan.json. --input
NAME=PATH replaces a fish's scan (for example the tuna as a NIfTI in slice-normal order, until the DICOM
loader keeps that order). Encoding saves each cube as it is made, so an interrupted run goes on where it
stopped.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.box import Box  # noqa: E402
from src.core.loaders import load_volume  # noqa: E402
from src.pipeline.metrics import compute_metrics  # noqa: E402

from src.models.foundation.sam_med3d import assets, finetune, network  # noqa: E402

PLAN = ROOT / "configs" / "experiments" / "sam_med3d_fish_plan.json"
RESULTS = ROOT / "results" / "sam_med3d_finetune"
HUMAN_ORGANS = {
    1: "spleen", 2: "right_kidney", 3: "left_kidney", 4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
    8: "aorta", 9: "postcava", 10: "pancreas", 11: "right_adrenal", 12: "left_adrenal", 13: "duodenum",
    14: "bladder", 15: "prostate_uterus",
}
logger = logging.getLogger("finetune_sam_med3d")


def cache_root() -> Path:
    return assets.locate().weights.parent.parent / "finetune-cache"


def plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))["fish"]


def fish_input(name: str, inputs: dict[str, str]) -> tuple[str, str | None]:
    entry = plan()[name]
    if name in inputs:
        return inputs[name], None
    return str(ROOT / entry["input_path"]), entry["format"]


def load_specimen(name: str, inputs: dict[str, str]) -> finetune.Specimen:
    """The scan and its reference on the scan's grid, exactly as the runner scores against it."""
    from src.pipeline.runner import _load_reference, match_grid

    path, fmt = fish_input(name, inputs)
    volume = load_volume(path, fmt)
    mask, mask_spacing = _load_reference(ROOT / plan()[name]["reference_mask_path"])
    return finetune.Specimen(name, volume, match_grid(mask, mask_spacing, volume.shape, volume.spacing_mm))


def git_commit() -> str | None:
    found = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
    return (found.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")) if found.returncode == 0 else None


# --- encode / train -------------------------------------------------------------------------------


def command_encode(arguments) -> int:
    # synthetic: enough for a fold whose other training fish has no room for one (training_files)
    count = arguments.count or finetune.RECIPE["cubes_per_fish"]
    net = network.get_network(arguments.device, arguments.threads)
    for name in arguments.fish:
        specimen = load_specimen(name, arguments.input)
        files = finetune.encode(specimen, arguments.kind, count, net, cache_root())
        logger.info("%s: %d %s cubes in %s", name, len(files), arguments.kind, finetune.cache_folder(cache_root(), name, arguments.kind))
    return 0


def command_train(arguments) -> int:
    kinds = ["real", "synthetic"] if arguments.synthetic else ["real"]
    try:
        files, counts = finetune.training_files(cache_root(), arguments.fish, arguments.synthetic)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    missing = [name for name in arguments.fish if counts[f"{name}/real"] == 0]
    if missing:
        raise SystemExit(f"error: no encoded cubes for {missing}; run encode first")
    if assets.tuned_folder(arguments.name).exists():
        raise SystemExit(f"error: tuned weights '{arguments.name}' exist already; choose another name")
    located = assets.require()
    net = network.SamMed3D(located.source, located.weights, arguments.device, arguments.threads)
    started = datetime.datetime.now()
    log_path = RESULTS / "training" / f"{arguments.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.datetime.now():%H:%M:%S} {message}\n")

    log(f"train {arguments.name} on {arguments.fish} ({kinds}), {len(files)} cubes, device {arguments.device}")
    history = finetune.train(files, net, finetune.RECIPE, log)
    folder = finetune.save_tuned(arguments.name, net, {
        "trained_on": list(arguments.fish),
        "kinds": kinds,
        "cubes": counts,
        "recipe": finetune.RECIPE,
        "recipe_hash": finetune.recipe_hash(),
        "loss_history": history,
        "device": arguments.device,
        "code_commit": git_commit(),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": {name: fish_input(name, arguments.input)[0] for name in arguments.fish},
    })
    log(f"saved {folder}")
    logger.info("saved %s", folder)
    return 0


# --- evaluate ------------------------------------------------------------------------------------


def verdict(dice: list[float], hd95: list[float], hd95_bar_mm: float) -> dict:
    """md-files/sam_med3d_protocol.md section 10: overlap, robustness and boundary, over the 9 clicks."""
    dice_values, hd_values = np.asarray(dice, float), np.asarray(hd95, float)
    overlap = float(np.median(dice_values))
    robust = int((dice_values >= 0.5).sum())
    boundary = float(np.median(np.where(np.isfinite(hd_values), hd_values, np.inf)))
    return {
        "median_dice": overlap,
        "dice_iqr": [float(np.percentile(dice_values, 25)), float(np.percentile(dice_values, 75))],
        "clicks_at_least_0_5": robust,
        "median_hd95_mm": boundary,
        "hd95_bar_mm": hd95_bar_mm,
        "overlap_ok": overlap >= 0.70,
        "robustness_ok": robust >= 7,
        "boundary_ok": boundary <= hd95_bar_mm,
        "usable": overlap >= 0.70 and robust >= 7 and boundary <= hd95_bar_mm,
    }


def evaluate_fish(name: str, weights: str, inputs: dict[str, str]) -> dict:
    from src.pipeline.runner import run_experiment

    entry = plan()[name]
    path, fmt = fish_input(name, inputs)
    out = RESULTS / weights
    dice, hd95, runs = [], [], []
    for index, click in enumerate(entry["settings"]["primary"]["clicks"]):
        run_id = f"{name}__click{index}"
        config = {
            "experiment": {"id": run_id, "repetitions": 1},
            "data": {"input_path": path, "format": fmt, "reference_mask_path": str(ROOT / entry["reference_mask_path"]),
                     "box": entry["settings"]["primary"]["box"], "seed_zyx": click},
            "algorithm": {"name": "sam_med3d", "parameters": {"weights": weights}},
            "mesh": {"smoothing_iterations": 0, "output_format": "glb"},
            "output": {"results_dir": str(out) + "/"},
        }
        config_path = out / "configs" / f"{run_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        metrics = run_experiment(config_path).metrics
        dice.append(float(metrics["dice"]))
        hd95.append(float(metrics["hd95_mm"]) if metrics.get("hd95_mm") is not None else float("inf"))
        runs.append({"click": click, "dice": dice[-1], "hd95_mm": hd95[-1]})
        logger.info("%s %s click %d: Dice %.3f, HD95 %.1f mm", weights, name, index, dice[-1], hd95[-1])
    return {"fish": name, "weights": weights, "input": path, "runs": runs, **verdict(dice, hd95, entry["hd95_bar_mm"])}


def test_clicks(mask: np.ndarray, spacing, count: int, seed: int = 42) -> list[tuple[int, int, int]]:
    """As the human checks chose them (protocol section 4): the deepest voxel of the largest part, then one
    voxel in each of `count - 1` slabs along its longest axis, at least half as deep as the deepest."""
    from scipy import ndimage as ndi

    labels, parts = ndi.label(mask)
    largest = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:])) if parts > 1 else np.asarray(mask, bool)
    depth = ndi.distance_transform_edt(largest, sampling=spacing)
    clicks = [tuple(int(v) for v in np.unravel_index(int(depth.argmax()), depth.shape))]
    interior = np.argwhere(depth >= 0.5 * depth.max())
    extent = (np.argwhere(largest).max(0) - np.argwhere(largest).min(0) + 1) * np.asarray(spacing)
    order = interior[np.argsort(interior[:, int(np.argmax(extent))], kind="stable")]
    rng = np.random.default_rng(seed)
    for slab in np.array_split(order, count - 1) if count > 1 else []:
        clicks.append(tuple(int(v) for v in slab[rng.integers(len(slab))]))
    return clicks


def score(prediction: np.ndarray, reference: np.ndarray, spacing) -> dict:
    """The metrics on a crop holding both masks and one voxel more, which gives the whole scan's distances."""
    either = np.asarray(prediction, bool) | np.asarray(reference, bool)
    found = np.argwhere(either)
    window = tuple(slice(max(int(a) - 1, 0), min(int(b) + 2, n)) for a, b, n in zip(found.min(0), found.max(0), either.shape))
    metrics = compute_metrics(np.asarray(prediction)[window], np.asarray(reference)[window], spacing)
    return {key: metrics[key] for key in ("dice", "hd95_mm")}


def evaluate_human(weights: str) -> dict:
    """The human control (protocol section 7): amos_0013, 15 organs, 3 clicks each, box = the organ's
    bounding box grown 25%, as `scripts/measure_contrast.py` draws it."""
    import nibabel as nib

    from scripts.measure_contrast import _box_around_mask
    from src.models.foundation.sam_med3d import segment

    source = assets.locate().source / "test_data" / "amos_val_toy_data"
    volume = load_volume(source / "imagesVa" / "amos_0013.nii.gz")
    labels = np.asarray(nib.load(str(source / "labelsVa" / "amos_0013.nii.gz")).dataobj).transpose(2, 1, 0)
    organs = {}
    for label, organ in HUMAN_ORGANS.items():
        mask = labels == label
        if not mask.any():
            continue
        box, _ = _box_around_mask(mask, volume.spacing_mm, 0.25)
        where = box.slices(volume.shape, volume.spacing_mm)
        results = []
        for click in test_clicks(mask, volume.spacing_mm, 3):
            prediction = segment(volume, {"seed_zyx": list(click), "box_zyx": [[w.start, w.stop] for w in where], "weights": weights})
            results.append({"click": list(click), **score(np.asarray(prediction) != 0, mask, volume.spacing_mm)})
        organs[organ] = {"runs": results, "median_dice": float(np.median([r["dice"] for r in results]))}
        logger.info("%s human %s: median Dice %.3f", weights, organ, organs[organ]["median_dice"])
    return {"weights": weights, "organs": organs, "median_of_organ_medians": float(np.median([o["median_dice"] for o in organs.values()]))}


def command_evaluate(arguments) -> int:
    out = RESULTS / arguments.weights
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"weights": arguments.weights, "fish": {}}
    summary["code_commit"] = git_commit()
    for name in arguments.fish:
        summary["fish"][name] = evaluate_fish(name, arguments.weights, arguments.input)
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")  # after every fish
    if arguments.human:
        summary["human"] = evaluate_human(arguments.weights)
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for name, result in summary["fish"].items():
        print(f"{arguments.weights:>16} {name:6} median Dice {result['median_dice']:.3f}, {result['clicks_at_least_0_5']}/9 >= 0.5, "
              f"median HD95 {result['median_hd95_mm']:.1f} mm (bar {result['hd95_bar_mm']:.1f}): {'usable' if result['usable'] else 'not usable'}")
    if "human" in summary:
        print(f"{arguments.weights:>16} human  median of organ medians {summary['human']['median_of_organ_medians']:.3f}")
    return 0


def parse_inputs(values: list[str]) -> dict[str, str]:
    inputs = {}
    for value in values or []:
        name, _, path = value.partition("=")
        if not name or not path:
            raise SystemExit(f"error: --input expects NAME=PATH, got {value!r}")
        inputs[name] = path
    return inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "train", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--fish", nargs="*", default=[], choices=sorted(plan()))
        sub.add_argument("--input", action="append", default=[], metavar="NAME=PATH", help="replace a fish's scan")
        if name != "evaluate":
            sub.add_argument("--device", default="cpu", help="cpu, xpu or cuda")
            sub.add_argument("--threads", type=int, default=None)
    commands.choices["encode"].add_argument("--kind", choices=finetune.KINDS, default="real")
    commands.choices["encode"].add_argument("--count", type=int, default=None, help="default: the recipe's")
    commands.choices["train"].add_argument("--name", required=True, help="the tuned set's name, e.g. s1-oreo")
    commands.choices["train"].add_argument("--synthetic", action="store_true", help="arm B: real and synthetic cubes")
    commands.choices["evaluate"].add_argument("--weights", default=assets.STANDARD)
    commands.choices["evaluate"].add_argument("--human", action="store_true", help="also the human control")
    arguments = parser.parse_args(argv)
    arguments.input = parse_inputs(arguments.input)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if arguments.command in ("encode", "train") and not arguments.fish:
        parser.error("--fish is required")
    return {"encode": command_encode, "train": command_train, "evaluate": command_evaluate}[arguments.command](arguments)


if __name__ == "__main__":
    raise SystemExit(main())
