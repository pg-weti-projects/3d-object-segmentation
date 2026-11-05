"""
scripts/sam_med3d_golden.py

Golden comparison of our SAM-Med3D wrapper against upstream's own code, on upstream's
test scan (amos_0013). Two halves, in two environments:

    # with torchio and medim, in an environment of its own
    python scripts/sam_med3d_golden.py upstream --out <folder>

    # with the project's requirements and torch (also works in the one above)
    python scripts/sam_med3d_golden.py compare --out <folder>

`upstream` runs validate_paired_img_gt as it is, then again with recorders around its
functions, to capture each organ's cube, click and output. `compare` then checks three
things separately, so that a difference can be traced:

    network      our network, on upstream's cube and click
    cube         our cube against upstream's, for the same placement
    end to end   our cube through our network, with upstream's click

and once more with our sampled values truncated to whole numbers. Upstream's resampler casts its
output to the scan's integer type, so on an int16 scan its values lose up to 1 HU towards zero; ours
stay as interpolated. This separates that one known difference from anything else.

`fixture` (run with `upstream`) writes tests/data/sam_med3d_golden.npz for the regression test.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.foundation.sam_med3d import assets, cube as geometry  # noqa: E402
from src.models.foundation.sam_med3d.network import SamMed3D  # noqa: E402
from src.models.foundation.sam_med3d.pipeline import prepare  # noqa: E402

SCAN = Path("test_data") / "amos_val_toy_data"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "data" / "sam_med3d_golden.npz"
STRIDE = 8  # of a 128^3 cube, in the fixture
PREPARE = {"clamp": None, "intensity_shift": 0.0, "padding": 0.0}


def scan_paths(source: Path) -> tuple[Path, Path]:
    return source / SCAN / "imagesVa" / "amos_0013.nii.gz", source / SCAN / "labelsVa" / "amos_0013.nii.gz"


def synthetic_scan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small scan (x, y, z), its label and its affine, reproducible anywhere, on a grid like AMOS's:
    thick slices, and the x axis running from the patient's left, so that canonical order flips it."""
    x, y, z = np.mgrid[:64, :72, :30].astype(np.float64)
    data = np.full(x.shape, -1000.0)
    data[((x - 32) / 28) ** 2 + ((y - 36) / 32) ** 2 + ((z - 15) / 13) ** 2 <= 1] = 40.0
    data += 30 * np.sin(x / 5) * np.cos(y / 7) * np.sin(z / 3 + 1)
    label = ((x - 24) / 10) ** 2 + ((y - 30) / 9) ** 2 + ((z - 12) / 4) ** 2 <= 1
    data[label] = 250.0
    affine = np.array([[-0.8, 0, 0, 120.0], [0, 0.8, 0, -40.0], [0, 0, 4.0, 10.0], [0, 0, 0, 1.0]])
    return data.astype(np.int16), label.astype(np.uint8), affine


def synthetic_cube() -> np.ndarray:
    """A normalised cube for the network part of the fixture, reproducible anywhere."""
    x, y, z = np.mgrid[:128, :128, :128].astype(np.float32)
    field = np.sin(x / 9) * np.cos(y / 13) * np.sin(z / 7)
    field[((x - 70) / 20) ** 2 + ((y - 60) / 25) ** 2 + ((z - 64) / 18) ** 2 <= 1] += 2.0
    return field.astype(np.float32)


# --- the upstream half: torchio and medim, nothing of ours in the computation ---------------------


def _upstream_modules(source: Path):
    sys.path.insert(0, str(source))
    import medim
    import torch
    from utils import infer_utils

    # The checkpoint holds tensors tagged for a GPU, and medim loads it with no map_location, which fails
    # on a CPU-only machine. This changes where they load and nothing else: run_upstream compares the
    # loaded weights with the checkpoint.
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "map_location": "cpu"})
    return medim, torch, infer_utils


def run_upstream(out: Path) -> int:
    located = assets.require()
    medim, torch, infer_utils = _upstream_modules(located.source)
    torch.set_num_threads(4)
    image, label = scan_paths(located.source)
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=str(located.weights))
    checkpoint = torch.load(str(located.weights), map_location="cpu", weights_only=True)["model_state_dict"]
    state = model.state_dict()
    same_keys = set(state) == set(checkpoint)
    same_values = same_keys and all(torch.equal(state[key].cpu(), checkpoint[key]) for key in state)
    environment = {
        "torch": torch.__version__,
        "medim": getattr(medim, "__version__", "unknown"),
        "weights_equal_checkpoint": bool(same_values),
        "decoder_forward": str(inspect.signature(model.mask_decoder.forward)),
        "load_s": time.perf_counter() - started,
    }

    # exactly upstream's call, nothing patched
    started = time.perf_counter()
    pure = out / "upstream_pure.nii.gz"
    infer_utils.validate_paired_img_gt(model, str(image), str(label), str(pure), num_clicks=1)
    environment["pure_run_s"] = time.perf_counter() - started

    # the same call with recorders around it
    original_preprocess, original_infer = infer_utils.data_preprocess, infer_utils.sam_model_infer
    original_click = infer_utils.random_sample_next_click
    current: dict = {}
    organs = []

    def preprocess(subject, meta_info, category_index, target_spacing, crop_size=128):
        current["organ"] = int(category_index)
        return original_preprocess(subject, meta_info, category_index, target_spacing, crop_size)

    def infer(model_, roi_image, roi_gt=None, num_clicks=1, prev_low_res_mask=None):
        clicks = []

        def recording(prev_mask, gt_mask):
            point, point_label = original_click(prev_mask, gt_mask)
            clicks.append(point.cpu().numpy().ravel().tolist())
            return point, point_label

        mask, low_resolution = original_infer(
            model_, roi_image, roi_gt=roi_gt, prompt_generator=recording, num_clicks=num_clicks, prev_low_res_mask=prev_low_res_mask
        )
        current.update(image=roi_image, gt=roi_gt, mask=mask, low=low_resolution, click=clicks[0])
        return mask, low_resolution

    def postprocess(roi_pred, meta_info):
        result = original_postprocess(roi_pred, meta_info)
        probability = torch.sigmoid(
            torch.nn.functional.interpolate(current["low"], size=(128, 128, 128), mode="trilinear", align_corners=False)
        )
        np.savez_compressed(
            out / f"organ_{current['organ']:02d}.npz",
            image=current["image"].cpu().numpy().squeeze().astype(np.float32),
            label=current["gt"].cpu().numpy().squeeze().astype(np.uint8),
            mask=np.asarray(current["mask"], dtype=np.uint8),
            probability=probability.cpu().numpy().squeeze().astype(np.float32),
            click=np.array(current["click"]),
            roi_affine=np.asarray(meta_info["roi_subject_affine"]),
            on_original_grid=result,
        )
        organs.append(current["organ"])
        print(f"upstream: organ {current['organ']}", flush=True)
        return result

    original_postprocess = infer_utils.data_postprocess
    infer_utils.data_preprocess, infer_utils.sam_model_infer, infer_utils.data_postprocess = preprocess, infer, postprocess
    recorded = out / "upstream_recorded.nii.gz"
    infer_utils.validate_paired_img_gt(model, str(image), str(label), str(recorded), num_clicks=1)
    infer_utils.data_preprocess, infer_utils.sam_model_infer, infer_utils.data_postprocess = (
        original_preprocess,
        original_infer,
        original_postprocess,
    )

    import nibabel as nib

    pure_mask = np.asarray(nib.load(str(pure)).dataobj)
    environment["recorded_equals_pure"] = bool(np.array_equal(pure_mask, np.asarray(nib.load(str(recorded)).dataobj)))
    truth = np.asarray(nib.load(str(label)).dataobj)
    dice = {
        int(organ): float(2 * ((pure_mask == organ) & (truth == organ)).sum() / ((pure_mask == organ).sum() + (truth == organ).sum()))
        for organ in np.unique(truth)
        if organ
    }
    environment["dice_per_organ"] = dice
    environment["median_dice"] = float(np.median(list(dice.values())))
    environment["organs"] = organs
    (out / "upstream.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(json.dumps(environment, indent=2))
    return 0


def run_fixture(out: Path) -> int:
    """Upstream's cube and upstream's network output on synthetic input, small enough to keep in git."""
    located = assets.require()
    medim, torch, infer_utils = _upstream_modules(located.source)
    import nibabel as nib

    data, label, affine = synthetic_scan()
    with tempfile.TemporaryDirectory() as folder:
        image_path, label_path = Path(folder) / "image.nii.gz", Path(folder) / "label.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(image_path))
        nib.save(nib.Nifti1Image(label, affine), str(label_path))
        subject, meta = infer_utils.get_subject_and_meta_info(str(image_path), str(label_path))
        roi_image, _, meta = infer_utils.data_preprocess(subject, meta, category_index=1, target_spacing=(1.5, 1.5, 1.5))
    roi = roi_image.cpu().numpy().squeeze().astype(np.float32)

    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=str(located.weights)).eval()
    cube_xyz = torch.from_numpy(synthetic_cube())[None, None]
    with torch.no_grad():
        embedding = model.image_encoder(cube_xyz)
        sparse, dense = model.prompt_encoder(
            points=[torch.tensor([[[70.0, 60.0, 64.0]]]), torch.tensor([[1]], dtype=torch.int64)],
            boxes=None,
            masks=torch.zeros(1, 1, 32, 32, 32),
        )
        low, _ = model.mask_decoder(
            image_embeddings=embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
        )
        probability = torch.sigmoid(torch.nn.functional.interpolate(low, size=(128, 128, 128), mode="trilinear", align_corners=False))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        cube=roi[::STRIDE, ::STRIDE, ::STRIDE],
        roi_affine=np.asarray(meta["roi_subject_affine"]),
        probability=probability.numpy().squeeze()[::STRIDE, ::STRIDE, ::STRIDE].astype(np.float32),
        click=np.array([70, 60, 64]),
    )
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    return 0


# --- our half ---------------------------------------------------------------------------------------


def canonical_scan(image_path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """The scan in the order upstream gives the network, as our (z, y, x) array."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(image_path)))
    return (
        np.asarray(image.dataobj, dtype=np.float32).transpose(2, 1, 0),
        tuple(float(z) for z in image.header.get_zooms()[:3])[::-1],
        image.affine,
    )


def cube_from_affine(roi_affine: np.ndarray, scan_affine: np.ndarray, spacing_zyx) -> geometry.Cube:
    """The placement of upstream's cube, in our mm: voxel i of an axis covers [i * s, (i + 1) * s)."""
    size = float(roi_affine[0, 0])
    corner_xyz = [
        roi_affine[axis, 3] - size / 2 - (scan_affine[axis, 3] - spacing_zyx[2 - axis] / 2)
        for axis in range(3)
    ]
    return geometry.Cube(tuple(corner_xyz[::-1]), size)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / total if total else 1.0


def run_compare(out: Path, threads: int = 4) -> int:
    located = assets.require()
    upstream = json.loads((out / "upstream.json").read_text(encoding="utf-8"))
    image_path, _ = scan_paths(located.source)
    data, spacing, scan_affine = canonical_scan(image_path)
    network = SamMed3D(located.source, located.weights, threads=threads)

    import torch

    rows = []
    for organ in upstream["organs"]:
        saved = np.load(out / f"organ_{organ:02d}.npz")
        theirs, their_probability = saved["image"], saved["probability"]
        click = tuple(int(v) for v in saved["click"])

        # 1. the network alone, on upstream's cube and click
        started = time.perf_counter()
        ours_on_theirs = network.predict(network.embed(theirs), click)
        seconds = time.perf_counter() - started
        # 2. our cube for upstream's placement; the arrays are (x, y, z) on their side
        cube = cube_from_affine(saved["roi_affine"], scan_affine, spacing)
        values, inside = geometry.sample(data, spacing, cube)
        ours_cube, normalisation = prepare(values, inside, PREPARE)
        ours_cube_xyz = geometry.to_network_axes(ours_cube)
        inside_xyz = geometry.to_network_axes(inside)
        difference = np.abs(ours_cube_xyz - theirs)
        # 3. end to end, with upstream's click
        ours_probability = network.predict(network.embed(ours_cube_xyz), click)
        # 4. the same with upstream's truncation to integers
        truncated = values.copy()
        truncated[inside] = np.trunc(np.asarray(truncated[inside], dtype=np.float64))
        truncated_xyz = geometry.to_network_axes(prepare(truncated, inside, PREPARE)[0])
        truncated_difference = np.abs(truncated_xyz - theirs)
        truncated_probability = network.predict(network.embed(truncated_xyz), click)
        rows.append(
            {
                "organ": organ,
                "click_xyz": list(click),
                "network_max_abs_prob_diff": float(np.abs(ours_on_theirs - their_probability).max()),
                "network_dice_of_masks": dice(ours_on_theirs > 0.5, their_probability > 0.5),
                "cube_max_abs_diff": float(difference.max()),
                "cube_mean_abs_diff": float(difference.mean()),
                "cube_voxels_over_0_01": int((difference > 0.01).sum()),
                "cube_outside_scan_share": float(1 - inside.mean()),
                "cube_max_abs_diff_inside_scan": float(difference[inside_xyz].max()) if inside.any() else 0.0,
                "cube_correlation": float(np.corrcoef(ours_cube_xyz.ravel(), theirs.ravel())[0, 1]),
                "norm_mean": normalisation["mean"],
                "norm_std": normalisation["std"],
                "end_to_end_max_abs_prob_diff": float(np.abs(ours_probability - their_probability).max()),
                "end_to_end_dice_of_masks": dice(ours_probability > 0.5, their_probability > 0.5),
                "truncated_cube_mean_abs_diff": float(truncated_difference.mean()),
                "truncated_cube_voxels_over_0_001": int((truncated_difference > 0.001).sum()),
                "truncated_max_abs_prob_diff": float(np.abs(truncated_probability - their_probability).max()),
                "truncated_dice_of_masks": dice(truncated_probability > 0.5, their_probability > 0.5),
                "seconds": seconds,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "torch": torch.__version__,
        "upstream_torch": upstream["torch"],
        "upstream_medim": upstream["medim"],
        "weights_equal_checkpoint": upstream["weights_equal_checkpoint"],
        "recorded_equals_pure": upstream["recorded_equals_pure"],
        "upstream_median_dice": upstream["median_dice"],
        "worst_network_prob_diff": max(row["network_max_abs_prob_diff"] for row in rows),
        "worst_cube_diff": max(row["cube_max_abs_diff"] for row in rows),
        "worst_cube_diff_inside_scan": max(row["cube_max_abs_diff_inside_scan"] for row in rows),
        "worst_end_to_end_prob_diff": max(row["end_to_end_max_abs_prob_diff"] for row in rows),
        "lowest_end_to_end_mask_dice": min(row["end_to_end_dice_of_masks"] for row in rows),
        "worst_truncated_cube_mean_diff": max(row["truncated_cube_mean_abs_diff"] for row in rows),
        "most_truncated_voxels_over_0_001": max(row["truncated_cube_voxels_over_0_001"] for row in rows),
        "worst_truncated_prob_diff": max(row["truncated_max_abs_prob_diff"] for row in rows),
        "lowest_truncated_mask_dice": min(row["truncated_dice_of_masks"] for row in rows),
    }
    (out / f"comparison_torch_{torch.__version__.split('+')[0]}.json").write_text(json.dumps({"summary": summary, "organs": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("upstream", "fixture", "compare"))
    parser.add_argument("--out", type=Path, default=Path("results/sam_med3d_checks/golden"))
    arguments = parser.parse_args()
    if arguments.step == "upstream":
        return run_upstream(arguments.out)
    if arguments.step == "fixture":
        return run_fixture(arguments.out)
    return run_compare(arguments.out)


if __name__ == "__main__":
    raise SystemExit(main())
