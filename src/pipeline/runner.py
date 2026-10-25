from __future__ import annotations

import importlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np

from src.core.loaders import load_volume
from src.core.preprocessor import PreprocessingConfig, preprocess
from src.core.box import crop_volume, restore_mask
from src.core.volume import Volume
from src.pipeline.experiment import (
    ExperimentConfig,
    load_experiment_config,
    prepare_result_dir,
)
from src.pipeline.mesh import export_mesh, reconstruct_mesh_from_settings
from src.pipeline.metrics import compute_metrics, write_metrics_json

SegmentFunction = Callable[[Volume, dict], np.ndarray]

# algorithm.name -> "module:function". Imported lazily, so a missing optional
# dependency (e.g. torch) only matters for the algorithm that needs it.
ALGORITHMS: dict[str, str] = {
    "box_region": "src.models.box_region:segment",
    "unet_2d": "src.models.unet:segment",
    "unet_25d": "src.models.unet:segment",
    "medsam": "src.models.foundation.medsam:segment",
    "sam_med3d": "src.models.foundation.sam_med3d:segment",
    "protosam3d": "src.models.foundation.protosam3d:segment",
}
_registered: dict[str, SegmentFunction] = {}


class RunnerError(RuntimeError):
    """Raised when an experiment cannot be completed."""


@dataclass
class RunResult:
    result_dir: Path
    mask: np.ndarray  # full grid of the preprocessed volume, uint8, (z, y, x)
    spacing_mm: tuple[float, float, float]
    box_prompted: bool
    runtimes_s: list[float]
    deterministic: bool
    metrics: dict | None = None
    mesh_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def register_algorithm(name: str, function: SegmentFunction) -> None:
    """Make `function` available as ``algorithm.name`` (tests, quick experiments)."""
    _registered[name] = function


def resolve_algorithm(name: str) -> SegmentFunction:
    if name in _registered:
        return _registered[name]
    target = ALGORITHMS.get(name)
    if target is None:
        known = sorted(set(ALGORITHMS) | set(_registered))
        raise RunnerError(f"Unknown algorithm '{name}'. Known algorithms: {', '.join(known)}")
    module_name, function_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RunnerError(f"Algorithm '{name}' is not available ({module_name}: {error})") from error
    function = getattr(module, function_name, None)
    if function is None:
        raise RunnerError(f"{module_name} has no function '{function_name}'")
    return function


def run_experiment(config_path: str | Path) -> RunResult:
    config_path = Path(config_path)
    config = load_experiment_config(config_path)
    result_dir = prepare_result_dir(config, config_path)
    logger = _file_logger(result_dir / "run.log", config.experiment.id)
    try:
        return _run(config, result_dir, logger)
    except Exception as error:
        logger.error("failed: %s", error)
        raise
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _run(config: ExperimentConfig, result_dir: Path, logger: logging.Logger) -> RunResult:
    segment = resolve_algorithm(config.algorithm.name)
    _seed(config.experiment.seed)

    volume = load_volume(config.data.input_path, config.data.format)
    logger.info("loaded %s: shape %s, spacing %s mm", config.data.input_path, volume.shape, volume.spacing_mm)
    settings = config.preprocessing
    volume = preprocess(
        volume,
        PreprocessingConfig(
            target_spacing=settings.target_spacing,
            hu_range=settings.hu_range,
            invert_hu=settings.invert_hu,
            denoise=settings.denoise,
        ),
    )
    logger.info("preprocessed: shape %s, spacing %s mm", volume.shape, volume.spacing_mm)

    where = None
    model_input = volume
    if config.data.box is not None:
        model_input, where = crop_volume(volume, config.data.box)
        logger.info(
            "box: the algorithm sees %s of %s voxels (%.1f%%)",
            model_input.shape,
            volume.shape,
            100 * model_input.data.size / volume.data.size,
        )

    parameters = dict(config.algorithm.parameters)
    if config.data.seed_zyx is not None and "seed_zyx" not in parameters:
        offset = [w.start for w in where] if where else [0, 0, 0]
        local_seed = tuple(v - o for v, o in zip(config.data.seed_zyx, offset))
        if any(not 0 <= v < n for v, n in zip(local_seed, model_input.shape)):
            raise RunnerError(f"data.seed_zyx {config.data.seed_zyx} lies outside the volume the algorithm sees")
        parameters["seed_zyx"] = local_seed  # the algorithms take it relative to the volume they receive

    masks, runtimes = [], []
    for repetition in range(max(config.experiment.repetitions, 1)):
        _seed(config.experiment.seed)  # every repetition starts from the same seed
        started = time.perf_counter()
        local_mask = np.asarray(segment(model_input, dict(parameters)))
        runtimes.append(time.perf_counter() - started)
        if local_mask.shape != model_input.shape:
            raise RunnerError(
                f"'{config.algorithm.name}' returned a mask of shape {local_mask.shape}, "
                f"expected {model_input.shape}"
            )
        masks.append(local_mask != 0)
        logger.info("repetition %d: %.2f s, %d voxels", repetition + 1, runtimes[-1], masks[-1].sum())
    deterministic = all(np.array_equal(masks[0], other) for other in masks[1:])
    local_mask = masks[-1]

    mask = restore_mask(local_mask, volume.shape, where) if where else local_mask.astype(np.uint8)
    _save_mask(mask, volume.spacing_mm, result_dir / "mask.nii.gz")

    result = RunResult(
        result_dir=result_dir,
        mask=mask,
        spacing_mm=volume.spacing_mm,
        box_prompted=where is not None,
        runtimes_s=runtimes,
        deterministic=deterministic,
    )
    if not deterministic:
        result.warnings.append("repetitions gave different masks")
        logger.warning(result.warnings[-1])

    if mask.any():
        mesh = reconstruct_mesh_from_settings(mask, volume.spacing_mm, config.mesh)
        result.mesh_path = export_mesh(mesh, result_dir / f"mesh.{config.mesh.output_format}")
    else:
        result.warnings.append("the mask is empty - no mesh was built")
        logger.warning(result.warnings[-1])

    if config.data.reference_mask_path:
        reference, reference_spacing = _load_reference(config.data.reference_mask_path)
        on_grid = match_grid(reference, reference_spacing, volume.shape, volume.spacing_mm)
        result.metrics = compute_metrics(mask, on_grid, volume.spacing_mm)
        result.metrics["box_prompted"] = result.box_prompted
        result.metrics["runtime_s"] = float(np.median(runtimes))
        write_metrics_json(result.metrics, result_dir / "metrics.json")
        logger.info("dice %.3f, hd95 %s mm", result.metrics["dice"], result.metrics["hd95_mm"])

    return result


def match_grid(
    reference: np.ndarray,
    reference_spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """The reference mask on the grid of the prediction (nearest neighbour, by
    voxel centres in mm). Needed when ``preprocessing.target_spacing`` differs
    from the spacing of the reference. Voxels outside the reference are 0."""
    reference = np.asarray(reference) != 0
    if reference.shape == tuple(shape) and np.allclose(reference_spacing, spacing, rtol=1e-4):
        return reference
    indices, inside = [], []
    for size, step, ref_size, ref_step in zip(shape, spacing, reference.shape, reference_spacing):
        position = (np.arange(size) + 0.5) * step / ref_step
        index = np.floor(position).astype(int)
        inside.append(index < ref_size)
        indices.append(np.clip(index, 0, ref_size - 1))
    result = reference[np.ix_(*indices)]
    in_z, in_y, in_x = inside
    return result & in_z[:, None, None] & in_y[None, :, None] & in_x[None, None, :]


def _load_reference(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
    except (OSError, nib.filebasedimages.ImageFileError) as error:
        raise RunnerError(f"Cannot read the reference mask {path}: {error}") from error
    zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
    return (data != 0).transpose(2, 1, 0), zooms[::-1]  # NIfTI (x,y,z) -> (z,y,x)


def _save_mask(mask: np.ndarray, spacing: tuple[float, float, float], path: Path) -> None:
    affine = np.diag([*spacing[::-1], 1.0])  # NIfTI axes are (x, y, z)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8).transpose(2, 1, 0), affine), str(path))


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def _file_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(f"runner.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python -m src.pipeline.runner <config.yaml>", file=sys.stderr)
        return 2
    try:
        result = run_experiment(arguments[0])
    except Exception as error:  # noqa: BLE001 - report any failure as a message, not a traceback
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = {
        "result_dir": str(result.result_dir),
        "box_prompted": result.box_prompted,
        "runtime_s": result.runtimes_s,
        "deterministic": result.deterministic,
        "metrics": result.metrics,
        "warnings": result.warnings,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
