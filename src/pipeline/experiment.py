"""
src/pipeline/experiment.py

Shared experiment configuration format (.yaml) and the results/<experiment_id>/
output convention. Used by the runner (#9), mesh reconstruction (#7), the
metrics module (#8), and eventually the frontend (to know where to find
generated artifacts).

results/<experiment_id>/ convention:
    config.yaml   - copy of the config used for this run (reproducibility)
    mask.nii.gz    - segmentation mask
    mesh.<format>   - mesh output, named after mesh.output_format
    metrics.json     - metrics report (only when reference_mask_path is set)
    run.log            - execution log
"""

from __future__ import annotations

import random
import shutil
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.core.box import Box, BoxError


class ExperimentConfigError(ValueError):
    """Raised when a config file is missing required fields or malformed."""


@dataclass(frozen=True)
class ExperimentMeta:
    id: str
    description: str = ""
    seed: int = 42
    repetitions: int = 3  # NFNC_04 - 3-run repetition requirement


@dataclass(frozen=True)
class DataConfig:
    input_path: str
    format: str | None = None  # 'dicom' | 'tiff' | 'voxels-json'; None = infer from suffix
    reference_mask_path: str | None = None  # optional - required only to compute metrics
    box: Box | None = None  # optional box around the bladder (semi-automatic mode), see core/box.py
    seed_zyx: tuple[int, int, int] | None = None  # optional voxel inside the bladder (preprocessed grid)


@dataclass(frozen=True)
class PreprocessingSettings:
    target_spacing: tuple | None = None
    hu_range: tuple | None = None
    invert_hu: bool = False
    denoise: bool = False


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str  # 'chan_vese' | 'unet_2d' | 'medsam' | 'sam_med3d' | 'protosam3d'
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MeshSettings:
    algorithm: str = "marching_cubes"
    smoothing_iterations: int = 10
    decimation_factor: float = 0.0  # 0.0 = no decimation, 0.5 = reduce to ~50%
    output_format: str = "glb"  # 'glb' | 'stl' | 'obj' | 'ply'


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str = "results/"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentMeta
    data: DataConfig
    preprocessing: PreprocessingSettings
    algorithm: AlgorithmConfig
    mesh: MeshSettings
    output: OutputConfig

    @property
    def results_dir(self) -> Path:
        return result_dir(self.output.results_dir, self.experiment.id)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Loads and validates a .yaml config. Missing fields fall back to the
    same defaults as configs/default_config.yaml, except 'data.input_path'
    and 'algorithm.name', which are required."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"Invalid YAML in {source}: {exc}") from exc

    data_raw = raw.get("data") or {}
    algorithm_raw = raw.get("algorithm") or {}

    if "input_path" not in data_raw:
        raise ExperimentConfigError(f"{source}: 'data.input_path' is required")
    if "name" not in algorithm_raw:
        raise ExperimentConfigError(f"{source}: 'algorithm.name' is required")

    experiment_raw = raw.get("experiment") or {}
    experiment_id = experiment_raw.get("id") or generate_experiment_id(algorithm_raw["name"])

    preprocessing_raw = raw.get("preprocessing") or {}
    mesh_raw = raw.get("mesh") or {}
    output_raw = raw.get("output") or {}

    try:
        return ExperimentConfig(
            experiment=ExperimentMeta(
                id=experiment_id,
                description=experiment_raw.get("description", ""),
                seed=int(experiment_raw.get("seed", 42)),
                repetitions=int(experiment_raw.get("repetitions", 3)),
            ),
            data=DataConfig(
                input_path=data_raw["input_path"],
                format=data_raw.get("format"),
                reference_mask_path=data_raw.get("reference_mask_path"),
                box=_parse_box(data_raw.get("box"), source),
                seed_zyx=_parse_seed(data_raw.get("seed_zyx"), source),
            ),
            preprocessing=PreprocessingSettings(
                target_spacing=_as_tuple(preprocessing_raw.get("target_spacing")),
                hu_range=_as_tuple(preprocessing_raw.get("hu_range")),
                invert_hu=bool(preprocessing_raw.get("invert_hu", False)),
                denoise=bool(preprocessing_raw.get("denoise", False)),
            ),
            algorithm=AlgorithmConfig(
                name=algorithm_raw["name"],
                parameters=algorithm_raw.get("parameters") or {},
            ),
            mesh=MeshSettings(
                algorithm=mesh_raw.get("algorithm", "marching_cubes"),
                smoothing_iterations=int(mesh_raw.get("smoothing_iterations", 10)),
                decimation_factor=float(mesh_raw.get("decimation_factor", 0.0)),
                output_format=mesh_raw.get("output_format", "glb"),
            ),
            output=OutputConfig(results_dir=output_raw.get("results_dir", "results/")),
        )
    except (TypeError, KeyError) as exc:
        raise ExperimentConfigError(f"{source}: unexpected field - {exc}") from exc


def generate_experiment_id(algorithm_name: str) -> str:
    """algorithm name + timestamp + random suffix - readable and unique
    without a central counter (important for parallel runs)."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{algorithm_name}_{timestamp}_{suffix}"


def result_dir(results_root: str, experiment_id: str) -> Path:
    return Path(results_root) / experiment_id


def prepare_result_dir(config: ExperimentConfig, config_source: Path) -> Path:
    """Creates results/<experiment_id>/ and copies the config used into it
    (reproducibility - so it's always clear which parameters produced a
    given result, even if configs/experiments/ changes later)."""
    directory = config.results_dir
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_source, directory / "config.yaml")
    return directory


def _as_tuple(value) -> tuple | None:
    if value is None:
        return None
    return tuple(value)


def _parse_box(value, source: Path) -> Box | None:
    if value is None:
        return None
    try:
        return Box.from_mapping(value)
    except BoxError as error:
        raise ExperimentConfigError(f"{source}: invalid 'data.box' - {error}") from error


def _parse_seed(value, source: Path) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in value)
    ):
        raise ExperimentConfigError(f"{source}: 'data.seed_zyx' must be three non-negative integers [z, y, x]")
    return tuple(value)
