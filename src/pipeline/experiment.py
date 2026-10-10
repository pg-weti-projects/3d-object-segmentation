"""
Experiment configuration (.yaml) loading and the results/<experiment_id>/
directory convention. Used by the runner, mesh reconstruction,
metrics, and eventually the frontend (to know where to find results).

results/<experiment_id>/ convention:
    mask.nii.gz          - segmentation mask
    mesh.<output_format>  - reconstructed mesh (format from mesh.output_format)
    metrics.json           - metrics report (only if reference_mask_path is set)
    config.yaml              - copy of the config used for this run (reproducibility)
    run.log                   - execution log
"""

from __future__ import annotations

import random
import shutil
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_RESULTS_ROOT = "results/"


class ExperimentConfigError(ValueError):
    """Raised when a config file is missing required fields or malformed."""


@dataclass(frozen=True)
class DataConfig:
    input_path: str
    format: str | None = None
    reference_mask_path: str | None = None


@dataclass(frozen=True)
class PreprocessingSettings:
    target_spacing: tuple | None = None
    hu_range: tuple | None = None
    invert_hu: bool = False
    denoise: bool = False


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MeshSettings:
    algorithm: str = "marching_cubes"
    smoothing_iterations: int = 15
    decimation_factor: float = 0.0
    output_format: str = "glb"


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    seed: int
    repetitions: int
    data: DataConfig
    preprocessing: PreprocessingSettings
    algorithm: AlgorithmConfig
    mesh: MeshSettings
    results_root: str

    @property
    def results_dir(self) -> Path:
        return Path(self.results_root) / self.experiment_id


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a .yaml config. Missing optional fields fall back to
    the defaults shown in configs/default_config.yaml. Required: data.input_path
    and algorithm.name."""
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
            experiment_id=experiment_id,
            description=experiment_raw.get("description", ""),
            seed=int(experiment_raw.get("seed", 42)),
            repetitions=int(experiment_raw.get("repetitions", 3)),
            data=DataConfig(
                input_path=data_raw["input_path"],
                format=data_raw.get("format"),
                reference_mask_path=data_raw.get("reference_mask_path"),
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
                smoothing_iterations=int(mesh_raw.get("smoothing_iterations", 15)),
                decimation_factor=float(mesh_raw.get("decimation_factor", 0.0)),
                output_format=mesh_raw.get("output_format", "glb"),
            ),
            results_root=output_raw.get("results_dir", DEFAULT_RESULTS_ROOT),
        )
    except (TypeError, KeyError) as exc:
        raise ExperimentConfigError(f"{source}: unexpected or missing field - {exc}") from exc


def generate_experiment_id(algorithm_name: str) -> str:
    """algorithm name + timestamp + random suffix - readable and unique
    without a central counter (important for parallel runs)."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{algorithm_name}_{timestamp}_{suffix}"


def prepare_result_dir(config: ExperimentConfig, config_source: Path) -> Path:
    """Creates results/<experiment_id>/ and copies the config used into it
    (reproducibility - so it's always clear which parameters produced a given
    result, even if configs/experiments/ changes later)."""
    directory = config.results_dir
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_source, directory / "config.yaml")
    return directory


def _as_tuple(value) -> tuple | None:
    if value is None:
        return None
    return tuple(value)
