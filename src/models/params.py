from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "range_estimator": "percentile",
    "roi_radius_mm": 3.0,
    "intensity_levels": None,
    "median_size_mm": 1.0,
    "sigmoid_gain": 10.0,
    "remove_outside_air": False,
    "init_size_px": 4,
    "init_step_px": 1,
    "init_anchor": "corner",
    "init_band_px": 1,
    "data_term_sign": "paper",
    "mu": 0.1,
    "nu": 0.0,
    "lambda_obj": 1.0,
    "lambda_bg": 1.0,
    "eps_mm": 1.0,
    "dt": 0.5,
    "tol": 1.0e-3,
    "tol_region": "band",
    "max_iterations": 100,
    "n_jobs": 1,
}

_CHOICES: dict[str, tuple[str, ...]] = {
    "range_estimator": ("percentile", "minmax"),
    "init_anchor": ("corner", "center"),
    "data_term_sign": ("paper", "chan_vese"),
    "tol_region": ("image", "band"),
}
PARAM_CHOICES: dict[str, tuple[str, ...]] = _CHOICES


def resolve_params(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS)
    if cfg:
        unknown = sorted(str(k) for k in cfg if k not in DEFAULT_PARAMS)
        if unknown:
            logger.warning(
                "Nieznane parametry algorytmu (literówka?): %s — zostaną zignorowane; znane klucze: %s",
                unknown,
                sorted(DEFAULT_PARAMS),
            )
        params.update(cfg)
    for key, allowed in _CHOICES.items():
        if params[key] not in allowed:
            raise ValueError(f"Parametr {key}={params[key]!r}; dozwolone: {allowed}")
    if params["intensity_levels"] is not None and int(params["intensity_levels"]) < 2:
        raise ValueError("intensity_levels musi być None albo >= 2")
    if int(params["init_size_px"]) < 1 or int(params["init_step_px"]) < 1 or int(params["init_band_px"]) < 1:
        raise ValueError("init_size_px, init_step_px i init_band_px muszą być >= 1")
    if int(params["max_iterations"]) < 1:
        raise ValueError("max_iterations musi być >= 1")
    return params


CHAN_VESE = "chan_vese"
GEODESIC_ACTIVE_CONTOUR = "geodesic_active_contour"
GROWCUT = "growcut"
ALGORITHMS: tuple[str, ...] = (CHAN_VESE, GEODESIC_ACTIVE_CONTOUR, GROWCUT)

GAC_DEFAULT_PARAMS: dict[str, Any] = {
    "domain_size_mm": 80.0,
    "compute_spacing_mm": 0.5,
    "gac_alpha": 100.0,
    "gac_sigma_mm": 1.0,
    "init_radius_mm": 3.0,
    "max_iterations": 300,
    "smoothing": 1,
    "threshold": "auto",
    "balloon": 1.0,
    "stable_iterations": 5,
}

GROWCUT_DEFAULT_PARAMS: dict[str, Any] = {
    "domain_size_mm": 80.0,
    "compute_spacing_mm": 0.5,
    "fg_radius_mm": 3.0,
    "bg_radius_mm": None,
    "neighborhood": "von_neumann",
    "max_iterations": 1000,
    "smooth_t1": None,
    "smooth_t2": None,
}

GROWCUT_NEIGHBORHOODS: tuple[str, ...] = ("von_neumann", "moore")

ALGORITHM_DEFAULTS: dict[str, dict[str, Any]] = {
    GEODESIC_ACTIVE_CONTOUR: GAC_DEFAULT_PARAMS,
    GROWCUT: GROWCUT_DEFAULT_PARAMS,
}


def algorithm_name(name: str) -> str:
    if name not in ALGORITHMS:
        raise ValueError(
            f"algorithm.name={name!r}: nieznana metoda analityczna; dozwolone: {list(ALGORITHMS)}"
        )
    return str(name)


def _positive(value: Any, key: str, *, allow_none: bool = False, allow_zero: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key}={value!r}: oczekiwano liczby" + (" albo null" if allow_none else ""))
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key}={value!r}: musi być {'>= 0' if allow_zero else '> 0'}")
    return float(value)


def _integer(value: Any, key: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key}={value!r}: oczekiwano liczby całkowitej >= {minimum}")
    return int(value)


def _merge_strict(cfg: Mapping[str, Any] | None, name: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    section: Any = cfg or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"algorithm.parameters dla {name} musi być słownikiem")
    unknown = sorted(str(k) for k in section if k not in defaults)
    if unknown:
        raise ValueError(f"nieznane parametry {name} {unknown} (literówka?); dozwolone: {sorted(defaults)}")
    return {**defaults, **section}


def _check_grid_vs_domain(out: Mapping[str, Any]) -> None:
    dom, cs = out["domain_size_mm"], out["compute_spacing_mm"]
    if dom is not None and cs is not None and cs >= dom:
        raise ValueError(f"compute_spacing_mm={cs} musi być mniejszy od domain_size_mm={dom}")


def resolve_gac_params(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    p = _merge_strict(cfg, GEODESIC_ACTIVE_CONTOUR, GAC_DEFAULT_PARAMS)
    out: dict[str, Any] = {
        "domain_size_mm": _positive(p["domain_size_mm"], "domain_size_mm", allow_none=True),
        "compute_spacing_mm": _positive(p["compute_spacing_mm"], "compute_spacing_mm", allow_none=True),
        "gac_alpha": _positive(p["gac_alpha"], "gac_alpha"),
        "gac_sigma_mm": _positive(p["gac_sigma_mm"], "gac_sigma_mm", allow_zero=True),
        "init_radius_mm": _positive(p["init_radius_mm"], "init_radius_mm"),
        "max_iterations": _integer(p["max_iterations"], "max_iterations", 1),
        "smoothing": _integer(p["smoothing"], "smoothing", 0),
        "stable_iterations": _integer(p["stable_iterations"], "stable_iterations", 1),
    }
    threshold = p["threshold"]
    out["threshold"] = (
        "auto"
        if threshold == "auto"
        else _positive(threshold, "threshold ('auto' albo liczba)", allow_zero=True)
    )
    balloon = p["balloon"]
    if isinstance(balloon, bool) or not isinstance(balloon, (int, float)) or not math.isfinite(balloon):
        raise ValueError(f"balloon={balloon!r}: oczekiwano liczby (+ rozszerzanie, − kurczenie, 0 = brak)")
    out["balloon"] = float(balloon)
    _check_grid_vs_domain(out)
    return out


def resolve_growcut_params(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    p = _merge_strict(cfg, GROWCUT, GROWCUT_DEFAULT_PARAMS)
    out: dict[str, Any] = {
        "domain_size_mm": _positive(p["domain_size_mm"], "domain_size_mm", allow_none=True),
        "compute_spacing_mm": _positive(p["compute_spacing_mm"], "compute_spacing_mm", allow_none=True),
        "fg_radius_mm": _positive(p["fg_radius_mm"], "fg_radius_mm", allow_zero=True),
        "bg_radius_mm": _positive(p["bg_radius_mm"], "bg_radius_mm", allow_none=True),
        "max_iterations": _integer(p["max_iterations"], "max_iterations", 1),
    }
    if p["neighborhood"] not in GROWCUT_NEIGHBORHOODS:
        raise ValueError(f"neighborhood={p['neighborhood']!r}; dozwolone: {GROWCUT_NEIGHBORHOODS}")
    out["neighborhood"] = str(p["neighborhood"])
    for key in ("smooth_t1", "smooth_t2"):
        out[key] = None if p[key] is None else _integer(p[key], f"{key} (albo null)", 1)
    if out["bg_radius_mm"] is not None and out["bg_radius_mm"] <= out["fg_radius_mm"]:
        raise ValueError(
            f"bg_radius_mm={out['bg_radius_mm']} musi być większe od fg_radius_mm={out['fg_radius_mm']}"
        )
    _check_grid_vs_domain(out)
    return out


def resolve_algorithm_params(name: str, cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if name == GEODESIC_ACTIVE_CONTOUR:
        return resolve_gac_params(cfg)
    if name == GROWCUT:
        return resolve_growcut_params(cfg)
    if name == CHAN_VESE:
        return resolve_params(cfg)
    raise ValueError(f"nieznana metoda analityczna {name!r}; dozwolone: {list(ALGORITHMS)}")
