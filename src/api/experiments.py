from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
MESH_FORMATS = ("glb", "stl", "obj", "ply")

# The columns of the table, in order: key, label, how to format, is more better?
METRICS: tuple[tuple[str, str, str, bool | None], ...] = (
    ("dice", "Dice", "ratio", True),
    ("iou", "IoU", "ratio", True),
    ("vs", "Volume sim.", "ratio", True),
    ("hd95_mm", "HD95", "mm", False),
    ("assd_mm", "ASSD", "mm", False),
    ("volume_pred_mm3", "Volume (pred.)", "mm3", None),
    ("volume_ref_mm3", "Volume (ref.)", "mm3", None),
    ("runtime_s", "Run time", "seconds", False),
)
CHARTABLE = tuple(key for key, _label, _kind, better in METRICS if better is not None)
_KIND = {key: kind for key, _label, kind, _better in METRICS}
_BETTER = {key: better for key, _label, _kind, better in METRICS}
_LABEL = {key: label for key, label, _kind, _better in METRICS}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _dataset_of(config: dict[str, Any]) -> str:
    """The scan a run was made from, as a name a reader recognises."""
    path = str(_section(config, "data").get("input_path", "") or "")
    if not path:
        return "-"
    parts = [part for part in Path(path).parts if part not in (".", "/")]
    for part in reversed(parts):  # the scan folder, not "density" or a series name
        if part not in ("density", "raw", "data"):
            return part
    return parts[-1] if parts else "-"


def _specimen_of(config: dict[str, Any]) -> str:
    """The specimen a run is about, named after the reference mask it is scored
    against rather than after the scan.

    One fish reaches the runner under several file names - a DICOM folder, a
    NIfTI export of that folder, a resampled copy - which spreads a single
    specimen over several labels and makes its methods impossible to compare at
    a glance. The masks keep one folder per fish, so that folder is the name.
    """
    reference = str(_section(config, "data").get("reference_mask_path", "") or "")
    if reference:
        folder = Path(reference).parent.name
        if folder and folder not in ("reference_masks", "data"):
            return folder
    return _dataset_of(config)


def read_experiment(directory: Path) -> dict[str, Any]:
    """One run, as the page needs it."""
    config = _read_yaml(directory / "config.yaml")
    metrics = _read_json(directory / "metrics.json") or {}
    stamp = max(
        (item.stat().st_mtime for item in directory.iterdir() if item.is_file()),
        default=directory.stat().st_mtime,
    )
    return {
        "id": directory.name,
        "dataset": _specimen_of(config),
        "algorithm": str(_section(config, "algorithm").get("name", "") or "-"),
        "repetitions": _section(config, "experiment").get("repetitions"),
        "metrics": {key: metrics.get(key) for key in _KIND},
        "box_prompted": metrics.get("box_prompted"),
        "has_metrics": bool(metrics),
        "meshes": [fmt for fmt in MESH_FORMATS if (directory / f"mesh.{fmt}").is_file()],
        "modified": datetime.fromtimestamp(stamp),
    }


def list_experiments(results_dir: Path) -> list[dict[str, Any]]:
    """Every run in ``results_dir``, newest first."""
    if not results_dir.is_dir():
        return []
    runs = [
        read_experiment(directory)
        for directory in sorted(results_dir.iterdir())
        if directory.is_dir() and _ID_PATTERN.match(directory.name)
    ]
    return sorted(runs, key=lambda run: run["modified"], reverse=True)


def format_value(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if kind == "ratio":
        return f"{number:.3f}"
    if kind == "mm":
        return f"{number:.2f} mm"
    if kind == "mm3":
        return f"{number / 1000:.1f} mL"
    if kind == "seconds":
        return f"{number:.2f} s" if number < 60 else f"{number / 60:.1f} min"
    return f"{number:g}"


def comparison(runs: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """The runs that have this metric, best first - which for HD95 or a run time
    means the smallest."""
    rows = []
    for run in runs:
        value = run["metrics"].get(metric)
        if value is None:
            continue
        try:
            rows.append({**run, "value": float(value)})
        except (TypeError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["value"], reverse=bool(_BETTER.get(metric, True)))


MAX_LABEL = 30


def _shorten(text: str, limit: int = MAX_LABEL) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def bar_chart(rows: list[dict[str, Any]], metric: str, width: int = 860, bar: int = 26) -> str:
    """A horizontal bar chart as inline SVG - built here so the page needs no
    chart library and still draws with scripts blocked.

    The bars are grouped by specimen, so that what sits next to each other is
    what was run on the same fish. Every bar is measured against the same scale,
    so the groups stay comparable with one another.
    """
    if not rows:
        return ""
    kind = _KIND.get(metric, "ratio")
    label_width, right, indent, heading = 250, 110, 14, 26
    span = max(width - label_width - right, 10)
    highest = max(row["value"] for row in rows)
    scale = 1.0 if (kind == "ratio" and highest <= 1.0) else (highest or 1.0)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:  # rows arrive best first, so the groups do too
        groups.setdefault(row["dataset"], []).append(row)
    height = len(rows) * (bar + 10) + len(groups) * heading + 16

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" '
        f'aria-label="{_escape(_LABEL.get(metric, metric))} per run, grouped by specimen" '
        f'class="metric-chart">'
    ]
    top = 8
    for specimen, members in groups.items():
        parts.append(
            f'<text x="0" y="{top + 15}" class="metric-chart-group">{_escape(specimen)}</text>'
        )
        top += heading
        for row in members:
            baseline = top + bar * 0.7
            length = max(2, round(span * (row["value"] / scale))) if scale else 2
            parts.append(
                f'<text x="{indent}" y="{baseline:.0f}" class="metric-chart-label">'
                f'{_escape(_shorten(row["algorithm"]))}</text>'
                f'<rect x="{label_width}" y="{top}" width="{length}" height="{bar}" rx="3" '
                f'class="metric-chart-bar"></rect>'
                f'<text x="{label_width + length + 8}" y="{baseline:.0f}" class="metric-chart-value">'
                f'{_escape(format_value(row["value"], kind))}</text>'
            )
            top += bar + 10
    parts.append("</svg>")
    return "".join(parts)


def _escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def page_context(results_dir: Path, metric: str | None = None) -> dict[str, Any]:
    """Everything the template needs."""
    chosen = metric if metric in CHARTABLE else CHARTABLE[0]
    runs = list_experiments(results_dir)
    rows = comparison(runs, chosen)
    return {
        "runs": runs,
        "columns": METRICS,
        "metric": chosen,
        "metric_label": _LABEL[chosen],
        "lower_is_better": not _BETTER[chosen],
        "chartable": [(key, _LABEL[key]) for key in CHARTABLE],
        "chart": bar_chart(rows, chosen),
        "format_value": format_value,
    }
