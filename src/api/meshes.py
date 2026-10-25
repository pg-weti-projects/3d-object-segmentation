from __future__ import annotations

import re
import secrets
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from src.pipeline.mesh import export_mesh, reconstruct_mesh

FORMATS = ("glb", "stl", "obj", "ply")  # first = preferred source and viewer format
MIME_TYPES = {
    "glb": "model/gltf-binary",
    "stl": "model/stl",
    "obj": "model/obj",
    "ply": "application/octet-stream",
}
MASK_SUFFIXES = (".nii.gz", ".nii")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class MeshNotFound(LookupError):
    """No mesh file exists for the requested result id."""


def list_meshes(results_dir: Path) -> list[dict]:
    """Results that contain a mesh file, newest first."""
    found = []
    if not results_dir.is_dir():
        return found
    for directory in results_dir.iterdir():
        if not directory.is_dir() or not _ID_PATTERN.match(directory.name):
            continue
        files = [directory / f"mesh.{fmt}" for fmt in FORMATS if (directory / f"mesh.{fmt}").is_file()]
        if not files:
            continue
        newest = max(file.stat().st_mtime for file in files)
        found.append(
            {
                "id": directory.name,
                "formats": [file.suffix[1:] for file in files],
                "modified": datetime.fromtimestamp(newest),
            }
        )
    return sorted(found, key=lambda item: item["modified"], reverse=True)


def mesh_info(results_dir: Path, mesh_id: str) -> dict:
    """Basic geometry of a result's mesh, for the viewer's info panel."""
    mesh = _load_mesh(_source_file(results_dir, mesh_id))
    return {
        "id": mesh_id,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "extents_mm": [round(float(value), 1) for value in mesh.extents],
        "volume_mm3": round(float(mesh.volume), 1) if mesh.is_watertight else None,
    }


def mesh_bytes(results_dir: Path, mesh_id: str, fmt: str) -> tuple[bytes, str]:
    """The mesh encoded as `fmt` (glb/stl/obj/ply) and its MIME type. A format
    that was not written by the pipeline is converted from the stored one."""
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported mesh format '{fmt}', use one of {FORMATS}")
    stored = Path(results_dir) / _checked_id(mesh_id) / f"mesh.{fmt}"
    if stored.is_file():
        return stored.read_bytes(), MIME_TYPES[fmt]

    exported = _load_mesh(_source_file(results_dir, mesh_id)).export(file_type=fmt)
    if isinstance(exported, str):
        exported = exported.encode("utf-8")
    return exported, MIME_TYPES[fmt]


def build_mesh_from_mask(upload: FileStorage, form, results_dir: Path) -> str:
    """Reconstruct a mesh from an uploaded NIfTI mask, store it as
    ``results/<new id>/mesh.glb`` and return the new id."""
    import nibabel as nib

    filename = secure_filename(upload.filename or "")
    if not filename.lower().endswith(MASK_SUFFIXES):
        raise ValueError("Upload a NIfTI mask (.nii or .nii.gz).")
    smoothing_iterations = _number(form, "smoothing_iterations", 10, int)
    decimation_factor = _number(form, "decimation_factor", 0.2, float)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        upload.save(path)
        try:
            image = nib.load(path)
            array = np.asanyarray(image.dataobj)
        except Exception as error:  # nibabel raises several unrelated types
            raise ValueError(f"Not a readable NIfTI file: {error}") from error
    if array.ndim != 3:
        raise ValueError(f"The mask must be 3-dimensional, got {array.ndim} dimensions.")

    # NIfTI is (x, y, z); the pipeline uses (z, y, x) like core.Volume
    mask = array.transpose(2, 1, 0)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])[::-1]
    mesh = reconstruct_mesh(
        mask,
        spacing,
        smoothing_iterations=smoothing_iterations,
        decimation_factor=decimation_factor,
    )

    mesh_id = f"mask_{time.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(2)}"
    export_mesh(mesh, Path(results_dir) / mesh_id / "mesh.glb")
    return mesh_id


def _number(form, name: str, default, cast):
    # some browsers send "0,2" in a Polish locale: accept a decimal comma
    raw = (form.get(name) or "").strip().replace(",", ".")
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as error:
        raise ValueError(f"'{name}' must be a number.") from error


def _checked_id(mesh_id: str) -> str:
    # The id becomes a directory name: reject anything that could leave results/
    if not _ID_PATTERN.match(mesh_id) or ".." in mesh_id:
        raise MeshNotFound(mesh_id)
    return mesh_id


def _source_file(results_dir: Path, mesh_id: str) -> Path:
    directory = Path(results_dir) / _checked_id(mesh_id)
    for fmt in FORMATS:
        candidate = directory / f"mesh.{fmt}"
        if candidate.is_file():
            return candidate
    raise MeshNotFound(mesh_id)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")
