"""
Triangle-mesh reconstruction from a 3D segmentation mask (task #7).

Flow: binary mask -> keep largest component -> Marching Cubes
      -> repair -> Taubin smoothing -> quadric decimation -> export.

The mask can come from any source: a segmentation model or a
reference mask from data/reference_masks/. Parameter names follow the
`mesh:` section of configs/default_config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

SUPPORTED_FORMATS = ("glb", "stl", "obj", "ply")


def reconstruct_mesh(
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    clean: bool = True,
    smoothing_iterations: int = 10,
    decimation_factor: float = 0.0,
) -> trimesh.Trimesh:
    """Convert a 3D binary mask into a triangle mesh."""
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-dimensional, got {mask.ndim}")
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != 3 or any(value <= 0 or not np.isfinite(value) for value in spacing):
        raise ValueError("spacing must contain three positive finite values")
    if smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be >= 0")
    if not 0.0 <= decimation_factor < 1.0:
        raise ValueError("decimation_factor must be in [0.0, 1.0)")

    binary = mask > 0
    if not binary.any():
        raise ValueError("Mask is empty - nothing to reconstruct.")

    if clean:
        binary = _keep_largest_component(binary)

    # Crop to the object's bounding box and pad with one empty voxel: keeps
    # Marching Cubes cheap on big, mostly-empty scans and guarantees a closed
    # surface even when the object touches the volume border.
    box = ndimage.find_objects(binary.astype(np.uint8))[0]
    start = np.array([s.start for s in box])
    cropped = np.pad(binary[box], 1, mode="constant", constant_values=False)

    verts, faces, _normals, _values = measure.marching_cubes(
        cropped.astype(np.uint8), level=0.5, spacing=spacing
    )
    verts = verts + (start - 1) * np.array(spacing)  # back to full-volume coordinates

    # (z, y, x) -> (x, y, z). Reversing the axis order is a reflection, which
    # turns skimage's winding into outward-facing normals (positive mesh
    # volume; checked in tests/test_mesh.py), so faces are left as they are.
    verts = verts[:, ::-1]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if smoothing_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)

    if decimation_factor > 0.0:
        n_faces = len(mesh.faces)
        target = max(round(n_faces * (1.0 - decimation_factor)), min(n_faces, 100))
        if target < n_faces:
            mesh = mesh.simplify_quadric_decimation(face_count=target)

    return mesh


def reconstruct_mesh_from_settings(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    settings,
) -> trimesh.Trimesh:
    """Same as `reconstruct_mesh`, driven by `experiment.MeshSettings`
    (anything with `smoothing_iterations` and `decimation_factor` works)."""
    return reconstruct_mesh(
        mask,
        spacing,
        smoothing_iterations=settings.smoothing_iterations,
        decimation_factor=settings.decimation_factor,
    )


def export_mesh(mesh: trimesh.Trimesh, output_path: str | Path) -> Path:
    """Write the mesh; the format comes from the extension (glb/stl/obj/ply).
    Creates parent directories, so it can write straight to results/<id>/."""
    path = Path(output_path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported mesh format '{fmt}', use one of {SUPPORTED_FORMATS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    labeled, n_components = ndimage.label(binary)
    if n_components <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
    return labeled == (int(np.argmax(sizes)) + 1)
