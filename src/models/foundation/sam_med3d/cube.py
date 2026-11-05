"""
Geometry of the 128^3 cube: where it goes, what it samples, and how its mask goes
back onto the scan. No torch, and in (z, y, x) like the rest of the project; only
`to_network_axes` and `network_click` know the network's (x, y, z).

Voxel i of an axis with spacing s covers [i * s, (i + 1) * s) mm (as in core/box.py),
so its centre is at (i + 0.5) * s. Both the scan and the cube follow this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 128  # fixed by the checkpoint: image_encoder.pos_embed is (1, 8, 8, 8, 768), patches of 16


@dataclass(frozen=True)
class Cube:
    corner_mm: tuple[float, float, float]  # (z, y, x) of the low corner of voxel (0, 0, 0)
    spacing_mm: float

    @property
    def side_mm(self) -> float:
        return SIZE * self.spacing_mm

    def bounds_mm(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        low = tuple(float(c) for c in self.corner_mm)
        return low, tuple(c + self.side_mm for c in low)

    def centres_mm(self, axis: int) -> np.ndarray:
        return self.corner_mm[axis] + (np.arange(SIZE) + 0.5) * self.spacing_mm


def voxel_centre_mm(voxel_zyx, spacing_zyx) -> tuple[float, float, float]:
    return tuple((v + 0.5) * s for v, s in zip(voxel_zyx, spacing_zyx))


def cube_on_box(box_min_mm, box_max_mm, margin: float = 0.0) -> Cube:
    """Centred on the box, isotropic, with its longest side (plus `margin` of it) filling the cube."""
    extent = [high - low for low, high in zip(box_min_mm, box_max_mm)]
    spacing = max(extent) * (1.0 + margin) / SIZE
    centre = [(low + high) / 2 for low, high in zip(box_min_mm, box_max_mm)]
    return Cube(tuple(c - SIZE * spacing / 2 for c in centre), float(spacing))


def sample(data: np.ndarray, spacing_zyx, cube: Cube) -> tuple[np.ndarray, np.ndarray]:
    """The scan on the cube's grid by trilinear interpolation, and where the cube lies in the scan.

    Interpolation stays inside the scan: from the last voxel centre to the scan's edge the
    edge value is held, and past the edge a voxel is simply outside. Padding is added
    afterwards, to voxels outside, so a voxel is never a blend of scan and padding."""
    positions, inside_axis = [], []
    for axis, (size, step) in enumerate(zip(data.shape, spacing_zyx)):
        mm = cube.centres_mm(axis)
        positions.append(mm / step - 0.5)
        inside_axis.append((mm >= 0) & (mm < size * step))
    inside = inside_axis[0][:, None, None] & inside_axis[1][None, :, None] & inside_axis[2][None, None, :]
    values = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    if not inside.any():
        return values, inside

    keep = [np.flatnonzero(flags) for flags in inside_axis]
    lows, highs, weights = [], [], []
    for axis, size in enumerate(data.shape):
        position = np.clip(positions[axis][keep[axis]], 0, size - 1)
        low = np.floor(position).astype(int)
        lows.append(low)
        highs.append(np.minimum(low + 1, size - 1))
        weights.append(position - low)
    start = [int(low.min()) for low in lows]
    stop = [int(high.max()) + 1 for high in highs]
    block = np.asarray(data[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]], dtype=np.float64)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = -1
        block = np.take(block, lows[axis] - start[axis], axis=axis) * (1 - weights[axis].reshape(shape)) + np.take(
            block, highs[axis] - start[axis], axis=axis
        ) * weights[axis].reshape(shape)
    values[np.ix_(*keep)] = block
    return values, inside


def paste_back(cube_mask: np.ndarray, cube: Cube, shape, spacing_zyx) -> np.ndarray:
    """The cube's mask on the scan's grid (nearest cube voxel to each scan voxel centre);
    scan voxels outside the cube are 0."""
    scan_axes, cube_axes = [], []
    for axis, (size, step) in enumerate(zip(shape, spacing_zyx)):
        centres = (np.arange(size) + 0.5) * step
        index = np.floor((centres - cube.corner_mm[axis]) / cube.spacing_mm).astype(int)
        valid = np.flatnonzero((index >= 0) & (index < SIZE))
        scan_axes.append(valid)
        cube_axes.append(index[valid])
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    if all(axis.size for axis in scan_axes):
        mask[np.ix_(*scan_axes)] = np.asarray(cube_mask)[np.ix_(*cube_axes)] != 0
    return mask


def to_network_axes(cube_zyx: np.ndarray) -> np.ndarray:
    """(z, y, x) to the (x, y, z) the network expects; the same call converts back."""
    return np.asarray(cube_zyx).transpose(2, 1, 0)


def network_click(click_mm, cube: Cube) -> tuple[int, int, int]:
    """The cube voxel holding the click, as the network's (x, y, z)."""
    voxel = [int(np.floor((c - low) / cube.spacing_mm)) for c, low in zip(click_mm, cube.corner_mm)]
    if any(not 0 <= v < SIZE for v in voxel):
        raise ValueError(f"The click at {tuple(round(c, 2) for c in click_mm)} mm lies outside the cube")
    return voxel[2], voxel[1], voxel[0]


def coverage(cube: Cube, shape, spacing_zyx, box_mm=None) -> dict:
    """How the cube and the scan (and the box, if any) overlap. Shares are of volume."""
    low, high = cube.bounds_mm()
    overlap, extent = [], []
    for axis, (size, step) in enumerate(zip(shape, spacing_zyx)):
        extent.append(size * step)
        overlap.append(max(min(high[axis], extent[axis]) - max(low[axis], 0.0), 0.0))
    side = cube.side_mm
    in_scan = min(float(np.prod([o / side for o in overlap])), 1.0)  # rounding can pass 1 by a hair
    report = {
        "view_mm": {"min_zyx": list(low), "max_zyx": list(high)},
        "scan_extent_mm_zyx": extent,
        "cube_share_in_scan": in_scan,
        "padding_share_of_cube": 1.0 - in_scan,
        "scan_share_in_cube": float(np.prod([o / e for o, e in zip(overlap, extent)])),
        "scan_axes_longer_than_cube": ["zyx"[a] for a in range(3) if extent[a] > side],
    }
    if box_mm is not None:
        box_low, box_high = box_mm
        size = [b - a for a, b in zip(box_low, box_high)]
        in_cube = [max(min(b, h) - max(a, l), 0.0) for a, b, l, h in zip(box_low, box_high, low, high)]
        in_scan = [max(min(b, e) - max(a, 0.0), 0.0) for a, b, e in zip(box_low, box_high, extent)]
        report["box_share_in_cube"] = float(np.prod([i / s for i, s in zip(in_cube, size)]))
        report["box_share_in_scan"] = float(np.prod([i / s for i, s in zip(in_scan, size)]))
    return report


def faces_touched_inside_scan(cube_mask: np.ndarray, cube: Cube, shape, spacing_zyx) -> list[str]:
    """Cube faces the mask reaches where the scan goes on past the face: there the target may be cut off."""
    low, high = cube.bounds_mm()
    touched = []
    for axis, (size, step) in enumerate(zip(shape, spacing_zyx)):
        first = np.take(cube_mask, 0, axis=axis).any()
        last = np.take(cube_mask, SIZE - 1, axis=axis).any()
        if first and low[axis] > 0:
            touched.append("low " + "zyx"[axis])
        if last and high[axis] < size * step:
            touched.append("high " + "zyx"[axis])
    return touched
