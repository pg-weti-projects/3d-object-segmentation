from __future__ import annotations

from typing import Any, Literal

import numpy as np

from src.models.result import signed_distance

Anchor = Literal["corner", "center"]


def contour_force(p: float, u: float, v: float) -> float:
    return float((p - u) ** 2 + (p - v) ** 2)


def _rect_bounds(center: int, size: int, anchor: Anchor) -> tuple[int, int]:
    if anchor == "corner":
        lo = center
    else:
        lo = center - size // 2
    return lo, lo + size


def auto_init_contour(
    binary_slice: np.ndarray,
    centroid: tuple[int, int],
    init_size_px: int = 4,
    step_px: int = 1,
    anchor: Anchor = "corner",
    band_px: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    img = np.asarray(binary_slice, dtype=np.float64)
    nx, ny = img.shape
    xc, yc = int(centroid[0]), int(centroid[1])
    if not (0 <= xc < nx and 0 <= yc < ny):
        raise ValueError(f"Środek ciężkości {centroid} poza obrazem {img.shape}")

    integral = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    integral[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)

    def box(x0: int, x1: int, y0: int, y1: int) -> tuple[float, int]:
        x0, x1 = max(x0, 0), min(x1, nx)
        y0, y1 = max(y0, 0), min(y1, ny)
        if x1 <= x0 or y1 <= y0:
            return 0.0, 0
        s = integral[x1, y1] - integral[x0, y1] - integral[x1, y0] + integral[x0, y0]
        return float(s), (x1 - x0) * (y1 - y0)

    p_value = float(img[xc, yc])
    size = int(init_size_px)
    steps = 0
    warnings: list[str] = []
    touched_edge = False
    f_value = u_mean = v_mean = float("nan")
    while True:
        x0, x1 = _rect_bounds(xc, size, anchor)
        y0, y1 = _rect_bounds(yc, size, anchor)
        touched_edge = x0 <= 0 or y0 <= 0 or x1 >= nx or y1 >= ny
        rect_sum, rect_n = box(x0, x1, y0, y1)
        core_sum, core_n = box(x0 + band_px, x1 - band_px, y0 + band_px, y1 - band_px)
        outer_sum, outer_n = box(x0 - band_px, x1 + band_px, y0 - band_px, y1 + band_px)
        inner_n = rect_n - core_n
        ring_n = outer_n - rect_n
        u_mean = (rect_sum - core_sum) / inner_n if inner_n > 0 else float("nan")
        v_mean = (outer_sum - rect_sum) / ring_n if ring_n > 0 else float("nan")
        f_value = contour_force(p_value, u_mean, v_mean) if inner_n > 0 and ring_n > 0 else float("nan")
        if np.isfinite(f_value) and f_value >= 1.0:
            break
        if touched_edge:
            msg = (
                f"Prostokąt startowy dotknął krawędzi obrazu przy rozmiarze {size} px (F={f_value:.3f} < 1); "
                "wzrost przerwany"
            )
            warnings.append(msg)
            break
        size += int(step_px)
        steps += 1

    inside = np.zeros((nx, ny), dtype=bool)
    inside[max(x0, 0) : min(x1, nx), max(y0, 0) : min(y1, ny)] = True
    phi0 = signed_distance(inside)
    if p_value == 0.0:
        warnings.append("P = 0: środek ciężkości leży w tle obrazu binarnego (obiekt wklęsły?)")
    info = {
        "size_px": size,
        "rect": [max(x0, 0), min(x1, nx), max(y0, 0), min(y1, ny)],
        "steps": steps,
        "force": None if not np.isfinite(f_value) else f_value,
        "p": p_value,
        "u": None if not np.isfinite(u_mean) else float(u_mean),
        "v": None if not np.isfinite(v_mean) else float(v_mean),
        "anchor": anchor,
        "touched_edge": bool(touched_edge),
        "warnings": warnings,
    }
    return phi0, info
