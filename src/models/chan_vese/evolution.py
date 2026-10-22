from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

DataTermSign = Literal["paper", "chan_vese"]
TolRegion = Literal["image", "band"]

_CURVATURE_EPS = 1e-8
_TOL_BAND_PX = 1.5


class ChanVeseResult(NamedTuple):
    phi: np.ndarray
    iterations: int
    converged: bool


def heaviside(z: np.ndarray, eps: float) -> np.ndarray:
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(z / eps))


def dirac(z: np.ndarray, eps: float) -> np.ndarray:
    return eps / (np.pi * (eps * eps + z * z))


def curvature(phi: np.ndarray) -> np.ndarray:
    p = np.pad(phi, 1, mode="edge")
    c = p[1:-1, 1:-1]
    xp, xm = p[2:, 1:-1], p[:-2, 1:-1]
    yp, ym = p[1:-1, 2:], p[1:-1, :-2]
    phi_x = xp - xm
    phi_x *= 0.5
    phi_y = yp - ym
    phi_y *= 0.5
    c2 = c + c
    phi_xx = xp + xm
    phi_xx -= c2
    phi_yy = yp + ym
    phi_yy -= c2
    phi_xy = p[2:, 2:] - p[2:, :-2]
    phi_xy -= p[:-2, 2:]
    phi_xy += p[:-2, :-2]
    phi_xy *= 0.25
    gx2 = phi_x * phi_x
    gy2 = phi_y * phi_y
    num = phi_xx * gy2
    phi_yy *= gx2
    num += phi_yy
    phi_x *= phi_y
    phi_x *= phi_xy
    phi_x *= 2.0
    num -= phi_x
    gx2 += gy2
    gx2 += _CURVATURE_EPS
    den = np.sqrt(gx2)
    den *= gx2
    num /= den
    return num


def chan_vese_2d(
    image: np.ndarray,
    phi0: np.ndarray,
    mu: float = 0.1,
    nu: float = 0.0,
    lambda_obj: float = 1.0,
    lambda_bg: float = 1.0,
    eps: float = 1.0,
    dt: float = 0.5,
    tol: float = 1e-3,
    max_iterations: int = 100,
    data_term_sign: DataTermSign = "paper",
    tol_region: TolRegion = "band",
) -> ChanVeseResult:
    bg_sign = -1.0 if data_term_sign == "paper" else 1.0
    if data_term_sign not in ("paper", "chan_vese"):
        raise ValueError(f"data_term_sign={data_term_sign!r}; dozwolone: 'paper', 'chan_vese'")
    if tol_region not in ("image", "band"):
        raise ValueError(f"tol_region={tol_region!r}; dozwolone: 'image', 'band'")
    img = np.asarray(image, dtype=np.float32)
    phi = np.array(phi0, dtype=np.float32, copy=True)
    if img.shape != phi.shape:
        raise ValueError(f"Kształt obrazu {img.shape} różni się od kształtu φ0 {phi.shape}")
    img_flat = img.ravel().astype(np.float64)
    img_sum = float(img_flat.sum())
    n_pixels = float(img.size)
    eps32 = np.float32(eps)

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):  # noqa: B007
        h = heaviside(phi, eps32)
        h_flat = h.ravel().astype(np.float64)
        h_sum = float(h_flat.sum())
        ih_sum = float(img_flat @ h_flat)
        bg_weight = n_pixels - h_sum
        b_obj = ih_sum / h_sum if h_sum > 1e-12 else 0.0
        b_bg = (img_sum - ih_sum) / bg_weight if bg_weight > 1e-12 else 0.0
        force = curvature(phi)
        force *= np.float32(mu)
        if nu:
            force -= np.float32(nu)
        d_obj = img - np.float32(b_obj)
        d_obj *= d_obj
        d_obj *= np.float32(lambda_obj)
        force -= d_obj
        d_bg = img - np.float32(b_bg)
        d_bg *= d_bg
        d_bg *= np.float32(bg_sign * lambda_bg)
        force += d_bg
        change = dirac(phi, eps32)
        change *= force
        change *= np.float32(dt)
        abs_change = np.abs(change)
        if tol_region == "image":
            mean_change = float(abs_change.mean(dtype=np.float64))
        else:
            band = np.abs(phi) <= _TOL_BAND_PX
            mean_change = float(abs_change[band].mean(dtype=np.float64)) if band.any() else 0.0
        phi += change
        if mean_change < tol:
            converged = True
            break
    return ChanVeseResult(phi=phi, iterations=iterations, converged=converged)
