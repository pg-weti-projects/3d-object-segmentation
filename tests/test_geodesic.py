from __future__ import annotations

import logging

import numpy as np
import pytest

from src.models.chan_vese import ImprovedChanVese
from src.models.factory import make_segmenter
from src.models.geodesic import (
    GeodesicActiveContour,
    compute_domain,
    compute_grid,
    compute_grid_spacing,
    evolve,
    init_ball,
    resolve_threshold,
)
from src.models.grid import SeedPoint, WorkVolume
from src.models.params import GAC_DEFAULT_PARAMS, algorithm_name, resolve_algorithm_params, resolve_gac_params
from src.models.result import unprocessed_phi_value
from src.models.roi import RoiResult


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * float((a & b).sum()) / float(a.sum() + b.sum())


def sphere_volume(
    n: int, spacing: float, radius_mm: float, kind: str, noise: float = 0.02
) -> tuple[WorkVolume, np.ndarray, tuple[int, int, int]]:
    idx = np.indices((n, n, n), dtype=np.float32)
    c = (n - 1) / 2.0
    r_mm = np.sqrt(((idx - c) ** 2).sum(axis=0)) * spacing
    del idx
    inside = r_mm < radius_mm
    data = np.full((n, n, n), 0.8, dtype=np.float32)
    if kind == "wall":
        data[(r_mm >= radius_mm) & (r_mm < radius_mm + 1.0)] = 1.0
    else:
        data[inside] = 0.0
    data += np.random.default_rng(0).normal(0.0, noise, data.shape).astype(np.float32)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    center = (n // 2, n // 2, n // 2)
    return WorkVolume(data=data, affine=affine, intensity_unit="normalized"), inside, center


def roi_at(voxel: tuple[int, int, int]) -> RoiResult:
    seed = SeedPoint("manual", voxel, tuple(float(v) for v in voxel))
    return RoiResult(seed=seed, roi_mask=np.zeros((0, 0, 0), dtype=bool), tmin=0.0, tmax=1.0, estimator="x")


class TestSegmentation:
    def test_wall_only_bladder_like_cod(self):
        volume, inside, center = sphere_volume(48, 0.5, 7.0, "wall")
        res = GeodesicActiveContour({"domain_size_mm": None}).segment(volume, roi_at(center))
        assert res.status == "ok"
        assert dice(res.mask.data, inside) > 0.7, res.params_used
        assert res.params_used["converged"] is True
        assert (
            res.params_used["touches_domain_boundary"] == [] and res.params_used["touches_volume_wall"] == []
        )

    def test_gas_bladder_with_resampled_compute_grid(self):
        volume, inside, center = sphere_volume(64, 0.25, 4.5, "gas")
        res = GeodesicActiveContour({"compute_spacing_mm": 0.5, "gac_sigma_mm": 0.5}).segment(
            volume, roi_at(center)
        )
        used = res.params_used
        assert used["compute_spacing_mm_used"] == 0.5 and used["compute_shape"] == [32, 32, 32]
        assert dice(res.mask.data, inside) > 0.8, used
        assert res.phi.dtype == np.float32 and res.phi.shape == volume.shape
        np.testing.assert_array_equal(res.phi > 0, res.mask.data)
        assert used["volume_mm3"] == pytest.approx(res.mask.data.sum() * 0.25**3)

    def test_domain_clipped_at_volume_border(self):
        volume, _, _ = sphere_volume(40, 0.5, 4.0, "gas")
        seed = (2, 20, 20)
        domain = compute_domain(volume.shape, volume.spacing, seed, 10.0)
        assert domain.clipped is True
        assert domain.lo == (0, 10, 10) and domain.hi == (13, 31, 31)
        res = GeodesicActiveContour({"domain_size_mm": 10.0, "max_iterations": 20}).segment(
            volume, roi_at(seed)
        )
        used = res.params_used
        assert used["domain"]["lo_voxel"] == [0, 10, 10] and used["domain"]["clipped_to_volume"] is True
        outside = np.ones(volume.shape, dtype=bool)
        outside[domain.slices] = False
        assert not res.mask.data[outside].any()
        assert np.all(res.phi[outside] == unprocessed_phi_value(volume.shape))

    def test_mask_touching_domain_boundary_is_reported(self, caplog):
        volume, _, center = sphere_volume(48, 0.5, 9.0, "gas")
        with caplog.at_level(logging.WARNING, logger="src.models.geodesic"):
            res = GeodesicActiveContour({"domain_size_mm": 6.0, "compute_spacing_mm": None}).segment(
                volume, roi_at(center)
            )
        assert len(res.params_used["touches_domain_boundary"]) == 6
        assert any("za mały" in w for w in res.params_used["warnings"])
        assert "za mały" in caplog.text

    def test_too_small_compute_grid_is_clear_error(self):
        volume, _, center = sphere_volume(20, 1.0, 4.0, "gas")
        with pytest.raises(ValueError, match="siatka obliczeń"):
            GeodesicActiveContour({"domain_size_mm": 2.5, "compute_spacing_mm": 2.0}).segment(
                volume, roi_at(center)
            )
        with pytest.raises(ValueError, match="compute_spacing_mm"):
            resolve_gac_params({"domain_size_mm": 5.0, "compute_spacing_mm": 5.0})
        assert (
            resolve_gac_params({"domain_size_mm": None, "compute_spacing_mm": 100.0})["compute_spacing_mm"]
            == 100.0
        )

    def test_no_resampling_when_compute_spacing_coarser_than_requested(self):
        assert compute_grid_spacing((1.0, 1.0, 1.0), 0.5) == 1.0
        assert compute_grid_spacing((0.2, 0.2, 0.2), 0.5) == 0.5
        assert compute_grid_spacing((0.8, 0.8, 3.0), None) == 0.8
        data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        dom = compute_domain((3, 3, 3), (1.0, 1.0, 1.0), (1, 1, 1), None)
        img, step = compute_grid(data, dom, (1.0, 1.0, 1.0), 1.0)
        np.testing.assert_array_equal(img, data)
        np.testing.assert_array_equal(step, [1.0, 1.0, 1.0])


class TestEvolution:
    def test_early_stop_on_convergence(self):
        volume, _, center = sphere_volume(40, 0.5, 5.0, "gas")
        res = GeodesicActiveContour({"max_iterations": 300, "stable_iterations": 3}).segment(
            volume, roi_at(center)
        )
        assert res.params_used["converged"] is True
        assert res.params_used["iterations"] < 300

    def test_iteration_limit_without_convergence(self):
        volume, _, center = sphere_volume(40, 0.5, 5.0, "gas")
        res = GeodesicActiveContour({"max_iterations": 2}).segment(volume, roi_at(center))
        assert res.params_used["iterations"] == 2 and res.params_used["converged"] is False
        assert any("brak zbieżności" in w for w in res.params_used["warnings"])

    def test_period_two_oscillation_counts_as_stable(self):
        g = np.ones((12, 12, 12), dtype=np.float32)
        init = init_ball(g.shape, (6, 6, 6), 3.0)
        evo = evolve(g, init, max_iterations=50, smoothing=0, threshold=0.5, balloon=0.0, stable_iterations=4)
        assert evo.converged is True and evo.iterations == 4
        np.testing.assert_array_equal(evo.mask, init.astype(bool))

    def test_auto_threshold_adjusted_for_flat_image(self):
        g = np.ones((10, 10, 10), dtype=np.float32)
        g[:2] = 0.5
        thr, adjusted = resolve_threshold(g, "auto", 1.0)
        assert adjusted is True and thr == pytest.approx(0.5)
        assert resolve_threshold(g, 0.3, 1.0) == (0.3, False)
        noisy = np.random.default_rng(1).uniform(0.0, 1.0, (10, 10, 10)).astype(np.float32)
        thr, adjusted = resolve_threshold(noisy, "auto", 1.0)
        assert adjusted is False and thr == pytest.approx(np.percentile(noisy, 40))

    def test_init_ball_contains_seed_for_tiny_radius(self):
        ball = init_ball((5, 5, 5), (2.2, 1.9, 3.0), 0.1)
        assert ball.sum() == 1 and ball[2, 2, 3] == 1


class TestParamsAndFactory:
    def test_defaults_and_validation(self):
        assert resolve_gac_params(None) == {**GAC_DEFAULT_PARAMS, "threshold": "auto"}
        p = resolve_gac_params({"domain_size_mm": None, "balloon": -1})
        assert p["domain_size_mm"] is None and p["balloon"] == -1.0
        with pytest.raises(ValueError, match="literówka"):
            resolve_gac_params({"domain_sise_mm": 50})
        for bad in ({"gac_alpha": 0}, {"max_iterations": 0}, {"threshold": "x"}, {"smoothing": 1.5}):
            with pytest.raises(ValueError):
                resolve_gac_params(bad)
        assert resolve_algorithm_params("chan_vese", None)["data_term_sign"] == "paper"

    def test_algorithm_name(self):
        assert algorithm_name("chan_vese") == "chan_vese"
        assert algorithm_name("geodesic_active_contour") == "geodesic_active_contour"
        with pytest.raises(ValueError, match="dozwolone"):
            algorithm_name("growcutt")

    def test_make_segmenter(self):
        cv = make_segmenter("chan_vese", {"init_anchor": "center"})
        assert isinstance(cv, ImprovedChanVese) and cv.params["init_anchor"] == "center"
        gac = make_segmenter("geodesic_active_contour", {"balloon": 2})
        assert isinstance(gac, GeodesicActiveContour) and gac.params["balloon"] == 2.0
        with pytest.raises(ValueError):
            make_segmenter("unet_2d")
        with pytest.raises(ValueError, match="literówka"):
            make_segmenter("geodesic_active_contour", {"balon": 1})
