from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.segmentation import chan_vese as sk_chan_vese

from src.models.chan_vese import ImprovedChanVese, ObjectNotFoundError
from src.models.chan_vese.binarize import binarize_slice, median_size_px, quantize, threshold_slice
from src.models.chan_vese.evolution import chan_vese_2d
from src.models.chan_vese.init_contour import auto_init_contour, contour_force
from src.models.chan_vese.object_region import select_object_region
from src.models.grid import SeedPoint, WorkVolume
from src.models.result import signed_distance, unprocessed_phi_value
from src.models.roi import roi_from_seed
from tests.test_roi import bladder_center, make_fish


def manual_roi(volume, bladder, cfg=None):
    return roi_from_seed(volume, SeedPoint("manual", bladder_center(bladder), (0.0, 0.0, 0.0)), cfg or {})


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def disk(n: int, radius: float, center: tuple[float, float] | None = None) -> np.ndarray:
    c = center if center is not None else ((n - 1) / 2, (n - 1) / 2)
    x, y = np.mgrid[:n, :n]
    return (x - c[0]) ** 2 + (y - c[1]) ** 2 <= radius**2


def rect(n: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    m = np.zeros((n, n), bool)
    m[x0:x1, y0:y1] = True
    return m


class TestThresholding:
    def test_gas_object_distinguishable(self):
        img = np.full((20, 20), 0.8, np.float32)
        img[5:10, 5:10] = 0.0
        thr = threshold_slice(img, 0.0, 0.0)
        assert np.all(thr[5:10, 5:10] == 0.0)
        assert np.all(thr[img == 0.8] == -1.0)
        binary = binarize_slice(img, 0.0, 0.0)
        assert np.array_equal(binary, img == 0.0)

    def test_inclusive_range(self):
        img = np.array([[0.1, 0.2, 0.3, 0.4]], np.float32)
        assert binarize_slice(img, 0.2, 0.3).tolist() == [[False, True, True, False]]

    def test_median_removes_speckle(self):
        img = np.full((15, 15), 0.8, np.float32)
        img[7, 7] = 0.0
        assert binarize_slice(img, 0.0, 0.0, (1, 1)).sum() == 1
        assert binarize_slice(img, 0.0, 0.0, (3, 3)).sum() == 0

    def test_median_size_px(self):
        assert median_size_px(1.0, (0.2065, 0.2065)) == (5, 5)
        assert median_size_px(1.0, (0.839, 0.839)) == (1, 1)
        assert median_size_px(1.0, (3.0, 3.0)) == (1, 1)
        assert median_size_px(1.0, (0.5, 0.5)) == (3, 3)

    def test_quantize(self):
        img = np.array([0.0, 0.5, 1.0, 0.1234], np.float32)
        q = quantize(img, 256)
        assert np.allclose(q * 255, np.round(q * 255))
        assert np.array_equal(quantize(q, 256), q)


class TestContourForce:
    def test_force_values(self):
        assert contour_force(1.0, 1.0, 0.0) == pytest.approx(1.0)
        assert contour_force(1.0, 1.0, 0.5) == pytest.approx(0.25)
        assert contour_force(1.0, 1.0, 1.0) == 0.0

    def test_corner_grows_only_positive(self):
        img = rect(128, 10, 50, 10, 50)
        centroid = (30, 30)
        phi0, info = auto_init_contour(img, centroid, anchor="corner")
        x0, x1, y0, y1 = info["rect"]
        assert (x0, y0) == centroid
        assert x1 - x0 == info["size_px"] and y1 - y0 == info["size_px"]
        assert info["size_px"] > 4 and info["steps"] == info["size_px"] - 4
        assert info["force"] >= 1.0 and info["p"] == 1.0
        assert not info["touched_edge"] and not info["warnings"]
        assert x1 > 50
        assert np.array_equal(phi0 > 0, rect(128, x0, x1, y0, y1))

    def test_center_symmetric(self):
        img = rect(64, 10, 50, 10, 50)
        _, info = auto_init_contour(img, (30, 30), anchor="center")
        x0, x1, y0, y1 = info["rect"]
        assert abs((30 - x0) - (x1 - 30)) <= 1 and abs((30 - y0) - (y1 - 30)) <= 1
        assert info["force"] >= 1.0
        assert x0 <= 11 and x1 >= 49

    def test_stops_when_band_outside(self):
        img = rect(32, 10, 14, 10, 14)
        _, info = auto_init_contour(img, (10, 10), anchor="corner")
        assert info["steps"] == 0 and info["force"] == pytest.approx(1.0)

    def test_edge_safety(self):
        img = np.ones((20, 20), bool)
        _, info = auto_init_contour(img, (10, 10), anchor="corner")
        assert info["touched_edge"] and info["warnings"]
        assert info["rect"][1] == 20

    def test_p_zero_warning(self):
        img = rect(40, 5, 35, 5, 35)
        img[15:25, 15:25] = False
        _, info = auto_init_contour(img, (20, 20))
        assert info["p"] == 0.0
        assert any("P = 0" in w for w in info["warnings"])


class TestChanVese:
    N, R = 64, 12

    def setup_method(self):
        self.circle = disk(self.N, self.R)
        c = self.N // 2
        self.init = rect(self.N, c - 4, c + 4, c - 4, c + 4)
        self.phi0 = signed_distance(self.init)

    def test_chan_vese_sign_recovers_circle(self):
        res = chan_vese_2d(
            self.circle.astype(float), self.phi0, data_term_sign="chan_vese", tol=0.0, max_iterations=1500
        )
        assert dice(res.phi > 0, self.circle) > 0.95

    def test_matches_skimage(self):
        ours = chan_vese_2d(
            self.circle.astype(float),
            self.phi0,
            mu=0.1,
            dt=0.5,
            eps=1.0,
            data_term_sign="chan_vese",
            tol=0.0,
            max_iterations=1500,
        )
        ref = sk_chan_vese(
            self.circle.astype(float),
            mu=0.1,
            lambda1=1.0,
            lambda2=1.0,
            tol=0.0,
            max_num_iter=1500,
            dt=0.5,
            init_level_set=self.phi0,
        )
        assert dice(ours.phi > 0, ref) > 0.9

    def test_paper_sign_does_not_grow(self):
        res = chan_vese_2d(
            self.circle.astype(float), self.phi0, data_term_sign="paper", tol=0.0, max_iterations=300
        )
        assert (res.phi > 0).sum() <= self.init.sum()

    def test_iteration_limit_and_convergence(self):
        limited = chan_vese_2d(
            self.circle.astype(float), self.phi0, data_term_sign="chan_vese", tol=0.0, max_iterations=7
        )
        assert limited.iterations == 7 and not limited.converged
        conv = chan_vese_2d(
            self.circle.astype(float), self.phi0, data_term_sign="chan_vese", tol=1e-3, max_iterations=5000
        )
        assert conv.converged and conv.iterations < 5000

    def test_band_tolerance_region(self):
        res = chan_vese_2d(
            self.circle.astype(float),
            self.phi0,
            data_term_sign="chan_vese",
            tol=1e-3,
            max_iterations=50,
            tol_region="band",
        )
        assert res.iterations <= 50

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            chan_vese_2d(np.zeros((5, 5)), np.zeros((5, 6)))


class TestSelectObjectRegion:
    def make_binary(self) -> np.ndarray:
        b = np.zeros((20, 20, 10), bool)
        b[5:10, 5:10, 3:7] = True
        b[0:3, :, :] = True
        return b

    def test_region_with_seed(self):
        b = self.make_binary()
        region = select_object_region(b, (7, 7, 5))
        assert region.object_mask.sum() == 5 * 5 * 4
        assert np.array_equal(region.binary, b)

    def test_outside_air_kept_by_default(self):
        b = self.make_binary()
        region = select_object_region(b, (1, 10, 5))
        assert region.object_mask.sum() == 3 * 20 * 10

    def test_remove_outside_air(self):
        b = self.make_binary()
        region = select_object_region(b, (7, 7, 5), remove_outside_air=True)
        assert region.n_removed_regions == 1
        assert not region.binary[0:3].any()
        with pytest.raises(ObjectNotFoundError):
            select_object_region(b, (1, 10, 5), remove_outside_air=True)

    def test_nearest_searched_after_outside_air_removal(self):
        b = np.zeros((20, 20, 10), bool)
        b[0:3, :, :] = True
        b[7:10, 8:12, 3:7] = True
        seed = (4, 10, 5)
        region = select_object_region(b, seed, True, spacing=(1, 1, 1), search_radius_mm=3.5)
        assert region.seed_voxel == (7, 10, 5)
        assert region.object_mask.sum() == 3 * 4 * 4 and region.warnings
        kept = select_object_region(b, seed, False, spacing=(1, 1, 1), search_radius_mm=3.5)
        assert kept.seed_voxel == (2, 10, 5)

    def test_seed_in_background_nearest(self):
        b = self.make_binary()
        region = select_object_region(b, (11, 7, 5), spacing=(1, 1, 1), search_radius_mm=3.0)
        assert region.seed_voxel == (9, 7, 5) and region.warnings
        with pytest.raises(ObjectNotFoundError):
            select_object_region(b, (16, 16, 5), spacing=(1, 1, 1), search_radius_mm=1.0)


class TestPhiScale:
    def test_unprocessed_slices_do_not_leak_when_interpolated(self):
        shape = (48, 48, 40)
        data = np.full(shape, 0.8, np.float32)
        x, y = np.mgrid[: shape[0], : shape[1]]
        cyl = (x - 23.5) ** 2 + (y - 23.5) ** 2 <= 100
        data[:, :, 5:25][cyl] = 0.0
        volume = WorkVolume(data=data, affine=np.eye(4), intensity_unit="normalized")
        roi = roi_from_seed(volume, SeedPoint("manual", (23, 23, 15), (23.0, 23.0, 15.0)), {})
        res = ImprovedChanVese({"data_term_sign": "chan_vese", "init_anchor": "center"}).segment(volume, roi)
        assert res.params_used["slice_range"] == [5, 24]

        def interpolated_mask(z: float) -> np.ndarray:
            coords = np.stack([x, y, np.full(x.shape, z)]).astype(float)
            return ndi.map_coordinates(res.phi, coords, order=1, mode="nearest") > 0

        assert interpolated_mask(24.7).sum() == 0
        assert interpolated_mask(28.7).sum() == 0
        assert interpolated_mask(16.7).sum() > 0.8 * cyl.sum()


class TestSegmentEndToEnd:
    def test_gas_bladder(self):
        volume, bladder = make_fish()
        roi = manual_roi(volume, bladder)
        cfg = {"data_term_sign": "chan_vese", "init_anchor": "center", "remove_outside_air": True}
        res = ImprovedChanVese(cfg).segment(volume, roi)
        assert res.status == "ok"
        assert res.mask.shape == volume.shape and np.array_equal(res.mask.affine, volume.affine)
        assert res.phi.shape == volume.shape and res.phi.dtype == np.float32
        assert dice(res.mask.data, bladder) > 0.8
        unprocessed = ~np.isin(np.arange(volume.shape[2]), [r["k"] for r in res.params_used["per_slice"]])
        assert np.all(res.phi[:, :, unprocessed] == unprocessed_phi_value(volume.shape))
        assert np.array_equal(res.mask.data, res.phi > 0)
        for k in np.flatnonzero(~unprocessed):
            assert np.allclose(res.phi[:, :, k], signed_distance(res.phi[:, :, k] > 0))

    def test_default_config_runs(self):
        volume, bladder = make_fish()
        roi = manual_roi(volume, bladder)
        res = ImprovedChanVese({}).segment(volume, roi)
        assert res.status in ("ok", "empty")
        assert res.params_used["init_anchor"] == "corner" and res.params_used["data_term_sign"] == "paper"

    def test_determinism_and_json(self):
        volume, bladder = make_fish()
        roi = manual_roi(volume, bladder)
        cfg = {"data_term_sign": "chan_vese", "remove_outside_air": True}
        a = ImprovedChanVese(cfg).segment(volume, roi)
        b = ImprovedChanVese(cfg).segment(volume, roi)
        assert np.array_equal(a.phi, b.phi) and np.array_equal(a.mask.data, b.mask.data)
        text = json.dumps(a.params_used)
        assert json.loads(text)["n_slices_processed"] == len(a.params_used["per_slice"]) > 0
        assert a.params_used["tmin"] == 0.0 and a.params_used["thresholding"]["outside_value"] == -1.0

    def test_tissue_bright_bladder_spills(self):
        volume, bladder = make_fish(bladder_value=0.8)
        seed_voxel = (29, 18, 13)
        roi = roi_from_seed(volume, SeedPoint("manual", seed_voxel, (0.0, 0.0, 0.0)), {})
        res = ImprovedChanVese({"data_term_sign": "chan_vese", "init_anchor": "center"}).segment(volume, roi)
        assert res.status == "ok"
        assert res.mask.data.sum() > bladder.sum()

    def test_object_not_found(self):
        volume, _ = make_fish()
        roi = roi_from_seed(volume, SeedPoint("manual", (1, 1, 1), (0.0, 0.0, 0.0)), {})
        with pytest.raises(ObjectNotFoundError):
            ImprovedChanVese({"remove_outside_air": True}).segment(volume, roi)

    def test_parallel_matches_serial(self):
        volume, bladder = make_fish()
        roi = manual_roi(volume, bladder)
        cfg = {"data_term_sign": "chan_vese", "remove_outside_air": True, "max_iterations": 20}
        serial = ImprovedChanVese(cfg).segment(volume, roi)
        parallel = ImprovedChanVese({**cfg, "n_jobs": 2}).segment(volume, roi)
        assert np.array_equal(serial.phi, parallel.phi)

    def test_intensity_levels(self):
        volume, bladder = make_fish()
        roi = manual_roi(volume, bladder, {"range_estimator": "minmax"})
        res = ImprovedChanVese(
            {
                "intensity_levels": 256,
                "data_term_sign": "chan_vese",
                "init_anchor": "center",
                "remove_outside_air": True,
            }
        ).segment(volume, roi)
        assert res.params_used["intensity_levels"] == 256
        assert dice(res.mask.data, bladder) > 0.8
