"""Tests for the contrast measurement and for starting the analytical methods
from it (``init_from: contrast``)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.preprocessor import PreprocessingConfig, preprocess
from src.core.volume import Volume
from src.models import contrast as C
from src.models.contrast_init import (
    ESTIMATOR,
    candidates_for,
    contrast_roi,
    growcut_seeds,
    region_on_grid,
)
from src.models.domain import compute_domain
from src.models.grid import SeedPoint
from src.models.params import (
    CHAN_VESE,
    GEODESIC_ACTIVE_CONTOUR,
    GROWCUT,
    resolve_algorithm_params,
)
from src.models.roi import RoiResult
from src.models.segment import seed_from_zyx, segment_volume, to_work_volume

SPACING = (0.5, 0.5, 0.5)
HU_RANGE = (-1000.0, 200.0)

METHOD_PARAMETERS = {
    CHAN_VESE: {},
    GEODESIC_ACTIVE_CONTOUR: {"domain_size_mm": None},
    GROWCUT: {"domain_size_mm": None},
}


def make_fish(
    tissue: float = 40.0,
    bladder: float = -1000.0,
    outside: float = -1000.0,
    spacing: tuple[float, float, float] = SPACING,
) -> tuple[Volume, np.ndarray, tuple[int, int, int]]:
    """A fish-shaped body with a bladder inside it, and air around the body with
    the same value as a gas-filled bladder - which is what makes the restriction
    to the body necessary."""
    shape = tuple(int(round(extent / s)) for extent, s in zip((14.0, 20.0, 28.0), spacing))
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)]
    z, y, x = np.meshgrid(*axes, indexing="ij")
    data = np.full(shape, outside, dtype=np.float32)
    data[(x / 12) ** 2 + (y / 8) ** 2 + (z / 6) ** 2 <= 1] = tissue
    inside = ((x - 1) / 6) ** 2 + ((y + 0.5) / 4) ** 2 + (z / 3.5) ** 2 <= 1
    data[inside] = bladder
    seed = tuple(int(round(float(c))) for c in np.argwhere(inside).mean(axis=0))
    return Volume(data, spacing, "HU", "dicom", "0" * 64), inside, seed  # type: ignore[return-value]


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction, reference = prediction != 0, reference != 0
    total = float(prediction.sum() + reference.sum())
    return 2 * float(np.count_nonzero(prediction & reference)) / total if total else 0.0


def work_volume(invert_hu: bool = False, **kwargs):
    volume, inside, seed = make_fish(**kwargs)
    volume = preprocess(volume, PreprocessingConfig(hu_range=HU_RANGE, invert_hu=invert_hu))
    work = to_work_volume(volume, HU_RANGE, invert_hu)
    return volume, work, inside, seed


class TestEstimate:
    def test_gas_bladder_is_high_contrast(self):
        volume, inside, seed = make_fish()
        estimate = C.estimate(volume.data, volume.spacing_mm, seed)
        assert estimate.regime == C.HIGH
        assert estimate.separability >= C.HIGH_SEPARABILITY
        assert estimate.object_level < estimate.threshold < estimate.tissue_level
        assert estimate.separated and estimate.usable
        assert estimate.seed_zyx == tuple(seed)

    def test_a_few_percent_is_the_low_regime(self):
        # A fifth of the dynamic range, as in the Gadus scan: no gas, the bladder
        # only slightly darker than the tissue around it.
        volume, _, seed = make_fish(tissue=40.0, bladder=0.0, outside=-150.0)
        estimate = C.estimate(volume.data, volume.spacing_mm, seed)
        assert estimate.regime == C.LOW
        assert C.LOW_SEPARABILITY <= estimate.separability < C.HIGH_SEPARABILITY
        assert estimate.separated
        assert any("low contrast" in note for note in estimate.notes)

    def test_seed_in_the_tissue_is_not_usable(self):
        volume, _, _ = make_fish()
        in_tissue = tuple(int(n) // 2 for n in volume.shape)
        in_tissue = (in_tissue[0], in_tissue[1] + 10, in_tissue[2])
        estimate = C.estimate(volume.data, volume.spacing_mm, in_tissue)
        assert estimate.regime == C.NONE
        assert not estimate.usable
        smooth = C.smooth_mm(volume.data, volume.spacing_mm, 0.8)
        assert not C.region(smooth, volume.spacing_mm, estimate).any()

    def test_bright_bladder_needs_dark_false(self):
        volume, inside, seed = make_fish(tissue=-1000.0, bladder=40.0, outside=40.0)
        assert C.estimate(volume.data, volume.spacing_mm, seed, dark=True).regime == C.NONE
        estimate = C.estimate(volume.data, volume.spacing_mm, seed, dark=False)
        assert estimate.regime == C.HIGH
        assert estimate.object_level > estimate.threshold > estimate.tissue_level

    def test_cnr_is_infinite_without_noise_and_finite_with_it(self):
        volume, _, seed = make_fish()
        assert C.estimate(volume.data, volume.spacing_mm, seed).cnr == float("inf")
        noisy = volume.data + np.random.default_rng(0).normal(0, 5, volume.shape).astype(np.float32)
        estimate = C.estimate(noisy, volume.spacing_mm, seed)
        assert np.isfinite(estimate.cnr) and estimate.cnr > 10

    def test_uniform_volume_is_rejected(self):
        with pytest.raises(ValueError, match="uniform"):
            C.estimate(np.ones((4, 4, 4), np.float32), (1.0, 1.0, 1.0), (2, 2, 2))

    def test_nan_is_rejected(self):
        data = np.zeros((4, 4, 4), np.float32)
        data[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            C.estimate(data, (1.0, 1.0, 1.0), (2, 2, 2))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
    def test_threshold_fraction_out_of_range_is_rejected(self, fraction):
        volume, _, seed = make_fish()
        with pytest.raises(ValueError, match="threshold_fraction"):
            C.estimate(volume.data, volume.spacing_mm, seed, threshold_fraction=fraction)

    def test_seed_outside_the_volume_is_rejected(self):
        volume, _, _ = make_fish()
        with pytest.raises(ValueError, match="outside"):
            C.estimate(volume.data, volume.spacing_mm, (0, 0, 10_000))

    def test_no_seed_means_the_centre(self):
        volume, _, _ = make_fish()
        estimate = C.estimate(volume.data, volume.spacing_mm)
        assert estimate.seed_zyx == tuple(int(n) // 2 for n in volume.shape)


class TestRegion:
    def test_region_finds_the_bladder_and_stays_in_the_body(self):
        volume, inside, seed = make_fish()
        estimate = C.estimate(volume.data, volume.spacing_mm, seed)
        smooth = C.smooth_mm(volume.data, volume.spacing_mm, 0.8)
        found = C.region(smooth, volume.spacing_mm, estimate)
        assert dice(found, inside) > 0.8
        # The air around the fish has the same value as the bladder; none of it
        # may end up in the region.
        body = (volume.data > estimate.threshold) | inside
        assert np.count_nonzero(found & ~body) == 0

    def test_without_the_hull_the_outside_air_is_followed(self):
        volume, inside, seed = make_fish()
        estimate = C.estimate(volume.data, volume.spacing_mm, seed)
        smooth = C.smooth_mm(volume.data, volume.spacing_mm, 0.8)
        limited = C.candidates(smooth, estimate, use_hull=True)
        everything = C.candidates(smooth, estimate, use_hull=False)
        assert int(everything.sum()) > 3 * int(limited.sum())

    def test_candidates_are_empty_when_nothing_is_separated(self):
        volume, _, seed = make_fish()
        estimate = C.estimate(volume.data, volume.spacing_mm, seed, dark=False)
        assert estimate.regime == C.NONE
        assert not C.candidates(volume.data, estimate).any()


class TestStrategy:
    def test_low_regime_smooths_more_than_high(self):
        assert C.strategy_for(C.LOW).sigma_mm > C.strategy_for(C.HIGH).sigma_mm
        assert C.strategy_for(C.LOW).open_mm >= C.strategy_for(C.HIGH).open_mm

    def test_unknown_regime_is_rejected(self):
        with pytest.raises(ValueError, match="regime"):
            C.strategy_for("something else")


class TestContrastRoi:
    def test_window_and_region(self):
        _, work, inside, seed = work_volume()
        roi = contrast_roi(work, seed_from_zyx(list(seed), work))
        assert roi.estimator == ESTIMATOR
        assert roi.tmin == pytest.approx(float(work.data.min()))
        assert roi.tmin < roi.tmax < 1.0
        assert roi.meta["contrast"]["regime"] == C.HIGH
        assert roi.meta["region_voxels"] > 0
        # The region is in the orientation of the work volume (x, y, z).
        assert roi.roi_mask.shape == work.shape
        assert dice(roi.roi_mask.transpose(2, 1, 0), inside) > 0.8

    def test_explicit_settings_win_over_the_regime(self):
        _, work, _, seed = work_volume()
        point = seed_from_zyx(list(seed), work)
        roi = contrast_roi(work, point, sigma_mm=0.0, threshold_fraction=0.2, open_mm=0.0)
        assert roi.meta["strategy"] == {
            "sigma_mm": 0.0,
            "threshold_fraction": 0.2,
            "open_mm": 0.0,
            "use_hull": True,
        }

    def test_inverted_preprocessing_is_undone_first(self):
        # to_work_volume undoes preprocessing's inversion, so the bladder is the
        # darker side here either way and the region is the same one.
        _, work, inside, seed = work_volume(invert_hu=True)
        roi = contrast_roi(work, seed_from_zyx(list(seed), work))
        assert roi.meta["contrast"]["bladder_is_dark"] is True
        assert roi.tmin == pytest.approx(float(work.data.min()))
        assert dice(roi.roi_mask.transpose(2, 1, 0), inside) > 0.8

    def test_candidates_for_needs_a_contrast_roi(self):
        _, work, _, seed = work_volume()
        other = RoiResult(
            seed=seed_from_zyx(list(seed), work),
            roi_mask=np.zeros((1, 1, 1), bool),
            tmin=0.0,
            tmax=1.0,
            estimator="percentile",
        )
        assert candidates_for(work, other) is None
        roi = contrast_roi(work, seed_from_zyx(list(seed), work))
        found = candidates_for(work, roi)
        assert found is not None and found.shape == work.shape
        assert int(found.sum()) >= int(roi.roi_mask.sum())


class TestRegionOnGrid:
    def _roi(self, mask: np.ndarray) -> RoiResult:
        seed = SeedPoint(source="manual", voxel=(1, 1, 1), mm=(1.0, 1.0, 1.0))
        return RoiResult(seed=seed, roi_mask=mask, tmin=0.0, tmax=0.5, estimator=ESTIMATOR, meta={})

    def test_none_for_another_estimator(self):
        mask = np.ones((4, 4, 4), bool)
        roi = self._roi(mask)
        roi.estimator = "percentile"
        domain = compute_domain(mask.shape, (1.0, 1.0, 1.0), (2, 2, 2), None)
        assert region_on_grid(roi, mask.shape, domain, np.ones(3), mask.shape) is None

    def test_none_for_an_empty_or_mismatched_region(self):
        domain = compute_domain((4, 4, 4), (1.0, 1.0, 1.0), (2, 2, 2), None)
        empty = self._roi(np.zeros((4, 4, 4), bool))
        assert region_on_grid(empty, (4, 4, 4), domain, np.ones(3), (4, 4, 4)) is None
        other_shape = self._roi(np.ones((2, 2, 2), bool))
        assert region_on_grid(other_shape, (4, 4, 4), domain, np.ones(3), (4, 4, 4)) is None

    def test_same_grid_keeps_the_region(self):
        mask = np.zeros((6, 6, 6), bool)
        mask[2:4, 2:4, 2:4] = True
        domain = compute_domain(mask.shape, (1.0, 1.0, 1.0), (3, 3, 3), None)
        on_grid = region_on_grid(self._roi(mask), mask.shape, domain, np.ones(3), mask.shape)
        assert on_grid is not None and np.array_equal(on_grid, mask)

    def test_domain_crops_the_region(self):
        mask = np.zeros((9, 9, 9), bool)
        mask[4:6, 4:6, 4:6] = True
        domain = compute_domain(mask.shape, (1.0, 1.0, 1.0), (4, 4, 4), 4.0)
        on_grid = region_on_grid(self._roi(mask), mask.shape, domain, np.ones(3), domain.shape)
        assert on_grid is not None and on_grid.shape == domain.shape
        assert int(on_grid.sum()) == 8

    def test_resampling_to_a_coarser_grid(self):
        mask = np.zeros((8, 8, 8), bool)
        mask[2:6, 2:6, 2:6] = True
        domain = compute_domain(mask.shape, (1.0, 1.0, 1.0), (4, 4, 4), None)
        on_grid = region_on_grid(self._roi(mask), mask.shape, domain, np.full(3, 2.0), (4, 4, 4))
        assert on_grid is not None and on_grid.shape == (4, 4, 4)
        assert on_grid.any()


class TestGrowcutSeeds:
    def test_object_and_background_seeds_are_disjoint(self):
        image = np.ones((12, 12, 12), np.float32)
        region = np.zeros((12, 12, 12), bool)
        region[4:8, 4:8, 4:8] = True
        image[region] = 0.0
        seed = SeedPoint(source="manual", voxel=(6, 6, 6), mm=(6.0, 6.0, 6.0))
        roi = RoiResult(
            seed=seed,
            roi_mask=region,
            tmin=0.0,
            tmax=0.5,
            estimator=ESTIMATOR,
            meta={"contrast": {"bladder_is_dark": True}},
        )
        inside, outside = growcut_seeds(region, image, roi)
        assert inside.any() and outside.any()
        assert not (inside & outside).any()
        assert np.count_nonzero(inside & ~region) == 0
        assert np.count_nonzero(outside & region) == 0


class TestParameters:
    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    def test_default_start_is_the_seed_ball(self, name):
        resolved = resolve_algorithm_params(name, METHOD_PARAMETERS[name])
        assert resolved["init_from"] == "seed_ball"

    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    def test_unknown_start_is_rejected(self, name):
        parameters = {**METHOD_PARAMETERS[name], "init_from": "magic"}
        with pytest.raises(ValueError, match="init_from"):
            resolve_algorithm_params(name, parameters)

    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    @pytest.mark.parametrize("fraction", [0.0, 1.0, 5])
    def test_contrast_fraction_must_be_between_zero_and_one(self, name, fraction):
        parameters = {**METHOD_PARAMETERS[name], "contrast_fraction": fraction}
        with pytest.raises(ValueError, match="contrast_fraction"):
            resolve_algorithm_params(name, parameters)

    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    def test_negative_contrast_sigma_is_rejected(self, name):
        parameters = {**METHOD_PARAMETERS[name], "contrast_sigma_mm": -1.0}
        with pytest.raises(ValueError, match="contrast_sigma_mm"):
            resolve_algorithm_params(name, parameters)

    def test_gac_turns_the_balloon_off_for_a_contrast_start(self):
        resolved = resolve_algorithm_params(GEODESIC_ACTIVE_CONTOUR, {"init_from": "contrast"})
        assert resolved["balloon"] == 0.0
        asked = resolve_algorithm_params(
            GEODESIC_ACTIVE_CONTOUR, {"init_from": "contrast", "balloon": 1.0}
        )
        assert asked["balloon"] == 1.0
        assert resolve_algorithm_params(GEODESIC_ACTIVE_CONTOUR, {})["balloon"] == 1.0

    def test_chan_vese_switches_the_data_term_for_a_contrast_start(self):
        resolved = resolve_algorithm_params(CHAN_VESE, {"init_from": "contrast"})
        assert resolved["data_term_sign"] == "chan_vese"
        asked = resolve_algorithm_params(
            CHAN_VESE, {"init_from": "contrast", "data_term_sign": "paper"}
        )
        assert asked["data_term_sign"] == "paper"
        assert resolve_algorithm_params(CHAN_VESE, {})["data_term_sign"] == "paper"


class TestMethodsFromContrast:
    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    def test_every_method_finds_the_bladder(self, name):
        volume, _, inside, seed = work_volume()
        parameters = {**METHOD_PARAMETERS[name], "init_from": "contrast"}
        preprocessing = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=False)
        result = segment_volume(
            volume, {"name": name, "parameters": parameters}, list(seed), preprocessing
        )
        assert result.status == "ok"
        assert dice(result.mask, inside) > 0.7
        assert result.params_used["seed_zyx"] == [int(v) for v in seed]

    @pytest.mark.parametrize("name", [GEODESIC_ACTIVE_CONTOUR, GROWCUT])
    def test_the_start_is_reported(self, name):
        volume, _, _, seed = work_volume()
        parameters = {**METHOD_PARAMETERS[name], "init_from": "contrast"}
        preprocessing = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=False)
        result = segment_volume(
            volume, {"name": name, "parameters": parameters}, list(seed), preprocessing
        )
        assert result.params_used["init_source"] == ESTIMATOR
        assert result.params_used["contrast"]["regime"] == C.HIGH

    @pytest.mark.parametrize("name", list(METHOD_PARAMETERS))
    def test_no_contrast_gives_an_empty_mask_and_says_why(self, name):
        # The Bluefin tuna case: what the reference mask outlines is as dense as
        # the tissue around it, so there is nothing for a threshold to find. The
        # method must say so instead of failing or inventing a shape.
        volume, _, _, seed = work_volume(tissue=40.0, bladder=45.0, outside=-1000.0)
        parameters = {**METHOD_PARAMETERS[name], "init_from": "contrast"}
        preprocessing = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=False)

        result = segment_volume(
            volume, {"name": name, "parameters": parameters}, list(seed), preprocessing
        )

        assert result.status == "empty"
        assert not result.mask.any()
        assert result.mask.shape == volume.shape
        assert result.params_used["contrast"]["regime"] == C.NONE
        assert result.params_used["n_warnings"] == len(result.params_used["warnings"]) >= 1
        assert "not darker" in " ".join(result.params_used["warnings"])

    def test_a_seed_in_the_tissue_gives_an_empty_mask_not_a_crash(self):
        volume, _, _, _ = work_volume()
        in_tissue = (volume.shape[0] // 2, volume.shape[1] // 2 + 10, volume.shape[2] // 2)
        preprocessing = PreprocessingConfig(hu_range=HU_RANGE, invert_hu=False)
        result = segment_volume(
            volume,
            {"name": GROWCUT, "parameters": {**METHOD_PARAMETERS[GROWCUT], "init_from": "contrast"}},
            list(in_tissue),
            preprocessing,
        )
        assert result.status == "empty"
        assert not result.mask.any()
