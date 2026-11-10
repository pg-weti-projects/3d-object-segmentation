from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.models.grid import SeedPoint, WorkVolume
from src.models.params import resolve_params
from src.models.roi import estimate_range, roi_from_seed

SPACING = 0.5


def make_fish(
    bladder_value: float = 0.0, shape: tuple[int, int, int] = (56, 40, 28), spacing: float = SPACING
) -> tuple[WorkVolume, np.ndarray]:
    axes = [(np.arange(n) - (n - 1) / 2) * spacing for n in shape]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    data = np.zeros(shape, dtype=np.float32)
    body = (x / 12) ** 2 + (y / 8) ** 2 + (z / 6) ** 2 <= 1
    data[body] = 0.8
    bladder = ((x - 1) / 6) ** 2 + ((y + 0.5) / 4) ** 2 + (z / 3.5) ** 2 <= 1
    data[bladder] = bladder_value
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = [-10.0, 5.0, 2.0]
    return WorkVolume(
        data=data, affine=affine, intensity_unit="normalized", meta={"hu_range": [-1000, 200]}
    ), bladder


def bladder_center(bladder: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(round(float(c))) for c in np.argwhere(bladder).mean(axis=0))  # type: ignore[return-value]


class TestEstimateRange:
    def test_percentile_ignores_outliers(self):
        values = np.concatenate([np.full(1000, 0.5), [0.0, 1.0]])
        assert estimate_range(values, "percentile") == pytest.approx((0.5, 0.5))

    def test_minmax(self):
        assert estimate_range(np.array([0.2, 0.4, 0.9]), "minmax") == (0.2, 0.9)

    def test_empty_and_unknown(self):
        with pytest.raises(ValueError):
            estimate_range(np.array([]))
        with pytest.raises(ValueError):
            estimate_range(np.array([1.0]), "median")  # type: ignore[arg-type]


class TestManualRoi:
    def test_ball_not_restricted(self):
        volume, bladder = make_fish()
        voxel = bladder_center(bladder)
        seed = SeedPoint(source="manual", voxel=voxel, mm=(0.0, 0.0, 0.0))
        roi = roi_from_seed(volume, seed, {"roi_radius_mm": 3.0})
        assert roi.seed.source == "manual"
        r_vox = 6
        assert roi.roi_mask.sum() == pytest.approx(4 / 3 * np.pi * r_vox**3, rel=0.1)
        json.dumps(roi.meta)

    def test_seed_mm_matches_affine(self):
        volume, bladder = make_fish()
        voxel = bladder_center(bladder)
        roi = roi_from_seed(volume, SeedPoint("manual", voxel, (0.0, 0.0, 0.0)), {})
        expected = volume.affine @ np.array([*voxel, 1.0])
        assert roi.seed.mm == pytest.approx(tuple(expected[:3]))

    def test_gas_range_and_hu(self):
        volume, bladder = make_fish()
        seed = SeedPoint("manual", bladder_center(bladder), (0.0, 0.0, 0.0))
        roi = roi_from_seed(volume, seed, {"roi_radius_mm": 1.0})
        assert (roi.tmin, roi.tmax) == (0.0, 0.0)
        assert roi.meta["tmin_hu"] == pytest.approx(-1000.0)
        assert roi.meta["hu_range"] == [-1000.0, 200.0]

    def test_roi_radius_limits_ball(self):
        volume, bladder = make_fish()
        seed = SeedPoint("manual", bladder_center(bladder), (0.0, 0.0, 0.0))
        small = roi_from_seed(volume, seed, {"roi_radius_mm": 0.5})
        large = roi_from_seed(volume, seed, {"roi_radius_mm": 2.0})
        assert small.roi_mask.sum() < large.roi_mask.sum()
        assert small.roi_mask.sum() == 7

    def test_seed_outside_volume(self):
        volume, _ = make_fish()
        with pytest.raises(ValueError):
            roi_from_seed(volume, SeedPoint("manual", (999, 0, 0), (0.0, 0.0, 0.0)), {})

    def test_range_from_tissue_seed(self):
        volume, _ = make_fish()
        seed = SeedPoint("manual", (10, 20, 14), (0.0, 0.0, 0.0))
        roi = roi_from_seed(volume, seed, {"roi_radius_mm": 1.0})
        assert roi.tmin == pytest.approx(0.8) and roi.tmax == pytest.approx(0.8)

    def test_no_hu_range_no_hu_meta(self):
        volume, bladder = make_fish()
        plain = WorkVolume(volume.data, volume.affine, "normalized")
        roi = roi_from_seed(plain, SeedPoint("manual", bladder_center(bladder), (0.0, 0.0, 0.0)), {})
        assert "tmin_hu" not in roi.meta


class TestParams:
    def test_merge_with_defaults(self):
        p = resolve_params({"mu": 0.2})
        assert p["mu"] == 0.2 and p["roi_radius_mm"] == 3.0

    def test_unknown_key_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.models.params"):
            p = resolve_params({"init_anchr": "center", "mu": 0.2})
        assert p["init_anchor"] == "corner"
        warned = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warned) == 1
        assert "['init_anchr']" in warned[0].getMessage()

    def test_known_keys_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.models.params"):
            resolve_params({"n_jobs": 2})
            resolve_params(resolve_params({"intensity_levels": 256}))
        assert not caplog.records

    def test_invalid_choice(self):
        with pytest.raises(ValueError):
            resolve_params({"init_anchor": "middle"})
