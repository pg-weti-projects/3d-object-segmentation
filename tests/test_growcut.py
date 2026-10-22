from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest

from src.models.domain import compute_domain
from src.models.factory import make_segmenter
from src.models.grid import SeedPoint, WorkVolume
from src.models.growcut import (
    BACKGROUND,
    OBJECT,
    GrowCut,
    attack_coefficient,
    background_seeds,
    growcut_evolve,
    neighborhood_offsets,
)
from src.models.params import GROWCUT_DEFAULT_PARAMS, resolve_algorithm_params, resolve_growcut_params
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


def seeds_1d(labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lab = np.array(labels, dtype=np.int8)
    return lab, (lab > 0).astype(np.float32)


def reference_step(img, lab, st, fixed, offsets, max_feature, t1, t2):
    shape = img.shape
    new_l, new_t = lab.copy(), st.copy()

    def neighbours(p):
        for d in offsets:
            q = tuple(a + b for a, b in zip(p, d, strict=True))
            if all(0 <= v < n for v, n in zip(q, shape, strict=True)):
                yield q

    def enemies(p):
        if lab[p] == 0:
            return 0
        return sum(1 for q in neighbours(p) if lab[q] not in (0, lab[p]))

    inv = np.float32(1.0 / max_feature)

    def g(p, q):
        return max(np.float32(0.0), np.float32(1.0) - np.abs(img[p] - img[q]) * inv)

    for p in itertools.product(*(range(n) for n in shape)):
        if fixed[p]:
            continue
        best, best_label = 0.0, 0
        for q in neighbours(p):
            if t1 is not None and enemies(q) >= t1:
                continue
            force = g(p, q) * st[q]
            if force > best or (force == best and force > 0 and lab[q] < best_label):
                best, best_label = force, lab[q]
        if best > st[p]:
            new_l[p], new_t[p] = best_label, best
        if t2 is not None and lab[p] != 0 and enemies(p) >= t2:
            forces = [g(p, q) * st[q] for q in neighbours(p) if lab[q] not in (0, lab[p])]
            new_l[p], new_t[p] = 3 - lab[p], min(forces)
    return new_l, new_t


def reference_run(img, lab, st, offsets, steps, t1=None, t2=None):
    fixed = (lab != 0) & (st >= 1.0)
    max_feature = float(img.max()) or 1.0
    for _ in range(steps):
        lab, st = reference_step(img, lab, st, fixed, offsets, max_feature, t1, t2)
    return lab, st


class TestRule:
    def test_g_function_formula_4(self):
        np.testing.assert_allclose(attack_coefficient(np.array([0.0, 0.25, 1.0]), 1.0), [1.0, 0.75, 0.0])
        np.testing.assert_allclose(attack_coefficient(np.array([-0.25, 0.25, 0.75]), 0.5), [0.5, 0.5, 0.0])

    def test_neighborhoods_formulas_1_2(self):
        assert len(neighborhood_offsets(3, "von_neumann")) == 6
        assert len(neighborhood_offsets(3, "moore")) == 26
        assert sorted(neighborhood_offsets(1, "moore")) == [(-1,), (1,)]
        with pytest.raises(ValueError):
            neighborhood_offsets(2, "hex")

    def test_hand_computed_1d(self):
        img = np.array([0.0, 0.2, 0.4, 1.0, 1.0], dtype=np.float32)
        lab, st = seeds_1d([1, 0, 0, 0, 2])
        one = growcut_evolve(img, lab, st, max_iterations=1)
        np.testing.assert_array_equal(one.labels, [1, 1, 0, 2, 2])
        np.testing.assert_allclose(one.strengths, [1.0, 0.8, 0.0, 1.0, 1.0], rtol=1e-6)
        full = growcut_evolve(img, lab, st)
        np.testing.assert_array_equal(full.labels, [1, 1, 1, 2, 2])
        np.testing.assert_allclose(full.strengths, [1.0, 0.8, 0.64, 1.0, 1.0], rtol=1e-6)
        assert full.converged and full.stop_reason == "stable" and full.iterations == 2

    def test_strongest_attack_wins_not_last_in_loop(self):
        img = np.array([0.0, 0.1, 0.6], dtype=np.float32)
        lab, st = seeds_1d([1, 0, 2])
        evo = growcut_evolve(img, lab, st, max_iterations=1, max_feature=1.0)
        assert evo.labels[1] == OBJECT and evo.strengths[1] == pytest.approx(0.9)
        mirror = growcut_evolve(
            img[::-1].copy(), lab[::-1].copy(), st[::-1].copy(), max_iterations=1, max_feature=1.0
        )
        assert mirror.labels[1] == OBJECT

    def test_hand_computed_2d_several_attackers(self):
        img = np.array([[1.0, 0.3, 1.0], [0.0, 0.5, 0.4], [1.0, 0.9, 1.0]], dtype=np.float32)
        lab = np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]], dtype=np.int8)
        st = np.array([[0, 0.5, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        evo = growcut_evolve(img, lab, st, max_iterations=1, max_feature=1.0)
        assert evo.labels[1, 1] == OBJECT and evo.strengths[1, 1] == pytest.approx(0.9)

    def test_weaker_attack_does_not_conquer_and_tie_goes_to_object(self):
        img = np.array([0.0, 0.2], dtype=np.float32)
        evo = growcut_evolve(
            img, np.array([1, 2], np.int8), np.array([1.0, 0.9], np.float32), max_iterations=5
        )
        np.testing.assert_array_equal(evo.labels, [1, 2])
        assert evo.iterations == 0 and evo.stop_reason == "stable"
        lab, st = seeds_1d([2, 0, 1])
        evo = growcut_evolve(np.zeros(3, np.float32), lab, st, max_iterations=1)
        assert evo.labels[1] == OBJECT and evo.strengths[1] == 1.0

    def test_synchronous_front_moves_one_cell_per_iteration(self):
        lab, st = seeds_1d([1] + [0] * 9)
        for k in (1, 3, 5):
            evo = growcut_evolve(np.zeros(10, np.float32), lab, st, max_iterations=k)
            assert int((evo.labels == OBJECT).sum()) == k + 1

    def test_zero_image_max_feature_guard(self):
        lab, st = seeds_1d([1, 0, 0])
        evo = growcut_evolve(np.zeros(3, np.float32), lab, st)
        np.testing.assert_array_equal(evo.labels, [1, 1, 1])
        np.testing.assert_array_equal(evo.strengths, [1.0, 1.0, 1.0])


class TestEvolution:
    @staticmethod
    def random_case(shape, seed=0):
        rng = np.random.default_rng(seed)
        img = rng.uniform(0.0, 1.0, shape).astype(np.float32)
        lab = np.zeros(shape, dtype=np.int8)
        lab[(0,) * len(shape)] = OBJECT
        lab[tuple(n // 2 for n in shape)] = OBJECT
        lab[tuple(n - 1 for n in shape)] = BACKGROUND
        lab[(shape[0] - 1,) + (0,) * (len(shape) - 1)] = BACKGROUND
        return img, lab, (lab > 0).astype(np.float32)

    def test_convergence_and_monotonic_strengths(self):
        img, lab, st = self.random_case((20, 18))
        cur_l, cur_t = lab, st
        for _ in range(200):
            nxt = growcut_evolve(
                img, cur_l, cur_t, max_iterations=1, max_feature=float(img.max()), fixed=lab > 0
            )
            assert np.all(nxt.strengths >= cur_t)
            if nxt.iterations == 0:
                break
            cur_l, cur_t = nxt.labels, nxt.strengths
        else:
            pytest.fail("brak zbieżności w 200 iteracjach")
        full = growcut_evolve(img, lab, st)
        assert full.converged and full.stop_reason == "stable"
        np.testing.assert_array_equal(full.labels, cur_l)
        np.testing.assert_array_equal(full.strengths, cur_t)
        assert not (full.labels == 0).any()

    @pytest.mark.parametrize(("shape", "neighborhood"), [((20, 18), "von_neumann"), ((9, 8, 7), "moore")])
    def test_stable_state_is_fixed_point_of_full_update(self, shape, neighborhood):
        img, lab, st = self.random_case(shape, seed=4)
        evo = growcut_evolve(img, lab, st, neighborhood=neighborhood)
        assert evo.stop_reason == "stable"
        fixed = (lab != 0) & (st >= 1.0)
        offsets = neighborhood_offsets(len(shape), neighborhood)
        new_l, new_t = reference_step(
            img, evo.labels, evo.strengths, fixed, offsets, float(img.max()), None, None
        )
        np.testing.assert_array_equal(new_l, evo.labels)
        np.testing.assert_array_equal(new_t, evo.strengths)

    def test_hard_seeds_never_change(self):
        img, lab, st = self.random_case((12, 10, 8), seed=2)
        seeds = lab > 0
        for t in (None, 3):
            evo = growcut_evolve(
                img, lab, st, neighborhood="moore", smooth_t1=t, smooth_t2=t, max_iterations=200
            )
            np.testing.assert_array_equal(evo.labels[seeds], lab[seeds])
            np.testing.assert_array_equal(evo.strengths[seeds], 1.0)

    @pytest.mark.parametrize("neighborhood", ["von_neumann", "moore"])
    @pytest.mark.parametrize("smooth", [None, (20, 22), (4, 5)])
    def test_matches_reference_implementation(self, neighborhood, smooth):
        img, lab, st = self.random_case((7, 6, 5), seed=1)
        t1, t2 = smooth if smooth else (None, None)
        evo = growcut_evolve(
            img, lab, st, neighborhood=neighborhood, smooth_t1=t1, smooth_t2=t2, max_iterations=60
        )
        offsets = neighborhood_offsets(3, neighborhood)
        ref_l, ref_t = reference_run(img, lab, st, offsets, evo.iterations, t1, t2)
        np.testing.assert_array_equal(evo.labels, ref_l)
        np.testing.assert_allclose(evo.strengths, ref_t, rtol=1e-6)

    def test_iteration_limit(self):
        img, lab, st = self.random_case((20, 20))
        evo = growcut_evolve(img, lab, st, max_iterations=2)
        assert evo.iterations == 2 and evo.converged is False and evo.stop_reason == "max_iterations"

    def test_boundary_smoothing_reduces_raggedness(self):
        img = np.random.default_rng(0).uniform(0.0, 1.0, (64, 64)).astype(np.float32)
        lab = np.zeros((64, 64), dtype=np.int8)
        lab[:, 0], lab[:, -1] = OBJECT, BACKGROUND
        st = (lab > 0).astype(np.float32)

        def perimeter(m: np.ndarray) -> int:
            return int((m[1:] != m[:-1]).sum() + (m[:, 1:] != m[:, :-1]).sum())

        plain = growcut_evolve(img, lab, st, neighborhood="moore")
        same = growcut_evolve(img, lab, st, neighborhood="moore", smooth_t1=9, smooth_t2=9)
        smooth = growcut_evolve(img, lab, st, neighborhood="moore", smooth_t1=6, smooth_t2=6)
        np.testing.assert_array_equal(same.labels, plain.labels)
        m0, m1 = plain.labels == OBJECT, smooth.labels == OBJECT
        assert perimeter(m1) < 0.85 * perimeter(m0)
        assert abs(int(m1.sum()) - int(m0.sum())) < 0.05 * m0.sum()
        assert smooth.converged and smooth.stop_reason in ("stable", "cycle", "label_cycle")


class TestSegmentation:
    def test_wall_only_bladder_like_cod(self):
        volume, inside, center = sphere_volume(48, 0.5, 7.0, "wall")
        res = GrowCut({"domain_size_mm": None}).segment(volume, roi_at(center))
        used = res.params_used
        assert res.status == "ok" and dice(res.mask.data, inside) > 0.7, used
        assert used["converged"] is True and used["stop_reason"] == "stable"
        assert used["touches_domain_boundary"] == [] and used["touches_volume_wall"] == []
        assert used["label_voxels"]["unreached"] == 0 and used["warnings"] == []

    @pytest.mark.parametrize("neighborhood", ["von_neumann", "moore"])
    def test_gas_bladder_both_neighborhoods(self, neighborhood):
        volume, inside, center = sphere_volume(40, 0.5, 6.0, "gas")
        res = GrowCut({"domain_size_mm": None, "neighborhood": neighborhood}).segment(volume, roi_at(center))
        assert dice(res.mask.data, inside) > 0.8, res.params_used
        assert res.params_used["neighborhood"] == neighborhood

    def test_resampled_compute_grid_and_phi_convention(self):
        volume, inside, center = sphere_volume(64, 0.25, 4.5, "gas")
        res = GrowCut({"compute_spacing_mm": 0.5}).segment(volume, roi_at(center))
        used = res.params_used
        assert used["compute_spacing_mm_used"] == 0.5 and used["compute_shape"] == [32, 32, 32]
        assert dice(res.mask.data, inside) > 0.8, used
        assert res.phi.dtype == np.float32 and res.phi.shape == volume.shape
        np.testing.assert_array_equal(res.phi > 0, res.mask.data)
        assert used["volume_mm3"] == pytest.approx(res.mask.data.sum() * 0.25**3)
        assert used["fg_radius_voxels"] == pytest.approx(6.0)

    def test_domain_clipped_phi_outside_is_far(self):
        volume, _, _ = sphere_volume(40, 0.5, 4.0, "gas")
        seed = (2, 20, 20)
        domain = compute_domain(volume.shape, volume.spacing, seed, 10.0)
        res = GrowCut({"domain_size_mm": 10.0}).segment(volume, roi_at(seed))
        assert res.params_used["domain"]["clipped_to_volume"] is True
        outside = np.ones(volume.shape, dtype=bool)
        outside[domain.slices] = False
        assert not res.mask.data[outside].any()
        assert np.all(res.phi[outside] == unprocessed_phi_value(volume.shape))

    def test_domain_too_small_is_reported(self, caplog):
        volume, _, center = sphere_volume(48, 0.5, 9.0, "gas")
        with caplog.at_level(logging.WARNING, logger="src.models.growcut"):
            res = GrowCut({"domain_size_mm": 6.0}).segment(volume, roi_at(center))
        assert len(res.params_used["touches_domain_boundary"]) == 6
        assert any("za mały" in w for w in res.params_used["warnings"])
        assert "za mały" in caplog.text

    def test_background_shell_limits_object(self):
        volume, _, center = sphere_volume(48, 0.5, 9.0, "gas")
        res = GrowCut({"domain_size_mm": None, "bg_radius_mm": 5.0}).segment(volume, roi_at(center))
        used = res.params_used
        assert used["touches_background_shell"] is True and used["seed_voxels"]["shell"] > 0
        idx = np.indices(volume.shape)
        dist_mm = np.sqrt(sum((idx[a] - center[a]) ** 2 for a in range(3))) * 0.5
        assert not res.mask.data[dist_mm > 5.5].any()
        with pytest.raises(ValueError, match="bg_radius_mm"):
            GrowCut({"bg_radius_mm": 2.0})

    def test_shell_closed_for_neighborhood(self):
        _, vn = background_seeds((21, 21, 21), (10, 10, 10), 6.0, "von_neumann")
        _, mo = background_seeds((21, 21, 21), (10, 10, 10), 6.0, "moore")
        assert vn.sum() < mo.sum()
        from scipy import ndimage as ndi

        for shell, structure in ((vn, ndi.generate_binary_structure(3, 1)), (mo, np.ones((3, 3, 3)))):
            comp, _ = ndi.label(~shell, structure=structure)
            assert comp[10, 10, 10] != comp[0, 0, 0]

    def test_unreached_voxels_are_reported(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[12:16, 12:16, 12:16] = 1.0
        volume = WorkVolume(data=data, affine=np.eye(4), intensity_unit="normalized")
        res = GrowCut({"domain_size_mm": None, "compute_spacing_mm": None}).segment(volume, roi_at((5, 5, 5)))
        assert res.params_used["label_voxels"]["unreached"] == 64
        assert any("nieosiągniętych" in w for w in res.params_used["warnings"])

    def test_too_small_compute_grid_is_clear_error(self):
        volume, _, center = sphere_volume(20, 1.0, 4.0, "gas")
        with pytest.raises(ValueError, match="siatka obliczeń"):
            GrowCut({"domain_size_mm": 2.5, "compute_spacing_mm": 2.0}).segment(volume, roi_at(center))

    def test_null_compute_spacing_is_isotropic_min_spacing(self):
        data = np.full((20, 20, 10), 0.8, dtype=np.float32)
        aniso = WorkVolume(data=data, affine=np.diag([0.5, 0.5, 1.0, 1.0]), intensity_unit="normalized")
        used = (
            GrowCut({"domain_size_mm": None, "compute_spacing_mm": None})
            .segment(aniso, roi_at((10, 10, 5)))
            .params_used
        )
        assert used["compute_spacing_mm_used"] == 0.5 and used["compute_shape"] == [20, 20, 19]
        iso = WorkVolume(data=data, affine=np.diag([0.5, 0.5, 0.5, 1.0]), intensity_unit="normalized")
        used = (
            GrowCut({"domain_size_mm": None, "compute_spacing_mm": None})
            .segment(iso, roi_at((10, 10, 5)))
            .params_used
        )
        assert used["compute_shape"] == [20, 20, 10]

    def test_iteration_limit_warning(self):
        volume, _, center = sphere_volume(32, 0.5, 4.0, "gas")
        res = GrowCut({"max_iterations": 2}).segment(volume, roi_at(center))
        assert res.params_used["iterations"] == 2 and res.params_used["converged"] is False
        assert any("brak zbieżności" in w for w in res.params_used["warnings"])


class TestParamsAndFactory:
    def test_defaults_and_validation(self):
        assert resolve_growcut_params(None) == GROWCUT_DEFAULT_PARAMS
        assert resolve_algorithm_params("growcut", None) == GROWCUT_DEFAULT_PARAMS
        p = resolve_growcut_params({"domain_size_mm": None, "neighborhood": "moore", "smooth_t1": 20})
        assert p["domain_size_mm"] is None and p["neighborhood"] == "moore" and p["smooth_t1"] == 20
        with pytest.raises(ValueError, match="literówka"):
            resolve_growcut_params({"fg_radius": 3.0})
        for bad in (
            {"neighborhood": "hex"},
            {"max_iterations": 0},
            {"smooth_t2": 0},
            {"smooth_t1": True},
            {"fg_radius_mm": -1},
            {"bg_radius_mm": 0},
            {"compute_spacing_mm": "0.5"},
            {"compute_spacing_mm": 80.0},
            {"domain_size_mm": 0.4},
        ):
            with pytest.raises(ValueError):
                resolve_growcut_params(bad)

    def test_make_segmenter(self):
        gc = make_segmenter("growcut", {"neighborhood": "moore"})
        assert isinstance(gc, GrowCut) and gc.params["neighborhood"] == "moore"
        with pytest.raises(ValueError, match="literówka"):
            make_segmenter("growcut", {"neighbourhood": "moore"})
