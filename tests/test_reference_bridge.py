"""The f2py bridge must be correct before anything can be validated against it.

These tests check the bridge itself, not the port. They exist because the
default f2py behaviour for this code is silently wrong: the Fortran declares
no `intent`, so without the hand-written .pyf every output array comes back
untouched -- full of zeros, with no error raised. A port validated against
that would agree perfectly with nothing.

Expected counts are taken from the emitted source, so they also serve as a
check that the vendored reference has not been swapped underneath us.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))

try:
    import reference
except ImportError as exc:  # the compiled bridge is absent
    # `pytest.importorskip` does not cover this: the package imports and
    # then raises because its .so is missing, which pytest treats as an
    # error rather than a missing module. CI without gfortran found it.
    pytest.skip(
        f"Fortran bridge not built ({exc}); "
        "run: uv run python validation/reference/build.py",
        allow_module_level=True,
    )

T_REF = 280.0
COEF = reference.make_coef(temperature=T_REF, cs_ref=1e-3, ipr_neg=3.0e6)


@pytest.fixture(scope="module")
def k() -> np.ndarray:
    return reference.get_coll(T_REF)


class TestCollisionCoefficients:
    def test_shape_and_symmetry(self, k: np.ndarray) -> None:
        assert k.shape == (54, 54)
        np.testing.assert_array_equal(k, k.T)

    def test_nonzero_count(self, k: np.ndarray) -> None:
        """2131 = 125 neutral-neutral + 587 ion-neutral + 361 recombination
        + 1058 symmetry copies, i.e. every `K(` assignment in get_coll."""
        assert np.count_nonzero(k) == 2131

    def test_hard_sphere_matches_emitted_literal(self, k: np.ndarray) -> None:
        """K(1,1) = 2.00300435981201e-17 * sqrt(T), 1A + 1A.

        Exact equality, not a tolerance: this is the same literal multiplied
        by the same sqrt in the same precision.
        """
        assert k[0, 0] == 2.00300435981201e-17 * np.sqrt(T_REF)

    def test_recombination_is_a_flat_constant(self, k: np.ndarray) -> None:
        """Every opposite-charge pair, no size dependence (Perl :371)."""
        assert k[reference.IDX_POS_ION, reference.IDX_NEG_ION] == 1.6e-12
        neg = list(range(16, 34)) + [reference.IDX_NEG_ION]
        pos = list(range(34, 52)) + [reference.IDX_POS_ION]
        block = k[np.ix_(neg, pos)]
        assert block.shape == (19, 19)
        assert np.all(block == 1.6e-12)

    def test_same_sign_pairs_are_forbidden(self, k: np.ndarray) -> None:
        neg = list(range(16, 34)) + [reference.IDX_NEG_ION]
        assert np.all(k[np.ix_(neg, neg)] == 0.0)

    def test_scales_as_sqrt_t_for_neutral_pairs(self) -> None:
        """Hard-sphere beta goes as sqrt(T); ion-neutral does not, so this
        is asserted only on the neutral-neutral block."""
        k1, k2 = reference.get_coll(250.0), reference.get_coll(1000.0)
        neutral = slice(0, 16)
        a, b = k1[neutral, neutral], k2[neutral, neutral]
        mask = a > 0
        ratio = b[mask] / a[mask]
        np.testing.assert_allclose(ratio, np.sqrt(1000.0 / 250.0), rtol=1e-12)


class TestEvaporationCoefficients:
    def test_shape_and_nonzero_count(self) -> None:
        """422 = 213 explicit formulas + 209 symmetry copies."""
        e = reference.get_evap(T_REF)
        assert e.shape == (54, 54)
        assert np.count_nonzero(e) == 422

    def test_finite_and_nonnegative(self) -> None:
        e = reference.get_evap(T_REF)
        assert np.all(np.isfinite(e))
        assert np.all(e >= 0.0)

    def test_strongly_temperature_dependent(self) -> None:
        """E carries exp(dG/kT), so it must move by orders of magnitude
        where K moves by a factor sqrt(T)."""
        cold, hot = reference.get_evap(250.0), reference.get_evap(320.0)
        mask = cold > 0
        assert (hot[mask] / cold[mask]).max() > 1e3


class TestLosses:
    def test_exp_loss_size_dependence(self) -> None:
        """cs_i = (d_i/d_ref)^-1.6, verified analytically for 2A.

        2A has twice the volume of 1A, so d ratio is 2^(1/3).
        """
        cs = reference.get_losses()
        assert cs.shape == (54,)
        np.testing.assert_allclose(cs[1], (2 ** (1 / 3)) ** -1.6, rtol=1e-12)

    def test_vapour_monomers_excluded(self) -> None:
        """Generated with --cs_only 1A,0 --cs_only 1N,0: indices 0 and 2."""
        cs = reference.get_losses()
        assert cs[0] == 0.0
        assert cs[2] == 0.0

    def test_fcr_is_unity(self) -> None:
        assert reference.get_fcr() == 1.0


class TestRHS:
    @pytest.fixture
    def c(self) -> np.ndarray:
        c = np.zeros(reference.NEQ)
        c[0] = 1e13  # [1A], m^-3
        c[2] = 1e15  # [1N], m^-3
        return c

    def test_reproducible(self, c: np.ndarray) -> None:
        """The whole point of controlling ipar.

        feval keeps its coefficient tensors in `save`d arrays gated on
        ipar(1), which it mutates. If the bridge leaked that state between
        calls, the second call would skip re-initialisation and could return
        something computed for different ambient conditions.
        """
        f1 = reference.feval(c, COEF)
        f2 = reference.feval(c, COEF)
        np.testing.assert_array_equal(f1, f2)

    def test_shape_and_finite(self, c: np.ndarray) -> None:
        f = reference.feval(c, COEF)
        assert f.shape == (reference.NEQ,)
        assert np.all(np.isfinite(f))

    def test_constant_monomers_have_zero_derivative(self, c: np.ndarray) -> None:
        """Under solve_ss the neutral monomers are pinned
        (acdc_simulation_setup.f90:84)."""
        f = reference.feval(c, COEF)
        assert f[0] == 0.0  # 1A
        assert f[2] == 0.0  # 1N

    def test_empty_state_gives_only_ion_source(self) -> None:
        """With nothing present, the only nonzero terms are the zeroth-order
        ion production sources on the two generic charger ions."""
        f = reference.feval(np.zeros(reference.NEQ), COEF)
        nonzero = np.flatnonzero(f)
        np.testing.assert_array_equal(
            nonzero, [reference.IDX_NEG_ION, reference.IDX_POS_ION]
        )
        assert f[reference.IDX_NEG_ION] == COEF[2]
        assert f[reference.IDX_POS_ION] == COEF[3]

    def test_dimer_forms_from_monomers(self, c: np.ndarray) -> None:
        """1A + 1A -> 2A must give a positive d[2A]/dt."""
        f = reference.feval(c, COEF)
        assert f[1] > 0.0

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            reference.feval(np.zeros(10), COEF)


class TestFormation:
    def test_zero_when_nothing_can_grow_out(self) -> None:
        c = np.zeros(reference.NEQ)
        c[0] = 1e13
        assert reference.formation(c, COEF)["j_tot"] == 0.0

    def test_returns_the_discarded_attribution(self) -> None:
        """j_by_cluster and j_all are computed by the reference and thrown
        away by its driver. The bridge exposes them."""
        c = np.full(reference.NEQ, 1e12)
        out = reference.formation(c, COEF)
        assert out["j_tot"] > 0.0
        assert out["j_by_charge"].shape == (4,)
        assert out["j_by_cluster"].shape == (reference.NEQ,)
        assert out["j_all"].shape == (reference.NEQ, 4)
        np.testing.assert_allclose(out["j_by_charge"].sum(), out["j_tot"], rtol=1e-12)


class TestRateCoefTensors:
    def test_shapes_and_sparsity(self) -> None:
        cq, cl = reference.get_rate_coefs(COEF)
        assert cq.shape == (54, 54, 63)
        assert cl.shape == (63, 63, 54)
        assert np.count_nonzero(cq) == 2679
        assert np.count_nonzero(cl) == 474

    def test_self_collision_carries_the_half_factor(self) -> None:
        """coef_quad(i,i,k) = 0.5*K(i,i) because the pair is enumerated twice
        (acdc_equations_AN_ions_example.f90:1017)."""
        cq, _ = reference.get_rate_coefs(COEF)
        k = reference.get_coll(T_REF)
        assert cq[0, 0, 1] == 0.5 * k[0, 0]  # 1A + 1A -> 2A

    def test_coagulation_sink_is_scaled_by_cs_ref(self) -> None:
        """get_rate_coefs:1011 does cs = coef(2)*cs."""
        _, cl = reference.get_rate_coefs(COEF)
        cs = reference.get_losses()
        assert cl[reference.IDX_COAG, reference.IDX_COAG, 1] == COEF[1] * cs[1]
