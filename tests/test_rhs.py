"""Coefficient assembly, the right-hand side and the formation rate.

Phase 5's gate. Three levels, because agreement at one level can hide
compensating errors at another:

1. the assembled tensors against ``get_rate_coefs``;
2. ``dc/dt`` against the ``feval`` goldens;
3. J against the ``formation`` goldens.

The dc/dt threshold is split. Physically plausible states are held to 1e-12;
the log-uniform random states span 19 orders of magnitude in concentration
and are held to 1e-11. That is not slack for the port -- see
``test_accuracy_is_driven_by_dynamic_range``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from acdc_jax import rates, rhs
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = REPO / "validation/goldens"

sys.path.insert(0, str(REPO / "validation"))
sys.path.insert(0, str(REPO / "validation/reference"))
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

TEMPERATURES = (250.0, 280.0, 298.15, 320.0)
CS_REF, IPR = 1e-3, 3.0e6

GATE_PHYSICAL = 1e-12
GATE_STRESS = 1e-11

# The first three golden states are structured; the rest are log-uniform
# random over 1e-6 to 1e14 m^-3.
N_STRUCTURED = 3


@pytest.fixture(scope="module")
def built():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    boundary = BoundarySystem(cluster_set)
    reactions = enumerate_reactions(system, boundary)
    energies = parse_energy_file(
        INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
    )
    inputs = rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )
    return system, reactions, inputs


@pytest.fixture(scope="module")
def goldens():
    path = GOLDENS / "rhs.npz"
    if not path.exists():
        pytest.skip("goldens not captured")
    return np.load(path)


def _coefficients(built, temperature):
    system, reactions, inputs = built
    return rhs.assemble(system, reactions, inputs, temperature, CS_REF, IPR, IPR)


def _worst(got: np.ndarray, expected: np.ndarray) -> float:
    scale = np.maximum(np.abs(got), np.abs(expected))
    mask = scale > 0
    if not mask.any():
        return 0.0
    return float((np.abs(got - expected)[mask] / scale[mask]).max())


class TestAssembly:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_coef_quad_matches(self, built, temperature: float) -> None:
        coefficients = _coefficients(built, temperature)
        expected, _ = reference.get_rate_coefs(
            reference.make_coef(temperature, CS_REF, IPR)
        )
        got = np.asarray(coefficients.coef_quad)
        np.testing.assert_array_equal(got != 0, expected != 0)
        assert _worst(got, expected) < GATE_PHYSICAL

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_coef_lin_matches(self, built, temperature: float) -> None:
        coefficients = _coefficients(built, temperature)
        _, expected = reference.get_rate_coefs(
            reference.make_coef(temperature, CS_REF, IPR)
        )
        got = np.asarray(coefficients.coef_lin)
        np.testing.assert_array_equal(got != 0, expected != 0)
        assert _worst(got, expected) < GATE_PHYSICAL

    def test_rate_is_set_not_accumulated_per_product(self, built) -> None:
        """A boundary collision can name the same species twice.

        `2N1P + neg` recombines to `2N`, which is outside the set and strips
        back to `1N + 1 N`, so 1N is both the surviving cluster and the
        stripped monomer. Accumulating the rate once per product makes that
        channel 2x too fast; the multiplicity must be summed instead. Four
        entries in this system, caught by the coef_quad comparison.
        """
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        got = np.asarray(coefficients.coef_quad)
        i = system.labels.index("2N1P")
        j = system.generic_neg
        k = system.labels.index("1N")
        assert got[i, j, k] == pytest.approx(1.6e-12, rel=1e-12)
        assert coefficients.multiplicity[i, j, k] == 2

    def test_pinned_vapours(self, built) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        assert coefficients.isconst[system.labels.index("1A")]
        assert coefficients.isconst[system.labels.index("1N")]
        assert not coefficients.isconst[system.generic_neg], (
            "ionic monomers stay free so they can respond to ion production"
        )

    def test_ion_sources(self, built) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        source = np.asarray(coefficients.source)
        assert source[system.generic_neg] == IPR
        assert source[system.generic_pos] == IPR
        assert np.count_nonzero(source) == 2


class TestRHS:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_structured_states(self, built, goldens, temperature: float) -> None:
        """Phase 5 gate on physically plausible states."""
        coefficients = _coefficients(built, temperature)
        expected = goldens[f"dcdt_{temperature:g}"]
        for c, want in zip(
            goldens["states"][:N_STRUCTURED], expected[:N_STRUCTURED], strict=True
        ):
            got = np.asarray(rhs.rhs(coefficients, c))
            assert _worst(got, want) < GATE_PHYSICAL

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_stress_states(self, built, goldens, temperature: float) -> None:
        """Log-uniform states spanning 19 orders of magnitude."""
        coefficients = _coefficients(built, temperature)
        expected = goldens[f"dcdt_{temperature:g}"]
        for c, want in zip(
            goldens["states"][N_STRUCTURED:], expected[N_STRUCTURED:], strict=True
        ):
            got = np.asarray(rhs.rhs(coefficients, c))
            assert _worst(got, want) < GATE_STRESS

    def test_accuracy_is_driven_by_dynamic_range(self, built, goldens) -> None:
        """Justifies the split threshold.

        The looser bound is a property of the test states, not of the port.
        Physical states agree to ~1e-14; the random ones reach ~1e-12 purely
        because a 1e19 spread in concentration maximises sensitivity to the
        order in which terms are summed, and this port sums by scatter-add
        while the reference sums by nested loop. Floating-point addition is
        not associative.
        """
        coefficients = _coefficients(built, 298.15)
        expected = goldens["dcdt_298.15"]
        states = goldens["states"]

        structured = max(
            _worst(np.asarray(rhs.rhs(coefficients, c)), want)
            for c, want in zip(
                states[:N_STRUCTURED], expected[:N_STRUCTURED], strict=True
            )
        )
        stress = max(
            _worst(np.asarray(rhs.rhs(coefficients, c)), want)
            for c, want in zip(
                states[N_STRUCTURED:], expected[N_STRUCTURED:], strict=True
            )
        )
        assert structured < 1e-13
        assert stress > structured * 10

    def test_empty_state_gives_only_ion_sources(self, built) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        f = np.asarray(rhs.rhs(coefficients, np.zeros(system.n_equations)))
        np.testing.assert_array_equal(
            np.flatnonzero(f), sorted([system.generic_neg, system.generic_pos])
        )

    def test_pinned_species_have_zero_derivative(self, built, goldens) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        for c in goldens["states"]:
            f = np.asarray(rhs.rhs(coefficients, c))
            assert f[system.labels.index("1A")] == 0.0
            assert f[system.labels.index("1N")] == 0.0

    def test_flux_counters_never_decrease(self, built, goldens) -> None:
        """out_* and coag are monotone accumulators: they appear only as
        products, so a non-negative state cannot make them fall."""
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        slots = [
            system.flux_index[name]
            for name in ("out_neu", "out_neg", "out_pos", "coag")
        ]
        for c in goldens["states"]:
            if np.any(c < 0):
                continue
            f = np.asarray(rhs.rhs(coefficients, c))
            assert np.all(f[slots] >= 0.0)


class TestFormationRate:
    def test_matches_reference(self, built) -> None:
        system, _, _ = built
        path = GOLDENS / "formation.npz"
        if not path.exists():
            pytest.skip("goldens not captured")
        goldens = np.load(path)
        coefficients = _coefficients(built, 280.0)
        for c, want in zip(goldens["states"], goldens["j_tot"], strict=True):
            got = float(rhs.formation_rate(system, coefficients, c)["j_tot"])
            if want > 0:
                assert abs(got - want) / want < GATE_PHYSICAL

    def test_channels_fold_recombination_into_neutral(self, built) -> None:
        """The reference returns four channels and the driver adds the
        recombination one to the neutral channel
        (``driver_acdc_J.f90:363``). This port returns the three folded
        channels directly.
        """
        system, _, _ = built
        path = GOLDENS / "formation.npz"
        if not path.exists():
            pytest.skip("goldens not captured")
        goldens = np.load(path)
        coefficients = _coefficients(built, 280.0)
        for c, want in zip(goldens["states"], goldens["j_by_charge"], strict=True):
            got = np.asarray(
                rhs.formation_rate(system, coefficients, c)["j_by_channel"]
            )
            folded = np.array([want[0] + want[3], want[1], want[2]])
            mask = folded > 0
            if mask.any():
                assert (np.abs(got - folded)[mask] / folded[mask]).max() < GATE_PHYSICAL

    def test_is_not_double_counted(self, built) -> None:
        """The dense coef_quad stores both orderings of every pair, so
        summing it whole nearly doubles J.

        Nearly, not exactly: a self-collision appears once rather than
        twice, and already carries a half. Computing J from the enumerated
        collision list sidesteps the whole question. This pins the size of
        the error the naive route would make -- it was a factor of 2 in the
        first draft, caught by the golden comparison.
        """
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.full(system.n_equations, 1e12)
        got = float(rhs.formation_rate(system, coefficients, c)["j_tot"])

        slots = [system.flux_index[n] for n in ("out_neu", "out_neg", "out_pos")]
        quad = np.asarray(coefficients.coef_quad)
        naive = sum(float((quad[:, :, s] * 1e12 * 1e12).sum()) for s in slots)
        assert 1.9 < naive / got <= 2.0

    def test_zero_when_nothing_can_grow_out(self, built) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.zeros(system.n_equations)
        c[system.labels.index("1A")] = 1e13
        assert float(rhs.formation_rate(system, coefficients, c)["j_tot"]) == 0.0


class TestJacobian:
    def test_jacfwd_is_finite_and_correctly_shaped(self, built) -> None:
        """The reference ships an EMPTY jeval and runs VODE with mf=22, a
        finite-difference Jacobian costing neqn extra RHS calls per build.
        Here it is exact and free."""
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.full(system.n_equations, 1e12)
        jacobian = jax.jacfwd(lambda x: rhs.rhs(coefficients, x))(c)
        assert jacobian.shape == (system.n_equations, system.n_equations)
        assert bool(np.all(np.isfinite(jacobian)))

    def test_matches_central_differences(self, built) -> None:
        """Compared at a moderate concentration, where finite differences
        are actually valid.

        The RHS is quadratic, so a central difference has no truncation
        error -- only roundoff. But roundoff in `f(+) - f(-)` scales with
        the magnitude of `f` itself, and at atmospheric concentrations
        (~1e12 m^-3) `f` is enormous compared with `c * df/dc`. Measured
        agreement degrades linearly with concentration:

            c = 1e4   4.5e-10
            c = 1e6   4.5e-08
            c = 1e8   4.5e-06
            c = 1e12  2.4e-05

        That is a statement about finite differences, not about `jacfwd`,
        which is exact. So the check is done at 1e4 where FD is trustworthy;
        `test_jacobian_is_exact_for_a_quadratic` covers the physical scale
        by a route that does not difference anything.
        """
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.full(system.n_equations, 1e4)
        jacobian = np.asarray(jax.jacfwd(lambda x: rhs.rhs(coefficients, x))(c))

        for k in (1, 5, 20, 40):
            plus, minus = c.copy(), c.copy()
            plus[k] *= 1.01
            minus[k] *= 0.99
            finite = (
                np.asarray(rhs.rhs(coefficients, plus))
                - np.asarray(rhs.rhs(coefficients, minus))
            ) / (2 * c[k] * 0.01)
            scale = max(np.abs(finite).max(), 1e-30)
            assert np.abs(jacobian[:, k] - finite).max() / scale < 1e-8

    def test_jacobian_is_exact_for_a_quadratic(self, built) -> None:
        """A check at the physical scale that never differences the RHS.

        The right-hand side is exactly quadratic in c, so for any step v

            f(c + v) = f(c) + J(c) v + Q(v, v)

        with Q the same quadratic form evaluated at v. Taking v = c gives
        f(2c) = f(c) + J(c)c + Q(c,c), and since Q(c,c) = f(c) - f(0) -
        J(0)c, everything is available without cancellation-prone small
        differences. Holding this identity at 1e12 m^-3 confirms the
        Jacobian at the concentrations that matter.
        """
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.full(system.n_equations, 1e12)
        zero = np.zeros(system.n_equations)

        f_of = lambda x: np.asarray(rhs.rhs(coefficients, x))  # noqa: E731
        jac = lambda x: np.asarray(  # noqa: E731
            jax.jacfwd(lambda y: rhs.rhs(coefficients, y))(x)
        )

        quadratic = f_of(c) - f_of(zero) - jac(zero) @ c
        predicted = f_of(c) + jac(c) @ c + quadratic
        actual = f_of(2 * c)

        scale = np.maximum(np.abs(predicted), np.abs(actual))
        mask = scale > 0
        assert (np.abs(predicted - actual)[mask] / scale[mask]).max() < 1e-12

    def test_pinned_rows_are_zero(self, built) -> None:
        system, _, _ = built
        coefficients = _coefficients(built, 280.0)
        c = np.full(system.n_equations, 1e12)
        jacobian = np.asarray(jax.jacfwd(lambda x: rhs.rhs(coefficients, x))(c))
        assert np.all(jacobian[system.labels.index("1A"), :] == 0.0)
