"""Time integration and steady state. Phase 6's gate.

The headline result of the port: the steady-state formation rate against the
Fortran binary, and the two solution paths against each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from acdc_jax import rates, rhs, solve
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = REPO / "validation/goldens"

sys.path.insert(0, str(REPO / "validation/reference"))

# The bundled example, whose answer the `run` binary prints.
EXAMPLE = {"c_a": 1e13, "c_n": 1e15, "temperature": 280.0, "cs": 1e-3, "ipr": 3.0e6}
EXAMPLE_J = 2217995.192415948

GATE_J = 1e-5
"""Against the reference. Cannot be tighter: its steady state is only
defined to within sstol -- see docs/validation.md."""

GATE_BETWEEN_PATHS = 1e-8
"""Between our own two paths, both deterministic."""


@pytest.fixture(scope="module")
def model():
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


def _setup(model, c_a, c_n, temperature, cs, ipr):
    system, reactions, inputs = model
    coefficients = rhs.assemble(system, reactions, inputs, temperature, cs, ipr, ipr)
    c0 = solve.set_vapours(
        system, jnp.zeros(system.n_equations), {"1A": c_a, "1N": c_n}
    )
    return system, coefficients, c0


class TestSteadyStateAgainstReference:
    def test_bundled_example(self, model) -> None:
        """The headline number: J for the example the `run` binary ships."""
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        result = solve.steady_state_by_integration(system, coefficients, c0)
        assert result.converged
        got = float(result.j_total)
        assert abs(got - EXAMPLE_J) / EXAMPLE_J < GATE_J

    def test_grid(self, model) -> None:
        """Phase 6 gate over the captured (vapour, T, CS, IPR) grid."""
        path = GOLDENS / "steadystate.npz"
        if not path.exists():
            pytest.skip("steady-state goldens not captured")
        goldens = np.load(path)
        conditions, expected = goldens["conditions"], goldens["j"]

        # A representative slice rather than all 60: each solve integrates a
        # stiff system to 1e5 s, and the grid is fully swept by
        # validation/compare_steadystate.py.
        for row in range(0, len(conditions), 11):
            c_a, c_n, temperature, cs, ipr = conditions[row]
            system, coefficients, c0 = _setup(
                model, c_a, c_n, float(temperature), float(cs), float(ipr)
            )
            result = solve.steady_state_by_integration(system, coefficients, c0)
            got = float(result.j_total)
            want = float(expected[row])
            assert abs(got - want) / want < GATE_J, (
                f"[A]={c_a:.1e} [N]={c_n:.1e} T={temperature} CS={cs}: "
                f"got {got:.6e}, want {want:.6e}"
            )


class TestTolerance:
    def test_j_is_insensitive_to_atol(self, model) -> None:
        """Backs fidelity F14 with a measurement rather than a claim.

        This port loosens `atol` from the reference's 1e-6 m^-3 to 1.0,
        because Kvaerno5 cannot meet 1e-6 on the evaporation-dominated
        corner of the grid while VODE can. That is only defensible if the
        answer does not depend on it -- so this asserts J is unchanged
        across four orders of magnitude.
        """
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        values = []
        for atol in (1e-2, 1.0, 1e2):
            result = solve.steady_state_by_integration(
                system, coefficients, c0, atol=atol
            )
            assert result.converged
            values.append(float(result.j_total))
        for value in values[1:]:
            assert abs(value - values[0]) / values[0] < 1e-6

    def test_hard_corner_needs_the_loosened_atol(self, model) -> None:
        """The specific point that forced F14: lowest vapour, lowest
        temperature, where clusters are stable and slow to equilibrate.

        At the reference's atol this exhausts the step budget; at the
        port's default it converges in about 100 steps to 5e-7 of the
        Fortran answer.
        """
        system, coefficients, c0 = _setup(
            model, c_a=1e12, c_n=1e15, temperature=250.0, cs=1e-3, ipr=3.0e6
        )
        result = solve.steady_state_by_integration(system, coefficients, c0)
        assert result.converged
        assert result.steps < 1000

        with pytest.raises(Exception, match="maximum number of solver steps"):
            solve.steady_state_by_integration(
                system, coefficients, c0, atol=1e-6, max_steps=20_000
            )


class TestTwoPathsAgree:
    def test_j_matches_between_paths(self, model) -> None:
        """The gate that justifies keeping the root-find at all.

        Both are deterministic and both are ours, so this can be tight where
        the comparison against the reference cannot.
        """
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        integrated = solve.solve_steady_state(
            system, coefficients, method="integrate", c0=c0
        )
        found = solve.solve_steady_state(system, coefficients, method="rootfind", c0=c0)
        a, b = float(integrated.j_total), float(found.j_total)
        assert abs(a - b) / a < GATE_BETWEEN_PATHS

    def test_concentrations_match_between_paths(self, model) -> None:
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        integrated = solve.solve_steady_state(
            system, coefficients, method="integrate", c0=c0
        )
        found = solve.solve_steady_state(system, coefficients, method="rootfind", c0=c0)
        a = np.asarray(integrated.concentrations[: system.n_clusters])
        b = np.asarray(found.concentrations[: system.n_clusters])
        scale = np.maximum(np.abs(a), np.abs(b))
        mask = scale > 0
        assert (np.abs(a - b)[mask] / scale[mask]).max() < 1e-6

    def test_rootfind_stays_positive(self, model) -> None:
        """Newton is solved in log space precisely so that it cannot produce
        a negative concentration, where the evaporation terms are
        meaningless."""
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        found = solve.solve_steady_state(system, coefficients, method="rootfind", c0=c0)
        assert bool(jnp.all(found.concentrations[: system.n_clusters] >= 0))

    def test_rootfind_underreports_convergence(self, model) -> None:
        """Documents fidelity F13 rather than pretending it is fixed.

        The relative residual floors near 6e-5 because f is a small
        difference of large opposing fluxes, so optimistix never certifies
        even when the answer is right. Judge this path by the inter-path
        agreement above.
        """
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        found = solve.solve_steady_state(system, coefficients, method="rootfind", c0=c0)
        assert not found.converged
        assert abs(float(found.j_total) - EXAMPLE_J) / EXAMPLE_J < GATE_J


class TestPurity:
    def test_solve_is_a_pure_function(self, model) -> None:
        """The reference is not.

        `acdc_plugin` warm-starts from a `save`d concentration vector, so
        the same inputs give different answers depending on call history --
        measured at 1.5e-6 in Phase 0. Here two identical calls must give
        bit-identical results.
        """
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        first = solve.steady_state_by_integration(system, coefficients, c0)
        second = solve.steady_state_by_integration(system, coefficients, c0)
        assert float(first.j_total) == float(second.j_total)
        np.testing.assert_array_equal(
            np.asarray(first.concentrations), np.asarray(second.concentrations)
        )

    def test_call_order_does_not_matter(self, model) -> None:
        """Solving a different point in between changes nothing."""
        system, coefficients, c0 = _setup(model, **EXAMPLE)
        first = solve.steady_state_by_integration(system, coefficients, c0)

        _, other_coefficients, other_c0 = _setup(
            model, c_a=1e12, c_n=1e16, temperature=300.0, cs=1e-2, ipr=3.0e6
        )
        solve.steady_state_by_integration(system, other_coefficients, other_c0)

        again = solve.steady_state_by_integration(system, coefficients, c0)
        assert float(first.j_total) == float(again.j_total)


class TestIntegration:
    def test_trajectory_is_positive_and_finite(self, model) -> None:
        import diffrax

        system, coefficients, c0 = _setup(model, **EXAMPLE)
        times = jnp.array([1e-6, 1e-2, 1.0, 100.0, 1000.0])
        solution = solve.integrate(
            system, coefficients, c0, t1=1000.0, saveat=diffrax.SaveAt(ts=times)
        )
        trajectory = np.asarray(solution.ys)
        assert np.all(np.isfinite(trajectory))
        assert np.all(trajectory >= -config_negtol())

    def test_j_rises_towards_steady_state(self, model) -> None:
        """A physical sanity check independent of the reference: starting
        from bare vapour, clusters must build up before any particles can
        grow out, so J increases monotonically towards its plateau."""
        import diffrax

        system, coefficients, c0 = _setup(model, **EXAMPLE)
        times = jnp.array([1.0, 10.0, 100.0, 1000.0])
        solution = solve.integrate(
            system, coefficients, c0, t1=1000.0, saveat=diffrax.SaveAt(ts=times)
        )
        js = []
        for y in solution.ys:
            full = jnp.zeros(system.n_equations).at[: system.n_clusters].set(y)
            js.append(float(rhs.formation_rate(system, coefficients, full)["j_tot"]))
        assert all(b >= a for a, b in zip(js, js[1:], strict=False))


def config_negtol() -> float:
    from acdc_jax import config

    return abs(config.NEGTOL)


class TestPhysicalBehaviour:
    def test_j_increases_with_sulfuric_acid(self, model) -> None:
        previous = 0.0
        for c_a in (1e12, 3e12, 1e13):
            system, coefficients, c0 = _setup(
                model, c_a=c_a, c_n=1e15, temperature=280.0, cs=1e-3, ipr=3.0e6
            )
            got = float(
                solve.steady_state_by_integration(system, coefficients, c0).j_total
            )
            assert got > previous
            previous = got

    def test_j_decreases_with_coagulation_sink(self, model) -> None:
        results = []
        for cs in (1e-3, 1e-2):
            system, coefficients, c0 = _setup(
                model, c_a=1e13, c_n=1e15, temperature=280.0, cs=cs, ipr=3.0e6
            )
            results.append(
                float(
                    solve.steady_state_by_integration(system, coefficients, c0).j_total
                )
            )
        assert results[1] < results[0]
