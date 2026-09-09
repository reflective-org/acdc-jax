"""Gradients through the steady-state solve. Phase 7's gate.

The capability the port exists for. Kept deliberately small -- each test
differentiates through a root-find, so the suite pays a few seconds per
assertion.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from acdc_jax import rates, sensitivity
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"

CONDITIONS = sensitivity.Conditions(
    c_a=1e13, c_n=1e15, temperature=280.0, cs_ref=1e-3, ipr=3.0e6
)
EXAMPLE_J = 2217995.192415948

GATE_GRADIENT = 1e-3
"""Against central finite differences. Measured ~2e-7."""


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


@pytest.fixture(scope="module")
def free_energy_sensitivity(model):
    system, reactions, inputs = model
    return sensitivity.sensitivity_to_free_energies(
        system, reactions, inputs, CONDITIONS
    )


class TestDifferentiableParameters:
    def test_reference_state_species_are_excluded(self, model) -> None:
        """Formation energies are relative to the free monomers, so a
        monomer's is zero by construction -- not something anyone can be
        wrong about, and not a valid parameter to differentiate.

        Left unmasked, 1A and 1N dominate dJ/dH for this system (+1.16 and
        +0.65 per kcal/mol, ahead of every real cluster), which reads as a
        physical result and is not one: it is the reference state moving.
        """
        system, _, inputs = model
        excluded = {system.labels[i] for i in np.flatnonzero(~inputs.has_energy_data)}
        assert excluded == {"1A", "1N", "1B", "neg", "pos"}
        assert inputs.has_energy_data.sum() == 49

    def test_masked_species_have_zero_gradient(
        self, model, free_energy_sensitivity
    ) -> None:
        system, _, inputs = model
        d_dh = np.asarray(free_energy_sensitivity["d_dh"])
        for i in np.flatnonzero(~inputs.has_energy_data):
            assert d_dh[i] == 0.0, system.labels[i]


class TestFreeEnergyGradients:
    def test_j_is_unchanged_by_differentiating(self, free_energy_sensitivity) -> None:
        got = float(free_energy_sensitivity["j"])
        assert abs(got - EXAMPLE_J) / EXAMPLE_J < 1e-5

    def test_finite_and_nonzero(self, free_energy_sensitivity) -> None:
        d_dh = np.asarray(free_energy_sensitivity["d_dh"])
        assert np.all(np.isfinite(d_dh))
        assert np.count_nonzero(d_dh) > 20

    def test_matches_finite_differences(self, model, free_energy_sensitivity) -> None:
        """Phase 7 gate. Checked on the most sensitive clusters, where the
        finite difference is best conditioned."""
        system, reactions, inputs = model
        gradient = np.asarray(free_energy_sensitivity["d_dh"])
        h0 = np.asarray(inputs.delta_h)

        for i in np.argsort(-np.abs(gradient))[:3]:
            eps = 1e-3  # kcal/mol
            plus, minus = h0.copy(), h0.copy()
            plus[i] += eps
            minus[i] -= eps
            j_plus = float(
                sensitivity.formation_rate_of(
                    system, reactions, inputs, CONDITIONS, delta_h=jnp.asarray(plus)
                )
            )
            j_minus = float(
                sensitivity.formation_rate_of(
                    system, reactions, inputs, CONDITIONS, delta_h=jnp.asarray(minus)
                )
            )
            finite = (j_plus - j_minus) / (2 * eps)
            relative = abs(gradient[i] - finite) / max(abs(finite), 1e-30)
            assert relative < GATE_GRADIENT, system.labels[i]

    def test_entropy_gradient_is_finite(self, free_energy_sensitivity) -> None:
        d_ds = np.asarray(free_energy_sensitivity["d_ds"])
        assert np.all(np.isfinite(d_ds))
        assert np.count_nonzero(d_ds) > 20

    def test_sensitivities_are_physically_modest(self, free_energy_sensitivity) -> None:
        """A sanity bound, not a reference comparison.

        The largest fractional change in J per kcal/mol is ~0.14 for this
        system. A value far above 1 would mean a single cluster's energy
        controls the answer outright, which would be a red flag about either
        the cluster set or the gradient.
        """
        relative = np.asarray(free_energy_sensitivity["relative_dh"])
        assert 0.01 < np.abs(relative).max() < 1.0


class TestConditionGradients:
    def test_apparent_nucleation_order(self, model) -> None:
        """d ln J / d ln [H2SO4] -- the quantity measurements report.

        Observed values are typically 1 to 3 for sulfuric-acid systems, so
        anything far outside that would indicate a problem. This is a
        plausibility check on physics, not a validation against the
        reference, which cannot produce this number at all.
        """
        system, reactions, inputs = model
        result = sensitivity.sensitivity_to_conditions(
            system, reactions, inputs, CONDITIONS
        )
        order = float(result["d_ln_j_d_ln_c_a"])
        assert 0.5 < order < 5.0

    def test_signs_are_physical(self, model) -> None:
        """More vapour and more ions increase J; a larger sink decreases it."""
        system, reactions, inputs = model
        result = sensitivity.sensitivity_to_conditions(
            system, reactions, inputs, CONDITIONS
        )
        assert float(result["d_c_a"]) > 0
        assert float(result["d_cs_ref"]) < 0

    def test_all_finite(self, model) -> None:
        system, reactions, inputs = model
        result = sensitivity.sensitivity_to_conditions(
            system, reactions, inputs, CONDITIONS
        )
        for key, value in result.items():
            assert np.isfinite(float(value)), key


class TestBatching:
    """Phase 7.4: the whole steady-state solve under `vmap`."""

    def test_matches_the_loop(self, model) -> None:
        """The gate the plan asked for was bitwise equality. It is not
        achievable and should not be: under `vmap` diffrax's adaptive
        controller drives the whole batch on one clock, so the step
        sequence differs from a solo solve and the trajectories part at the
        last couple of digits. Measured worst case over the sweep is 6e-13,
        which is eight orders below the 1e-5 the steady-state criterion
        itself defines J to."""
        system, reactions, inputs = model
        c_a = np.logspace(12, 13, 3)
        args = (system, reactions, inputs, c_a, 1e15, 280.0, 1e-3, 3.0e6)
        loop = sensitivity.sweep(*args)
        batched = sensitivity.sweep(*args, batched=True)
        assert np.all(np.isfinite(batched))
        worst = float(np.max(np.abs(batched - loop) / np.abs(loop)))
        assert worst < 1e-9, worst

    def test_broadcasts_a_temperature_grid(self, model) -> None:
        """Any field may be the batched one; a scalar rides along."""
        import jax.numpy as jnp

        system, reactions, inputs = model
        temperatures = jnp.array([270.0, 280.0, 290.0])
        j = sensitivity.formation_rate_batch(
            system,
            reactions,
            inputs,
            sensitivity.Conditions(1e13, 1e15, temperatures, 1e-3, 3.0e6),
        )
        assert j.shape == (3,)
        assert np.all(np.isfinite(j))
        # Evaporation grows with temperature faster than collision does, so
        # at fixed vapour the formation rate falls.
        assert j[0] > j[1] > j[2]

    def test_result_takes_the_broadcast_shape(self, model) -> None:
        import jax.numpy as jnp

        system, reactions, inputs = model
        j = sensitivity.formation_rate_batch(
            system,
            reactions,
            inputs,
            sensitivity.Conditions(
                jnp.array([[1e13], [2e13]]),
                1e15,
                jnp.array([275.0, 285.0]),
                1e-3,
                3.0e6,
            ),
        )
        assert j.shape == (2, 2)
