"""Rate coefficients. Phase 4's gate.

K, E and cs must match the emitted literals to better than 1e-12 relative,
across a temperature sweep rather than at one point: the reference bakes the
temperature dependence into per-pair constants plus a piecewise branch whose
switch point differs for every pair, so a single-temperature check would
pass with the branch on the wrong side for most of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, rates
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"

sys.path.insert(0, str(REPO / "validation"))
sys.path.insert(0, str(REPO / "validation/reference"))
reference = pytest.importorskip("reference")

TEMPERATURES = (250.0, 280.0, 298.15, 320.0)
GATE = 1e-12


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


def _relative(got: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> float:
    return float((np.abs(got - expected)[mask] / np.abs(expected)[mask]).max())


class TestCollisionCoefficients:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_matches_reference(self, built, temperature: float) -> None:
        """Phase 4 gate for K."""
        _, _, inputs = built
        got = np.asarray(rates.collision_coefficients(inputs, temperature))
        expected = reference.get_coll(temperature)
        assert _relative(got, expected, expected != 0) < GATE

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_sparsity_is_identical(self, built, temperature: float) -> None:
        """Which pairs collide is combinatorial, so exact.

        Charge compatibility alone is not enough: `neg + 1N` has no acid to
        convert and the reference leaves it at zero, as it does the no-op
        boundary collisions the enumeration prunes. Without the validity
        mask this differs by 63 entries.
        """
        _, _, inputs = built
        got = np.asarray(rates.collision_coefficients(inputs, temperature))
        expected = reference.get_coll(temperature)
        np.testing.assert_array_equal(got != 0, expected != 0)

    def test_neutral_pairs_scale_as_sqrt_temperature(self, built) -> None:
        _, _, inputs = built
        cold = np.asarray(rates.collision_coefficients(inputs, 250.0))
        hot = np.asarray(rates.collision_coefficients(inputs, 1000.0))
        neutral = (inputs.pair_kind == rates.PAIR_NEUTRAL) & (cold > 0)
        ratio = hot[neutral] / cold[neutral]
        np.testing.assert_allclose(ratio, np.sqrt(4.0), rtol=1e-12)

    def test_recombination_is_a_flat_constant(self, built) -> None:
        _, _, inputs = built
        k = np.asarray(rates.collision_coefficients(inputs, 280.0))
        recombination = (inputs.pair_kind == rates.PAIR_RECOMBINATION) & (k > 0)
        assert np.all(k[recombination] == config.RECOMB_COEFF)

    def test_same_sign_pairs_are_exactly_zero(self, built) -> None:
        _, _, inputs = built
        k = np.asarray(rates.collision_coefficients(inputs, 280.0))
        assert np.all(k[inputs.pair_kind == rates.PAIR_FORBIDDEN] == 0.0)

    def test_symmetric(self, built) -> None:
        _, _, inputs = built
        k = np.asarray(rates.collision_coefficients(inputs, 280.0))
        np.testing.assert_allclose(k, k.T, rtol=1e-15)

    def test_ion_enhancement_never_falls_below_hard_sphere(self, built) -> None:
        """The reference takes max(ionic, hard sphere) (Perl :8886).

        Not a safety clamp: the Su parameterization can fall below the
        geometric rate for a large, weakly polar neutral, and a collision
        cannot be slower than hard spheres.
        """
        _, _, inputs = built
        k = np.asarray(rates.collision_coefficients(inputs, 280.0))
        beta = np.asarray(rates.hard_sphere(inputs, 280.0))
        ion_neutral = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (k > 0)
        assert np.all(k[ion_neutral] >= beta[ion_neutral] * (1 - 1e-12))

    def test_ion_enhancement_exceeds_the_manual_s_stated_range(self, built) -> None:
        """The manual says the enhancement is "a factor between one and ten".
        It is not a bound, and the REFERENCE breaks it too.

        Measured on the reference's own K at 280 K: the ratio to hard sphere
        runs from 1.17 to 16.95, with 6% of ion-neutral pairs above 10 (the
        largest is 2A1N + pos). So the manual is an approximate
        characterisation of typical magnitude, not a constraint the code
        enforces -- port from the code, not the manual. Recorded as F12.

        Asserted here rather than deleted, so that a future change which
        silently clamped the enhancement to 10 would fail.
        """
        _, _, inputs = built
        k = np.asarray(rates.collision_coefficients(inputs, 280.0))
        beta = np.asarray(rates.hard_sphere(inputs, 280.0))
        mask = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (k > 0)
        ratio = k[mask] / beta[mask]
        assert ratio.min() >= 1.0 - 1e-12, "never below hard sphere"
        assert ratio.max() > 10.0, "the manual's upper bound is not enforced"
        assert ratio.max() < 20.0, "but it stays the right order of magnitude"

    def test_su82_branch_is_exercised_on_both_sides(self, built) -> None:
        """Otherwise the piecewise formula is only half tested."""
        _, _, inputs = built
        for temperature in TEMPERATURES:
            x_low = x_high = False
            enhancement = np.asarray(rates.su82_enhancement(inputs, temperature))
            # Recover x from the low branch to see which side each pair took.
            mask = inputs.pair_kind == rates.PAIR_ION_NEUTRAL
            values = enhancement[mask]
            x_low |= bool((values > config.SU82_C).any())
            x_high |= bool((values > 0).any())
            assert x_low and x_high


class TestEvaporation:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_matches_reference(self, built, temperature: float) -> None:
        """Phase 4 gate for E.

        Looser in absolute terms than K (~1e-13 rather than ~1e-15) because
        the exponent amplifies: Delta-G/kT reaches ~50, so a last-digit
        difference in the tabulated free energies is magnified by exp().
        Still two orders inside the gate.
        """
        _, reactions, inputs = built
        parents = np.array([e.k for e in reactions.evaporations])
        di = np.array([e.i for e in reactions.evaporations])
        dj = np.array([e.j for e in reactions.evaporations])

        collision = rates.collision_coefficients(inputs, temperature)
        got = np.asarray(
            rates.evaporation_for_pairs(inputs, collision, parents, di, dj, temperature)
        )
        expected = reference.get_evap(temperature)[di, dj]
        assert _relative(got, expected, expected != 0) < GATE

    def test_uses_the_ion_enhanced_collision_rate(self, built) -> None:
        """Detailed balance must use the SAME beta as the forward reaction.

        Using the enhanced rate forward and the bare one here breaks
        equilibrium while still producing finite, plausible numbers -- the
        single easiest thing to get wrong in the whole port
        (Perl :9698-9701). Verified by showing that the bare rate gives a
        materially different answer for a charged channel.
        """
        _, reactions, inputs = built
        temperature = 280.0
        charged = [
            e
            for e in reactions.evaporations
            if inputs.charge[e.i] != 0 or inputs.charge[e.j] != 0
        ]
        assert charged, "no charged evaporation channels to check"

        parents = np.array([e.k for e in charged])
        di = np.array([e.i for e in charged])
        dj = np.array([e.j for e in charged])

        enhanced = rates.collision_coefficients(inputs, temperature)
        bare = rates.hard_sphere(inputs, temperature)

        with_enhancement = np.asarray(
            rates.evaporation_for_pairs(inputs, enhanced, parents, di, dj, temperature)
        )
        without = np.asarray(
            rates.evaporation_for_pairs(inputs, bare, parents, di, dj, temperature)
        )
        expected = reference.get_evap(temperature)[di, dj]

        assert _relative(with_enhancement, expected, expected != 0) < GATE
        assert _relative(without, expected, expected != 0) > 1e-3

    def test_symmetric_channels_carry_a_half(self, built) -> None:
        """k -> i + i is counted once, so it takes a factor of one half."""
        _, reactions, _ = built
        symmetric = [e for e in reactions.evaporations if e.is_symmetric]
        assert len(symmetric) == 4

    def test_strongly_temperature_dependent(self, built) -> None:
        _, reactions, inputs = built
        parents = np.array([e.k for e in reactions.evaporations])
        di = np.array([e.i for e in reactions.evaporations])
        dj = np.array([e.j for e in reactions.evaporations])

        def at(temperature):
            k = rates.collision_coefficients(inputs, temperature)
            return np.asarray(
                rates.evaporation_for_pairs(inputs, k, parents, di, dj, temperature)
            )

        cold, hot = at(250.0), at(320.0)
        assert (hot / cold).max() > 1e3


class TestCoagulationSink:
    def test_matches_reference_shape(self, built) -> None:
        """The reference emits only the size dependence; cs_ref multiplies
        it at runtime (get_rate_coefs:1011)."""
        _, _, inputs = built
        got = np.asarray(rates.coagulation_sink(inputs, 1.0))
        expected = reference.get_losses()
        assert _relative(got, expected, expected != 0) < GATE
        np.testing.assert_array_equal(got == 0, expected == 0)

    def test_scales_linearly_with_cs_ref(self, built) -> None:
        _, _, inputs = built
        one = np.asarray(rates.coagulation_sink(inputs, 1.0))
        ten = np.asarray(rates.coagulation_sink(inputs, 10.0))
        np.testing.assert_allclose(ten, 10.0 * one, rtol=1e-15)

    def test_analytic_value_for_the_dimer(self, built) -> None:
        """2A has twice the volume of 1A, so d_2A/d_1A = 2^(1/3) and the
        sink shape is (2^(1/3))^-1.6 -- checkable without the reference."""
        system, _, inputs = built
        got = np.asarray(rates.coagulation_sink(inputs, 1.0))
        assert got[system.labels.index("2A")] == pytest.approx(
            (2 ** (1 / 3)) ** config.CS_EXPONENT_DEFAULT, rel=1e-12
        )

    def test_vapour_monomers_are_excluded(self, built) -> None:
        system, _, inputs = built
        got = np.asarray(rates.coagulation_sink(inputs, 1.0))
        assert got[system.labels.index("1A")] == 0.0
        assert got[system.labels.index("1N")] == 0.0

    def test_ion_enhancement_applies_only_to_charged(self, built) -> None:
        """fcs is 1.0 in the bundled setup so the branch is invisible there,
        but it exists and must not touch neutrals."""
        _, _, inputs = built
        plain = np.asarray(rates.coagulation_sink(inputs, 1.0, fcs=1.0))
        boosted = np.asarray(rates.coagulation_sink(inputs, 1.0, fcs=3.3))
        neutral = inputs.charge == 0
        np.testing.assert_allclose(boosted[neutral], plain[neutral], rtol=1e-15)
        charged = (inputs.charge != 0) & (plain > 0)
        np.testing.assert_allclose(boosted[charged], 3.3 * plain[charged], rtol=1e-15)


class TestGeometry:
    def test_generic_ion_properties_come_from_the_generator(self, built) -> None:
        """Their pseudo-composition has a negative entry, so deriving mass or
        volume from it would give nonsense -- a negative volume and a NaN
        radius. The generator hardcodes O2-like and H3O-like properties."""
        system, _, inputs = built
        assert inputs.mass[system.generic_neg] == pytest.approx(
            config.MASS_NEG_ION * config.MASS_CONV, rel=1e-15
        )
        assert inputs.mass[system.generic_pos] == pytest.approx(
            config.MASS_POS_ION * config.MASS_CONV, rel=1e-15
        )
        assert np.all(np.isfinite(inputs.radius))
        assert np.all(inputs.radius > 0)
