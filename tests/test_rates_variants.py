"""Ion-neutral collision methods beyond the shipped Su82 default. Phase 8.

Each method is validated against its own Perl-generated fixture, captured to a
golden by ``validation/capture_variants.py``. Gate is the same 1e-12 as
Phase 4, across the same temperature sweep.
"""

from __future__ import annotations

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
GOLDENS = REPO / "validation/goldens"

TEMPERATURES = (250.0, 280.0, 298.15, 320.0)
GATE = 1e-12


@pytest.fixture(scope="module")
def inputs():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    reactions = enumerate_reactions(system, BoundarySystem(cluster_set))
    energies = parse_energy_file(
        INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
    )
    return rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )


def _golden(name: str):
    path = GOLDENS / f"rates_variant_{name}.npz"
    if not path.exists():
        pytest.skip(
            f"{path.name} not captured: uv run python validation/capture_variants.py"
        )
    return np.load(path)


def _worst(got, expected) -> float:
    mask = expected != 0
    return float((np.abs(got - expected)[mask] / np.abs(expected)[mask]).max())


class TestSu73:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_matches_fixture(self, inputs, temperature: float) -> None:
        golden = _golden("su73")
        fidelity = config.FidelityConfig(ion_collision_method="su73")
        got = np.asarray(rates.collision_coefficients(inputs, temperature, fidelity))
        expected = golden[f"K_{temperature:g}"]
        np.testing.assert_array_equal(got != 0, expected != 0)
        assert _worst(got, expected) < GATE

    def test_uses_the_locked_dipole(self, inputs) -> None:
        """Su73 damps the dipole by the header's locking coefficients; Su82
        reads them and ignores them. The two must therefore differ on
        ion-neutral pairs and agree everywhere else."""
        su82 = np.asarray(rates.collision_coefficients(inputs, 280.0))
        su73 = np.asarray(
            rates.collision_coefficients(
                inputs, 280.0, config.FidelityConfig(ion_collision_method="su73")
            )
        )
        ion_neutral = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (su82 > 0)
        # Relative, with atol=0. The default np.allclose atol of 1e-8 is eight
        # orders larger than a collision coefficient (~1e-16 m^3/s), so it
        # declares ANY two K matrices equal -- which is how the first draft
        # of this test passed against itself.
        relative = np.abs(su82[ion_neutral] - su73[ion_neutral]) / su82[ion_neutral]
        assert relative.max() > 1e-3, "Su73 and Su82 must actually differ"
        np.testing.assert_array_equal(su82[~ion_neutral], su73[~ion_neutral])

    def test_locked_dipole_is_damped(self, inputs) -> None:
        """Locking coefficients are 0.15 in the bundled file, so the locked
        dipole is 15% of the raw one wherever there is data."""
        has = inputs.dipole > 0
        np.testing.assert_allclose(
            inputs.dipole_locked[has], 0.15 * inputs.dipole[has], rtol=1e-15
        )

    def test_never_below_hard_sphere(self, inputs) -> None:
        fidelity = config.FidelityConfig(ion_collision_method="su73")
        k = np.asarray(rates.collision_coefficients(inputs, 280.0, fidelity))
        beta = np.asarray(rates.hard_sphere(inputs, 280.0))
        mask = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (k > 0)
        assert np.all(k[mask] >= beta[mask] * (1 - 1e-12))


class TestConstant:
    def test_documented_factor_of_ten(self, inputs) -> None:
        """`constant` applies the manual's size-independent factor of 10 --
        which is also exactly what upstream does at FIXED temperature."""
        fidelity = config.FidelityConfig(ion_collision_method="constant")
        k = np.asarray(rates.collision_coefficients(inputs, 280.0, fidelity))
        beta = np.asarray(rates.hard_sphere(inputs, 280.0))
        mask = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (k > 0)
        np.testing.assert_allclose(k[mask], 10.0 * beta[mask], rtol=1e-15)

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_no_enhancement_reproduces_upstream_variable_temp(
        self, inputs, temperature: float
    ) -> None:
        """Fidelity F15: with --variable_temp, upstream silently drops the
        factor and emits the bare hard-sphere rate for every ion-neutral pair.
        Verified against the generated fixture -- zero of 587 pairs enhanced.
        """
        golden = _golden("constant")
        fidelity = config.FidelityConfig(ion_collision_method="constant_no_enhancement")
        got = np.asarray(rates.collision_coefficients(inputs, temperature, fidelity))
        expected = golden[f"K_{temperature:g}"]
        np.testing.assert_array_equal(got != 0, expected != 0)
        assert _worst(got, expected) < GATE

    def test_matches_the_fixed_temperature_fixture(self, inputs) -> None:
        """At fixed temperature upstream DOES apply the factor of 10; this is
        the fixture for the `constant` setting itself (the F15 flag's other
        side), literal K values at 280 K."""
        golden = _golden("constant_fixed")
        fidelity = config.FidelityConfig(ion_collision_method="constant")
        got = np.asarray(rates.collision_coefficients(inputs, 280.0, fidelity))
        expected = golden["K"]
        np.testing.assert_array_equal(got != 0, expected != 0)
        assert _worst(got, expected) < GATE

    def test_upstream_fixture_really_has_no_enhancement(self, inputs) -> None:
        """The finding itself, asserted on the fixture rather than on this
        port: the constant-method fixture's ion-neutral rates equal plain
        hard sphere. If upstream ever fixes this, the test tells us."""
        golden = _golden("constant")
        beta = np.asarray(rates.hard_sphere(inputs, 280.0))
        fixture = golden["K_280"]
        mask = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (fixture > 0)
        np.testing.assert_allclose(fixture[mask], beta[mask], rtol=1e-12)

    def test_the_two_constant_options_differ_by_exactly_ten(self, inputs) -> None:
        with_factor = np.asarray(
            rates.collision_coefficients(
                inputs, 280.0, config.FidelityConfig(ion_collision_method="constant")
            )
        )
        without = np.asarray(
            rates.collision_coefficients(
                inputs,
                280.0,
                config.FidelityConfig(ion_collision_method="constant_no_enhancement"),
            )
        )
        mask = (inputs.pair_kind == rates.PAIR_ION_NEUTRAL) & (without > 0)
        np.testing.assert_allclose(with_factor[mask] / without[mask], 10.0, rtol=1e-15)


def test_default_is_still_su82(inputs) -> None:
    """The Phase 4 gate must be unaffected by adding options."""
    default = np.asarray(rates.collision_coefficients(inputs, 280.0))
    explicit = np.asarray(
        rates.collision_coefficients(
            inputs, 280.0, config.FidelityConfig(ion_collision_method="su82")
        )
    )
    np.testing.assert_array_equal(default, explicit)
    assert config.DEFAULT.ion_collision_method == "su82"


class TestDipoleLocking:
    """Su73 damps the dipole by a MONOMER or a CLUSTER coefficient (Perl
    :7880-7885). The bundled file has both at 0.15, so only a hand-built
    table can tell the branch is wired the right way round."""

    def test_monomer_and_cluster_coefficients_go_to_the_right_species(self) -> None:
        from acdc_jax.thermo import DipoleTable

        cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
        system = build_system(cluster_set)
        energies = parse_energy_file(
            INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
        )
        real = parse_dipole_file(
            INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
        )
        table = DipoleTable(
            monomer_locking=0.1,
            cluster_locking=0.3,
            dipole=real.dipole,
            polarizability=real.polarizability,
        )
        inputs = rates.build_rate_inputs(system, cluster_set, energies, table)
        for label, factor in (
            ("1A", 0.1),
            ("1N", 0.1),
            ("1B", 0.1),
            ("2A", 0.3),
            ("1A1N", 0.3),
        ):
            i = system.index(label)
            assert inputs.dipole_locked[i] == pytest.approx(factor * inputs.dipole[i])
