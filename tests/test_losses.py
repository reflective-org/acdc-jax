"""External loss processes. Phase 8.

`bg_loss` -- coagulation onto a monodisperse background population -- is
validated against a fixture the generator produced at FIXED temperature,
because upstream refuses to combine it with --variable_temp. The port has no
such restriction, so the temperature dependence is also checked for physical
sense here even though nothing upstream can confirm it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, losses, rates
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = REPO / "validation/goldens"

GATE = 1e-12


@pytest.fixture(scope="module")
def model():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    reactions = enumerate_reactions(system, BoundarySystem(cluster_set))
    energies = parse_energy_file(
        INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
    )
    inputs = rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )
    excluded = np.array([label in ("1A", "1N") for label in system.labels])
    return system, inputs, excluded


@pytest.fixture(scope="module")
def golden():
    path = GOLDENS / "losses_variant_bgloss.npz"
    if not path.exists():
        pytest.skip("run: uv run python validation/capture_variants.py")
    return np.load(path)


class TestBackgroundCoagulationSink:
    def test_matches_fixture(self, model, golden) -> None:
        """Phase 8.2 gate, at the fixture's fixed temperature."""
        _, inputs, excluded = model
        temperature = float(golden["temperature"])
        got = np.asarray(
            losses.background_coagulation_sink(inputs, temperature, excluded=excluded)
        )
        expected = golden["cs"]
        np.testing.assert_array_equal(got == 0, expected == 0)
        mask = expected != 0
        assert (np.abs(got - expected)[mask] / expected[mask]).max() < GATE

    def test_vapour_monomers_excluded(self, model) -> None:
        system, inputs, excluded = model
        sink = np.asarray(
            losses.background_coagulation_sink(inputs, 280.0, excluded=excluded)
        )
        assert sink[system.labels.index("1A")] == 0.0
        assert sink[system.labels.index("1N")] == 0.0
        assert np.count_nonzero(sink) == 52

    def test_linear_in_background_concentration(self, model) -> None:
        _, inputs, excluded = model
        one = np.asarray(
            losses.background_coagulation_sink(
                inputs, 280.0, bg_concentration=1e9, excluded=excluded
            )
        )
        ten = np.asarray(
            losses.background_coagulation_sink(
                inputs, 280.0, bg_concentration=1e10, excluded=excluded
            )
        )
        np.testing.assert_allclose(ten, 10.0 * one, rtol=1e-15)

    def test_larger_clusters_are_lost_more_slowly(self, model) -> None:
        """Small clusters diffuse fast and are scavenged fast. The kernel
        must fall with cluster size across the set -- the same monotonicity
        exp_loss encodes with its negative exponent."""
        _, inputs, excluded = model
        sink = np.asarray(
            losses.background_coagulation_sink(inputs, 280.0, excluded=excluded)
        )
        radius = inputs.radius
        keep = sink > 0
        order = np.argsort(radius[keep])
        # Correlation, not strict monotonicity: mass enters through the
        # thermal speed independently of radius, so an ion and a neutral of
        # equal radius sit at different sinks. Measured -0.88; the bound is
        # loose enough to be a trend check, tight enough to catch a sign flip.
        assert np.corrcoef(radius[keep][order], sink[keep][order])[0, 1] < -0.8

    def test_temperature_dependence_is_physical(self, model) -> None:
        """Upstream cannot produce this -- the generator dies on
        bg_loss + variable temperature. Here the sink follows T, and a
        warmer gas diffuses faster, so the sink should grow with T."""
        _, inputs, excluded = model
        cold = np.asarray(
            losses.background_coagulation_sink(inputs, 250.0, excluded=excluded)
        )
        hot = np.asarray(
            losses.background_coagulation_sink(inputs, 320.0, excluded=excluded)
        )
        mask = cold > 0
        assert np.all(hot[mask] > cold[mask])
        # ... but only modestly: it is a transport rate, not an Arrhenius one.
        assert (hot[mask] / cold[mask]).max() < 2.0

    def test_default_background_matches_generator_defaults(self) -> None:
        """1e3 cm^-3, 100 nm, 1000 kg/m^3 (Perl :325-327), in SI."""
        assert config.BG_CONCENTRATION_DEFAULT == 1e9
        assert config.BG_DIAMETER_DEFAULT == 100e-9
        assert config.BG_DENSITY_DEFAULT == 1000.0


class TestAirProperties:
    def test_viscosity_is_the_dman_fit(self) -> None:
        """2.5277e-7 * T^0.75302, not Sutherland (Perl :6668)."""
        assert float(losses.air_viscosity(280.0)) == pytest.approx(
            2.5277e-7 * 280.0**0.75302, rel=1e-15
        )
        # Order of magnitude sanity: air is ~1.8e-5 Pa s at room temperature.
        assert 1.5e-5 < float(losses.air_viscosity(293.0)) < 2.0e-5

    def test_mean_free_path_is_tens_of_nanometres(self) -> None:
        """~65 nm at standard conditions is the textbook value."""
        assert 5e-8 < float(losses.air_mean_free_path(293.0)) < 8e-8

    def test_slip_correction_reduces_to_stokes_einstein(self) -> None:
        """For a particle much larger than the mean free path the Phillips
        factor -> 1 and D -> k_B T / (6 pi mu r)."""
        radius = 10e-6  # 10 um: lambda/r ~ 0.007
        temperature = 293.0
        got = float(losses.slip_corrected_diffusivity(radius, temperature))
        stokes_einstein = (
            config.K_B
            * temperature
            / (6.0 * config.PI * float(losses.air_viscosity(temperature)) * radius)
        )
        assert got == pytest.approx(stokes_einstein, rel=2e-2)

    def test_slip_correction_grows_for_small_particles(self) -> None:
        """Free-molecular limit: D grows much faster than 1/r."""
        temperature = 293.0
        big = float(losses.slip_corrected_diffusivity(1e-6, temperature))
        small = float(losses.slip_corrected_diffusivity(1e-9, temperature))
        # Pure Stokes-Einstein would give exactly 1e3; the slip correction
        # makes the small particle far more mobile than that.
        assert small / big > 1e4
