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
    # The --cs_only exclusion is read from the inputs, so this is the mask
    # that the production path uses; tests keep it only to reason about it.
    excluded = np.asarray(inputs.cs_excluded)
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
        got = np.asarray(losses.background_coagulation_sink(inputs, temperature))
        expected = golden["cs"]
        np.testing.assert_array_equal(got == 0, expected == 0)
        mask = expected != 0
        assert (np.abs(got - expected)[mask] / expected[mask]).max() < GATE

    def test_vapour_monomers_excluded(self, model) -> None:
        system, inputs, excluded = model
        sink = np.asarray(losses.background_coagulation_sink(inputs, 280.0))
        assert sink[system.labels.index("1A")] == 0.0
        assert sink[system.labels.index("1N")] == 0.0
        assert np.count_nonzero(sink) == 52

    def test_linear_in_background_concentration(self, model) -> None:
        _, inputs, excluded = model
        one = np.asarray(
            losses.background_coagulation_sink(inputs, 280.0, bg_concentration=1e9)
        )
        ten = np.asarray(
            losses.background_coagulation_sink(inputs, 280.0, bg_concentration=1e10)
        )
        np.testing.assert_allclose(ten, 10.0 * one, rtol=1e-15)

    def test_larger_clusters_are_lost_more_slowly(self, model) -> None:
        """Small clusters diffuse fast and are scavenged fast. The kernel
        must fall with cluster size across the set -- the same monotonicity
        exp_loss encodes with its negative exponent."""
        _, inputs, excluded = model
        sink = np.asarray(losses.background_coagulation_sink(inputs, 280.0))
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
        cold = np.asarray(losses.background_coagulation_sink(inputs, 250.0))
        hot = np.asarray(losses.background_coagulation_sink(inputs, 320.0))
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


WALL_VARIANTS = {
    "CLOUD4_JA": "cloud4_ja",
    "CLOUD4_JK": "cloud4_jk",
    "CLOUD4_AK": "cloud4_ak",
    "CLOUD4_simple": "cloud4_simple",
    "CLOUD3": "cloud3",
    "ift": "ift",
    "diffusion": "diffusion",
}


def _wall_golden(name: str):
    path = GOLDENS / f"losses_variant_wl_{name}.npz"
    if not path.exists():
        pytest.skip("run: uv run python validation/capture_variants.py")
    return np.load(path)


class TestWallLosses:
    @pytest.mark.parametrize("name", sorted(WALL_VARIANTS))
    def test_matches_fixture(self, model, name: str) -> None:
        """Phase 8.5 gate. The fixture emits the BARE wl(k); the ion factor
        fwl is applied downstream in get_rate_coefs, so compare with fwl=1."""
        system, inputs, _ = model
        golden = _wall_golden(name)
        temperature = float(golden["temperature"])
        got = np.asarray(
            losses.wall_loss(WALL_VARIANTS[name], inputs, temperature, fwl=1.0)
        )
        expected = golden["wl"]
        np.testing.assert_array_equal(got == 0, expected == 0)
        mask = expected != 0
        assert (np.abs(got - expected)[mask] / expected[mask]).max() < GATE

    def test_vapour_monomers_are_NOT_excluded(self, model) -> None:
        """Unlike the coagulation sink, there is no --cs_only mechanism for
        walls: every cluster hits the wall, 1A and 1N included. Confirmed in
        every fixture (wl(1) and wl(3) nonzero)."""
        system, inputs, _ = model
        wl = np.asarray(losses.wall_loss("cloud4_ja", inputs, 280.0))
        assert wl[system.labels.index("1A")] > 0
        assert wl[system.labels.index("1N")] > 0

    def test_charged_clusters_take_fwl(self, model) -> None:
        """coef_lin(57,57,k) reads fwl*wl(k) for ions and wl(k) for neutrals."""
        system, inputs, _ = model
        bare = np.asarray(losses.wall_loss("cloud4_ja", inputs, 280.0, fwl=1.0))
        boosted = np.asarray(losses.wall_loss("cloud4_ja", inputs, 280.0, fwl=3.3))
        neutral = inputs.charge == 0
        np.testing.assert_array_equal(boosted[neutral], bare[neutral])
        charged = inputs.charge != 0
        np.testing.assert_allclose(boosted[charged], 3.3 * bare[charged], rtol=1e-15)
        assert config.FWL_DEFAULT == 3.3

    def test_diffusion_excludes_the_generic_ions(self, model) -> None:
        """Fidelity F16. The diffusion branch alone loops to $max_cluster,
        the real clusters, where every other loss branch loops to
        $max_cluster_number, which includes the two generic ions. The fixture
        has exactly 52 nonzero entries, zero at the generic-ion slots."""
        system, inputs, _ = model
        wl = np.asarray(losses.wall_loss("diffusion", inputs, 280.0))
        assert wl[system.generic_neg] == 0.0
        assert wl[system.generic_pos] == 0.0
        assert np.count_nonzero(wl) == 52
        # ... and no other branch does this.
        other = np.asarray(losses.wall_loss("cloud4_ja", inputs, 280.0))
        assert np.count_nonzero(other) == 54

    def test_mobility_diameter_correction_is_live_here(self, model) -> None:
        """The same Tammet expression that F11 found defeated in the metadata
        emission is applied CORRECTLY in the wall-loss code, where $mass1 is
        the raw kg value. So CLOUD4_JA must show a mass dependence beyond
        d + 0.3 nm -- which is exactly what CLOUD4_simple lacks."""
        _, inputs, _ = model
        d_mob = np.asarray(losses.mobility_diameter(inputs))
        d_geom = 2.0 * inputs.radius + config.MOB_DIAMETER_OFFSET
        ratio = d_mob / d_geom
        assert np.all(ratio > 1.0)
        assert ratio.max() > 1.5, "the lightest species should be corrected by >50%"

    def test_ift_is_flat(self, model) -> None:
        _, inputs, _ = model
        wl = np.asarray(losses.wall_loss("ift", inputs, 280.0, fwl=1.0))
        assert np.all(wl == config.WL_IFT)

    def test_unknown_method_raises(self, model) -> None:
        _, inputs, _ = model
        with pytest.raises(ValueError, match="unknown wall-loss method"):
            losses.wall_loss("cloud5", inputs, 280.0)

    def test_diffusion_2024_setting_includes_the_generic_ions(self, model) -> None:
        """F16 at its other setting: the 2024 generator loops to
        $max_cluster_number, so neg/pos get a finite loss like every other
        cluster. Everything else is untouched."""
        system, inputs, _ = model
        fid = config.FidelityConfig(diffusion_wall_loss_generic_ions="included_2024")
        wl_2024 = np.asarray(losses.wall_loss("diffusion", inputs, 280.0, fidelity=fid))
        wl_2020 = np.asarray(losses.wall_loss("diffusion", inputs, 280.0))
        assert wl_2024[system.generic_neg] > 0 and wl_2024[system.generic_pos] > 0
        assert np.count_nonzero(wl_2024) == 54
        real = ~np.asarray(inputs.is_generic_ion)
        np.testing.assert_array_equal(wl_2024[real], wl_2020[real])
        assert config.DEFAULT.diffusion_wall_loss_generic_ions == "excluded_2020"

    def test_diffusion_tube_parameters_are_forwarded(self, model) -> None:
        """Halving the tube radius quadruples the loss (3.65/R^2); doubling
        the pressure halves it."""
        _, inputs, _ = model
        base = np.asarray(losses.wall_loss("diffusion", inputs, 280.0))
        narrow = np.asarray(
            losses.wall_loss(
                "diffusion",
                inputs,
                280.0,
                tube_radius=config.WL_DIFFUSION_TUBE_RADIUS / 2,
            )
        )
        dense = np.asarray(
            losses.wall_loss(
                "diffusion",
                inputs,
                280.0,
                tube_pressure=2 * config.WL_DIFFUSION_TUBE_PRESSURE,
            )
        )
        mask = base != 0
        np.testing.assert_allclose(narrow[mask] / base[mask], 4.0, rtol=1e-14)
        np.testing.assert_allclose(dense[mask] / base[mask], 0.5, rtol=1e-14)


class TestDilution:
    def test_matches_fixture(self, model) -> None:
        _, inputs, _ = model
        path = GOLDENS / "losses_variant_dilution.npz"
        if not path.exists():
            pytest.skip("run: uv run python validation/capture_variants.py")
        expected = np.load(path)["dil"]
        got = np.asarray(losses.dilution(inputs))
        np.testing.assert_array_equal(got, expected)

    def test_flat_over_all_species_including_generic_ions(self, model) -> None:
        """A scalar upstream, coef_lin(58,58,k) = dil for ALL k, no ion factor."""
        system, inputs, _ = model
        dil = np.asarray(losses.dilution(inputs))
        assert np.all(dil == config.DILUTION_DEFAULT)
        assert dil[system.generic_neg] == config.DILUTION_DEFAULT
        assert config.DILUTION_DEFAULT == 9.6e-5


class TestWiring:
    """The losses must reach the right-hand side, each on its own flux slot."""

    @pytest.fixture(scope="class")
    def full(self):
        from acdc_jax import rhs

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
        return rhs, system, reactions, inputs

    def test_default_is_the_exp_loss_sink_alone(self, full) -> None:
        rhs, system, reactions, inputs = full
        co = rhs.assemble(system, reactions, inputs, 280.0, 1e-3)
        assert co.loss_slots == (system.flux_index["coag"],)
        np.testing.assert_array_equal(
            np.asarray(co.losses[0]), np.asarray(rates.coagulation_sink(inputs, 1e-3))
        )

    def test_every_loss_lands_on_its_slot(self, full) -> None:
        rhs, system, reactions, inputs = full
        settings = losses.LossSettings(
            coagulation="bg_loss", wall="cloud4_ja", dilution=config.DILUTION_DEFAULT
        )
        co = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, loss_settings=settings
        )
        slots = {name: system.flux_index[name] for name in ("coag", "wall", "dilution")}
        assert co.loss_slots == tuple(slots.values())
        c = np.zeros(system.n_equations)
        c[: system.n_clusters] = 1e12
        f = np.asarray(rhs.rhs(co, c))
        for name, vector in zip(slots, co.losses, strict=True):
            # The slot accumulates exactly what the clusters lose to it.
            assert f[slots[name]] == pytest.approx(
                float(np.sum(np.asarray(vector) * c[: system.n_clusters])), rel=1e-12
            )
        # Dense layout matches the reference's coef_lin(slot,slot,k) booking.
        lin = np.asarray(co.coef_lin)
        for name, vector in zip(slots, co.losses, strict=True):
            np.testing.assert_array_equal(lin[slots[name], slots[name], :], vector)
        # and the bg_loss sink is the one that was asked for
        np.testing.assert_array_equal(
            np.asarray(co.losses[0]),
            np.asarray(losses.background_coagulation_sink(inputs, 280.0)),
        )

    def test_fidelity_reaches_the_solver_path(self, full) -> None:
        """The review's first finding: ion_collision_method used to stop at
        the unit tests. Now assemble forwards it."""
        rhs, system, reactions, inputs = full
        su82 = rhs.assemble(system, reactions, inputs, 280.0, 1e-3)
        su73 = rhs.assemble(
            system,
            reactions,
            inputs,
            280.0,
            1e-3,
            fidelity=config.FidelityConfig(ion_collision_method="su73"),
        )
        a, b = np.asarray(su82.collision_rate), np.asarray(su73.collision_rate)
        # relative: np.allclose's atol would call any two K vectors equal
        assert np.max(np.abs(a - b) / a) > 1e-3

    def test_losses_average_over_hydrates(self, full) -> None:
        from acdc_jax import hydrates as hydrate_module

        rhs, system, reactions, _ = full
        cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
        hyd = REPO / "validation/fixtures/hydrates"
        energies = parse_energy_file(
            hyd / "HS298.15K_with_hydrates.txt", cluster_set.molecule_names
        )
        dipoles = parse_dipole_file(
            hyd / "dip_pol_298.15K_with_hydrates.txt", cluster_set.molecule_names
        )
        inputs = rates.build_rate_inputs(
            system, cluster_set, energies, dipoles, reactions=reactions
        )
        hm = hydrate_module.build_hydrate_model(
            system, energies, dipoles, inputs, reactions, 20.0
        )
        settings = losses.LossSettings(coagulation="bg_loss", wall="cloud4_ja")
        wet = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, hydrates=hm, loss_settings=settings
        )
        dry = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, loss_settings=settings
        )
        a = system.index("2A")
        # 2A has hydrates (and, unlike 1A, no --cs_only zero): its averaged
        # bg_loss and wall loss differ from dry; 3A1N has none and is identical.
        for w, d in zip(wet.losses, dry.losses, strict=True):
            assert np.asarray(w)[a] != np.asarray(d)[a]
            # a weight matrix row of exactly [1.0] still rounds once
            assert np.asarray(w)[system.index("3A1N")] == pytest.approx(
                np.asarray(d)[system.index("3A1N")], rel=1e-14
            )
