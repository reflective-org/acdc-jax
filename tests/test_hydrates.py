"""Hydrate averaging (--rh). Phase 8.4.

Gate: K, E and the coagulation sink at 280 K, RH 20%, against a fixture
generated from the bundled thermodynamics plus four synthetic monohydrates
(1A, 2A, 1A1N, 2A1N). Every averaged number in the fixture is a literal, so
the gate is 1e-12 like the others.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from acdc_jax import config, hydrates, rates, rhs
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
HYD = REPO / "validation/fixtures/hydrates"
GOLDEN = REPO / "validation/goldens/rates_variant_hydr.npz"
T = 280.0
RH = 20.0
GATE = 1e-12


@pytest.fixture(scope="module")
def model():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    reactions = enumerate_reactions(system, BoundarySystem(cluster_set))
    energies = parse_energy_file(
        HYD / "HS298.15K_with_hydrates.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        HYD / "dip_pol_298.15K_with_hydrates.txt", cluster_set.molecule_names
    )
    inputs = rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )
    hm = hydrates.build_hydrate_model(system, energies, dipoles, inputs, reactions, RH)
    return system, reactions, inputs, hm


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("hydrate golden not captured")
    return np.load(GOLDEN)


def _worst(got, expected) -> float:
    mask = expected != 0
    return float((np.abs(got - expected)[mask] / np.abs(expected)[mask]).max())


def _dense_e(system, reactions, values):
    e = np.zeros((system.n_clusters, system.n_clusters))
    for ev, v in zip(reactions.evaporations, values, strict=True):
        e[ev.i, ev.j] = e[ev.j, ev.i] = v
    return e


class TestWater:
    def test_wexler_at_the_boiling_point_and_room_temperature(self) -> None:
        # 101325 Pa at 373.15 K; ~3169 Pa at 298.15 K (tables give 3169.9)
        assert float(hydrates.water_saturation_pressure(373.15)) == pytest.approx(
            101325, rel=2e-3
        )
        assert float(hydrates.water_saturation_pressure(298.15)) == pytest.approx(
            3169, rel=2e-3
        )


class TestExpansion:
    def test_species_ladder(self, model) -> None:
        system, _, _, hm = model
        assert hm.n_species == system.n_clusters + 4
        hydrated = {system.labels[i] for i in np.flatnonzero(hm.has_hydrates)}
        assert hydrated == {"1A", "2A", "1A1N", "2A1N"}

    def test_hydrate_mass_and_volume(self, model) -> None:
        system, _, inputs, hm = model
        a = system.index("1A")
        s = int(np.flatnonzero((hm.owner == a) & (hm.waters == 1))[0])
        assert hm.expanded.mass[s] == pytest.approx(
            inputs.mass[a] + config.MASS_WATER * config.MASS_CONV
        )
        dry_volume = 4 / 3 * np.pi * inputs.radius[a] ** 3
        water = config.MASS_WATER * config.MASS_CONV / config.DENS_WATER
        assert 4 / 3 * np.pi * hm.expanded.radius[s] ** 3 == pytest.approx(
            dry_volume + water, rel=1e-14
        )

    def test_weights_normalise_and_respond_to_humidity(self, model) -> None:
        system, _, _, hm = model
        w = np.asarray(hydrates.hydrate_weights(hm, T))
        np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-14)
        a = system.index("1A")
        dry_s = int(np.flatnonzero((hm.owner == a) & (hm.waters == 0))[0])
        drier = hydrates.HydrateModel(**{**hm.__dict__, "rh_percent": 5.0})
        assert np.asarray(hydrates.hydrate_weights(drier, T))[a, dry_s] > w[a, dry_s]

    def test_zero_humidity_is_dry(self, model) -> None:
        system, _, inputs, hm = model
        dry = hydrates.HydrateModel(**{**hm.__dict__, "rh_percent": 0.0})
        k = np.asarray(hydrates.collision_coefficients(dry, T))
        np.testing.assert_allclose(
            k, np.asarray(rates.collision_coefficients(inputs, T)), rtol=1e-14
        )

    def test_water_conserving_combinations(self, model) -> None:
        """2A1N(1W) -> 1N + 2A(1W) and -> 1N(dry) ... only combinations whose
        waters add up; 1N has no hydrates so the monohydrate parent has one
        route per daughter pair."""
        system, reactions, _, hm = model
        n, aa, aan = (system.index(x) for x in ("1N", "2A", "2A1N"))
        c = next(
            idx
            for idx, e in enumerate(reactions.evaporations)
            if e.k == aan and {e.i, e.j} == {n, aa}
        )
        combos = hm.combo_channel == c
        assert combos.sum() == 2  # dry parent -> dry pair; 1W parent -> 1N + 2A1W
        parent_waters = hm.waters[hm.combo_parent[combos]]
        daughter_waters = hm.waters[hm.combo_i[combos]] + hm.waters[hm.combo_j[combos]]
        np.testing.assert_array_equal(parent_waters, daughter_waters)


class TestFixture:
    def test_collision_coefficients(self, model, golden) -> None:
        _, _, _, hm = model
        k = np.asarray(hydrates.collision_coefficients(hm, T))
        np.testing.assert_array_equal(k != 0, golden["K"] != 0)
        assert _worst(k, golden["K"]) < GATE

    def test_evaporation(self, model, golden) -> None:
        system, reactions, _, hm = model
        e = _dense_e(system, reactions, np.asarray(hydrates.evaporation(hm, T)))
        np.testing.assert_array_equal(e != 0, golden["E"] != 0)
        assert _worst(e, golden["E"]) < GATE

    def test_coagulation_sink(self, model, golden) -> None:
        _, _, _, hm = model
        # Without --variable_cs the fixture folds the generator's default
        # exp_loss coefficient (2.6e-3 1/s) into the literals.
        cs = np.asarray(
            hydrates.coagulation_sink(hm, config.CS_COEFFICIENT_DEFAULT, temperature=T)
        )
        np.testing.assert_array_equal(cs != 0, golden["cs"] != 0)
        assert _worst(cs, golden["cs"]) < GATE

    def test_averaging_actually_moves_the_numbers(self, model, golden) -> None:
        """Not a tautology: the hydrated pairs differ from dry by percent."""
        system, _, inputs, _ = model
        dry = np.asarray(rates.collision_coefficients(inputs, T))
        a, n = system.index("1A"), system.index("1N")
        assert abs(golden["K"][n, a] / dry[n, a] - 1) > 0.01


class TestOverflow:
    def test_a_too_stable_hydrate_falls_back_to_dry_with_a_finite_gradient(
        self, model
    ) -> None:
        """F7's discard branch. Upstream normalises in linear space, so a
        very stable hydrate overflows to inf and the cluster reverts to
        dry. Computing it that way gives a finite value through the final
        `where` but leaves reverse mode crossing inf/inf, so `jax.grad`
        came back NaN."""
        import dataclasses

        system, _, _, hm = model

        def total(scale):
            delta_h = jnp.where(
                jnp.asarray(hm.waters) > 0, -1000.0 * scale, hm.expanded.delta_h
            )
            expanded = dataclasses.replace(hm.expanded, delta_h=delta_h)
            return hydrates.hydrate_weights(
                dataclasses.replace(hm, expanded=expanded), T
            ).sum()

        value = float(total(1.0))
        gradient = float(jax.grad(total)(1.0))
        # every cluster reverted to dry, so each row sums to exactly one
        assert value == pytest.approx(float(system.n_clusters))
        assert np.isfinite(gradient)

    def test_the_ordinary_distribution_still_differentiates(self, model) -> None:
        _, _, _, hm = model
        gradient = float(jax.grad(lambda t: hydrates.hydrate_weights(hm, t).sum())(T))
        assert np.isfinite(gradient)


class TestTraceable:
    def test_temperature_gradient_exists(self, model) -> None:
        """Upstream refuses --rh with --variable_temp; here it differentiates."""
        _, _, _, hm = model

        def total(t):
            return hydrates.collision_coefficients(hm, t).sum()

        g = float(jax.grad(total)(T))
        assert np.isfinite(g) and g != 0

    def test_assemble_with_hydrates(self, model) -> None:
        system, reactions, inputs, hm = model
        co = rhs.assemble(system, reactions, inputs, T, 1e-3, 3e6, 3e6, hydrates=hm)
        c = np.full(system.n_equations, 1e12)
        f = np.asarray(rhs.rhs(co, c))
        assert np.all(np.isfinite(f))
        plain = rhs.assemble(system, reactions, inputs, T, 1e-3, 3e6, 3e6)
        assert not np.allclose(f, np.asarray(rhs.rhs(plain, c)))
