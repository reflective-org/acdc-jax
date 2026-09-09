"""Flux matrix, monomer-source back-solve, growth pathways, dG surfaces. Phase 10."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, free_energy, pathways, rates, rhs, solve
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
T = 280.0
CM3 = 1e6


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
    return system, reactions, inputs


@pytest.fixture(scope="module")
def steady(model):
    """The QuickGuide's conditions: 5e6 cm^-3 acid, 100 ppt ammonia, 280 K."""
    system, reactions, inputs = model
    co = rhs.assemble(system, reactions, inputs, T, 1e-3, 3e6, 3e6)
    c_n = 100e-12 * config.P_ATM / (config.K_B * T)  # 100 ppt at 1 atm
    c0 = solve.set_vapours(
        system, np.zeros(system.n_equations), {"1A": 5e6 * CM3, "1N": c_n}
    )
    result = solve.solve_steady_state(system, co, c0=c0)
    assert result.converged
    return co, np.asarray(result.concentrations)


class TestFluxMatrix:
    def test_bookings_follow_the_matlab_convention(self, model, steady) -> None:
        """Per party: 1A + 1N -> 1A1N books K c_A c_N from 1A and from 1N;
        1A1N -> 1A + 1N books E c_{1A1N} from 1A1N to each daughter."""
        system, reactions, _ = model
        co, c = steady
        gross = pathways.gross_flux_matrix(system, co, c)
        a, n, an = (system.index(x) for x in ("1A", "1N", "1A1N"))
        k = next(
            i
            for i, (ci, cj) in enumerate(
                zip(co.collision_i, co.collision_j, strict=True)
            )
            if {int(ci), int(cj)} == {a, n}
        )
        phi = float(co.collision_rate[k]) * c[a] * c[n]
        assert gross[a, an] == pytest.approx(phi, rel=1e-12)
        assert gross[n, an] == pytest.approx(phi, rel=1e-12)
        e = next(
            float(co.evaporation_rate[m])
            for m, (ek, ei, ej) in enumerate(
                zip(co.evaporation_k, co.evaporation_i, co.evaporation_j, strict=True)
            )
            if ek == an and {int(ei), int(ej)} == {a, n}
        )
        assert gross[an, a] == pytest.approx(e * c[an], rel=1e-12)
        assert gross[an, n] == pytest.approx(e * c[an], rel=1e-12)
        assert gross.shape == (system.n_equations, system.n_equations)
        assert np.all(gross[: system.n_clusters, : system.n_clusters].diagonal() == 0)

    def test_boundary_collisions_go_through_the_bound_node(self, model, steady) -> None:
        """2N1P + neg recombines to 2N, outside the set, which strips back to
        two 1N: the parties are consumed once each, the pieces formed twice."""
        system, _, _ = model
        co, c = steady
        gross = pathways.gross_flux_matrix(system, co, c)
        bound = system.flux_index["bound"]
        assert gross[:, bound].sum() > 0
        assert gross[bound, :].sum() > 0
        # a party never flows to more than one destination per collision:
        # the neg ion's total consumption equals its production at steady state
        neg = system.generic_neg
        assert gross[neg, :].sum() == pytest.approx(3e6, rel=1e-6)

    def test_outgoing_flux_is_the_formation_rate(self, model, steady) -> None:
        system, _, _ = model
        co, c = steady
        gross = pathways.gross_flux_matrix(system, co, c)
        slots = [system.flux_index[s] for s in ("out_neu", "out_neg", "out_pos")]
        j = float(rhs.formation_rate(system, co, c)["j_tot"])
        # each out collision is booked from BOTH colliders, so the column
        # sum is twice the flux (once per party); self-collisions 2x once.
        assert gross[:, slots].sum() == pytest.approx(2 * j, rel=1e-10)

    def test_net_is_positive_part(self, model, steady) -> None:
        system, _, _ = model
        co, c = steady
        gross = pathways.gross_flux_matrix(system, co, c)
        net = pathways.net_flux_matrix(gross)
        assert np.all(net >= 0)
        assert np.all((net > 0) <= ~(net.T > 0))


class TestSources:
    def test_back_solved_source_sustains_the_steady_state(self, model, steady) -> None:
        """Replace the constant-vapour assumption by the back-solved sources
        with the monomers free: the state must still be stationary."""
        system, reactions, inputs = model
        co, c = steady
        gross = pathways.gross_flux_matrix(system, co, c)
        sources = pathways.monomer_sources(system, gross)
        assert set(sources) >= {"1A", "1N", "neg", "pos"}
        assert sources["1A"] > 0 and sources["1N"] > 0
        free = rhs.assemble(
            system, reactions, inputs, T, 1e-3, 3e6, 3e6, constant_vapours=()
        )
        source = np.asarray(free.source).copy()
        for label, value in sources.items():
            source[system.index(label)] = value
        import dataclasses

        free = dataclasses.replace(free, source=source)
        f = np.asarray(rhs.rhs(free, c))
        turnover = np.asarray(solve._turnover(free, c))
        n = system.n_clusters
        assert np.max(np.abs(f[:n]) / np.maximum(turnover[:n], 1e-300)) < 1e-6

    def test_ion_sources_recover_the_production_rate(self, model, steady) -> None:
        system, _, _ = model
        co, c = steady
        sources = pathways.monomer_sources(
            system, pathways.gross_flux_matrix(system, co, c)
        )
        assert sources["neg"] == pytest.approx(3e6, rel=1e-6)
        assert sources["pos"] == pytest.approx(3e6, rel=1e-6)


class TestPathways:
    def test_neutral_main_route(self, model, steady) -> None:
        system, reactions, _ = model
        co, c = steady
        pw = pathways.track_pathways(system, reactions, co, c, charge=0)
        assert pw.exits and pw.edges
        assert pw.total_out == pytest.approx(
            float(rhs.formation_rate(system, co, c)["j_tot"]), rel=1e-10
        )
        route = pw.main_route
        # starts inside the system, ends at a composition outside it
        assert len(route) >= 3
        assert all(label in system.labels for label in route[:-1])
        assert route[-1] not in system.labels
        # the exit cluster is neutral and, as the QuickGuide observes for this
        # system, carries at least as much acid as ammonia
        exit_cluster = system.index(route[-2])
        assert system.charges[exit_cluster] == 0
        comp = system.compositions[exit_cluster]
        assert comp[system.order.index("A")] >= comp[system.order.index("N")]
        # every intermediate is reached by a recorded edge
        ends = {e.end for e in pw.edges}
        assert all(label in ends for label in route[1:-1])
        # growth_only (default): molecule counts never decrease along the route
        counts = [sum(system.compositions[system.index(label)]) for label in route[:-1]]
        assert counts == sorted(counts)
        # the faithful argmax may take an evaporation step; it must still be
        # a chain of recorded edges ending at the same exit
        faithful = pathways.track_pathways(
            system, reactions, co, c, charge=0, growth_only=False
        )
        assert faithful.main_route[-2:] == route[-2:]

    def test_edges_are_significant_inflows(self, model, steady) -> None:
        system, reactions, _ = model
        co, c = steady
        pw = pathways.track_pathways(
            system, reactions, co, c, charge=0, crit_clust=0.05
        )
        for end in {e.end for e in pw.edges}:
            into = [e for e in pw.edges if e.end == end]
            assert all(e.value > 0 for e in into)
            assert len({e.start for e in into}) == len(into)

    def test_charged_pathways(self, model, steady) -> None:
        system, reactions, _ = model
        co, c = steady
        for charge in (-1, 1):
            pw = pathways.track_pathways(system, reactions, co, c, charge=charge)
            for label in pw.main_route[:-1]:
                assert system.charges[system.index(label)] == charge

    def test_exit_criterion_filters(self, model, steady) -> None:
        system, reactions, _ = model
        co, c = steady
        loose = pathways.track_pathways(system, reactions, co, c, crit_out=0.0)
        tight = pathways.track_pathways(system, reactions, co, c, crit_out=0.3)
        assert len(tight.exits) <= len(loose.exits)
        assert loose.exits[0] == tight.exits[0]  # the largest is always kept


class TestFreeEnergy:
    def test_reference_pressure_recovers_reference(self, model) -> None:
        system, _, inputs = model
        p_ref = config.P_ATM
        c_ref = p_ref / (config.K_B * T)
        actual = free_energy.actual_free_energy(
            system, inputs, T, {"1A": c_ref, "1N": c_ref}
        )
        np.testing.assert_allclose(
            actual, np.asarray(rates.gibbs_at(inputs, T)), rtol=1e-12, atol=1e-12
        )

    def test_lower_vapour_raises_the_surface_by_n_kT_ln(self, model) -> None:
        system, _, inputs = model
        c_ref = config.P_ATM / (config.K_B * T)
        base = free_energy.actual_free_energy(
            system, inputs, T, {"1A": c_ref, "1N": c_ref}
        )
        diluted = free_energy.actual_free_energy(
            system, inputs, T, {"1A": c_ref / 10, "1N": c_ref}
        )
        kt_kcal = config.K_B * T / config.KCAL_PER_MOL_TO_J
        a = system.order.index("A")
        n_a = np.asarray(system.compositions)[:, a]
        np.testing.assert_allclose(diluted - base, kt_kcal * n_a * np.log(10.0))

    def test_ions_are_not_molecules(self, model) -> None:
        """1B (bisulfate) has no neutral molecule: no pressure term."""
        system, _, inputs = model
        c_ref = config.P_ATM / (config.K_B * T)
        base = free_energy.actual_free_energy(
            system, inputs, T, {"1A": c_ref, "1N": c_ref}
        )
        diluted = free_energy.actual_free_energy(
            system, inputs, T, {"1A": c_ref / 10, "1N": c_ref / 10}
        )
        i = system.index("1B")
        assert diluted[i] == base[i]
        # ...while 1A1B carries one acid molecule's worth
        j = system.index("1A1B")
        kt_kcal = config.K_B * T / config.KCAL_PER_MOL_TO_J
        assert diluted[j] - base[j] == pytest.approx(kt_kcal * np.log(10.0))

    def test_grid_layout(self, model) -> None:
        system, reactions, inputs = model
        co = rhs.assemble(system, reactions, inputs, T, 1e-3)
        total = free_energy.total_evaporation_rate(
            system.n_clusters, co.evaporation_k, co.evaporation_rate
        )
        grid = free_energy.composition_grid(system, total, "A", "N", charge=0)
        assert grid.shape == (6, 6)  # 0..5 acids, 0..5 ammonia
        assert np.isnan(grid[0, 0])
        assert grid[2, 0] == total[system.index("2A")]
        assert grid[3, 2] == total[system.index("3A2N")]
