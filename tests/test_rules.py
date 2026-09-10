"""Sticking factors and Delta-G scaling. Phase 8.7.

Rule semantics are unit-tested against the Perl's branch structure; the
matrices are then gated against three generated fixtures at 1e-12 across the
temperature sweep, for both K and E.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, rates, rules
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
    return system, cluster_set, reactions, energies, dipoles


def _inputs(model, **kw):
    system, cluster_set, reactions, energies, dipoles = model
    return rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions, **kw
    )


def _golden(name: str):
    path = GOLDENS / f"rates_variant_{name}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} not captured")
    return np.load(path)


def _worst(got, expected) -> float:
    mask = expected != 0
    return float((np.abs(got - expected)[mask] / np.abs(expected)[mask]).max())


def _evaporation(model, inputs, temperature, fidelity=config.DEFAULT):
    system, _, reactions, _, _ = model
    k = rates.collision_coefficients(inputs, temperature, fidelity)
    parents = np.array([e.k for e in reactions.evaporations])
    di = np.array([e.i for e in reactions.evaporations])
    dj = np.array([e.j for e in reactions.evaporations])
    rate = np.asarray(
        rates.evaporation_for_pairs(
            inputs, k, parents, di, dj, temperature, fidelity=fidelity
        )
    )
    e = np.zeros((system.n_clusters, system.n_clusters))
    e[di, dj] = rate
    e[dj, di] = rate
    return np.asarray(k), e


class TestRuleOrdering:
    def test_bare_factor(self) -> None:
        assert rules.sticking_rules(0.5) == ((0.5,),)

    def test_ion_neutral_alone_replaces_slot_zero(self) -> None:
        assert rules.sticking_rules(1.0, 2.0) == ((2.0, "ion-neutral"),)

    def test_both_keep_order(self) -> None:
        assert rules.sticking_rules(0.5, 2.0) == ((0.5,), (2.0, "ion-neutral"))

    def test_file_rows_come_last(self) -> None:
        out = rules.sticking_rules(0.5, file_rules=((0.1, "1A", "1N"),))
        assert out == ((0.5,), (0.1, "1A", "1N"))

    def test_scale_zero_means_no_global_rule(self) -> None:
        assert rules.evap_scale_rules(0.0, ((1.0, "1A"),)) == ((1.0, "1A"),)
        assert rules.evap_scale_rules(2.0) == ((2.0,),)

    def test_parse_file(self, tmp_path: Path) -> None:
        f = tmp_path / "stick.txt"
        f.write_text("# comment\n0.5\n2 ion-neutral\n0.1 1A 1N\n\n")
        assert rules.parse_rule_file(f) == (
            (0.5,),
            (2.0, "ion-neutral"),
            (0.1, "1A", "1N"),
        )

    def test_parse_rejects_nonpositive(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("0 1A\n")
        with pytest.raises(ValueError):
            rules.parse_rule_file(f)


class TestStickingMatrix:
    def test_bare_factor_is_neutral_neutral_only(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((0.5,),))
        charged = np.asarray(system.charges) != 0
        neutral_pair = ~charged[:, None] & ~charged[None, :]
        assert np.all(s[neutral_pair] == 0.5)
        assert np.all(s[~neutral_pair] == 1.0)

    def test_ion_neutral_keyword(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((2.0, "ion-neutral"),))
        charged = np.asarray(system.charges) != 0
        ion_neutral = charged[:, None] != charged[None, :]
        assert np.all(s[ion_neutral] == 2.0)
        assert np.all(s[~ion_neutral] == 1.0)
        # generic ions count as ions: neg + 1A carries the factor
        assert s[system.generic_neg, system.index("1A")] == 2.0

    def test_molecule_name_restricts_to_clusters_containing_it(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((0.3, "N"),))
        a, aa, n, an = (system.index(x) for x in ("1A", "2A", "1N", "1A1N"))
        assert s[a, aa] == 1.0  # no N in either
        assert s[a, n] == 0.3
        assert s[aa, an] == 0.3
        # never on ion pairs, even with N present
        assert s[system.index("1N1P"), n] == 1.0 if "1N1P" in system.labels else True

    def test_specific_pair_in_either_order(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((0.2, "1N", "2A"),))
        n, aa = system.index("1N"), system.index("2A")
        assert s[n, aa] == 0.2 and s[aa, n] == 0.2
        assert s[n, system.index("1A")] == 1.0
        # composition order in the rule label does not matter
        s2 = rules.sticking_matrix(system, ((0.2, "1N1A"),))
        assert s2[n, system.index("1A1N")] == 0.2

    def test_last_match_wins(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((0.5,), (0.7, "1A", "1N")))
        assert s[system.index("1A"), system.index("1N")] == 0.7
        assert s[system.index("1A"), system.index("2A")] == 0.5

    def test_rounds_to_four_decimals_like_the_generator(self, model) -> None:
        system = model[0]
        s = rules.sticking_matrix(system, ((1 / 3,),))
        a, n = system.index("1A"), system.index("1N")
        assert s[a, n] == float("3.3333e-01")

    def test_no_rules_is_ones(self, model) -> None:
        assert np.all(rules.sticking_matrix(model[0], ()) == 1.0)


class TestScaleMatrix:
    def test_global(self, model) -> None:
        system = model[0]
        sc = rules.evap_scale_matrix(system, ((1.0,),))
        assert sc[system.index("1A"), system.index("1N")] == 1.0
        # a pair whose sum is not in the system stays zero
        assert (
            sc[system.index("2A"), system.index("2A")] == 0.0 or "4A" in system.labels
        )

    def test_daughter_and_parent(self, model) -> None:
        system = model[0]
        sc = rules.evap_scale_matrix(system, ((2.0, "1A", "1A1N"),))
        assert sc[system.index("1A"), system.index("1N")] == 2.0
        assert sc[system.index("1A"), system.index("1A")] == 0.0  # parent 2A


class TestFixtures:
    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_stick05(self, model, temperature: float) -> None:
        golden = _golden("stick05")
        inputs = _inputs(model, sticking_rules=rules.sticking_rules(0.5))
        k, e = _evaporation(model, inputs, temperature)
        np.testing.assert_array_equal(k != 0, golden[f"K_{temperature:g}"] != 0)
        assert _worst(k, golden[f"K_{temperature:g}"]) < GATE
        np.testing.assert_array_equal(e != 0, golden[f"E_{temperature:g}"] != 0)
        assert _worst(e, golden[f"E_{temperature:g}"]) < GATE

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_stickion2(self, model, temperature: float) -> None:
        golden = _golden("stickion2")
        inputs = _inputs(model, sticking_rules=rules.sticking_rules(1.0, 2.0))
        k, e = _evaporation(model, inputs, temperature)
        assert _worst(k, golden[f"K_{temperature:g}"]) < GATE
        assert _worst(e, golden[f"E_{temperature:g}"]) < GATE

    @pytest.mark.parametrize("temperature", TEMPERATURES)
    def test_scaleevap1(self, model, temperature: float) -> None:
        golden = _golden("scaleevap1")
        inputs = _inputs(model, evap_scale_rules=rules.evap_scale_rules(1.0))
        k, e = _evaporation(model, inputs, temperature)
        assert _worst(k, golden[f"K_{temperature:g}"]) < GATE
        assert _worst(e, golden[f"E_{temperature:g}"]) < GATE

    def test_upstream_squares_the_sticking_factor_on_evaporation(self, model) -> None:
        """F19, asserted on the fixture itself: E(stick05)/E(base) is 0.25
        where K(stick05)/K(base) is 0.5."""
        golden = _golden("stick05")
        # The unmodified port is the Phase 4/5 validated baseline.
        k_plain, e_plain = _evaporation(model, _inputs(model), 280.0)
        system = model[0]
        a, n = system.index("1A"), system.index("1N")
        assert golden["K_280"][n, a] / k_plain[n, a] == pytest.approx(0.5, rel=1e-12)
        e_ratio = golden["E_280"][n, a] / e_plain[n, a]
        assert e_ratio == pytest.approx(0.25, rel=1e-12)

    def test_detailed_balance_option_applies_it_once(self, model) -> None:
        inputs = _inputs(model, sticking_rules=rules.sticking_rules(0.5))
        plain = _inputs(model)
        fid = config.FidelityConfig(sticking_on_evaporation="detailed_balance")
        _, e_once = _evaporation(model, inputs, 280.0, fid)
        _, e_plain = _evaporation(model, plain, 280.0)
        system = model[0]
        a, n = system.index("1A"), system.index("1N")
        assert e_once[n, a] / e_plain[n, a] == pytest.approx(0.5, rel=1e-14)
        assert config.DEFAULT.sticking_on_evaporation == "upstream"
