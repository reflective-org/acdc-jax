"""--nst: forbidden collisions and product overrides. Phase 8.8.

The oracle is the emitted reaction GRAPH -- every ``coef_quad(i,j,k)``
triple, every extra-product multiplicity, every ``E(i,j)`` channel -- read
off the generated Fortran and stored as a golden. Rates do not enter; the
whole point of --nst is to change which reactions exist. The same
comparison is run on the unmodified system, which is the first exact,
whole-graph check of the Phase 3 enumeration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, rates, rhs
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import (
    ReactionSet,
    enumerate_reactions,
    parse_nonstandard_file,
)
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = REPO / "validation/goldens"
NST = REPO / "validation/fixtures/nst"


@pytest.fixture(scope="module")
def model():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    return system, cluster_set, BoundarySystem(cluster_set)


def _graph(reactions: ReactionSet):
    """The port's reaction graph in the golden's three shapes."""
    triples: set[tuple[int, int, int]] = set()
    extras: set[tuple[int, int, int, int]] = set()
    for c in reactions.collisions:
        for k, _ in c.products:
            triples.add((c.i, c.j, k))
            triples.add((c.j, c.i, k))
        for k, mult in c.products[1:]:
            extras.add((min(c.i, c.j), max(c.i, c.j), k, mult))
    pairs: set[tuple[int, int]] = set()
    for e in reactions.evaporations:
        pairs.add((e.i, e.j))
        pairs.add((e.j, e.i))
    return triples, extras, pairs


def _golden(name: str):
    path = GOLDENS / f"reactions_variant_{name}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} not captured")
    g = np.load(path)
    return (
        {tuple(int(x) for x in row) for row in g["triples"]},
        {tuple(int(x) for x in row) for row in g["extras"]},
        {tuple(int(x) for x in row) for row in g["evaporations"]},
    )


def _assert_same_graph(reactions: ReactionSet, name: str) -> None:
    got = _graph(reactions)
    want = _golden(name)
    for label, g, w in zip(("coef_quad", "extras", "E"), got, want, strict=True):
        assert g == w, (
            f"{label}: {len(g - w)} only in port {sorted(g - w)[:5]}, "
            f"{len(w - g)} only in fixture {sorted(w - g)[:5]}"
        )


class TestBaselineGraph:
    def test_unmodified_system_matches_the_fixture_exactly(self, model) -> None:
        system, _, boundary = model
        _assert_same_graph(enumerate_reactions(system, boundary), "base")


class TestMonomerDefinition:
    """Perl's check_monomer: everything counts except the proton pseudo-species."""

    @pytest.mark.parametrize("label", ["1A", "1N", "1B", "1N1P"])
    def test_monomers(self, model, label: str) -> None:
        system = model[0]
        assert system.is_monomer(system.index(label))

    @pytest.mark.parametrize("label", ["2A", "1A1B", "1A1N", "2N1P", "neg", "pos"])
    def test_not_monomers(self, model, label: str) -> None:
        system = model[0]
        assert not system.is_monomer(system.index(label))


class TestParser:
    def test_override(self, model) -> None:
        system, _, _ = model
        nst = parse_nonstandard_file(NST / "override.txt", system)
        a, an, n, aan = (system.index(x) for x in ("1A", "1A1N", "1N", "2A1N"))
        assert nst.is_forbidden(a, an) and nst.is_forbidden(an, a)
        assert nst.override(n, aan) == ((an, 2),)
        assert nst.override(aan, n) == ((an, 2),)

    def test_clusters_rule_spares_monomers_and_generic_ions(self, model) -> None:
        system, _, _ = model
        nst = parse_nonstandard_file(NST / "clusters.txt", system)
        aa = system.index("2A")
        assert nst.is_forbidden(aa, aa)
        assert nst.is_forbidden(aa, system.index("1A1B"))
        for label in ("1A", "1N", "1B", "1N1P", "neg", "pos"):
            assert not nst.is_forbidden(aa, system.index(label))
        assert not nst.overrides

    def test_unknown_reactant_warns_and_skips(self, model, tmp_path: Path) -> None:
        system, _, _ = model
        f = tmp_path / "nst.txt"
        f.write_text("9A9N 1A\n1A 1A1N\n")
        with pytest.warns(UserWarning, match="9A9N"):
            nst = parse_nonstandard_file(f, system)
        assert len(nst.forbidden) == 1

    def test_decimal_coefficient_rejected(self, model, tmp_path: Path) -> None:
        system, _, _ = model
        f = tmp_path / "nst.txt"
        f.write_text("1N 2A1N 1.5 1A1N\n")
        with pytest.raises(ValueError, match="non-integer"):
            parse_nonstandard_file(f, system)

    def test_out_slots(self, model, tmp_path: Path) -> None:
        system, _, _ = model
        f = tmp_path / "nst.txt"
        f.write_text("1N 2A1N 1 out\n1A 2A 1 out_neg\n")
        nst = parse_nonstandard_file(f, system)
        n, aan = system.index("1N"), system.index("2A1N")
        assert nst.override(n, aan) == ((system.flux_index["out_neu"], 1),)
        a, aa = system.index("1A"), system.index("2A")
        assert nst.override(a, aa) == ((system.flux_index["out_neg"], 1),)

    def test_useless_line_is_forbidden(self, model, tmp_path: Path) -> None:
        """`1A 1A1N 1 1A 1 1A1N` spells x + y -> x + y (Perl :2076-2083)."""
        system, _, _ = model
        f = tmp_path / "nst.txt"
        f.write_text("1A 1A1N 1 1A 1 1A1N\n")
        nst = parse_nonstandard_file(f, system)
        assert nst.is_forbidden(system.index("1A"), system.index("1A1N"))
        assert not nst.overrides


class TestGraphs:
    @pytest.mark.parametrize(
        "rule_file, golden",
        [
            ("override.txt", "nst_override"),
            ("duplicate_product.txt", "nst_duplicate"),
            ("clusters.txt", "nst_clusters"),
        ],
    )
    def test_matches_fixture(self, model, rule_file: str, golden: str) -> None:
        system, _, boundary = model
        nst = parse_nonstandard_file(NST / rule_file, system)
        _assert_same_graph(
            enumerate_reactions(system, boundary, nonstandard=nst), golden
        )

    def test_override_has_no_reverse_evaporation(self, model) -> None:
        system, _, boundary = model
        nst = parse_nonstandard_file(NST / "override.txt", system)
        reactions = enumerate_reactions(system, boundary, nonstandard=nst)
        n, aan, aann = (system.index(x) for x in ("1N", "2A1N", "2A2N"))
        assert not any(
            e.k == aann and {e.i, e.j} == {n, aan} for e in reactions.evaporations
        )
        # and 2A2N is no longer formed from that pair at all
        assert not any(
            {c.i, c.j} == {n, aan} and any(k == aann for k, _ in c.products)
            for c in reactions.collisions
        )


class TestMainCoefficient:
    """F20: the Fortran drops the first product's coefficient."""

    def _collision(self, model, fidelity):
        system, _, boundary = model
        nst = parse_nonstandard_file(NST / "override.txt", system)
        reactions = enumerate_reactions(
            system, boundary, nonstandard=nst, fidelity=fidelity
        )
        n, aan = system.index("1N"), system.index("2A1N")
        return next(c for c in reactions.collisions if {c.i, c.j} == {n, aan})

    def test_fortran_default_forms_one(self, model) -> None:
        c = self._collision(model, config.DEFAULT)
        assert c.products == ((model[0].index("1A1N"), 1),)
        assert config.DEFAULT.nonstandard_main_coefficient == "fortran"

    def test_literal_honours_two(self, model) -> None:
        c = self._collision(
            model, config.FidelityConfig(nonstandard_main_coefficient="literal")
        )
        assert c.products == ((model[0].index("1A1N"), 2),)

    def test_duplicate_form_forms_two_on_both_paths(self, model) -> None:
        """`1 1A1N 1 1A1N` is how to get two in the Fortran: the extra lands
        in ind_quad_form_extra with multiplicity 1."""
        system, _, boundary = model
        nst = parse_nonstandard_file(NST / "duplicate_product.txt", system)
        reactions = enumerate_reactions(system, boundary, nonstandard=nst)
        n, aan, an = (system.index(x) for x in ("1N", "2A1N", "1A1N"))
        c = next(c for c in reactions.collisions if {c.i, c.j} == {n, aan})
        assert c.products == ((an, 1), (an, 1))

    def test_flows_through_the_rhs(self, model) -> None:
        """The duplicated product must double the formation term: with only
        1N and 2A1N present, the two right-hand sides differ for 1A1N by
        exactly one K*c*c."""
        system, cluster_set, boundary = model
        energies = parse_energy_file(
            INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
        )
        dipoles = parse_dipole_file(
            INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
        )
        n, aan, an = (system.index(x) for x in ("1N", "2A1N", "1A1N"))
        c = np.zeros(system.n_equations)
        c[n], c[aan] = 1e14, 1e10
        f = {}
        for name in ("override.txt", "duplicate_product.txt"):
            nst = parse_nonstandard_file(NST / name, system)
            reactions = enumerate_reactions(system, boundary, nonstandard=nst)
            inputs = rates.build_rate_inputs(
                system, cluster_set, energies, dipoles, reactions=reactions
            )
            co = rhs.assemble(
                system, reactions, inputs, 280.0, 1e-3, 0.0, 0.0, constant_vapours=()
            )
            f[name] = np.asarray(rhs.rhs(co, c))
            k = np.asarray(rates.collision_coefficients(inputs, 280.0))[n, aan]
        extra = f["duplicate_product.txt"][an] - f["override.txt"][an]
        assert extra == pytest.approx(k * c[n] * c[aan], rel=1e-12)
        assert extra > 0
