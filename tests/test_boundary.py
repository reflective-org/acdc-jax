"""The boundary cascade. Phase 2's gate.

Exact match against every decision the Perl generator logged under
``--print_boundary``, for four cluster sets. Not a tolerance: the cascade is
combinatorial, so a decision either matches or it does not.

The coverage tests matter as much as the match tests. A pass that never
enters the stripping stages would be meaningless, and three of the four
fixtures never reach the fallback at all -- so its coverage is asserted
explicitly rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from acdc_jax import labels
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
compare_boundary = pytest.importorskip("compare_boundary")

SETS = ("AN_narrow", "AN", "AD", "AN_narrow_nostrength")


def _system(name: str) -> BoundarySystem:
    return BoundarySystem(parse_cluster_set(compare_boundary.CLUSTER_SETS[name]))


@pytest.fixture(scope="module")
def an_narrow() -> BoundarySystem:
    return _system("AN_narrow")


@pytest.fixture(scope="module")
def no_strength() -> BoundarySystem:
    return _system("AN_narrow_nostrength")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SETS)
def test_every_decision_matches_the_oracle(name: str) -> None:
    """Phase 2 gate: exact match, all four cluster sets."""
    if not (compare_boundary.GOLDENS / f"boundary_{name}.json").exists():
        pytest.skip("oracle not captured")
    checked, _skipped, failures = compare_boundary.replay(name)
    assert not failures, "\n".join(failures[:20])
    assert checked > 500, f"only {checked} decisions replayed for {name}"


def test_total_decisions_replayed() -> None:
    """Guards against the harness silently skipping most of its input."""
    total = sum(compare_boundary.replay(name)[0] for name in SETS)
    assert total > 3500, f"only {total} decisions replayed across all sets"


# ---------------------------------------------------------------------------
# Coverage of the stages -- a pass that skips them proves nothing
# ---------------------------------------------------------------------------


def _stage_hits(name: str) -> dict[str, int]:
    """Count which stages fire while replaying one cluster set."""
    import acdc_jax.boundary as module

    system = _system(name)
    hits = dict.fromkeys(
        ("strength", "fallback", "charged_fallback", "ion_transfer"), 0
    )

    original_strength = module.BoundarySystem._strip_by_strength
    original_count = module.BoundarySystem._strip_by_count
    original_forms = module.BoundarySystem._forms_ion_with

    def strength(self, working, monomers):
        hits["strength"] += 1
        return original_strength(self, working, monomers)

    def count(self, working, monomers, charge):
        hits["fallback"] += 1
        if charge != 0:
            hits["charged_fallback"] += 1
        return original_count(self, working, monomers, charge)

    def forms(self, i, ion, positive):
        result = original_forms(self, i, ion, positive)
        hits["ion_transfer"] += bool(result)
        return result

    module.BoundarySystem._strip_by_strength = strength
    module.BoundarySystem._strip_by_count = count
    module.BoundarySystem._forms_ion_with = forms
    try:
        import json

        oracle = json.loads(
            (compare_boundary.GOLDENS / f"boundary_{name}.json").read_text()
        )
        for decision in oracle["decisions"]:
            if any(labels.is_generic_ion(r) for r in decision["reactants"]):
                continue
            raw = decision.get("raw_product")
            if not raw:
                continue
            counts = tuple(labels.composition_vector(raw, system.order))
            if system.is_member(counts):
                continue
            system.check_boundary(counts)
    finally:
        module.BoundarySystem._strip_by_strength = original_strength
        module.BoundarySystem._strip_by_count = original_count
        module.BoundarySystem._forms_ion_with = original_forms
    return hits


def test_strength_ladder_is_exercised() -> None:
    assert _stage_hits("AN_narrow")["strength"] > 100


def test_shipped_cluster_sets_never_reach_the_fallback() -> None:
    """Measured, and the reason the no-strength fixture exists.

    All three shipped sets define acid and base strengths, so
    `l_strength_neutral` is true and the ladder always succeeds. Without a
    fixture that disables it, the entire count-based stage -- including its
    protonation-transfer accounting -- would be dead code in every test.
    """
    for name in ("AN_narrow", "AN", "AD"):
        assert _stage_hits(name)["fallback"] == 0, name


def test_fallback_is_exercised_by_the_nostrength_fixture() -> None:
    hits = _stage_hits("AN_narrow_nostrength")
    assert hits["strength"] == 0, "the ladder must be disabled"
    assert hits["fallback"] > 50
    assert hits["charged_fallback"] > 20, "charged clusters must reach it"
    assert hits["ion_transfer"] > 20, (
        "the protonation-transfer accounting must actually fire"
    )


def test_both_strategies_agree_on_this_system() -> None:
    """Worth recording: for AN_narrow the strength ladder and the count
    fallback reach identical answers on all 745 decisions.

    So the fixture validates that the fallback *code* is right, but it does
    not distinguish the two strategies -- a set where they disagree would be
    needed for that, and none of the shipped ones does.
    """
    import json

    a = json.loads((compare_boundary.GOLDENS / "boundary_AN_narrow.json").read_text())[
        "decisions"
    ]
    b = json.loads(
        (compare_boundary.GOLDENS / "boundary_AN_narrow_nostrength.json").read_text()
    )["decisions"]
    assert a == b


# ---------------------------------------------------------------------------
# combine_labels
# ---------------------------------------------------------------------------


class TestCombine:
    def _counts(self, system: BoundarySystem, label: str) -> tuple[int, ...]:
        return tuple(labels.composition_vector(label, system.order))

    def test_neutral_pair_sums(self, an_narrow) -> None:
        result = an_narrow.combine(
            self._counts(an_narrow, "1A"), self._counts(an_narrow, "1A1N")
        )
        assert result.label == "2A1N"
        assert result.valid_coll and result.valid_evap and result.in_system

    def test_same_sign_ions_cannot_collide(self, an_narrow) -> None:
        result = an_narrow.combine(
            self._counts(an_narrow, "1B"), self._counts(an_narrow, "1A1B")
        )
        assert not result.valid_coll
        assert result.label is None

    def test_out_of_system_product_cannot_evaporate_back(self, an_narrow) -> None:
        """A boundary collision has no reverse (Perl :10199)."""
        result = an_narrow.combine(
            self._counts(an_narrow, "2A"), self._counts(an_narrow, "2A")
        )
        assert result.valid_coll
        assert not result.in_system
        assert not result.valid_evap

    def test_recombination_neutralises_before_summing(self, an_narrow) -> None:
        """The negative ion reverts to its neutral (B -> A) and the positive
        cluster loses a proton, and only then are compositions added.

        1B + 1N1P: naive summing gives 1A1B1N1P; neutralising first gives
        1A + 1N = 1A1N.
        """
        result = an_narrow.combine(
            self._counts(an_narrow, "1B"), self._counts(an_narrow, "1N1P")
        )
        assert result.label == "1A1N"
        assert result.valid_coll
        assert not result.valid_evap, "recombination has no reverse"

    def test_charge_of_a_neutral_cluster(self, an_narrow) -> None:
        assert an_narrow.cluster_charge(self._counts(an_narrow, "2A1N")) == 0

    def test_charge_of_ions(self, an_narrow) -> None:
        assert an_narrow.cluster_charge(self._counts(an_narrow, "1A1B")) == -1
        assert an_narrow.cluster_charge(self._counts(an_narrow, "1N1P")) == 1


# ---------------------------------------------------------------------------
# check_boundary behaviour
# ---------------------------------------------------------------------------


class TestCheckBoundary:
    def _counts(self, system: BoundarySystem, label: str) -> tuple[int, ...]:
        return tuple(labels.composition_vector(label, system.order))

    def test_grow_out_returns_the_product_unchanged(self, an_narrow) -> None:
        result = an_narrow.check_boundary(self._counts(an_narrow, "6A5N"))
        assert result.lout == 1
        assert result.label == "6A5N"
        assert result.monomers == {}

    def test_grow_out_channels_by_charge(self, an_narrow) -> None:
        assert an_narrow.check_boundary(self._counts(an_narrow, "6A5N")).lout == 1
        assert an_narrow.check_boundary(self._counts(an_narrow, "5A1B3N")).lout == 2
        assert an_narrow.check_boundary(self._counts(an_narrow, "5A6N1P")).lout == 3

    def test_brought_back_conserves_molecules(self, an_narrow) -> None:
        """4A -> 2A + 2 A. The stripped monomers plus the surviving cluster
        must contain exactly what went in."""
        result = an_narrow.check_boundary(self._counts(an_narrow, "4A"))
        assert result.lout == 0
        assert result.label == "2A"
        assert result.monomers == {"A": 2}

    def test_rejects_a_cluster_already_in_the_system(self, an_narrow) -> None:
        """The reference dies here too: it means the caller should not have
        asked."""
        with pytest.raises(ValueError, match="inside the system"):
            an_narrow.check_boundary(self._counts(an_narrow, "2A1N"))


class TestSystemProperties:
    def test_per_charge_maxima_differ(self, an_narrow) -> None:
        """Maxima are per charge class, not global. Using the global maximum
        would let a negative cluster keep more acid than any negative cluster
        in the set actually has, silently."""
        assert an_narrow.n_max_type != an_narrow.n_max_type_neg

    def test_neutral_maxima_match_the_reference(self, an_narrow) -> None:
        """ij_ind_max = (5, 1, 5, 1) in the generated system module is the
        per-type maximum across all charges; the neutral block must not
        exceed it."""
        assert all(
            n <= m for n, m in zip(an_narrow.n_max_type, (5, 1, 5, 1), strict=True)
        )

    def test_strength_flag_reflects_the_input(self, an_narrow, no_strength) -> None:
        assert an_narrow.strength_neutral
        assert not no_strength.strength_neutral

    def test_proton_is_identified(self, an_narrow) -> None:
        assert an_narrow.proton == an_narrow.order.index("P")

    def test_no_missing_proton_in_these_files(self, an_narrow) -> None:
        assert an_narrow.missing_proton == -1
