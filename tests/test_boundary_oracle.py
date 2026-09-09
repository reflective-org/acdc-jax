"""The boundary oracle must be well-formed before Phase 2 can be gated on it.

These check the captured decision log, not the port. The Phase 2 gate is
exact match against this data, so a malformed or truncated oracle would let
a wrong port pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

GOLDENS = Path(__file__).resolve().parents[1] / "validation/goldens"
SETS = ("AN_narrow", "AN", "AD")

# Cluster labels are runs of <count><molecule>. The two generic charger
# ions are bare words instead -- they are pseudo-species, not compositions.
LABEL = re.compile(r"^(?:\d+[A-Z][a-z]?)+$")
GENERIC_IONS = {"pos", "neg"}


def _load(name: str) -> dict:
    path = GOLDENS / f"boundary_{name}.json"
    if not path.exists():
        pytest.skip("oracle not captured: uv run python validation/capture_boundary.py")
    return json.loads(path.read_text())


@pytest.fixture(scope="module", params=SETS)
def oracle(request) -> dict:
    return _load(request.param)


def test_all_three_sets_captured() -> None:
    """AD matters as much as the two AN sets: it swaps ammonia for
    dimethylamine, which drives a different path through the acid/base
    strength ladder in the stripping stage."""
    for name in SETS:
        assert (GOLDENS / f"boundary_{name}.json").exists(), name


class TestStructure:
    def test_counts_match_decisions(self, oracle: dict) -> None:
        tallied: dict[str, int] = {}
        for d in oracle["decisions"]:
            tallied[d["kind"]] = tallied.get(d["kind"], 0) + 1
        assert tallied == oracle["counts"]

    def test_every_decision_has_two_reactants(self, oracle: dict) -> None:
        for d in oracle["decisions"]:
            assert len(d["reactants"]) == 2

    def test_kinds_are_known(self, oracle: dict) -> None:
        assert {d["kind"] for d in oracle["decisions"]} <= {
            "out",
            "brought_back",
            "unused",
        }

    def test_labels_are_wellformed(self, oracle: dict) -> None:
        """Cluster labels are runs of <count><molecule>, e.g. 4A1B3N.

        The exceptions are the two generic charger ions, `pos` and `neg`,
        which are pseudo-species carrying only a charge and a mass.
        """
        for d in oracle["decisions"]:
            for label in d["reactants"]:
                assert LABEL.match(label) or label in GENERIC_IONS, f"{label!r} in {d}"


class TestGrowOut:
    def test_channels_are_the_three_flux_counters(self, oracle: dict) -> None:
        channels = {d["channel"] for d in oracle["decisions"] if d["kind"] == "out"}
        assert channels <= {"out_neu", "out_neg", "out_pos"}

    def test_all_three_channels_are_exercised(self, oracle: dict) -> None:
        """Every set here includes ions, so neutral, negative and positive
        growth must all appear -- otherwise the ion pathways are untested."""
        channels = {d["channel"] for d in oracle["decisions"] if d["kind"] == "out"}
        assert channels == {"out_neu", "out_neg", "out_pos"}


class TestBroughtBack:
    def test_products_are_nonempty_with_positive_multiplicity(
        self, oracle: dict
    ) -> None:
        for d in oracle["decisions"]:
            if d["kind"] != "brought_back":
                continue
            assert d["products"], d
            for label, mult in d["products"]:
                assert mult >= 1, d
                assert label, d

    def test_conserves_molecules(self, oracle: dict) -> None:
        """The stripped-back products must contain exactly the molecules the
        out-of-set product had. This is the one invariant of the cascade that
        holds regardless of which removal path it took, so it checks the
        parser against the physics rather than against itself.
        """
        for d in oracle["decisions"]:
            if d["kind"] != "brought_back":
                continue
            before = _composition(d["raw_product"])
            after: dict[str, int] = {}
            for label, mult in d["products"]:
                for mol, n in _composition(label).items():
                    after[mol] = after.get(mol, 0) + n * mult
            assert before == after, f"{d['raw_product']} -> {d['products']}"

    def test_neutral_reactants_sum_to_the_raw_product(self, oracle: dict) -> None:
        """For two neutral colliders the raw product is just the sum.

        Restricted to neutral pairs on purpose. Charged collisions are not
        simple sums: the generic ions carry charge without composition
        (`3A1D + pos -> 3A1D1P` gains a proton from nowhere), and
        recombination chemically neutralises both partners before summing --
        the negative ion reverts to its corresponding neutral molecule and
        the positive cluster loses a proton (Perl :10723-10767).
        """
        checked = 0
        for d in oracle["decisions"]:
            if d["kind"] != "brought_back":
                continue
            if any(r in GENERIC_IONS for r in d["reactants"]):
                continue
            i, j = (_composition(r) for r in d["reactants"])
            if {"B", "P"} & (set(i) | set(j)):
                continue
            summed = {m: i.get(m, 0) + j.get(m, 0) for m in set(i) | set(j)}
            assert summed == _composition(d["raw_product"]), d
            checked += 1
        assert checked > 0, "no neutral-neutral boundary collisions to check"


def _composition(label: str) -> dict[str, int]:
    """`4A1B3N` -> {'A': 4, 'B': 1, 'N': 3}.

    The count is optional. Stripped monomers are printed by the generator as
    `<multiplicity> <molecule>` with a bare molecule name, so `A` means one
    A. Requiring a leading digit here silently returned {} for every such
    product and made molecule conservation appear to fail on all 278
    brought-back decisions.
    """
    out: dict[str, int] = {}
    for count, mol in re.findall(r"(\d*)([A-Z][a-z]?)", label):
        out[mol] = out.get(mol, 0) + (int(count) if count else 1)
    return out


def test_composition_helper() -> None:
    assert _composition("4A1B3N") == {"A": 4, "B": 1, "N": 3}
    assert _composition("2A") == {"A": 2}
    assert _composition("A") == {"A": 1}  # bare name, implicit count of 1


def test_expected_decision_counts() -> None:
    """Pinned so a regenerated oracle that quietly changes size is caught.

    These are the Phase 2 targets: 3034 decisions in total, every one of
    which must match exactly.
    """
    expected = {
        "AN_narrow": {"out": 438, "brought_back": 278, "unused": 29},
        "AN": {"out": 636, "brought_back": 742, "unused": 29},
        "AD": {"out": 514, "brought_back": 347, "unused": 21},
    }
    for name, counts in expected.items():
        assert _load(name)["counts"] == counts, name


def test_wider_cluster_set_strips_back_more() -> None:
    """A physical consistency check between two sets of the same chemistry.

    The wide AN set reaches further in composition space before its
    boundary, so more collisions land outside it and get stripped rather
    than counted as grown out.
    """
    narrow, wide = _load("AN_narrow")["counts"], _load("AN")["counts"]
    assert wide["brought_back"] > narrow["brought_back"]
