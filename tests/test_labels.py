"""Cluster label parsing and canonicalisation.

Label handling has no tolerance: a mis-parsed label does not raise, it
silently creates a second entry for a cluster that already exists, and the
error surfaces much later as a missing reaction.
"""

from __future__ import annotations

import pytest

from acdc_jax import labels

ORDER = ("A", "B", "N", "P")


class TestParse:
    def test_basic(self) -> None:
        assert labels.parse("4A1B3N") == {"A": 4, "B": 1, "N": 3}
        assert labels.parse("2A") == {"A": 2}
        assert labels.parse("1N1P") == {"N": 1, "P": 1}

    def test_implicit_count(self) -> None:
        """The generator prints stripped monomers with a bare molecule name:
        `2 A` in a boundary log means two of molecule A, not one cluster
        named `2A`. So `A` must parse as one A."""
        assert labels.parse("A") == {"A": 1}
        assert labels.parse("AN") == {"A": 1, "N": 1}

    def test_two_letter_molecule_names(self) -> None:
        """Molecule symbols may be two characters, so `1Na` is one Na, not
        one N followed by a stray."""
        assert labels.parse("1Na2Cl") == {"Na": 1, "Cl": 2}

    def test_repeated_molecule_accumulates(self) -> None:
        assert labels.parse("1A1A") == {"A": 2}

    def test_generic_ions_have_no_composition(self) -> None:
        assert labels.parse("neg") == {}
        assert labels.parse("pos") == {}
        assert labels.is_generic_ion("neg")
        assert not labels.is_generic_ion("1A")

    @pytest.mark.parametrize("bad", ["", "1", "2a", "A-1", "1A 2N", "!"])
    def test_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(ValueError, match="malformed"):
            labels.parse(bad)


class TestCanonical:
    def test_reorders(self) -> None:
        assert labels.canonical("1N2A", ORDER) == "2A1N"

    def test_idempotent(self) -> None:
        for label in ("2A1N", "1A", "4A1B3N"):
            assert labels.canonical(labels.canonical(label, ORDER), ORDER) == label

    def test_normalises_implicit_counts(self) -> None:
        assert labels.canonical("AN", ORDER) == "1A1N"

    def test_generic_ions_pass_through(self) -> None:
        assert labels.canonical("neg", ORDER) == "neg"

    def test_rejects_unknown_molecule(self) -> None:
        with pytest.raises(ValueError, match="not in the system"):
            labels.canonical("1A1D", ORDER)


class TestFormat:
    def test_omits_zero_counts(self) -> None:
        """`2A`, never `2A0B0N0P` -- the generator omits absent molecules."""
        assert labels.format_label({"A": 2, "B": 0, "N": 0, "P": 0}, ORDER) == "2A"

    def test_round_trip(self) -> None:
        for label in ("1A", "2A1N", "4A1B3N", "5A5N1P"):
            assert labels.format_label(labels.parse(label), ORDER) == label


def test_total_molecules_counts_the_proton() -> None:
    """`1N1P` is protonated ammonia -- a monomer, but two by this count.

    Documented rather than special-cased: callers that need the physical
    molecule count must exclude the proton themselves, as the generator does
    when computing volumes.
    """
    assert labels.total_molecules("1N1P") == 2
    assert labels.total_molecules("2A1N") == 3


def test_composition_vector() -> None:
    assert labels.composition_vector("2A1N", ORDER) == [2, 0, 1, 0]
    assert labels.composition_vector("neg", ORDER) == [0, 0, 0, 0]
