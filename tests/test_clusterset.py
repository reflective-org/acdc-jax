"""Cluster-set file parsing, checked against the generated Fortran.

The gate for Phase 1.1/1.2: the parser must reproduce the 52 declared
clusters of the bundled AN system, in order, with the same molecule types
the generator kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from acdc_jax.clusterset import parse_cluster_set

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
CLUSTER_SETS = REPO / "fortran/cluster_sets"

sys.path.insert(0, str(REPO / "validation/reference"))
metadata = pytest.importorskip("metadata")


@pytest.fixture(scope="module")
def an_narrow():
    return parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")


class TestMoleculeHeader:
    def test_all_declared_types_parsed(self, an_narrow) -> None:
        assert an_narrow.molecule_names == ("A", "B", "N", "D", "P")

    def test_unused_types_are_dropped(self, an_narrow) -> None:
        """The header declares dimethylamine, but no cluster contains it.

        The generator drops declared-but-unused types, giving
        `n_mol_types = 4`. Keeping all five would silently widen every
        composition vector and every index array downstream.
        """
        assert an_narrow.used_molecule_names == tuple(metadata.molecule_names())
        assert len(an_narrow.used_molecule_names) == metadata.sizes()["n_mol_types"]

    def test_charges(self, an_narrow) -> None:
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["A"].charge == 0
        assert by_name["B"].charge == -1
        assert by_name["P"].charge == 1

    def test_proton_identified(self, an_narrow) -> None:
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["P"].is_proton
        assert not by_name["A"].is_proton

    def test_no_missing_proton_in_this_file(self, an_narrow) -> None:
        """A negative mass marks the missing-proton pseudo-species. The
        bundled file has none, so nothing should be flagged as one."""
        assert not any(m.is_missing_proton for m in an_narrow.molecules)

    def test_masses_and_densities(self, an_narrow) -> None:
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["A"].mass == 98.08
        assert by_name["N"].mass == 17.04
        assert by_name["A"].density == 1830.0
        assert by_name["P"].density is None, "the proton has mass but no volume"

    def test_correspondence_maps(self, an_narrow) -> None:
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["A"].corr_negative == "B"
        assert by_name["B"].corr_neutral == "A"
        assert by_name["N"].corr_positive == "1N1P", (
            "may name a cluster, not a molecule"
        )
        assert by_name["A"].corr_positive is None

    def test_strengths(self, an_narrow) -> None:
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["A"].acid_strength == 2
        assert by_name["A"].base_strength == -1
        assert by_name["N"].base_strength == 1
        assert by_name["D"].base_strength == 2, "dimethylamine is the stronger base"

    def test_density_row_uses_spaces_not_tabs(self, an_narrow) -> None:
        """The bundled file is nominally tab-separated but the density row
        separates its label from the first value with spaces. Splitting on
        any whitespace handles both; splitting on tabs alone drops it."""
        by_name = {m.name: m for m in an_narrow.molecules}
        assert by_name["B"].density == 1830.0


class TestClusterBody:
    def test_cluster_count(self, an_narrow) -> None:
        """52 declared clusters. The reference has 54: the generator appends
        the two generic charger ions, which are not in the .inp."""
        assert len(an_narrow.compositions) == 52
        assert metadata.sizes()["nclust"] == 54

    def test_labels_match_the_reference_exactly(self, an_narrow) -> None:
        """The Phase 1 gate."""
        assert list(an_narrow.labels()) == metadata.cluster_names()[:52]

    def test_range_expansion(self, an_narrow) -> None:
        """`1-2  0  0  0  0` expands to 1A and 2A."""
        assert an_narrow.labels()[:2] == ("1A", "2A")

    def test_range_need_not_be_the_first_column(self) -> None:
        """Perl :1451-1457 scans every column for a range."""
        from acdc_jax.clusterset import _expand_ranges

        rows = _expand_ranges(["2", "0", "1-3", "0"], Path("x"), "")
        assert rows == [(2, 0, 1, 0), (2, 0, 2, 0), (2, 0, 3, 0)]

    def test_row_without_a_range_is_one_cluster(self) -> None:
        from acdc_jax.clusterset import _expand_ranges

        assert _expand_ranges(["4", "1", "4", "0"], Path("x"), "") == [(4, 1, 4, 0)]

    def test_rejects_two_ranges(self) -> None:
        from acdc_jax.clusterset import _expand_ranges

        with pytest.raises(ValueError, match="more than one range"):
            _expand_ranges(["1-2", "0", "1-3", "0"], Path("x"), "")


class TestOutRules:
    def test_all_three_charges_present(self, an_narrow) -> None:
        assert set(an_narrow.out_rules) == {"neutral", "negative", "positive"}

    def test_thresholds_match_the_reference(self, an_narrow) -> None:
        """Compared over the USED molecules: the reference emits 4-wide
        thresholds, the file declares 5 columns."""
        used = an_narrow.used_molecule_names
        keep = [an_narrow.molecule_names.index(n) for n in used]
        expected = metadata.out_thresholds()
        for charge, rules in an_narrow.out_rules.items():
            assert len(rules) == 1
            trimmed = tuple(rules[0][i] for i in keep)
            assert trimmed == tuple(expected[charge]), charge

    def test_out_rules_are_not_clusters(self, an_narrow) -> None:
        """`out neutral 6 0 5 0 0` is a threshold, not a cluster definition.

        6A5N is outside the enumerated set -- if it had been parsed as a
        cluster there would be 53, and it would appear in the labels.
        """
        assert "6A5N" not in an_narrow.labels()


class TestOtherClusterSets:
    @pytest.mark.parametrize(
        "name", ["input_AN_neutral_neg_pos.inp", "input_AD_neutral_neg_pos.inp"]
    )
    def test_parses(self, name: str) -> None:
        cs = parse_cluster_set(CLUSTER_SETS / name)
        assert len(cs.compositions) > 0
        assert set(cs.out_rules) == {"neutral", "negative", "positive"}
        assert all(len(row) == len(cs.molecules) for row in cs.compositions)

    def test_ad_set_uses_dimethylamine(self) -> None:
        """The AD set is the mirror image of AN: D is used, N is not.

        This is what makes AD worth carrying as a second test system -- it
        exercises the other side of the acid/base strength ladder.
        """
        cs = parse_cluster_set(CLUSTER_SETS / "input_AD_neutral_neg_pos.inp")
        assert "D" in cs.used_molecule_names
        assert "N" not in cs.used_molecule_names

    def test_wide_an_set_is_larger_than_narrow(self, an_narrow) -> None:
        wide = parse_cluster_set(CLUSTER_SETS / "input_AN_neutral_neg_pos.inp")
        assert len(wide.compositions) > len(an_narrow.compositions)


class TestValidation:
    def test_rejects_missing_name_row(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.inp"
        f.write_text("#\tA\tB\ncharge:\t0\t-1\n\t1-2\t0\n")
        with pytest.raises(ValueError, match="no `name:` row"):
            parse_cluster_set(f)

    def test_rejects_bad_charge(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.inp"
        f.write_text("#\tA\tB\nname:\tA\tB\ncharge:\t0\t-2\n\t1-2\t0\n")
        with pytest.raises(ValueError, match="charge -2"):
            parse_cluster_set(f)

    def test_rejects_no_clusters(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.inp"
        f.write_text("#\tA\tB\nname:\tA\tB\ncharge:\t0\t-1\n")
        with pytest.raises(ValueError, match="no cluster definitions"):
            parse_cluster_set(f)

    def test_rejects_duplicate_cluster(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.inp"
        f.write_text("#\tA\tB\nname:\tA\tB\ncharge:\t0\t-1\n\t2\t0\n\t2\t0\n")
        with pytest.raises(ValueError, match="duplicate cluster"):
            parse_cluster_set(f)
