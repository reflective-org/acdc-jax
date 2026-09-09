"""Free-energy and dipole/polarizability file parsing.

Gate for Phase 1.5/1.6. The energy parser is validated against the values
the generator inlined into `get_evap` -- it copies them verbatim from the
file, so agreement is exact equality, not a tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from acdc_jax import thermo
from acdc_jax.clusterset import parse_cluster_set

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
ENERGY_FILE = INPUTS / "HS298.15K_example.txt"
DIPOLE_FILE = INPUTS / "dip_pol_298.15K_example.txt"

sys.path.insert(0, str(REPO / "validation/reference"))
metadata = pytest.importorskip("metadata")

ORDER = ("A", "B", "N", "D", "P")


@pytest.fixture(scope="module")
def energies():
    return thermo.parse_energy_file(ENERGY_FILE, ORDER)


@pytest.fixture(scope="module")
def dipoles():
    return thermo.parse_dipole_file(DIPOLE_FILE, ORDER)


@pytest.fixture(scope="module")
def cluster_labels():
    cs = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    return cs.labels()


class TestEnergyFile:
    def test_preamble(self, energies) -> None:
        """Line 1 is the reference pressure in Pa, line 2 the temperature."""
        assert energies.pressure == 101325.0
        assert energies.temperature == 298.15

    def test_entry_count(self, energies) -> None:
        """117 clusters, spanning positive ions, neutrals and negatives."""
        assert len(energies.delta_h) == 117

    def test_units_are_kept_as_written(self, energies) -> None:
        """H in kcal/mol, S in cal/(mol K), unconverted -- so the comparison
        against the generator's inlined literals can be exact."""
        assert energies.delta_h["2A"] == -17.8487481487
        assert energies.delta_s["2A"] == -33.418352

    def test_matches_the_emitted_literals_exactly(
        self, energies, cluster_labels
    ) -> None:
        """The Phase 1.5 gate.

        The generator copies H and S straight from the file into every
        evaporation expression, so every cluster that has data must agree to
        the last digit.
        """
        emitted = metadata.energy_data()
        checked = 0
        for label in cluster_labels:
            if label not in emitted:
                continue
            h, s = emitted[label]
            if (h, s) == (0.0, 0.0):
                continue  # zero-reference monomer, no file entry needed
            assert energies.delta_h[label] == h, label
            assert energies.delta_s[label] == s, label
            checked += 1
        assert checked >= 45, f"only {checked} clusters compared"

    def test_gibbs_from_h_and_s(self, energies) -> None:
        """G = H - T*S/1000, the 1000 converting cal to kcal."""
        expected = -17.8487481487 - 298.15 * (-33.418352) / 1000.0
        assert energies.gibbs("2A") == pytest.approx(expected, rel=1e-15)

    def test_monomers_default_to_zero(self, energies) -> None:
        """Formation energies are relative to the free monomers, so a
        monomer is zero by definition and needs no entry."""
        assert "1A" not in energies.delta_h
        assert energies.gibbs("1A") == 0.0

    def test_reference_state_entry_is_zero(self, energies) -> None:
        """1D1P is the reference for the positive ions and is written
        explicitly as `0 0`."""
        assert energies.delta_h["1D1P"] == 0.0
        assert energies.gibbs("1D1P") == 0.0

    def test_protonated_ammonia_is_not_a_reference(self, energies) -> None:
        """1N1P is a monomer but has genuinely nonzero data -- so 'monomer
        implies zero' is a default, not a rule."""
        assert energies.delta_h["1N1P"] == 18.2459260893

    def test_gibbs_at_another_temperature(self, energies) -> None:
        g298 = energies.gibbs("2A", 298.15)
        g250 = energies.gibbs("2A", 250.0)
        assert g250 != g298

    def test_gibbs_array(self, energies) -> None:
        values = energies.gibbs_array(["1A", "2A", "neg"])
        assert values[0] == 0.0
        assert values[2] == 0.0, "generic ions have no formation energy"
        assert values[1] != 0.0

    def test_labels_are_canonicalised(self) -> None:
        """A file may write `1N2A` where the cluster set says `2A1N`."""
        table = thermo.parse_energy_file(ENERGY_FILE, ORDER)
        assert "2A1N" in table.delta_h


class TestEnergyValidation:
    def test_rejects_duplicate(self, tmp_path: Path) -> None:
        f = tmp_path / "e.txt"
        f.write_text("101325\n298.15\n2A\t-1\t-2\n2A\t-3\t-4\n")
        with pytest.raises(ValueError, match="duplicate entry"):
            thermo.parse_energy_file(f)

    def test_rejects_missing_preamble(self, tmp_path: Path) -> None:
        f = tmp_path / "e.txt"
        f.write_text("101325\n")
        with pytest.raises(ValueError, match="missing pressure"):
            thermo.parse_energy_file(f)

    def test_rejects_nonpositive_pressure(self, tmp_path: Path) -> None:
        f = tmp_path / "e.txt"
        f.write_text("0\n298.15\n2A\t-1\t-2\n")
        with pytest.raises(ValueError, match="pressure must be positive"):
            thermo.parse_energy_file(f)

    def test_rejects_malformed_row(self, tmp_path: Path) -> None:
        f = tmp_path / "e.txt"
        f.write_text("101325\n298.15\n2A\t-1\t-2\t-3\t-4\n")
        with pytest.raises(ValueError, match="expected"):
            thermo.parse_energy_file(f)

    def test_gibbs_only_entry_cannot_be_retemperatured(self, tmp_path: Path) -> None:
        """Upstream dies in this case (Perl :2540): with only G at one
        temperature there is no way to re-evaluate it at another."""
        f = tmp_path / "e.txt"
        f.write_text("101325\n298.15\n2A\t-7.5\n")
        table = thermo.parse_energy_file(f)
        assert table.gibbs("2A", 298.15) == -7.5
        with pytest.raises(ValueError, match="cannot be re-evaluated"):
            table.gibbs("2A", 250.0)

    def test_missing_energy_is_reported_with_the_full_list(self, energies) -> None:
        """Upstream accumulates every missing cluster before dying
        (Perl :2560-2590), which is more useful than failing on the first."""
        with pytest.raises(ValueError, match="9A9N.*9A8N|9A8N.*9A9N"):
            thermo.check_energies_available(
                energies, ["2A", "9A9N", "9A8N"], monomer_labels={"1A"}
            )

    def test_monomers_and_generic_ions_are_exempt(self, energies) -> None:
        thermo.check_energies_available(
            energies, ["1A", "neg", "pos"], monomer_labels={"1A"}
        )


class TestDipoleFile:
    def test_locking_coefficients(self, dipoles) -> None:
        """Lines 1 and 2: monomer and cluster dipole locking. Su73 only --
        Su82 reads them but does not use them."""
        assert dipoles.monomer_locking == 0.15
        assert dipoles.cluster_locking == 0.15

    def test_entry_count(self, dipoles) -> None:
        assert len(dipoles.dipole) == 53

    def test_units(self, dipoles) -> None:
        """Dipole in Debye, polarizability in Angstrom^3."""
        assert dipoles.dipole["1A"] == 2.9643
        assert dipoles.polarizability["1A"] == 6.2

    def test_every_neutral_cluster_has_data(self, dipoles, cluster_labels) -> None:
        """Only neutrals need it: in an ion-neutral collision the neutral is
        the partner being polarised."""
        neutral = [
            label for label in cluster_labels if "B" not in label and "P" not in label
        ]
        thermo.check_dipoles_available(dipoles, neutral)

    def test_missing_dipole_is_reported(self, dipoles) -> None:
        with pytest.raises(ValueError, match="9A9N"):
            thermo.check_dipoles_available(dipoles, ["1A", "9A9N"])

    def test_rejects_out_of_range_locking(self, tmp_path: Path) -> None:
        f = tmp_path / "d.txt"
        f.write_text("1.5\n0.15\n1A\t2.9\t6.2\n")
        with pytest.raises(ValueError, match="outside"):
            thermo.parse_dipole_file(f)

    def test_rejects_malformed_row(self, tmp_path: Path) -> None:
        f = tmp_path / "d.txt"
        f.write_text("0.15\n0.15\n1A\t2.9\n")
        with pytest.raises(ValueError, match="expected"):
            thermo.parse_dipole_file(f)


def test_reference_number_density() -> None:
    """The 7.33893243358348e27/T prefactor in every get_evap expression."""
    value = thermo.reference_number_density(101325.0, 1.0)
    assert value == pytest.approx(7.33893243358348e27, rel=1e-14)
