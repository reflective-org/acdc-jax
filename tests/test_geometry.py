"""Cluster mass, volume, radius and diameters.

Gate for Phase 1.4: masses, mass-equivalent diameters and mobility
diameters must reproduce `acdc_system_AN_ions_example.f90` exactly, after
applying the same 2-decimal rounding the generator uses when emitting them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from acdc_jax import config, geometry
from acdc_jax.clusterset import parse_cluster_set

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation/reference"))
metadata = pytest.importorskip("metadata")

N_DECLARED = 52  # the .inp clusters; the reference adds 2 generic ions


@pytest.fixture(scope="module")
def system():
    cs = parse_cluster_set(
        REPO / "fortran/src/Perl_input/input_ANnarrow_neutral_neg_pos.inp"
    )
    return cs, np.array(cs.compositions), cs.molecules


class TestAgainstReference:
    def test_mass(self, system) -> None:
        _, comp, mols = system
        computed = geometry.emitted(geometry.cluster_mass(comp, mols))
        np.testing.assert_array_equal(computed, metadata.mass()[:N_DECLARED])

    def test_diameter(self, system) -> None:
        _, comp, mols = system
        computed = geometry.emitted(geometry.cluster_diameter(comp, mols))
        np.testing.assert_array_equal(computed, metadata.diameter()[:N_DECLARED])

    def test_mobility_diameter(self, system) -> None:
        _, comp, mols = system
        computed = geometry.emitted(geometry.mobility_diameter(comp, mols))
        np.testing.assert_array_equal(computed, metadata.mob_diameter()[:N_DECLARED])

    def test_rounding_is_required(self, system) -> None:
        """Justifies `emitted()`.

        The generator writes these through sprintf("%.2f"), so a
        full-precision comparison misses by ~1e-3 -- far outside anything
        that looks like a tolerance failure, and easy to misdiagnose as a
        formula error.
        """
        _, comp, mols = system
        raw = geometry.cluster_diameter(comp, mols)
        gap = np.abs(raw - metadata.diameter()[:N_DECLARED]).max()
        assert 1e-4 < gap < 1e-2


class TestMass:
    def test_additive(self, system) -> None:
        cs, comp, mols = system
        mass = geometry.cluster_mass(comp, mols)
        i = cs.labels().index("2A1N")
        assert mass[i] == pytest.approx(2 * 98.08 + 17.04)

    def test_proton_contributes_mass(self, system) -> None:
        """1N1P is protonated ammonia: 17.04 + 1.00."""
        cs, comp, mols = system
        mass = geometry.cluster_mass(comp, mols)
        assert mass[cs.labels().index("1N1P")] == pytest.approx(18.04)


class TestVolume:
    def test_proton_contributes_no_volume(self, system) -> None:
        """The proton is charge bookkeeping, not matter (Perl :11498).

        So 1N1P must have exactly the volume of 1N -- if the proton were
        given a density it would inflate every positive cluster.
        """
        cs, comp, mols = system
        volume = geometry.cluster_volume(comp, mols)
        labels = cs.labels()
        assert volume[labels.index("1N1P")] == pytest.approx(
            volume[labels.index("1N")], rel=1e-15
        )

    def test_additive_bulk_volumes(self, system) -> None:
        cs, comp, mols = system
        volume = geometry.cluster_volume(comp, mols)
        labels = cs.labels()
        assert volume[labels.index("2A")] == pytest.approx(
            2 * volume[labels.index("1A")], rel=1e-15
        )

    def test_radius_from_volume(self, system) -> None:
        _, comp, mols = system
        v = geometry.cluster_volume(comp, mols)
        r = geometry.cluster_radius(comp, mols)
        np.testing.assert_allclose(4 / 3 * config.PI * r**3, v, rtol=1e-14)

    def test_dimer_diameter_scales_as_cube_root_of_two(self, system) -> None:
        """The relation the exp_loss coagulation sink depends on."""
        cs, comp, mols = system
        d = geometry.cluster_diameter(comp, mols)
        labels = cs.labels()
        ratio = d[labels.index("2A")] / d[labels.index("1A")]
        assert ratio == pytest.approx(2 ** (1 / 3), rel=1e-14)


class TestMobilityDiameterBug:
    """The reference's mass correction is defeated by a unit bug (F11)."""

    def test_fortran_default_is_a_flat_offset(self, system) -> None:
        """Every emitted mobility diameter is d_mass + 0.30 nm exactly, with
        no mass dependence -- which is the observable signature of the bug."""
        _, comp, mols = system
        d = geometry.cluster_diameter(comp, mols)
        dm = geometry.mobility_diameter(comp, mols)
        np.testing.assert_allclose(dm - d, 0.3, rtol=0, atol=1e-15)

    def test_reference_arrays_show_the_same_signature(self) -> None:
        """Not our computation -- the committed Fortran itself."""
        gap = metadata.mob_diameter() - metadata.diameter()
        np.testing.assert_allclose(gap, 0.30, atol=1e-12)

    def test_tammet_mode_applies_the_correction(self, system) -> None:
        _, comp, mols = system
        cfg = config.FidelityConfig(mobility_diameter="tammet")
        dm = geometry.mobility_diameter(comp, mols, cfg)
        d = geometry.cluster_diameter(comp, mols)
        assert np.all(dm > d + 0.3)

    def test_the_two_modes_differ_materially(self, system) -> None:
        """~14% on the smallest clusters: large enough that it would matter
        for size-bin classification, which is why it is flagged rather than
        quietly fixed."""
        cs, comp, mols = system
        i = cs.labels().index("1A")
        fortran = geometry.mobility_diameter(comp, mols)[i]
        tammet = geometry.mobility_diameter(
            comp, mols, config.FidelityConfig(mobility_diameter="tammet")
        )[i]
        assert fortran == pytest.approx(0.854, abs=5e-4)
        assert tammet == pytest.approx(0.971, abs=5e-4)

    def test_the_buggy_factor_is_exactly_one(self) -> None:
        """The mechanism, asserted directly: with $mass1 in g/mol, the term
        28.8*mass_conv/mass is ~5e-28 and 1 + it is exactly 1.0."""
        factor = np.sqrt(1.0 + config.MOB_MASS_N2 * config.MASS_CONV / 98.08)
        assert factor == 1.0
