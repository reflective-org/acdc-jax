"""--charge_balance: one generic ion set algebraically to balance the rest.

Phase 8.9. The generator's own flag (not the MATLAB driver's per-iteration
projection recorded as F2). Validated against the emitted block at the top
of feval/formation in the cb1 fixture (acdc_equations_cb1.f90:93-99):

    if (c(53)+sum(c(17:34))-sum(c(35:52))>0) then
        c(54) = c(53)+sum(c(17:34))-sum(c(35:52))
    else
        c(53) = -(sum(c(17:34))-sum(c(35:52)))
        c(54) = 0
    end if

plus isconst(54)=.true., isconst(53)=.false., source(53)=coef(3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acdc_jax import rates, rhs
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions, parse_nonstandard_file
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
FIXTURE = REPO / "validation/fixtures/generated/acdc_equations_cb1.f90"


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


def _reference_projection(system, c: np.ndarray, mode: int) -> np.ndarray:
    """A literal NumPy transcription of the emitted Fortran block, as the oracle.

    Compared at 1e-13: the two sides sum the 18 ion concentrations in a
    different order, and the sums are ~1e11 with terms spanning 1e4-1e12.
    """
    c = c.copy()
    neg_i, pos_i = system.generic_neg, system.generic_pos
    charges = np.asarray(system.charges)
    n = system.n_clusters
    cl = np.ones(n, dtype=bool)
    cl[[neg_i, pos_i]] = False
    neg_sum = c[:n][cl & (charges < 0)].sum()
    pos_sum = c[:n][cl & (charges > 0)].sum()
    if mode > 0:
        if c[neg_i] + neg_sum - pos_sum > 0:
            c[pos_i] = c[neg_i] + neg_sum - pos_sum
        else:
            c[neg_i] = -(neg_sum - pos_sum)
            c[pos_i] = 0.0
    else:
        if c[pos_i] + pos_sum - neg_sum > 0:
            c[neg_i] = c[pos_i] + pos_sum - neg_sum
        else:
            c[pos_i] = -(pos_sum - neg_sum)
            c[neg_i] = 0.0
    return c


class TestProjection:
    @pytest.mark.parametrize("mode", [1, -1])
    def test_matches_the_emitted_algebra(self, model, mode: int) -> None:
        """Both branches of the guard, on random states."""
        system, _, _ = model
        rng = np.random.default_rng(8)
        hit_true = hit_false = False
        for _ in range(40):
            c = 10.0 ** rng.uniform(4, 12, size=system.n_equations)
            # Bias some states so the else-branch fires too.
            if rng.random() < 0.5:
                c[system.generic_neg] = 0.0
                c[[i for i in range(system.n_clusters) if system.charges[i] < 0]] *= (
                    1e-6
                )
            got = np.asarray(rhs.charge_balance_projection(system, c, mode))
            want = _reference_projection(system, c, mode)
            np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
            excess = want[system.generic_pos] if mode > 0 else want[system.generic_neg]
            hit_true |= excess > 0
            hit_false |= excess == 0
        assert hit_true and hit_false, "both guard branches must be exercised"

    def test_mode_zero_is_identity(self, model) -> None:
        system, _, _ = model
        c = np.full(system.n_equations, 1e10)
        np.testing.assert_array_equal(
            np.asarray(rhs.charge_balance_projection(system, c, 0)), c
        )

    def test_result_is_charge_neutral(self, model) -> None:
        """The whole point: after projection, total negative charge equals
        total positive charge (in the true-branch)."""
        system, _, _ = model
        c = np.full(system.n_equations, 1e10)
        projected = np.asarray(rhs.charge_balance_projection(system, c, 1))
        charges = np.asarray(system.charges)
        n = system.n_clusters
        neg_total = projected[:n][charges < 0].sum()
        pos_total = projected[:n][charges > 0].sum()
        assert neg_total == pytest.approx(pos_total, rel=1e-12)


class TestAssembly:
    def test_fitted_ion_is_algebraic(self, model) -> None:
        """isconst(54)=.true. and its source dropped; neg still sourced."""
        system, reactions, inputs = model
        co = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6, charge_balance=1
        )
        assert co.isconst[system.generic_pos]
        assert not co.isconst[system.generic_neg]
        source = np.asarray(co.source)
        assert source[system.generic_pos] == 0.0
        assert source[system.generic_neg] == 3e6

    def test_mirror_mode(self, model) -> None:
        system, reactions, inputs = model
        co = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6, charge_balance=-1
        )
        assert co.isconst[system.generic_neg]
        assert not co.isconst[system.generic_pos]

    def test_fixture_marks_the_same_species(self) -> None:
        """Read straight from the generated Fortran."""
        if not FIXTURE.exists():
            pytest.skip("cb1 fixture not generated")
        text = FIXTURE.read_text()
        assert "isconst(54) = .true." in text
        assert "isconst(53) = .false." in text
        assert "source(53) = coef(3)" in text
        assert "c(54) = c(53)+sum(c(17:34))-sum(c(35:52))" in text

    def test_rhs_applies_the_projection(self, model) -> None:
        """f for the fitted ion must be zero (isconst), and the projected state
        must be what the other terms were evaluated with."""
        system, reactions, inputs = model
        co = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6, charge_balance=1
        )
        c = np.full(system.n_equations, 1e10)
        f = np.asarray(rhs.rhs(co, c))
        assert f[system.generic_pos] == 0.0
        # Same RHS as evaluating the unprojected path on the projected state.
        co0 = rhs.assemble(system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6)
        projected = np.asarray(rhs.charge_balance_projection(system, c, 1))
        f0 = np.asarray(rhs.rhs(co0, projected))
        keep = np.ones(system.n_equations, dtype=bool)
        keep[system.generic_pos] = False  # isconst differs by construction
        np.testing.assert_allclose(f[keep], f0[keep], rtol=1e-14)

    def test_formation_rate_projects_too(self, model, tmp_path) -> None:
        """The emitted code projects at the top of BOTH `feval` and
        `formation` (fixture acdc_equations_cb1.f90:93-99); this had it in
        `rhs` only.

        No collision in the bundled set sends a charger ion out of the
        system -- measured, zero of them -- so J there is independent of
        both generic ions and the omission could not change a number. To
        make the guarantee testable, `--nst` (Phase 8.8) is used to route
        `neg + 1A` to the outgoing flux, which is what a wider cluster set
        would do on its own. J from a state whose fitted ion is stale must
        then equal J from the same state projected by hand.
        """
        system, _, inputs = model
        cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
        boundary = BoundarySystem(cluster_set)
        rule = tmp_path / "nst.txt"
        rule.write_text("neg 1A 1 out_neg\n")
        reactions = enumerate_reactions(
            system, boundary, nonstandard=parse_nonstandard_file(rule, system)
        )

        # charge_balance=-1 pins `neg`, which is now a collider that exits.
        co = rhs.assemble(
            system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6, charge_balance=-1
        )
        c = np.full(system.n_equations, 1e10)
        c[system.generic_neg] = 0.0  # stale: not the balanced value
        projected = np.asarray(rhs.charge_balance_projection(system, c, -1))
        assert projected[system.generic_neg] > 0

        j_stale = float(rhs.formation_rate(system, co, c)["j_tot"])
        j_projected = float(rhs.formation_rate(system, co, projected)["j_tot"])
        assert j_stale == pytest.approx(j_projected, rel=1e-14)

        # ...and it is not vacuous: reading the raw state, as the
        # unprojected path did, gives a materially different J.
        plain = rhs.assemble(system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6)
        j_unprojected = float(rhs.formation_rate(system, plain, c)["j_tot"])
        assert abs(j_unprojected - j_stale) / j_stale > 1e-6

    def test_default_is_unchanged(self, model) -> None:
        system, reactions, inputs = model
        co = rhs.assemble(system, reactions, inputs, 280.0, 1e-3, 3e6, 3e6)
        assert co.charge_balance == 0
        assert not co.isconst[system.generic_pos]
