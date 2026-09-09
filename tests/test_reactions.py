"""System assembly and reaction enumeration. Phase 3's gate.

Enumeration decides *where* the nonzero rate coefficients are; Phase 4
decides their values. So the gate here is the sparsity pattern of the
reference's assembled tensors, compared as exact sets of index triples.
That separates a wrong reaction list from a wrong formula, which a single
end-to-end number would not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import FLUX_SLOTS, build_system

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "fortran/src/Perl_input/input_ANnarrow_neutral_neg_pos.inp"

sys.path.insert(0, str(REPO / "validation/reference"))
sys.path.insert(0, str(REPO / "validation"))
metadata = pytest.importorskip("metadata")
reference = pytest.importorskip("reference")


@pytest.fixture(scope="module")
def built():
    cluster_set = parse_cluster_set(INPUT)
    system = build_system(cluster_set)
    boundary = BoundarySystem(cluster_set)
    return system, boundary, enumerate_reactions(system, boundary)


@pytest.fixture(scope="module")
def tensors():
    coef = reference.make_coef(280.0, 1e-3, 3.0e6)
    return reference.get_rate_coefs(coef)


class TestSystemAssembly:
    def test_sizes_match_the_reference(self, built) -> None:
        system, _, _ = built
        sizes = metadata.sizes()
        assert system.n_clusters == sizes["nclust"] == 54
        assert system.n_equations == sizes["neq"] == 63

    def test_labels_match_the_reference(self, built) -> None:
        system, _, _ = built
        assert list(system.labels) == metadata.cluster_names()

    def test_charge_states_match_the_reference(self, built) -> None:
        """The reference codes charge as 1/2/3, not as a signed value."""
        system, _, _ = built
        coded = [1 if c == 0 else (2 if c < 0 else 3) for c in system.charges]
        np.testing.assert_array_equal(coded, metadata.charging_state())

    def test_flux_slot_order(self, built) -> None:
        """Slot order is fixed by the generator; out_neu must land on the
        state index the reference calls 60 (1-based)."""
        system, _, _ = built
        assert system.flux_index["out_neu"] == 59
        assert system.flux_index["out_neg"] == 60
        assert system.flux_index["out_pos"] == 61
        assert len(FLUX_SLOTS) == 9

    def test_generic_negative_ion_has_a_negative_count(self, built) -> None:
        """The whole charge-transfer mechanism.

        `neg` is `1B-1A`: adding it to `1A` yields `1B`, so charge transfer
        is ordinary vector addition. Adding it to `1N`, which has no acid to
        convert, yields a negative count -- which is exactly how the
        reference detects that the charge has nowhere to go.
        """
        system, _, _ = built
        composition = system.compositions[system.generic_neg]
        assert composition == (-1, 1, 0, 0)
        assert min(composition) < 0

    def test_generic_positive_ion_is_a_bare_proton(self, built) -> None:
        system, _, _ = built
        assert system.compositions[system.generic_pos] == (0, 0, 0, 1)

    def test_generic_ion_charges(self, built) -> None:
        system, _, _ = built
        assert system.charges[system.generic_neg] == -1
        assert system.charges[system.generic_pos] == 1


class TestEnumerationGate:
    def test_quadratic_sparsity_matches_exactly(self, built, tensors) -> None:
        """Phase 3 gate: every (i, j, k) with a nonzero coef_quad."""
        _, _, reactions = built
        coef_quad, _ = tensors
        expected = set(map(tuple, np.argwhere(coef_quad != 0)))
        got = reactions.quad_triples()
        assert got == expected
        assert len(expected) == 2679

    def test_evaporation_sparsity_matches_exactly(self, built, tensors) -> None:
        """coef_lin holds evaporation and the coagulation sink together; the
        sink lives entirely in the (coag, coag, k) plane, so the rest is
        evaporation."""
        system, _, reactions = built
        _, coef_lin = tensors
        coag = system.flux_index["coag"]
        nonzero = set(map(tuple, np.argwhere(coef_lin != 0)))
        evaporation = {t for t in nonzero if not (t[0] == coag and t[1] == coag)}

        got = set()
        for e in reactions.evaporations:
            got.add((e.i, e.j, e.k))
            got.add((e.j, e.i, e.k))
        assert got == evaporation
        assert len(evaporation) == 422

    def test_evaporation_count(self, built) -> None:
        """213 distinct evaporation channels, 4 of them symmetric (k -> i+i).

        422 = (213 - 4)*2 + 4: an asymmetric channel is written in both
        orderings, a symmetric one only once.
        """
        _, _, reactions = built
        assert len(reactions.evaporations) == 213
        symmetric = sum(1 for e in reactions.evaporations if e.is_symmetric)
        assert symmetric == 4
        assert (213 - symmetric) * 2 + symmetric == 422

    def test_coagulation_sink_covers_all_but_the_vapour_monomers(
        self, built, tensors
    ) -> None:
        """52 of 54: the system was generated with --cs_only 1A,0 --cs_only
        1N,0, which excludes the two neutral vapour monomers."""
        system, _, _ = built
        _, coef_lin = tensors
        coag = system.flux_index["coag"]
        entries = np.argwhere(coef_lin[coag, coag, :] != 0).ravel()
        assert len(entries) == 52
        assert system.labels.index("1A") not in entries
        assert system.labels.index("1N") not in entries


class TestReactionKinds:
    def test_counts(self, built) -> None:
        _, _, reactions = built
        kinds: dict[str, int] = {}
        for collision in reactions.collisions:
            kinds[collision.kind] = kinds.get(collision.kind, 0) + 1
        assert kinds["out"] == 438, "must match the boundary oracle"
        assert kinds["boundary"] == 278
        assert kinds["recombination"] == 1

    def test_grow_out_counts_match_the_boundary_oracle(self, built) -> None:
        """Cross-check between two independently derived numbers: the
        enumeration's 'out' collisions and the Perl decision log."""
        import json

        _, _, reactions = built
        oracle = json.loads(
            (REPO / "validation/goldens/boundary_AN_narrow.json").read_text()
        )
        out_here = sum(1 for c in reactions.collisions if c.kind == "out")
        assert out_here == oracle["counts"]["out"]

    def test_recombination_goes_to_its_own_counter(self, built) -> None:
        """neg + pos leaves nothing behind, so the flux is booked to the
        recombination slot rather than producing a cluster."""
        system, _, reactions = built
        rec = [c for c in reactions.collisions if c.kind == "recombination"]
        assert len(rec) == 1
        assert rec[0].products == ((system.flux_index["rec"], 1),)

    def test_boundary_collisions_can_have_several_products(self, built) -> None:
        """2A + 2A -> 4A -> 2A + 2 A: one cluster plus stripped monomers."""
        _, _, reactions = built
        multi = [c for c in reactions.collisions if len(c.products) > 1]
        assert multi, "no multi-product boundary collisions found"

    def test_no_op_boundary_collisions_are_pruned(self, built) -> None:
        """A collision returning exactly its reactants changes nothing.

        The reference drops these by default (Perl :2777-2787); keeping them
        would add equal and opposite terms and, more visibly here, change
        the sparsity pattern away from coef_quad.
        """
        system, boundary, _ = built
        pruned = enumerate_reactions(system, boundary, prune_useless=True)
        kept = enumerate_reactions(system, boundary, prune_useless=False)
        assert len(kept.collisions) > len(pruned.collisions)

    def test_generic_ion_collisions_produce_no_evaporation(self, built) -> None:
        """A product cannot evaporate back into a charger ion (Perl :10219)."""
        system, _, reactions = built
        generic = {system.generic_neg, system.generic_pos}
        for e in reactions.evaporations:
            assert e.i not in generic and e.j not in generic
