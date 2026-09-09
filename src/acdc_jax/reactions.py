"""Enumerating collisions and evaporations.

The pair loop and decision ladder of the Perl generator (`:2680-2836`),
producing the reaction list that the right-hand side is built from.

Every unordered pair is considered once, over the upper triangle; the
symmetric entries are written out afterwards, which is why self-collisions
end up carrying a factor of one half.

Plain Python: runs once per cluster set.
"""

from __future__ import annotations

import dataclasses

from acdc_jax.boundary import BoundarySystem
from acdc_jax.system import AcdcSystem


@dataclasses.dataclass(frozen=True)
class Collision:
    """One collision ``i + j -> products``.

    ``products`` maps a state-vector index to its multiplicity. A boundary
    collision that is stripped back has several entries: the surviving
    cluster plus the monomers that came off. A collision that grows out of
    the system has a single entry, the relevant ``out_*`` flux slot.
    """

    i: int
    j: int
    products: tuple[tuple[int, int], ...]
    kind: str
    """``"in_system"``, ``"out"``, ``"boundary"`` or ``"recombination"``."""

    @property
    def is_self_collision(self) -> bool:
        return self.i == self.j


@dataclasses.dataclass(frozen=True)
class Evaporation:
    """One evaporation ``k -> i + j``, the reverse of an in-system collision."""

    k: int
    i: int
    j: int

    @property
    def is_symmetric(self) -> bool:
        return self.i == self.j


@dataclasses.dataclass(frozen=True)
class ReactionSet:
    collisions: tuple[Collision, ...]
    evaporations: tuple[Evaporation, ...]

    def quad_triples(self) -> set[tuple[int, int, int]]:
        """Every ``(i, j, k)`` with a nonzero quadratic coefficient.

        Both orderings of each pair, matching the reference, which writes
        ``coef_quad(i,j,k)`` and ``coef_quad(j,i,k)`` separately.
        """
        out: set[tuple[int, int, int]] = set()
        for collision in self.collisions:
            for product, _multiplicity in collision.products:
                out.add((collision.i, collision.j, product))
                out.add((collision.j, collision.i, product))
        return out


def enumerate_reactions(
    system: AcdcSystem,
    boundary: BoundarySystem,
    prune_useless: bool = True,
) -> ReactionSet:
    """Enumerate every collision and evaporation in the system.

    Args:
        prune_useless: drop boundary collisions whose products are exactly
            their reactants. The reference does this by default
            (``disable_useless_collisions = 1``, Perl `:243`); such a
            reaction changes nothing but would still contribute equal and
            opposite terms to the right-hand side.
    """
    collisions: list[Collision] = []
    evaporations: list[Evaporation] = []

    n = system.n_clusters
    generic = {system.generic_neg, system.generic_pos} - {-1}

    for i in range(n):
        for j in range(i, n):
            # Two charger ions: they recombine and leave nothing behind, so
            # the flux is booked to the recombination counter rather than
            # producing a cluster (Perl :2714-2717).
            if i in generic and j in generic:
                if i != j:
                    collisions.append(
                        Collision(
                            i, j, ((system.flux_index["rec"], 1),), "recombination"
                        )
                    )
                continue

            result = boundary.combine(system.compositions[i], system.compositions[j])
            if not result.valid_coll:
                continue

            product_counts = _product_counts(system, boundary, i, j)
            if product_counts is None:
                continue

            if boundary.is_member(product_counts):
                k = system.labels.index(boundary.label_of(product_counts))
                collisions.append(Collision(i, j, ((k, 1),), "in_system"))
                if result.valid_evap and not (i in generic or j in generic):
                    evaporations.append(Evaporation(k, i, j))
                continue

            outcome = boundary.check_boundary(product_counts)
            if outcome.lout:
                slot = {1: "out_neu", 2: "out_neg", 3: "out_pos"}[outcome.lout]
                collisions.append(
                    Collision(i, j, ((system.flux_index[slot], 1),), "out")
                )
                continue

            products = _brought_back_products(system, outcome)
            if prune_useless and _is_no_op(products, i, j):
                continue
            collisions.append(Collision(i, j, products, "boundary"))

    return ReactionSet(tuple(collisions), tuple(evaporations))


def _product_counts(
    system: AcdcSystem, boundary: BoundarySystem, i: int, j: int
) -> tuple[int, ...] | None:
    """Composition of the collision product, or None if impossible."""
    result = boundary.combine(system.compositions[i], system.compositions[j])
    if not result.valid_coll or result.label is None:
        return None
    from acdc_jax import labels as label_module

    return tuple(label_module.composition_vector(result.label, system.order))


def _brought_back_products(system: AcdcSystem, outcome) -> tuple[tuple[int, int], ...]:
    """Turn a boundary outcome into (index, multiplicity) product pairs."""
    products = [(system.labels.index(outcome.label), 1)]
    for name, count in outcome.monomers.items():
        if not count:
            continue
        products.append((system.labels.index(f"1{name}"), count))
    return tuple(products)


def _is_no_op(products: tuple[tuple[int, int], ...], i: int, j: int) -> bool:
    """Whether a boundary collision returns exactly its own reactants.

    ``1A + 2A -> 3A -> 2A + 1 A`` puts back precisely what went in. The
    reference drops these (Perl :2777-2787); keeping them would add equal
    and opposite terms to the right-hand side, which is harmless but
    changes the sparsity pattern and so the comparison against
    ``coef_quad``.
    """
    bag: dict[int, int] = {}
    for index, multiplicity in products:
        bag[index] = bag.get(index, 0) + multiplicity
    reactants: dict[int, int] = {}
    reactants[i] = reactants.get(i, 0) + 1
    reactants[j] = reactants.get(j, 0) + 1
    return bag == reactants
