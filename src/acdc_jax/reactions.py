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
import re
import warnings
from pathlib import Path

from acdc_jax import config
from acdc_jax import labels as label_module
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


@dataclasses.dataclass(frozen=True)
class NonStandardReactions:
    """The ``--nst`` file: forbidden collisions and product overrides.

    Perl :2009-2117 parses it; the pair ladder consults it first
    (:2496-2507). A forbidden pair has neither the collision nor the
    reverse evaporation. An overridden pair collides -- whatever the
    boundary logic would have said -- into the listed products and never
    evaporates back.
    """

    forbidden: frozenset[tuple[int, int]]
    """Unordered pairs as ``(min, max)``."""
    overrides: dict[tuple[int, int], tuple[tuple[int, int], ...]]
    """``(min, max) -> ((state index, coefficient), ...)`` in file order."""

    def is_forbidden(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self.forbidden

    def override(self, i: int, j: int) -> tuple[tuple[int, int], ...] | None:
        return self.overrides.get((min(i, j), max(i, j)))


_OUT_SLOTS = {
    "out_neu": "out_neu",
    "out": "out_neu",
    "out_neg": "out_neg",
    "out_pos": "out_pos",
}


def parse_nonstandard_file(
    path: str | Path,
    system: AcdcSystem,
    disable_useless_collisions: bool = True,
) -> NonStandardReactions:
    """Read a ``--nst`` file (Perl :2009-2117).

    Line forms, whitespace-separated, ``#`` comments:

    * ``X clusters`` -- X (not a monomer) collides with no other
      non-monomer cluster (generic ions are outside that loop);
    * ``X Y`` -- the collision X + Y is forbidden;
    * ``X Y c1 P1 [c2 P2 ...]`` -- X + Y forms ``c1 P1 + c2 P2 ...``. A
      product is a cluster label or ``out_neu``/``out``/``out_neg``/
      ``out_pos``. Coefficients must be integers: the Fortran path dies on
      a decimal point (:2088).

    A reactant that is not in the system skips the line with a warning, as
    upstream prints and continues. With ``disable_useless_collisions`` a
    line spelling ``X + Y -> X + Y`` is treated as forbidden (:2076-2083;
    copied literally, including its raw-string label comparison).
    """
    n = system.n_clusters
    generic = {system.generic_neg, system.generic_pos} - {-1}

    def find(label: str) -> int | None:
        if label_module.is_generic_ion(label):
            return system.labels.index(label) if label in system.labels else None
        try:
            wanted = label_module.canonical(label, system.order)
        except ValueError:
            return None
        return system.labels.index(wanted) if wanted in system.labels else None

    def same(a: str, b: str) -> bool:
        return find(a) is not None and find(a) == find(b)

    forbidden: set[tuple[int, int]] = set()
    overrides: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}

    for raw in Path(path).read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        columns = raw.split()
        if len(columns) % 2:
            raise ValueError(f"--nst line needs an even number of entries: {raw!r}")
        i = find(columns[0])
        if i is None:
            warnings.warn(
                f"--nst: {columns[0]} not in the system, skipping {raw!r}", stacklevel=2
            )
            continue
        if (
            not system.is_monomer(i)
            and len(columns) == 2
            and re.search("clusters", columns[1], re.IGNORECASE)
        ):
            for j in range(n):
                if j not in generic and not system.is_monomer(j):
                    forbidden.add((min(i, j), max(i, j)))
            continue
        j = find(columns[1])
        if j is None:
            warnings.warn(
                f"--nst: {columns[1]} not in the system, skipping {raw!r}", stacklevel=2
            )
            continue
        pair = (min(i, j), max(i, j))
        if len(columns) == 2:
            forbidden.add(pair)
            continue
        if disable_useless_collisions:
            li, lj = system.labels[i], system.labels[j]
            useless = (
                len(columns) == 6
                and (same(columns[3], li) or same(columns[5], li))
                and float(columns[2]) == 1.0
            ) or (
                len(columns) == 4 and system.is_monomer(i) and float(columns[2]) == 2.0
            )
            if useless and (columns[3] == li or columns[3] == lj):
                forbidden.add(pair)
                continue
        products: list[tuple[int, int]] = []
        for col in range(2, len(columns) - 1, 2):
            coefficient, label = columns[col], columns[col + 1]
            if "." in coefficient:
                raise ValueError(
                    "non-integer --nst coefficients do not work in the Fortran "
                    f"version (Perl :2088): {raw!r}"
                )
            count = int(coefficient)
            if count <= 0:
                raise ValueError(f"--nst coefficient must be positive: {raw!r}")
            slot = _OUT_SLOTS.get(label.lower())
            if slot is not None:
                k = system.flux_index[slot]
            else:
                found = find(label)
                if found is None:
                    raise ValueError(f"--nst product {label} is not a valid cluster")
                k = found
            products.append((k, count))
        overrides[pair] = tuple(products)

    return NonStandardReactions(frozenset(forbidden), overrides)


def enumerate_reactions(
    system: AcdcSystem,
    boundary: BoundarySystem,
    prune_useless: bool = True,
    nonstandard: NonStandardReactions | None = None,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> ReactionSet:
    """Enumerate every collision and evaporation in the system.

    Args:
        prune_useless: drop boundary collisions whose products are exactly
            their reactants. The reference does this by default
            (``disable_useless_collisions = 1``, Perl `:243`); such a
            reaction changes nothing but would still contribute equal and
            opposite terms to the right-hand side.
        nonstandard: a parsed ``--nst`` file. Consulted before anything
            else for each pair, as the reference does (Perl :2496-2507).
        fidelity: ``nonstandard_main_coefficient`` decides whether the
            first product's coefficient is honoured or dropped (F20).
    """
    collisions: list[Collision] = []
    evaporations: list[Evaporation] = []

    n = system.n_clusters
    generic = {system.generic_neg, system.generic_pos} - {-1}

    for i in range(n):
        for j in range(i, n):
            if nonstandard is not None:
                if nonstandard.is_forbidden(i, j):
                    continue
                products = nonstandard.override(i, j)
                if products is not None:
                    (main, count), *extras = products
                    if fidelity.nonstandard_main_coefficient == "fortran":
                        count = 1
                    collisions.append(
                        Collision(i, j, ((main, count), *extras), "nonstandard")
                    )
                    continue
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
