"""Flux matrices, growth pathways and the monomer-source back-solve.

The MATLAB side of ACDC computes a *flux matrix* -- gross rates of every
process from species i to species j, including the bookkeeping slots for
sources, external sinks and the outgoing flux -- and post-processes it with
``track_fluxes.m`` / ``plotflux.m`` into the growth pathways that the setup
QuickGuide's whole methodology rests on. The Fortran path has none of this.

Here the matrix is built from the port's explicit reaction list
(:class:`~acdc_jax.rhs.Coefficients`), so the partner in every collision is
known rather than searched for by combining labels. Plain NumPy: runs once
per solved state.

Semantics follow the generated ``get_fluxes.m`` (Perl `:5548-5745`) and
``plotflux.m`` (`:150-300`):

* a collision ``i + j -> k`` books ``K c_i c_j`` from i to k and from j to k
  (``2 K_half c_i^2`` when i == j -- the matrix is per party);
* an evaporation ``k -> i + j`` books ``E c_k`` from k to i and from k to j;
* a first-order loss books ``L c_i`` from i to its slot;
* sources sit in the ``source`` row.

The net flux is ``gross - gross^T``, kept where positive.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from acdc_jax import labels as label_module
from acdc_jax.reactions import ReactionSet
from acdc_jax.rhs import Coefficients
from acdc_jax.system import AcdcSystem


def gross_flux_matrix(
    system: AcdcSystem, coefficients: Coefficients, c: np.ndarray
) -> np.ndarray:
    """Gross process rates from species i to species j, 1/m^3/s.

    Shape ``(neq, neq)`` over clusters and flux slots. The right-hand side
    is recoverable as column sum minus row sum, which is what the tests
    check it against.
    """
    c = np.asarray(c, dtype=float)
    neq = system.n_equations
    gross = np.zeros((neq, neq))

    ci = np.asarray(coefficients.collision_i)
    cj = np.asarray(coefficients.collision_j)
    phi = np.asarray(coefficients.collision_rate) * c[ci] * c[cj]
    product = np.asarray(coefficients.product_index)
    owner = np.asarray(coefficients.product_owner)
    multiplicity = np.asarray(coefficients.product_multiplicity)
    bound = system.flux_index["bound"]
    for o in range(len(ci)):
        i, j, f = int(ci[o]), int(cj[o]), float(phi[o])
        rows = np.flatnonzero(owner == o)
        direct = len(rows) == 1 and multiplicity[rows[0]] == 1
        if direct:
            destination = int(product[rows[0]])
        else:
            # A boundary collision: the product breaks up. Upstream books the
            # colliders into a synthetic 'bound' node and the pieces out of it
            # (Perl :5599-5612), so a party is consumed once whatever the
            # number of pieces.
            destination = bound
            for r in rows:
                gross[bound, int(product[r])] += f * float(multiplicity[r])
        if i == j:
            gross[i, destination] += 2.0 * f
        else:
            gross[i, destination] += f
            gross[j, destination] += f

    ek = np.asarray(coefficients.evaporation_k)
    ei = np.asarray(coefficients.evaporation_i)
    ej = np.asarray(coefficients.evaporation_j)
    if ek.size:
        rate = np.asarray(coefficients.evaporation_rate) * c[ek]
        for k, i, j, r in zip(ek, ei, ej, rate, strict=True):
            if i == j:
                gross[k, i] += 2.0 * r
            else:
                gross[k, i] += r
                gross[k, j] += r

    n = system.n_clusters
    for vector, slot in zip(coefficients.losses, coefficients.loss_slots, strict=True):
        gross[:n, slot] += np.asarray(vector) * c[:n]

    source = np.asarray(coefficients.source)
    gross[system.flux_index["source"], :] += source
    return gross


def net_flux_matrix(gross: np.ndarray) -> np.ndarray:
    """``max(gross - gross^T, 0)``: the net flow between every pair."""
    net = gross - gross.T
    return np.where(net > 0, net, 0.0)


def monomer_sources(system: AcdcSystem, gross: np.ndarray) -> dict[str, float]:
    """The source each monomer would need to sustain this state, 1/m^3/s.

    The MATLAB driver's ``Sources_out`` (Perl `:5731-5741`): for a neutral
    monomer, everything it flows into minus everything flowing into it; for
    a generic charger ion, everything it flows into (its only source is the
    ion production). Meaningful at steady state with the monomers held
    constant, where it inverts the constant-concentration assumption into
    the emission rate that would produce it.
    """
    n = system.n_clusters
    out: dict[str, float] = {}
    for i in range(n):
        label = system.labels[i]
        if i in (system.generic_neg, system.generic_pos):
            out[label] = float(gross[i, :].sum())
        elif system.charges[i] == 0 and system.is_monomer(i):
            out[label] = float(gross[i, :].sum() - gross[:, i].sum())
    return out


# ---------------------------------------------------------------------------
# Pathways
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Edge:
    start: str
    end: str
    value: float
    """1/m^3/s."""


@dataclasses.dataclass(frozen=True)
class Pathways:
    exits: tuple[Edge, ...]
    """Significant collisions out of the system: growing cluster -> product
    composition (outside the set)."""
    edges: tuple[Edge, ...]
    """Significant fluxes into every tracked cluster: source -> cluster."""
    main_route: tuple[str, ...]
    """Greedy back-walk from the largest exit: monomer(s) ... -> boundary."""
    charge: int
    total_out: float
    """Total outgoing flux, all charges, 1/m^3/s."""


def _molecule_count(system: AcdcSystem, i: int) -> int:
    counts = system.compositions[i]
    return sum(n for j, n in enumerate(counts) if not system.molecule_is_pseudo[j])


def _larger(system: AcdcSystem, i: int, j: int, prefer_charge: int | None) -> int:
    """``max_mols`` with the ``lsame_ch`` preference (plotflux.m :221-250):
    the collider carrying the target's charge if exactly one does, else the
    one with more molecules (the first on a tie)."""
    if prefer_charge is not None:
        same = [k for k in (i, j) if system.charges[k] == prefer_charge]
        if len(same) == 1:
            return same[0]
    return i if _molecule_count(system, i) >= _molecule_count(system, j) else j


def _significant(
    items: list[tuple[str, float]], crit: float
) -> list[tuple[str, float]]:
    """``get_significant``: keep entries at or above ``crit`` of the total,
    aggregated by name, largest first."""
    total = sum(v for _, v in items)
    merged: dict[str, float] = {}
    for name, v in items:
        merged[name] = merged.get(name, 0.0) + v
    kept = [(k, v) for k, v in merged.items() if total > 0 and v >= crit * total]
    return sorted(kept, key=lambda kv: -kv[1])


def track_pathways(
    system: AcdcSystem,
    reactions: ReactionSet,
    coefficients: Coefficients,
    c: np.ndarray,
    charge: int = 0,
    crit_out: float = 0.05,
    crit_clust: float = 0.05,
    growth_only: bool = True,
) -> Pathways:
    """``track_fluxes.m`` on the port's reaction list.

    1. Exit channels: collisions whose product leaves the system. A channel
       counts for ``charge`` if a collider has that charge -- for neutral
       pathways only when the neutral collider is at least as large as the
       ion (`:87-101`). The pathway is attributed to the GROWING cluster
       (the larger collider, or the one of the wanted charge), and the
       channel's end is the product composition outside the set. Channels
       below ``crit_out`` of the total are dropped.
    2. Backward breadth-first traversal (`:134-194`): for every tracked
       cluster of the wanted charge, the fluxes INTO it -- collisions
       forming it (attributed as above), evaporations of a larger cluster
       into it, and the source term -- keeping those at or above
       ``crit_clust`` of the total inflow; new cluster sources are queued
       once.
    3. The main route (`:196-207`): from the largest exit's start, repeatedly
       follow the largest inflow while it comes from a cluster of the same
       charge that has not been visited.

    Boundary collisions (the product breaks up, Phase 2) are attributed to
    the growing collider directly; MATLAB routes them through a synthetic
    'bound' node that it then declines to track.

    ``growth_only`` (a port choice, default on) makes the main-route
    back-walk consider only inflows from SMALLER clusters, so the route is a
    growth sequence. MATLAB's argmax also accepts an evaporation from a
    larger cluster, which can put a shrinking step into what is presented
    as a growth route; ``growth_only=False`` reproduces that.
    """
    c = np.asarray(c, dtype=float)
    ci = np.asarray(coefficients.collision_i)
    cj = np.asarray(coefficients.collision_j)
    phi = np.asarray(coefficients.collision_rate) * c[ci] * c[cj]
    product = np.asarray(coefficients.product_index)
    owner = np.asarray(coefficients.product_owner)
    multiplicity = np.asarray(coefficients.product_multiplicity)
    out_slots = {system.flux_index[s] for s in ("out_neu", "out_neg", "out_pos")}
    compositions = np.asarray(system.compositions)

    def combined_label(i: int, j: int) -> str:
        summed = compositions[i] + compositions[j]
        return label_module.format_label(
            {m: int(k) for m, k in zip(system.order, summed, strict=True)}, system.order
        )

    # 1. exit channels
    exits_raw: list[tuple[tuple[str, str], float]] = []
    total_out = 0.0
    for p, o, m in zip(product, owner, multiplicity, strict=True):
        if p not in out_slots:
            continue
        i, j, f = int(ci[o]), int(cj[o]), float(phi[o] * m)
        total_out += f
        charges = (system.charges[i], system.charges[j])
        if charge not in charges:
            continue
        if charge == 0 and charges[0] != charges[1]:
            neutral, ion = (i, j) if charges[0] == 0 else (j, i)
            if _molecule_count(system, neutral) < _molecule_count(system, ion):
                continue
            start = neutral
        else:
            start = _larger(system, i, j, charge)
        exits_raw.append(((system.labels[start], combined_label(i, j)), f))

    keyed = [(f"{s}|{e}", v) for (s, e), v in exits_raw]
    exits = tuple(
        Edge(*name.split("|"), value) for name, value in _significant(keyed, crit_out)
    )
    if not exits:
        return Pathways((), (), (), charge, total_out)

    # 2. backward traversal
    ek = np.asarray(coefficients.evaporation_k)
    ei = np.asarray(coefficients.evaporation_i)
    ej = np.asarray(coefficients.evaporation_j)
    erate = (
        np.asarray(coefficients.evaporation_rate) * c[ek] if ek.size else np.zeros(0)
    )
    source = np.asarray(coefficients.source)

    def inflows(target: int) -> list[tuple[str, float]]:
        items: list[tuple[str, float]] = []
        for p, o, m in zip(product, owner, multiplicity, strict=True):
            if p != target:
                continue
            i, j = int(ci[o]), int(cj[o])
            start = _larger(system, i, j, system.charges[target])
            items.append((system.labels[start], float(phi[o] * m)))
        for k, i, j, r in zip(ek, ei, ej, erate, strict=True):
            if target in (i, j):
                items.append((system.labels[k], float(r) * (2.0 if i == j else 1.0)))
        if source[target] > 0:
            items.append(("source", float(source[target])))
        return items

    edges: list[Edge] = []
    tracked: set[str] = set()
    queue = [e.start for e in exits]
    while queue:
        label = queue.pop(0)
        if label in tracked or label not in system.labels:
            continue
        tracked.add(label)
        target = system.labels.index(label)
        if system.charges[target] != charge:
            continue
        for start, value in _significant(inflows(target), crit_clust):
            edges.append(Edge(start, label, value))
            if start in system.labels and start not in tracked and start not in queue:
                queue.append(start)

    # 3. main route
    route = [exits[0].start]
    visited = {exits[0].start}
    while True:
        current = route[0]
        into = [e for e in edges if e.end == current]
        if growth_only:
            size = _molecule_count(system, system.labels.index(current))
            into = [
                e
                for e in into
                if e.start not in system.labels
                or _molecule_count(system, system.labels.index(e.start)) < size
            ]
        if not into:
            break
        best = max(into, key=lambda e: e.value).start
        if best not in system.labels or best in visited:
            break
        if system.charges[system.labels.index(best)] != charge:
            break
        route.insert(0, best)
        visited.add(best)
    route.append(exits[0].end)

    return Pathways(exits, tuple(edges), tuple(route), charge, total_out)


__all__ = [
    "Edge",
    "Pathways",
    "gross_flux_matrix",
    "monomer_sources",
    "net_flux_matrix",
    "track_pathways",
]
