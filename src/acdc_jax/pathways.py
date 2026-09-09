"""Flux matrices, growth pathways and the monomer-source back-solve.

The MATLAB side of ACDC computes a *flux matrix* -- gross rates of every
process from species i to species j, including the bookkeeping slots for
sources, external sinks and the outgoing flux -- nets it, and post-processes
it with ``track_fluxes.m`` / ``plotflux.m`` into the growth pathways that
the setup QuickGuide's whole methodology rests on. The Fortran path has none
of this.

Here the matrix is built from the port's explicit reaction list
(:class:`~acdc_jax.rhs.Coefficients`), so the partner in every collision is
known rather than searched for by combining labels. Plain NumPy, vectorised:
a 2000-cluster loop grid has two million collisions.

Semantics follow the generated ``get_fluxes.m`` (Perl `:5548-5745`) and
``plotflux.m`` (`:150-300`):

* a collision ``i + j -> k`` books ``K c_i c_j`` from i to k and from j to k
  (``2 K_half c_i^2`` when i == j -- the matrix is per party); a boundary
  collision books the parties into the ``bound`` node and the pieces out of
  it (`:5599-5612`);
* an evaporation ``k -> i + j`` books ``E c_k`` from k to i and from k to j;
* a first-order loss books ``L c_i`` from i to its slot;
* given sources sit in the ``source`` row.

The pathways are tracked on the NET matrix, ``max(gross - gross^T, 0)``
(`:5723-5729`), which is what the driver hands ``track_fluxes`` (`:10511`).
"""

from __future__ import annotations

import collections
import dataclasses

import numpy as np

from acdc_jax import config
from acdc_jax import labels as label_module
from acdc_jax.rhs import Coefficients, charge_balance_projection
from acdc_jax.system import AcdcSystem


def _projected(coefficients: Coefficients, c: np.ndarray) -> np.ndarray:
    """The state the right-hand side actually sees: under --charge_balance
    the fitted ion is set algebraically before every evaluation (rhs.rhs),
    and the fluxes must be booked with that value, not the raw entry."""
    c = np.asarray(c, dtype=float)
    if coefficients.charge_balance and coefficients.system is not None:
        c = np.asarray(
            charge_balance_projection(
                coefficients.system, c, coefficients.charge_balance
            )
        )
    return c


def _process_rates(coefficients: Coefficients, c: np.ndarray):
    """Per-collision and per-evaporation-channel fluxes, 1/m^3/s."""
    ci = np.asarray(coefficients.collision_i)
    cj = np.asarray(coefficients.collision_j)
    phi = np.asarray(coefficients.collision_rate) * c[ci] * c[cj]
    ek = np.asarray(coefficients.evaporation_k)
    erate = (
        np.asarray(coefficients.evaporation_rate) * c[ek] if ek.size else np.zeros(0)
    )
    return ci, cj, phi, ek, erate


def gross_flux_matrix(
    system: AcdcSystem, coefficients: Coefficients, c: np.ndarray
) -> np.ndarray:
    """Gross process rates from species i to species j, 1/m^3/s.

    Shape ``(neq, neq)`` over clusters and flux slots. Per party: a
    collision is booked from each collider, so a self-collision appears
    twice and the column sum of a cluster is not its formation rate; use
    :func:`net_flux_matrix` for flows and the right-hand side for rates.
    """
    c = _projected(coefficients, c)
    neq = system.n_equations
    gross = np.zeros((neq, neq))
    ci, cj, phi, ek, erate = _process_rates(coefficients, c)

    product = np.asarray(coefficients.product_index)
    owner = np.asarray(coefficients.product_owner)
    multiplicity = np.asarray(coefficients.product_multiplicity)
    bound = system.flux_index["bound"]

    # A collision is direct when it has one product entry with multiplicity
    # one; otherwise the product broke up at the boundary.
    rows_per_collision = np.bincount(owner, minlength=len(ci))
    direct_row = (rows_per_collision[owner] == 1) & (multiplicity == 1)
    destination = np.full(len(ci), bound)
    destination[owner[direct_row]] = product[direct_row]

    np.add.at(gross, (ci, destination), np.where(ci == cj, 2.0, 1.0) * phi)
    other = ci != cj
    np.add.at(gross, (cj[other], destination[other]), phi[other])
    pieces = ~direct_row
    np.add.at(
        gross,
        (np.full(int(pieces.sum()), bound), product[pieces]),
        phi[owner[pieces]] * multiplicity[pieces],
    )

    if ek.size:
        ei = np.asarray(coefficients.evaporation_i)
        ej = np.asarray(coefficients.evaporation_j)
        np.add.at(gross, (ek, ei), np.where(ei == ej, 2.0, 1.0) * erate)
        asym = ei != ej
        np.add.at(gross, (ek[asym], ej[asym]), erate[asym])

    n = system.n_clusters
    for vector, slot in zip(coefficients.losses, coefficients.loss_slots, strict=True):
        gross[:n, slot] += np.asarray(vector) * c[:n]

    gross[system.flux_index["source"], :] += np.asarray(coefficients.source)
    return gross


def net_flux_matrix(gross: np.ndarray) -> np.ndarray:
    """``max(gross - gross^T, 0)``: the net flow between every pair."""
    net = gross - gross.T
    return np.where(net > 0, net, 0.0)


def monomer_sources(system: AcdcSystem, gross: np.ndarray) -> dict[str, float]:
    """The source each monomer would need to sustain this state, 1/m^3/s.

    The MATLAB driver's ``Sources_out`` (Perl `:5731-5741`): for a neutral
    monomer, everything it flows into minus everything flowing into it from
    the processes; for a generic charger ion, everything it flows into (its
    only source is the ion production). The given-source row is excluded,
    as upstream works on ``coll_evap_2d`` which never carries it -- so the
    answer is the TOTAL source needed, whatever was already supplied.
    """
    n = system.n_clusters
    source_row = system.flux_index["source"]
    out: dict[str, float] = {}
    for i in range(n):
        label = system.labels[i]
        if i in (system.generic_neg, system.generic_pos):
            out[label] = float(gross[i, :].sum())
        elif system.charges[i] == 0 and system.is_monomer(i):
            inflow = gross[:, i].sum() - gross[source_row, i]
            out[label] = float(gross[i, :].sum() - inflow)
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
    """Significant net fluxes into every tracked cluster: source -> cluster."""
    main_route: tuple[str, ...]
    """Greedy back-walk from the largest exit, smallest cluster first, ending
    at the composition outside the set."""
    total_out: float
    """Total outgoing flux, all charges, 1/m^3/s."""


def _significant(items, crit: float):
    """``get_significant``: keep entries at or above ``crit`` of the total,
    aggregated by key, largest first."""
    total = sum(v for _, v in items)
    merged: dict = {}
    for key, v in items:
        merged[key] = merged.get(key, 0.0) + v
    kept = [(k, v) for k, v in merged.items() if total > 0 and v >= crit * total]
    return sorted(kept, key=lambda kv: -kv[1])


def track_pathways(
    system: AcdcSystem,
    coefficients: Coefficients,
    c: np.ndarray,
    charge: int = 0,
    crit_out: float = config.PATHWAY_CRIT_OUT,
    crit_clust: float = config.PATHWAY_CRIT_CLUST,
) -> Pathways:
    """``track_fluxes.m`` on the port's reaction list.

    1. Exit channels (``plotflux_out.m``): collisions whose product leaves
       the system, from the gross out-fluxes -- one exit per collision, as
       the Fortran ``formation`` and :func:`~acdc_jax.rhs.formation_rate`
       count them. A channel counts for ``charge`` if a collider has that
       charge -- for neutral pathways only when the neutral collider is at
       least as large as the ion (`track_fluxes.m:87-101`). The pathway is
       attributed to the GROWING cluster (the collider of the wanted charge,
       else the larger one) and ends at the product composition outside the
       set. Channels below ``crit_out`` of the total are folded away.
    2. Backward breadth-first traversal (`:134-194`) on the NET flux matrix:
       for every tracked cluster of the wanted charge, the rows flowing into
       it (``plotflux.m`` direction ``'to'``): a larger cluster is a source
       in its own right (evaporation into the target); a smaller one is a
       collision party, paired with the partner that completes the target
       and attributed to the growing collider (`:221-250`), the partner then
       taken; flows from the generic charger ions are discarded (`:153-158`,
       the corresponding recombination is not a net flux); the ``source``
       and ``bound`` rows are sources by name. Inflows below ``crit_clust``
       of the total are folded away; new cluster sources are queued once.
       Generic ions are never tracked (``check_cluster`` is false for them).
    3. The main route (`:196-207`): from the largest exit's start, follow the
       largest inflow while it comes from a cluster of the same charge that
       has not been visited.
    """
    c = _projected(coefficients, c)
    n = system.n_clusters
    labels = system.labels
    index = {label: i for i, label in enumerate(labels)}
    generic = {system.generic_neg, system.generic_pos} - {-1}
    charges = np.asarray(system.charges)
    compositions = np.asarray(system.compositions)
    pseudo = np.asarray(system.molecule_is_pseudo, dtype=bool)
    mols = compositions[:, ~pseudo].sum(axis=1)
    by_composition = {
        tuple(int(x) for x in row): i for i, row in enumerate(compositions)
    }
    slot_names = {slot: name for name, slot in system.flux_index.items()}

    def larger(i: int, j: int, prefer_charge: int) -> int:
        same = [k for k in (i, j) if charges[k] == prefer_charge]
        if len(same) == 1:
            return same[0]
        return i if mols[i] >= mols[j] else j

    def combined_label(i: int, j: int) -> str:
        summed = compositions[i] + compositions[j]
        return label_module.format_label(
            {m: int(k) for m, k in zip(system.order, summed, strict=True)},
            system.order,
        )

    # 1. exit channels
    ci, cj, phi, _, _ = _process_rates(coefficients, c)
    product = np.asarray(coefficients.product_index)
    owner = np.asarray(coefficients.product_owner)
    out_slots = np.array(
        [system.flux_index[s] for s in ("out_neu", "out_neg", "out_pos")]
    )
    out_rows = np.isin(product, out_slots)
    out_flux = phi[owner[out_rows]]
    total_out = float(out_flux.sum())
    exits_raw = []
    for o, f in zip(owner[out_rows], out_flux, strict=True):
        i, j = int(ci[o]), int(cj[o])
        pair_charges = (int(charges[i]), int(charges[j]))
        if charge not in pair_charges:
            continue
        if charge == 0 and pair_charges[0] != pair_charges[1]:
            neutral, ion = (i, j) if pair_charges[0] == 0 else (j, i)
            if mols[neutral] < mols[ion]:
                continue
            start = neutral
        else:
            start = larger(i, j, charge)
        exits_raw.append(((labels[start], combined_label(i, j)), float(f)))
    exits = tuple(Edge(s, e, v) for (s, e), v in _significant(exits_raw, crit_out))
    if not exits:
        return Pathways((), (), (), total_out)

    # 2. backward traversal on the net matrix
    net = net_flux_matrix(gross_flux_matrix(system, coefficients, c))

    def inflows(target: int) -> list[tuple[str, float]]:
        items: list[tuple[str, float]] = []
        taken: set[int] = set()
        target_comp = compositions[target]
        for s in np.flatnonzero(net[:, target]):
            s = int(s)
            if s in taken:
                continue
            value = float(net[s, target])
            if s >= n:
                items.append((slot_names[s], value))
                continue
            if s in generic:
                taken.add(s)
                continue
            if mols[s] > mols[target]:
                items.append((labels[s], value))
                continue
            partner = by_composition.get(
                tuple(int(x) for x in target_comp - compositions[s])
            )
            if partner is not None and partner != s and partner not in taken:
                taken.add(partner)
                start = larger(s, partner, int(charges[target]))
                items.append((labels[start], value))
            elif partner == s:
                items.append((labels[s], value / 2.0))
            else:
                items.append((labels[s], value))
        return items

    edges: list[Edge] = []
    tracked: set[str] = set()
    queue = collections.deque(e.start for e in exits)
    queued = set(queue)
    while queue:
        label = queue.popleft()
        if label in tracked or label not in index or index[label] in generic:
            continue
        tracked.add(label)
        target = index[label]
        if charges[target] != charge:
            continue
        for start, value in _significant(inflows(target), crit_clust):
            edges.append(Edge(start, label, value))
            if start in index and start not in tracked and start not in queued:
                queue.append(start)
                queued.add(start)

    # 3. main route
    route = [exits[0].start]
    visited = {exits[0].start}
    while True:
        into = [e for e in edges if e.end == route[0]]
        best = max(into, key=lambda e: e.value, default=None)
        if (
            best is None
            or best.start not in index
            or index[best.start] in generic
            or best.start in visited
            or charges[index[best.start]] != charge
        ):
            break
        route.insert(0, best.start)
        visited.add(best.start)
    route.append(exits[0].end)

    return Pathways(exits, tuple(edges), tuple(route), total_out)


__all__ = [
    "Edge",
    "Pathways",
    "gross_flux_matrix",
    "monomer_sources",
    "net_flux_matrix",
    "track_pathways",
]
