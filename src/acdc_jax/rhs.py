"""The right-hand side, assembled from the reaction list and the rates.

The reference's ``feval`` is already data-driven -- 58 lines looping over
integer index arrays. This module reproduces the same structure with dense
tensors and two ``einsum`` calls, which is what JAX wants and what makes the
Jacobian exact and free.

At 54 clusters the dense coefficient tensors are about 1.5 MB, so density is
the cheap option. A sparse path would be needed for a much larger set.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from acdc_jax import config, losses, rates
from acdc_jax import hydrates as hydrate_module
from acdc_jax.hydrates import HydrateModel
from acdc_jax.reactions import ReactionSet
from acdc_jax.system import AcdcSystem


@dataclasses.dataclass(frozen=True)
class Coefficients:
    """Assembled rate tensors, in the reference's own layout.

    ``coef_quad[i, j, k]`` is the rate of ``i + j -> k`` and
    ``coef_lin[a, b, k]`` the rate of ``k -> a + b``. Both carry the
    half-factors the reference applies to self-collisions and symmetric
    evaporations.
    """

    n_clusters: int
    coef_quad: jnp.ndarray | None
    """(nclust, nclust, neq). None when assembled without the dense tensors
    (loop mode: n^2 * neq does not fit)."""
    coef_lin: jnp.ndarray | None
    """(neq, neq, nclust), or None."""
    multiplicity: np.ndarray | None
    """(nclust, nclust, neq) int, or None.

    How many of product ``k`` a single ``i + j`` collision yields. Almost
    always one; a boundary collision that strips two identical monomers off
    gives two. The reference keeps this OUT of `coef_quad` and applies it at
    right-hand-side time from `n_quad_form_extra`, so it is separate here
    too -- otherwise the assembled tensor would not match `get_rate_coefs`.
    """
    source: jnp.ndarray
    """(neq,) zeroth-order source terms, 1/m^3/s."""
    isconst: np.ndarray
    """(neq,) bool: species whose derivative is forced to zero."""

    # Flat reaction arrays, one entry per enumerated reaction. These drive
    # the right-hand side; the dense tensors above exist so the assembly can
    # be compared against the reference's get_rate_coefs.
    collision_i: np.ndarray
    collision_j: np.ndarray
    collision_rate: jnp.ndarray
    """Already halved for self-collisions."""
    product_index: np.ndarray
    """One entry per (collision, product) pair."""
    product_owner: np.ndarray
    """Which collision each product entry belongs to."""
    product_multiplicity: jnp.ndarray
    evaporation_k: np.ndarray
    evaporation_i: np.ndarray
    evaporation_j: np.ndarray
    evaporation_rate: jnp.ndarray
    losses: tuple[jnp.ndarray, ...]
    """First-order external losses, each (nclust,) in 1/s, one per slot in
    ``loss_slots`` (coagulation, wall, dilution -- whichever are on)."""
    loss_slots: tuple[int, ...]
    charge_balance: int = 0
    """--charge_balance mode; see charge_balance_projection."""
    system: AcdcSystem | None = None
    """Needed only when charge_balance != 0, for the projection inside rhs()."""

    @property
    def sink(self) -> jnp.ndarray:
        """Total first-order external loss per cluster, 1/s."""
        return sum(self.losses)


def assemble(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    temperature,
    cs_ref,
    ipr_neg: float = 0.0,
    ipr_pos: float = 0.0,
    constant_vapours: tuple[str, ...] = ("1A", "1N"),
    fcs: float = 1.0,
    charge_balance: int = 0,
    fidelity: config.FidelityConfig = config.DEFAULT,
    hydrates: HydrateModel | None = None,
    loss_settings: losses.LossSettings | None = None,
    dense: bool = True,
) -> Coefficients:
    """Build the coefficient tensors for one set of ambient conditions.

    Args:
        constant_vapours: species pinned to their input concentration. Under
            the steady-state assumption the reference pins the neutral
            vapour monomers (``acdc_simulation_setup.f90:84``); the ionic
            monomers stay free so they can respond to ion production.
        fidelity: forwarded to every rate formula.
        hydrates: a :func:`~acdc_jax.hydrates.build_hydrate_model` result.
            When given, K, E and every loss are the hydrate-averaged ones
            (``--rh``); the model must have been built from these same
            reactions.
        loss_settings: which first-order losses to include. Default: the
            exp_loss coagulation sink only, as in the shipped example.
        dense: also build ``coef_quad``/``coef_lin``/``multiplicity``. They
            exist to be compared against the reference's ``get_rate_coefs``;
            nothing in the right-hand side reads them. Building them costs
            one scatter per reaction, which dominates tracing when the
            assembly is inside a ``jit`` or ``vmap``, so a batched caller
            passes False.
    """
    n, neq = system.n_clusters, system.n_equations
    if loss_settings is None:
        loss_settings = losses.LossSettings()

    if hydrates is None:
        collision = rates.collision_coefficients(inputs, temperature, fidelity)
        loss_vectors = losses.first_order_losses(
            loss_settings, inputs, temperature, cs_ref, fcs, fidelity
        )
    else:
        collision = hydrate_module.collision_coefficients(
            hydrates, temperature, fidelity
        )
        per_species = losses.first_order_losses(
            loss_settings, hydrates.expanded, temperature, cs_ref, fcs, fidelity
        )
        loss_vectors = {
            name: hydrate_module.average_vector(hydrates, vector, temperature, fidelity)
            for name, vector in per_species.items()
        }
    loss_slots = tuple(system.flux_index[name] for name in loss_vectors)
    loss_list = tuple(loss_vectors.values())

    quad = jnp.zeros((n, n, neq)) if dense else None
    mult = np.ones((n, n, neq), dtype=np.int32) if dense else None

    # Collisions. The reference enumerates the upper triangle and writes
    # both orderings, so a self-collision -- listed once but reachable two
    # ways -- carries a half.
    #
    # The rate is SET once per (i, j, k) and the multiplicities are summed,
    # never accumulated into the rate. A boundary collision can name the
    # same species twice: `2N1P + neg` recombines to `2N`, which is outside
    # the set and strips back to `1N + 1 N`, so 1N is both the surviving
    # cluster and the stripped monomer. Adding the rate twice there makes
    # that channel 2x too fast -- four entries in the bundled system, which
    # is exactly what the get_rate_coefs comparison caught.
    if dense:
        for c in reactions.collisions:
            rate = collision[c.i, c.j]
            if c.is_self_collision:
                rate = 0.5 * rate

            totals: dict[int, int] = {}
            for product, multiplicity in c.products:
                totals[product] = totals.get(product, 0) + multiplicity

            for product, multiplicity in totals.items():
                quad = quad.at[c.i, c.j, product].set(rate)
                mult[c.i, c.j, product] = multiplicity
                if c.i != c.j:
                    quad = quad.at[c.j, c.i, product].set(rate)
                    mult[c.j, c.i, product] = multiplicity

    lin = jnp.zeros((neq, neq, n)) if dense else None

    if reactions.evaporations:
        parents = np.array([e.k for e in reactions.evaporations])
        di = np.array([e.i for e in reactions.evaporations])
        dj = np.array([e.j for e in reactions.evaporations])
        if hydrates is None:
            evaporation = rates.evaporation_for_pairs(
                inputs, collision, parents, di, dj, temperature, fidelity=fidelity
            )
        else:
            same_channels = (
                np.array_equal(hydrates.channel_parent, parents)
                and np.array_equal(hydrates.channel_i, di)
                and np.array_equal(hydrates.channel_j, dj)
            )
            if not same_channels:
                raise ValueError("hydrate model was built from different reactions")
            evaporation = hydrate_module.evaporation(
                hydrates, temperature, fidelity=fidelity
            )
        if dense:
            lin = lin.at[di, dj, parents].add(evaporation)
            asymmetric = di != dj
            lin = lin.at[dj[asymmetric], di[asymmetric], parents[asymmetric]].add(
                evaporation[asymmetric]
            )

    # External first-order losses, each booked to its own flux slot
    # (coef_lin(56,56,k) coagulation, (57,57,k) wall, (58,58,k) dilution).
    if dense:
        for vector, slot in zip(loss_list, loss_slots, strict=True):
            lin = lin.at[slot, slot, :].add(vector)

    source = jnp.zeros(neq)
    if system.generic_neg >= 0:
        source = source.at[system.generic_neg].set(ipr_neg)
    if system.generic_pos >= 0:
        source = source.at[system.generic_pos].set(ipr_pos)

    isconst = np.zeros(neq, dtype=bool)
    for label in constant_vapours:
        if label in system.labels:
            isconst[system.labels.index(label)] = True

    # Under --charge_balance the fitted ion is algebraic, not integrated:
    # the generator marks it isconst and drops its source (cb1.f90:227-230).
    if charge_balance > 0 and system.generic_pos >= 0:
        isconst[system.generic_pos] = True
        source = source.at[system.generic_pos].set(0.0)
    elif charge_balance < 0 and system.generic_neg >= 0:
        isconst[system.generic_neg] = True
        source = source.at[system.generic_neg].set(0.0)

    coll_i, coll_j, coll_rate = [], [], []
    prod_index, prod_owner, prod_mult = [], [], []
    for slot, c in enumerate(reactions.collisions):
        r = collision[c.i, c.j]
        coll_i.append(c.i)
        coll_j.append(c.j)
        coll_rate.append(0.5 * r if c.is_self_collision else r)
        totals: dict[int, int] = {}
        for product, multiplicity in c.products:
            totals[product] = totals.get(product, 0) + multiplicity
        for product, multiplicity in totals.items():
            prod_index.append(product)
            prod_owner.append(slot)
            prod_mult.append(multiplicity)

    if reactions.evaporations:
        evap_k, evap_i, evap_j = parents, di, dj
        evap_rate = evaporation
    else:
        evap_k = evap_i = evap_j = np.zeros(0, dtype=int)
        evap_rate = jnp.zeros(0)

    return Coefficients(
        n_clusters=n,
        coef_quad=quad,
        coef_lin=lin,
        multiplicity=mult,
        source=source,
        isconst=isconst,
        collision_i=np.array(coll_i),
        collision_j=np.array(coll_j),
        collision_rate=jnp.array(coll_rate),
        product_index=np.array(prod_index),
        product_owner=np.array(prod_owner),
        product_multiplicity=jnp.array(prod_mult, dtype=float),
        evaporation_k=evap_k,
        evaporation_i=evap_i,
        evaporation_j=evap_j,
        evaporation_rate=evap_rate,
        losses=loss_list,
        loss_slots=loss_slots,
        charge_balance=charge_balance,
        system=system if charge_balance else None,
    )


def charge_balance_projection(
    system: AcdcSystem, c: jnp.ndarray, mode: int
) -> jnp.ndarray:
    """Set one generic ion to balance the net charge of everything else.

    Reproduces the block the generator emits at the top of ``feval`` AND
    ``formation`` under ``--charge_balance`` (fixture cb1.f90:93-99)::

        excess = c(neg) + sum(c(negative clusters)) - sum(c(positive clusters))
        if excess > 0:  c(pos) = excess
        else:           c(neg) = -(sum(neg clusters) - sum(pos clusters)); c(pos) = 0

    for ``mode = +1``; ``mode = -1`` is the mirror. ``mode = 0`` is the
    identity. Note the else-branch OVERWRITES the sourced ion too.

    Upstream mutates ``c`` in place inside the RHS -- a state change through
    the ODE solver. Here it is a pure projection applied to the state before
    the RHS and before J, which is the same computation without the side
    effect. Branch-free via jnp.where so it stays traceable.
    """
    c = jnp.asarray(c)
    if mode == 0:
        return c
    charges = jnp.asarray(system.charges)
    n = system.n_clusters
    neg_i, pos_i = system.generic_neg, system.generic_pos
    # Charged CLUSTERS only -- the generic ions are handled explicitly.
    is_cluster = jnp.arange(n) < n
    is_cluster = is_cluster.at[neg_i].set(False).at[pos_i].set(False)
    neg_sum = jnp.sum(jnp.where(is_cluster & (charges < 0), c[:n], 0.0))
    pos_sum = jnp.sum(jnp.where(is_cluster & (charges > 0), c[:n], 0.0))

    if mode > 0:
        excess = c[neg_i] + neg_sum - pos_sum
        c_pos = jnp.where(excess > 0, excess, 0.0)
        c_neg = jnp.where(excess > 0, c[neg_i], -(neg_sum - pos_sum))
    else:
        excess = c[pos_i] + pos_sum - neg_sum
        c_neg = jnp.where(excess > 0, excess, 0.0)
        c_pos = jnp.where(excess > 0, c[pos_i], -(pos_sum - neg_sum))
    return c.at[neg_i].set(c_neg).at[pos_i].set(c_pos)


def rhs(coefficients: Coefficients, c: jnp.ndarray) -> jnp.ndarray:
    """dc/dt for the cluster birth-death system, 1/m^3/s.

    Written as explicit scatter-adds over the reaction list rather than as
    reductions over the dense tensors. Reducing `coef_quad` along its
    product axis to get the loss term is wrong twice over: a boundary
    collision that yields several products would be counted once per
    product, and an asymmetric evaporation is stored at both (i,j,k) and
    (j,i,k) so it would be counted twice. Both errors are invisible in the
    coefficient tensors -- which match the reference exactly -- and only
    show up in dc/dt.

    Conventions, matching ``feval``:

    - A collision flux is ``rate * c_i * c_j`` with ``rate`` already halved
      for a self-collision, and is subtracted from BOTH reactants. For
      ``i == j`` that subtracts it twice, giving the correct ``K c^2`` loss
      of two molecules per collision.
    - An evaporation flux is ``rate * c_k``, added to BOTH daughters, so a
      symmetric channel delivers two.
    """
    if coefficients.charge_balance:
        c = charge_balance_projection(
            coefficients.system, c, coefficients.charge_balance
        )

    quad_i = coefficients.collision_i
    quad_j = coefficients.collision_j
    rate = coefficients.collision_rate

    f = jnp.zeros(c.shape[0])

    # Collisions.
    flux = rate * c[quad_i] * c[quad_j]
    f = f.at[quad_i].add(-flux)
    f = f.at[quad_j].add(-flux)
    f = f.at[coefficients.product_index].add(
        coefficients.product_multiplicity * flux[coefficients.product_owner]
    )

    # Evaporations.
    if coefficients.evaporation_rate.size:
        evaporation = coefficients.evaporation_rate * c[coefficients.evaporation_k]
        f = f.at[coefficients.evaporation_k].add(-evaporation)
        f = f.at[coefficients.evaporation_i].add(evaporation)
        f = f.at[coefficients.evaporation_j].add(evaporation)

    # External losses: first-order, each booked to its own flux slot.
    n = coefficients.n_clusters
    for vector, slot in zip(coefficients.losses, coefficients.loss_slots, strict=True):
        lost = vector * c[:n]
        f = f.at[:n].add(-lost)
        f = f.at[slot].add(jnp.sum(lost))

    f = f + coefficients.source

    # Pinned species do not evolve. Masked rather than branched.
    return jnp.where(jnp.asarray(coefficients.isconst), 0.0, f)


def formation_rate(
    system: AcdcSystem, coefficients: Coefficients, c: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """Particle formation rate J and its attribution, 1/m^3/s.

    J is the flux into the three ``out_*`` counters, computed from the
    enumerated collision list rather than by reducing `coef_quad`. The dense
    tensor stores both orderings of every pair, so summing it whole counts
    each unordered collision twice -- which is exactly the factor of two
    that the golden comparison caught.

    The reference computes a per-cluster and per-charge-pair breakdown and
    then discards both (``driver_acdc_J.f90:359-363``). They cost nothing
    once the flux exists, so they are returned.

    Under ``--charge_balance`` the fitted generic ion is pinned, so its
    stored entry is not the balanced value; the emitted code projects at
    the top of BOTH ``feval`` and ``formation`` (fixture
    ``acdc_equations_cb1.f90``), so this does too. Without it a grow-out
    collision involving that ion gives a J the Fortran would not.
    """
    if coefficients.charge_balance:
        c = charge_balance_projection(
            coefficients.system, c, coefficients.charge_balance
        )

    slots = [system.flux_index[name] for name in ("out_neu", "out_neg", "out_pos")]

    flux = (
        coefficients.collision_rate
        * c[coefficients.collision_i]
        * c[coefficients.collision_j]
    )
    product = coefficients.product_index
    owner = coefficients.product_owner
    per_product = flux[owner]

    per_channel = jnp.stack(
        [jnp.sum(jnp.where(product == slot, per_product, 0.0)) for slot in slots]
    )

    out_mask = jnp.isin(product, jnp.array(slots))
    contributing = jnp.where(out_mask, per_product, 0.0)
    j_by_cluster = (
        jnp.zeros(c.shape[0])
        .at[coefficients.collision_i[owner]]
        .add(contributing)
        .at[coefficients.collision_j[owner]]
        .add(contributing)
    )

    return {
        "j_tot": per_channel.sum(),
        "j_by_channel": per_channel,
        "j_by_cluster": j_by_cluster,
    }
