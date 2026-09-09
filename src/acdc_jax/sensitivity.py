"""Gradients and batching -- what the port exists for.

ACDC's dominant uncertainty is the quantum-chemical cluster free energies
supplied as input. This module makes ``dJ/d(Delta-H)`` and
``dJ/d(Delta-S)`` available directly, which turns "which clusters control
the formation rate" from a manual sensitivity study into one call, and makes
inverting measured J for cluster thermodynamics a well-posed problem rather
than an intractable one.

Everything differentiates through the root-find path. Gradients there come
from the implicit function theorem rather than from unrolling an integration
history, so they cost about one extra linear solve instead of one per step.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from acdc_jax import rates, rhs, solve
from acdc_jax.reactions import ReactionSet
from acdc_jax.system import AcdcSystem


@dataclasses.dataclass(frozen=True)
class Conditions:
    """Ambient conditions, all SI."""

    c_a: float
    """Sulfuric acid concentration, m^-3."""
    c_n: float
    """Base concentration, m^-3."""
    temperature: float
    cs_ref: float
    """Reference coagulation sink, 1/s."""
    ipr: float
    """Ion production rate, 1/m^3/s."""


def formation_rate_of(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    conditions: Conditions,
    delta_h: jnp.ndarray | None = None,
    delta_s: jnp.ndarray | None = None,
    method: str = "rootfind",
) -> jnp.ndarray:
    """Steady-state J as a differentiable function of thermodynamics.

    ``delta_h`` and ``delta_s`` override the tabulated free energies, which
    is what makes them differentiable inputs. Both default to the values
    already in ``inputs``.

    Args:
        method: ``"rootfind"`` differentiates cleanly through the implicit
            function theorem. ``"integrate"`` would have to differentiate
            the whole solver history.
    """
    if delta_h is None:
        delta_h = jnp.asarray(inputs.delta_h)
    if delta_s is None:
        delta_s = jnp.asarray(inputs.delta_s)

    perturbed = dataclasses.replace(inputs, delta_h=delta_h, delta_s=delta_s)

    coefficients = rhs.assemble(
        system,
        reactions,
        perturbed,
        conditions.temperature,
        conditions.cs_ref,
        conditions.ipr,
        conditions.ipr,
        # J needs only the flat reaction arrays; the dense tensors exist for
        # the get_rate_coefs comparison and cost a scatter per reaction.
        dense=False,
    )
    c0 = solve.set_vapours(
        system,
        jnp.zeros(system.n_equations),
        {"1A": conditions.c_a, "1N": conditions.c_n},
    )
    result = solve.solve_steady_state(system, coefficients, method=method, c0=c0)
    return result.j_total


def sensitivity_to_free_energies(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    conditions: Conditions,
) -> dict[str, jnp.ndarray]:
    """``dJ/dH`` and ``dJ/dS`` per cluster, and the normalised version.

    Returns ``d_dh`` (per kcal/mol), ``d_ds`` (per cal/mol/K) and
    ``relative_dh`` -- the fractional change in J per kcal/mol, which is the
    form worth reading: it says how much a 1 kcal/mol error in a given
    cluster's formation enthalpy moves the answer, and quantum chemistry
    errors are of that order.

    **Restricted to clusters with tabulated energies.** Monomers and the
    generic charger ions are masked to zero: formation energies are defined
    relative to the free monomers, so a monomer's is zero by construction
    and not a parameter anyone can be wrong about. Differentiating with
    respect to it shifts the reference state itself and yields a large
    derivative that looks like a physical result. Left unmasked, 1A and 1N
    top the list for the bundled system with sensitivities of +1.16 and
    +0.65 per kcal/mol, ahead of every real cluster.
    """

    def j_of_h(h):
        return formation_rate_of(system, reactions, inputs, conditions, delta_h=h)

    def j_of_s(s):
        return formation_rate_of(system, reactions, inputs, conditions, delta_s=s)

    h0 = jnp.asarray(inputs.delta_h)
    s0 = jnp.asarray(inputs.delta_s)

    j = formation_rate_of(system, reactions, inputs, conditions)
    d_dh = jax.grad(j_of_h)(h0)
    d_ds = jax.grad(j_of_s)(s0)

    # Mask out the reference-state species. Applied to the RESULT rather
    # than by restricting the differentiated argument, so the returned
    # arrays stay aligned with system.labels.
    parameters = jnp.asarray(inputs.has_energy_data)
    d_dh = jnp.where(parameters, d_dh, 0.0)
    d_ds = jnp.where(parameters, d_ds, 0.0)

    # Guarded division: J can be numerically zero in the far
    # evaporation-limited corner, and a bare divide would give NaN
    # cotangents on any further differentiation.
    safe_j = jnp.where(j > 0, j, 1.0)
    relative = jnp.where(j > 0, d_dh / safe_j, 0.0)

    return {"j": j, "d_dh": d_dh, "d_ds": d_ds, "relative_dh": relative}


def sensitivity_to_conditions(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    conditions: Conditions,
) -> dict[str, jnp.ndarray]:
    """``dJ/d`` each ambient condition.

    ``d_ln_j_d_ln_c_a`` is the quantity usually quoted from measurements as
    the "apparent order" of nucleation in sulfuric acid -- typically
    between 1 and 3 -- so it is the natural thing to compare against
    observations.
    """

    def j_of(c_a, c_n, temperature, cs_ref, ipr):
        return formation_rate_of(
            system,
            reactions,
            inputs,
            Conditions(c_a, c_n, temperature, cs_ref, ipr),
        )

    args = (
        conditions.c_a,
        conditions.c_n,
        conditions.temperature,
        conditions.cs_ref,
        conditions.ipr,
    )
    j = j_of(*args)
    gradients = jax.grad(j_of, argnums=(0, 1, 2, 3, 4))(*args)

    safe_j = jnp.where(j > 0, j, 1.0)
    return {
        "j": j,
        "d_c_a": gradients[0],
        "d_c_n": gradients[1],
        "d_temperature": gradients[2],
        "d_cs_ref": gradients[3],
        "d_ipr": gradients[4],
        # d ln J / d ln [A] -- the apparent nucleation order.
        "d_ln_j_d_ln_c_a": jnp.where(
            j > 0, gradients[0] * conditions.c_a / safe_j, 0.0
        ),
    }


def formation_rate_batch(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    conditions: Conditions,
    method: str = "integrate",
    jit: bool = False,
) -> jnp.ndarray:
    """Steady-state J over a grid of conditions, in one batched solve.

    Every field of ``conditions`` may be a scalar or an array; they are
    broadcast against each other, so a temperature sweep at fixed vapour is
    ``Conditions(c_a=5e12, c_n=..., temperature=jnp.linspace(240, 300, 25),
    ...)``. The result has the broadcast shape.

    This is Phase 7.4. What made it possible was making the steady-state
    criterion traceable (`solve.steady_state_by_integration`) and letting
    the assembly skip the dense tensors: the checkpoint grid is fixed in
    advance, so which pairs the criterion considers is a compile-time fact
    and only the comparison depends on the data. The batched solve steps
    every point on a shared clock -- diffrax runs its loop until the last
    member is done -- so the win is in the per-step vectorisation, not in
    skipping work.

    Measured against the Python loop on the bundled system (CPU, one acid
    sweep; the loop is linear in the grid, this is nearly flat)::

        points     4     16     32     64
        loop     4.6 s  14.4 s  25.5 s  49.2 s
        batched  2.8 s   3.2 s   4.0 s   5.3 s

    Args:
        method: ``"integrate"`` (the validated default) or ``"rootfind"``.
        jit: compile the batched function. **Off by default, deliberately.**
            XLA's CPU compile time grows steeply with the batch: at 32
            points compiling took 3.5 minutes and the run came out slower
            than the plain loop, where the same batch without ``jit`` takes
            4 seconds. The solve is already one traced graph, so there is
            little left for ``jit`` to fuse. Worth trying on an accelerator,
            where the batched dense algebra is the right shape.

    Returns:
        J, 1/m^3/s, with the conditions' broadcast shape.
    """
    fields = jnp.broadcast_arrays(
        *(
            jnp.asarray(getattr(conditions, name), dtype=float)
            for name in ("c_a", "c_n", "temperature", "cs_ref", "ipr")
        )
    )
    shape = fields[0].shape
    flat = [field.reshape(-1) for field in fields]

    def one(c_a, c_n, temperature, cs_ref, ipr):
        return formation_rate_of(
            system,
            reactions,
            inputs,
            Conditions(c_a, c_n, temperature, cs_ref, ipr),
            method=method,
        )

    batched = jax.vmap(one)
    if jit:
        batched = jax.jit(batched)
    return batched(*flat).reshape(shape)


def sweep(
    system: AcdcSystem,
    reactions: ReactionSet,
    inputs: rates.RateInputs,
    c_a: np.ndarray,
    c_n: float,
    temperature: float,
    cs_ref: float,
    ipr: float,
    method: str = "integrate",
    batched: bool = False,
) -> np.ndarray:
    """J over a range of vapour concentrations.

    ``batched=True`` runs the whole sweep as one ``vmap``-ed solve
    (:func:`formation_rate_batch`); the default keeps the Python loop, which
    is what the reference does with a shell loop over a serial binary,
    except that here the compiled right-hand side is reused across points.
    The two agree to solver noise -- see ``test_sensitivity.py``.
    """
    if batched:
        return np.asarray(
            formation_rate_batch(
                system,
                reactions,
                inputs,
                Conditions(jnp.asarray(c_a), c_n, temperature, cs_ref, ipr),
                method=method,
            )
        )

    out = np.empty(len(c_a))
    for i, value in enumerate(c_a):
        out[i] = float(
            formation_rate_of(
                system,
                reactions,
                inputs,
                Conditions(float(value), c_n, temperature, cs_ref, ipr),
                method=method,
            )
        )
    return out


__all__ = [
    "Conditions",
    "formation_rate_batch",
    "formation_rate_of",
    "sensitivity_to_conditions",
    "sensitivity_to_free_energies",
    "sweep",
]
