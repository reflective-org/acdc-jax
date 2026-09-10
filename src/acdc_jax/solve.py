"""Time integration and steady state.

Two paths to the steady state:

- **integration** -- the reference's own approach, marching until no cluster
  changes by more than ``sstol`` over a 600 s window. This is the parity
  path, and the default.
- **root-finding** -- solving ``f(c) = 0`` directly. This is the manual's
  own definition of steady state (Nomenclature, p. iv), is far cheaper, and
  differentiates cleanly through the implicit function theorem. Gated
  against the integration path rather than trusted on its own, because
  Newton can converge to a non-physical root.

Everything here is a pure function. The reference threads a mutable ``ipar``
through the ODE solver as a hidden "recompute the rates" flag and keeps the
concentration vector in a ``save``d array, so identical inputs give
different answers depending on call history; none of that survives here.
"""

from __future__ import annotations

import dataclasses

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix

from acdc_jax import config, rhs
from acdc_jax.system import AcdcSystem


@dataclasses.dataclass(frozen=True)
class SteadyStateResult:
    """Outcome of a steady-state solve."""

    concentrations: jnp.ndarray
    j_total: jnp.ndarray
    """Formation rate, 1/m^3/s."""
    j_by_channel: jnp.ndarray
    """Neutral, negative, positive."""
    converged: jnp.ndarray | bool
    """Explicit, unlike the reference.

    ``driver_acdc_J.f90:178-187`` returns ``j_out = 0`` with ``ok = .true.``
    when the steady state is not reached, so a caller cannot tell "converged
    to zero" from "gave up". Here the two are distinguishable.
    """
    steps: jnp.ndarray | int
    """Solver steps taken. Left as an array so the solve traces."""


def integrate(
    system: AcdcSystem,
    coefficients: rhs.Coefficients,
    c0: jnp.ndarray,
    t1: float,
    t0: float = 0.0,
    rtol: float = config.RTOL_SOLVER,
    atol: float = config.ATOL_INTEGRATION,
    max_steps: int = 100_000,
    saveat: diffrax.SaveAt | None = None,
) -> diffrax.Solution:
    """Integrate the birth-death equations from ``t0`` to ``t1``.

    Kvaerno5, a 5th-order ESDIRK method, with the reference's own tolerances
    (``rtol = 1e-5``, ``atol = 1e-6`` m^-3). The system is stiff: monomer
    collisions equilibrate in microseconds while the largest clusters evolve
    over minutes.

    The implicit solver gets the exact Jacobian by autodiff, where the
    reference builds one by finite differences at a cost of ``neqn`` extra
    right-hand-side evaluations per step.
    """
    n = system.n_clusters

    # Integrate the CLUSTERS only. The nine flux accumulators are write-only
    # quadrature -- out_neu, coag and the rest are never read back by the
    # right-hand side -- but they grow without bound, so leaving them under
    # the error controller makes it fight to resolve a quantity that has no
    # steady state. With atol at 1e-6 m^-3 that is enough to exhaust the
    # step budget somewhere past 600 s, while the clusters themselves are
    # already settling. J is computed as an instantaneous flux from the
    # converged concentrations anyway, exactly as the reference does under
    # the steady-state assumption (driver_acdc_J.f90:356-365), so the
    # accumulators are not needed for it.
    def vector_field(t, y, args):
        full = jnp.zeros(system.n_equations).at[:n].set(y)
        return rhs.rhs(coefficients, full)[:n]

    term = diffrax.ODETerm(vector_field)
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Kvaerno5(),
        t0=t0,
        t1=t1,
        dt0=None,
        y0=c0[:n],
        stepsize_controller=controller,
        saveat=saveat or diffrax.SaveAt(t1=True),
        max_steps=max_steps,
    )
    return solution


def _turnover(coefficients: rhs.Coefficients, c: jnp.ndarray) -> jnp.ndarray:
    """Total absolute flux through each species, 1/m^3/s.

    The natural scale for a steady-state residual: at steady state the net
    rate is the small difference of these large opposing terms.
    """
    flux = (
        coefficients.collision_rate
        * c[coefficients.collision_i]
        * c[coefficients.collision_j]
    )
    total = (
        jnp.zeros(c.shape[0])
        .at[coefficients.collision_i]
        .add(flux)
        .at[coefficients.collision_j]
        .add(flux)
        .at[coefficients.product_index]
        .add(coefficients.product_multiplicity * flux[coefficients.product_owner])
    )
    if coefficients.evaporation_rate.size:
        evaporation = coefficients.evaporation_rate * c[coefficients.evaporation_k]
        total = (
            total.at[coefficients.evaporation_k]
            .add(evaporation)
            .at[coefficients.evaporation_i]
            .add(evaporation)
            .at[coefficients.evaporation_j]
            .add(evaporation)
        )
    n = coefficients.n_clusters
    return total.at[:n].add(coefficients.sink * c[:n])


def _relative_change(new: jnp.ndarray, old: jnp.ndarray, n: int) -> jnp.ndarray:
    """The reference's convergence metric (``driver_acdc_J.f90:174``).

    Largest relative change over the CLUSTERS only -- the flux accumulators
    grow without bound by construction and would never converge. The
    denominator is floored at ``chtol`` so a concentration of numerically
    zero does not produce a spurious infinity.
    """
    denominator = jnp.maximum(old[:n], config.CHTOL)
    return jnp.max(jnp.abs(new[:n] - old[:n]) / denominator)


def steady_state_by_integration(
    system: AcdcSystem,
    coefficients: rhs.Coefficients,
    c0: jnp.ndarray | None = None,
    sstol: float = config.SSTOL,
    sstimech: float = config.SSTIMECH,
    sstimetot: float = config.SSTIMETOT,
    max_time: float = 1e5,
    rtol: float = config.RTOL_SOLVER,
    atol: float = config.ATOL_INTEGRATION,
    max_steps: int = 20_000,
) -> SteadyStateResult:
    """March to steady state, reproducing the reference's own criterion.

    Steady state is declared when no cluster changes by more than ``sstol``
    relative across a window of at least ``sstimech`` seconds, with at least
    ``sstimetot`` seconds elapsed in total (``driver_acdc_J.f90:174``). Note
    that this makes the answer only defined to within ``sstol`` -- see
    docs/validation.md.

    The whole function traces, so it can be ``vmap``-ed over a grid of
    conditions (7.4): the checkpoint grid is fixed in advance, which makes
    WHICH pairs the criterion considers a compile-time fact, and only the
    comparison itself depends on the data.

    Integrated as a SINGLE solve with checkpoints, not as a sequence of
    restarts. The reference restarts because VODE is handed a fixed 20-point
    time grid, but restarting an adaptive implicit solver throws away its
    step-size and Jacobian history and makes it re-estimate an initial step
    from scratch each time. Doing that here was enough to exhaust a 200,000
    step budget on the 600-1200 s interval, while one continuous solve over
    the whole 1e5 s reaches the same answer in about 560 steps.
    """
    n = system.n_clusters
    if c0 is None:
        c0 = jnp.zeros(system.n_equations)

    # Checkpoints spaced by at least sstimech, so consecutive pairs can be
    # compared on the reference's own terms. Logarithmic early, where the
    # dynamics are fast, then linear.
    early = [1e-8, 1e-4, 1e-2, 1.0, 60.0]
    late = list(np.arange(sstimetot, max_time + sstimech, sstimech * 20))
    # Kept as Python floats: the grid is a compile-time fact, and reading
    # the last one back out of a jnp array would concretize under jit.
    times = [float(t) for t in early + late if t <= max_time]
    checkpoints = jnp.asarray(times)

    solution = integrate(
        system,
        coefficients,
        c0,
        t1=times[-1],
        t0=0.0,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(ts=checkpoints),
    )
    trajectory = solution.ys

    # Which checkpoint pairs the criterion may consider is fixed by the
    # grid, so it is settled here rather than inside the traced comparison.
    eligible = np.zeros(len(times), dtype=bool)
    for k in range(1, len(times)):
        eligible[k] = (times[k] >= sstimetot) and (times[k] - times[k - 1] >= sstimech)

    change = jax.vmap(_relative_change, in_axes=(0, 0, None))(
        trajectory[1:], trajectory[:-1], n
    )
    settled = jnp.concatenate([jnp.array([False]), change <= sstol])
    settled = settled & jnp.asarray(eligible)
    converged = jnp.any(settled)
    # argmax picks the FIRST True, which is the checkpoint the reference's
    # loop would have stopped at.
    index = jnp.where(converged, jnp.argmax(settled), len(times) - 1)

    c = jnp.zeros(system.n_equations).at[:n].set(trajectory[index])
    formation = rhs.formation_rate(system, coefficients, c)
    return SteadyStateResult(
        concentrations=c,
        j_total=formation["j_tot"],
        j_by_channel=formation["j_by_channel"],
        converged=converged,
        steps=solution.stats["num_steps"],
    )


def steady_state_by_rootfind(
    system: AcdcSystem,
    coefficients: rhs.Coefficients,
    c0: jnp.ndarray | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 500,
) -> SteadyStateResult:
    """Solve ``f(c) = 0`` directly.

    The manual defines steady state as exactly this (Nomenclature, p. iv):
    for every cluster, formation fluxes balance loss fluxes. Solving it as a
    root is both far cheaper than marching and cleanly differentiable --
    gradients come through the implicit function theorem rather than through
    the whole integration history.

    Solved in **log space**. Concentrations span thirty orders of magnitude
    and must stay positive; Newton on the raw variables walks straight into
    negative concentrations, where the evaporation terms are meaningless.
    Working in ``log c`` makes positivity structural rather than something to
    be clamped.

    The flux accumulators are excluded from the unknowns: they have no
    steady state, growing linearly forever by construction.

    **Convergence is under-reported.** ``converged`` comes from optimistix
    and is routinely ``False`` even when the answer is right, because the
    relative residual ``f/c`` has a floor set by cancellation -- at steady
    state ``f`` is the small difference of large opposing fluxes, and the
    worst-conditioned cluster floors at about 6e-5. Newton then exhausts its
    step budget without meeting a 1e-10 tolerance, while J is already
    correct to 2e-6 and agrees with the integration path to 2e-9. Judge this
    path by that agreement, which is what the tests gate on, rather than by
    the flag. Fixing it properly needs a residual scaled by cluster turnover;
    a first attempt at that destabilised Newton (the scaling moves with the
    iterate) and it is left for later.
    """
    n = system.n_clusters
    if c0 is None:
        c0 = jnp.full(system.n_equations, 1e6)

    # Newton needs a good starting point. From a cold start on a
    # 52-dimensional stiff system it diverges to NaN within two steps, so
    # callers normally seed this with a short integration -- which is what
    # `solve_steady_state(method="rootfind")` does.

    # Indices of the unknowns, resolved ONCE outside the traced residual.
    # `jnp.where` on a mask needs a statically known output size, so it
    # cannot be called inside a function optimistix will trace.
    free_index = np.flatnonzero(~np.asarray(coefficients.isconst)[:n])
    pinned_values = c0[:n]

    floor = 1e-30

    def residual(log_free, _args):
        c_clusters = pinned_values.at[free_index].set(jnp.exp(log_free))
        full = jnp.zeros(system.n_equations).at[:n].set(c_clusters)
        f = rhs.rhs(coefficients, full)[:n][free_index]
        # Scaled by concentration, so every cluster is weighted comparably
        # rather than the solve being dominated by the few most abundant.
        return f / jnp.maximum(jnp.exp(log_free), floor)

    solver = optimistix.Newton(rtol=rtol, atol=atol)
    initial = jnp.log(jnp.maximum(c0[:n][free_index], floor))

    solution = optimistix.root_find(
        residual,
        solver,
        initial,
        throw=False,
        max_steps=max_steps,
    )

    c_clusters = pinned_values.at[free_index].set(jnp.exp(solution.value))
    c = jnp.zeros(system.n_equations).at[:n].set(c_clusters)

    formation = rhs.formation_rate(system, coefficients, c)
    return SteadyStateResult(
        concentrations=c,
        j_total=formation["j_tot"],
        j_by_channel=formation["j_by_channel"],
        converged=solution.result == optimistix.RESULTS.successful,
        # Left as an array, like the integration path: `int()` here made
        # the whole root-find path unbatchable, even though its docstring
        # offered `method="rootfind"` to `formation_rate_batch`.
        steps=solution.stats.get("num_steps", 0),
    )


def solve_steady_state(
    system: AcdcSystem,
    coefficients: rhs.Coefficients,
    method: str = "integrate",
    **kwargs,
) -> SteadyStateResult:
    """Steady state by the requested method.

    ``"integrate"`` is the default because it is the path validated against
    the Fortran. ``"rootfind"`` is faster and differentiates better, but is
    gated against the integration path rather than trusted alone.
    """
    if method == "integrate":
        return steady_state_by_integration(system, coefficients, **kwargs)
    if method == "rootfind":
        # Globalise with a short integration first. Newton alone diverges
        # from a cold start; seeded, it polishes the answer to a true root
        # rather than to wherever the sstol window happened to stop.
        seed = kwargs.pop("c0", None)
        warm = steady_state_by_integration(
            system, coefficients, seed, max_time=kwargs.pop("seed_time", 1e3)
        )
        return steady_state_by_rootfind(
            system, coefficients, c0=warm.concentrations, **kwargs
        )
    raise ValueError(f"unknown method {method!r}; use 'integrate' or 'rootfind'")


def set_vapours(
    system: AcdcSystem, c: jnp.ndarray, concentrations: dict[str, float]
) -> jnp.ndarray:
    """Set named species concentrations, m^-3.

    Accepts a NumPy array as well as a JAX one; callers building an initial
    condition reach for `np.zeros` as often as `jnp.zeros`, and the
    `.at[].set()` idiom only exists on the latter.
    """
    c = jnp.asarray(c, dtype=float)
    for label, value in concentrations.items():
        c = c.at[system.labels.index(label)].set(value)
    return c


__all__ = [
    "SteadyStateResult",
    "integrate",
    "set_vapours",
    "solve_steady_state",
    "steady_state_by_integration",
    "steady_state_by_rootfind",
]
