# Phase 6 — Solving

**Goal:** time integration and steady state, both differentiable.

**Gate:** J matches the Fortran binary to < 1e-5 relative across the
60-point (vapour, T, CS, IPR) golden grid; root-find vs integration < 1e-8.

The J threshold is 1e-5 rather than 1e-6 because the reference's own
steady-state J is only defined to within `sstol` -- measured at 1.5e-6
between warm and cold starts of the same point. See `docs/validation.md`.

## 6.1 Time integration

diffrax `Kvaerno5` (5th-order ESDIRK, stiff) with
`PIDController(rtol=1e-5, atol=1e-6)` — ACDC's own tolerances from
`solution_settings.f90`. The exact Jacobian from Phase 5.2 goes to the
implicit solver rather than the finite-difference one VODE builds.

**Verify:** trajectories match the reference to < 1e-6 at matched output
times.

## 6.2 Steady state by integration (the parity default)

Reproduce the reference test: no cluster changes by more than
`sstol = 1e-5` relative over a ≥ `sstimech` = 600 s window, with total
elapsed ≥ `sstimetot` = 1200 s. The reference's 20-point time grid —
`(1e-8, 1e-2, 600, 1200)` then clusters of four spaced 600 s starting at
10× the previous — is a fixed-size array, so it precomputes and masks
rather than needing a dynamic loop.

Note the reference's ambiguity, reproduced but flagged: on non-convergence
it returns `j_out = 0` with `ok = .true.`, so a caller cannot distinguish
"converged to zero" from "failed". We return an explicit status.

**Verify:** J < 1e-5 vs the golden grid, which was captured cold.

## 6.3 Steady state by root-find (the fast path)

`optimistix.Newton` on `f(c) = 0` subject to `isconst`. The manual's own
definition of steady state (Nomenclature, p. iv) is exactly this. Gradients
come through the implicit function theorem rather than through the
integration history, which is both faster and lower-memory.

Risk: Newton can find a non-physical root. Gate it against the integration
path and keep integration as the default.

**Verify:** root-find vs integration < 1e-8 on the whole grid; negative
concentrations rejected.

## 6.4 Fitted concentrations

`fix_fitted_c` as an explicit projection, not a mutating callback. Upstream
assigns the residual to the monomer and rescales multiplicatively **only**
when that would go negative (`driver_acdc_J.f90:405-410`); MATLAB **always**
rescales. Both behind a `FidelityConfig` flag defaulting to Fortran.

**Verify:** both modes tested; the constraint holds after each step.

## 6.5 Replacing the hidden state

Upstream threads `ipar` as a mutable side channel *through the ODE solver*
to gate rate recomputation, and keeps `save`d `c` and `t_iter` across calls
so identical inputs give different outputs. Here: an immutable params pytree
built once per conditions change, and explicitly threaded state.

**Verify:** `solve(...)` is a pure function — same inputs, same outputs,
asserted by calling twice.
