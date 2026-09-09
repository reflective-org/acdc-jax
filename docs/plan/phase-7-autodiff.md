# Phase 7 — Differentiability and batching

**Goal:** the payoff. `jax.grad` through the solve, `vmap` over conditions.

**Gate:** grad vs central finite difference < 1e-3; `vmap` result equals the
serial loop exactly; jit changes no number.

Per the house rule, this phase is **separate commits from the port**:
correctness was established eagerly in float64 in Phases 1–6, and the eager
driver stays permanently as the debugger.

## 7.1 Gradient audit

Sweep the physics modules for the two NaN-cotangent patterns: every guarded
division gets the double-where idiom, and every masked reduction uses
`where(mask, term, 0.0)` rather than `mask * term`. Rate arrays are
evaluated over the full `(nclust, nclust)` extent and `0.0 * inf = NaN`.

Non-smooth operations inherited from the reference that will poison
gradients if left in the differentiated path: `minval(c) < negtol`,
`max(ci, chtol)`, the fitted-`c` negative branch, and the
`j_acdc = 1e-100` clamp at `get_acdc_J.f90:117`.

**Verify:** `test_grad_finite` — no NaN in any cotangent for any module.

## 7.2 ∂J/∂ΔG

Gradient of the steady-state formation rate with respect to the cluster
free energies. This is the motivating capability: ΔH/ΔS from quantum
chemistry is ACDC's dominant uncertainty, and this turns a manual
sensitivity study into an inverse problem.

**Verify:** vs central finite difference < 1e-3, on both the integration
and root-find paths.

## 7.3 ∂J/∂conditions

Gradients w.r.t. vapour concentrations, T, CS and IPR.

**Verify:** vs finite difference < 1e-3.

## 7.4 vmap over condition grids

Batch the whole steady-state solve. This is what makes lookup-table
generation practical.

**Verify:** `vmap` result equals a Python loop over the same conditions,
exactly (not to a tolerance — same computation, same order).

## 7.5 jit

`eqx.filter_jit` at the solver boundary, `ClusterSystem` as a pytree with
string/tuple metadata `static=True`.

**Verify:** jit and eager agree bit-for-bit; compile time recorded in
`benchmarks/`.
