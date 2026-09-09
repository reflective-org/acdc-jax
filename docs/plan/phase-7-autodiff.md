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

## 7.4 vmap over condition grids ✅

Batch the whole steady-state solve. This is what makes lookup-table
generation practical.

Two things had to change first. The steady-state criterion was a Python
loop with `float()` and `break`; since the checkpoint grid is fixed in
advance, *which* pairs the criterion may consider is a compile-time fact,
so the scan became `argmax` over a mask and the whole solve traces. And
`rhs.assemble` gained `dense=False`: the `(n, n, neq)` tensors exist to be
compared against `get_rate_coefs`, nothing in the right-hand side reads
them, and building them costs a scatter per reaction — 1.6 s of tracing per
condition, against 0.04 s without.

**Verify (revised):** the plan asked for bitwise equality with a Python
loop. That is not achievable and should not be: under `vmap` diffrax drives
the whole batch on one adaptive clock, so the step sequence differs from a
solo solve. Measured worst-case relative difference over an acid sweep is
**6e-13** — eight orders below the 1e-5 to which the steady-state criterion
defines J at all. The gate is 1e-9.

| points | 4 | 16 | 32 | 64 |
|---|---|---|---|---|
| loop | 4.6 s | 14.4 s | 25.5 s | 49.2 s |
| batched | 2.8 s | 3.2 s | 4.0 s | 5.3 s |

## 7.5 jit ✅ (off by default, measured)

Available as `formation_rate_batch(jit=True)`. It is **not** the default:
XLA's CPU compile time grows steeply with the batch — 3.5 minutes at 32
points, after which the run was slower than the serial loop, where the same
batch untraced takes 4 seconds. The solve is already a single traced graph,
so there is little left to fuse. Worth revisiting on an accelerator, where
the batched dense algebra is the right shape.

**Verify:** jit and eager agree bit-for-bit; compile time recorded in
`benchmarks/`.
