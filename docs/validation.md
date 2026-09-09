# Validation

The port is checked against the vendored Fortran at every level, not only
end-to-end. A single end-to-end number agreeing can hide two compensating
errors; the per-leaf gates cannot.

## Acceptance thresholds

| Gate | Threshold | Phase |
|---|---|---|
| Rate constants `K`, `E`, `cs` vs emitted literals | < 1e-12 rel | 4 |
| `dc/dt` vs f2py-wrapped `feval` | < 1e-12 rel | 5 |
| `jacfwd` vs the analytic quadratic Jacobian | < 1e-12 rel | 5 |
| Conservation (molecules, charge) | < 1e-10 rel | 5 |
| Steady-state J vs the `run` binary | < 1e-6 rel | 6 |
| Root-find vs time-integration steady state | < 1e-8 rel | 6 |
| `jax.grad` vs central finite difference | < 1e-3 rel | 7 |
| `vmap` vs a serial Python loop | exact | 7 |
| jit vs eager | exact | 7 |
| **Boundary-cascade decisions vs `--print_boundary`** | **exact match** | 2 |

## Why the boundary gate is exact

The boundary cascade decides which out-of-set collision products count as
grown out (contributing to J) and which are stripped back into the system,
and into exactly what. It is combinatorial: a decision is either the same or
it isn't. Any divergence is a different model, not a rounding difference —
and because boundary reactions carry a large fraction of the mass flux in
narrow cluster sets, a wrong decision produces a plausible-looking wrong
answer rather than an obvious one.

## Why the rate gate is 1e-12 and the J gate is 1e-6

Rate constants are closed-form functions of the same inputs, evaluated in
the same precision — they should agree to round-off, and anything looser
would hide a genuine formula error. Steady-state J is the output of two
*different* adaptive integrators with different error control, so agreement
is limited by the tolerance of the loosest one (`rtol=1e-5`), not by
arithmetic.

## Temperature sweeps, not single points

Rate constants are validated across 250–320 K rather than at the reference
temperature alone. The emitted Fortran bakes the temperature dependence into
per-pair literals and a piecewise branch whose switch point `T0_ij` differs
for every pair; a single-temperature check would pass with the branch on the
wrong side for most pairs.

## Goldens

Generated once on a pinned toolchain and committed as plain `.npz` (no Git
LFS), with the compiler version and flags recorded in
`validation/goldens/MANIFEST.md`. Regenerating on the same toolchain must
reproduce them bit-for-bit.

## Reference environment

| | |
|---|---|
| Compiler | GNU Fortran 16.1.0 (Homebrew GCC) |
| Flags | `-O3 -fcheck=bounds -finit-local-zero -fallow-argument-mismatch` |
| Platform | macOS, Apple M4 Pro (arm64) |
| JAX | 0.9.2, CPU |
