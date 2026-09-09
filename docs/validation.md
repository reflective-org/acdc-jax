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
| Steady-state J vs the `run` binary | **< 1e-5 rel** | 6 |
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

## Why the rate gate is 1e-12 and the J gate is 1e-5

Rate constants are closed-form functions of the same inputs, evaluated in
the same precision — they should agree to round-off, and anything looser
would hide a genuine formula error.

The J gate is looser for a reason that is a property of the reference, not
of the port. **The reference's steady-state J is only defined to within
`sstol = 1e-5`.** Measured during Phase 0.5 on this machine:

| | |
|---|---|
| Same point, fresh process each time | **bit-identical** (`2217995.192415948`) |
| Same point, re-solved after a different point in one process | **1.5e-6 relative shift** |

The cause is that `acdc_plugin` keeps the concentration vector in a `save`d
array (`get_acdc_J.f90:31`) and warm-starts from it, while the convergence
criterion is a *relative-change* test rather than an exact root: no cluster
may move by more than `sstol` over a 600 s window. Where you start therefore
determines where inside that tolerance band you stop.

So the reference does not have a single steady-state J to agree with to
1e-6 — it has a band of width ~`sstol`. Gating tighter than that would be
gating against an arbitrary point in the band, and would fail or pass on
irrelevant details of the integration path. 1e-5 is the honest threshold.

The root-find path has no such ambiguity (`f(c) = 0` is exact), which is why
it is gated against the integration path at 1e-8 — that comparison is
between two of *our* results, both deterministic.

### Consequence for golden capture

`validation/capture_steadystate.py` runs **one fresh subprocess per grid
point**. Sweeping in a single process would bake the warm-start path into
the goldens and make them unreproducible in any other order.

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
