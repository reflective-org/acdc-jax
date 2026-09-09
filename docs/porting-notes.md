# Porting notes

Running commentary on what changed and why. Numerical divergences and their
flags live in [`fidelity.md`](fidelity.md); this file is for the reasoning
that does not fit a table row.

## The 10,669 lines are rate tables, not an unrolled RHS

The first impression of `acdc_equations_AN_ions_example.f90` — 10,669 lines
for a 54-cluster system — suggests a fully unrolled right-hand side, and
suggests the port's job is to re-vectorize it. That is wrong, and acting on
it would have produced a worse port.

The RHS (`feval`) is **58 lines**, already loop-based over integer index
arrays. The bulk is `get_coll` (5076 lines), `get_rate_coefs` (3226),
`get_evap` (1283) and `initialize_parameters` (807) — hardcoded *numbers*.

So the sparse index-array structure is reproduced rather than replaced, and
what the port removes is the numeric emission. That is also what makes
term-by-term validation against the reference possible.

For contrast, the *MATLAB* path really is fully unrolled — one explicit
algebraic expression per cluster, generated inside the `if(!$lfortran)`
branch. Had the port targeted the MATLAB output the first impression would
have been right.

## Branchless temperature code disappears

The emitted collision rates are full of constructs like

```fortran
(.5d0+sign(.5d0,temperature-1.28315437332313d+03))*(...)
+(.5d0-sign(.5d0,temperature-1.28315437332313d+03))*(...)
```

This is a branchless Heaviside, and `1283.15 K` is precisely the temperature
at which that pair's Su82 reduced dipole parameter `x` equals 2 (verified:
`a=71.6423`, `a/√1283.15 = 2.0000`). Perl had to emit it this way because
under `--variable_temp` it could not evaluate the branch at generation time
and had no other way to express a runtime conditional in a numeric literal.

JAX traces temperature natively, so the whole apparatus collapses to one
`jnp.where` per parameterization, with `T0` never appearing. This is the
clearest case where the port is genuinely simpler than its source rather
than merely different.

## `ipar` is a side channel through the ODE solver

Upstream passes a 4-element integer array through DVODE as user data, and
`feval` **mutates it** to record that it has initialised the rate tables.
`get_acdc_J.f90:96-97` resets elements 1 and 3 before every call, which is
how "conditions changed, recompute" is signalled. `feval` and `formation`
keep *separate* `save`d copies of the coefficient tensors, each with its own
gate, so `initialize_parameters` runs twice per conditions change and ~3 MB
of tensors is stored twice.

In a functional port this becomes an immutable params pytree built once per
conditions change. The subtlety worth flagging: `fix_fitted_c` reads
`ipar(1)` to decide whether the incoming monomer concentration means "the
monomer" or "the sum over the fitted group" — an order-dependent protocol
that must become explicit data (`is_first`, `c_fit_target`).

## What upstream computes and discards

`formation` returns `j_by_cluster(neqn)` and `j_all(neqn,4)` —
per-cluster, per-charge-pair attribution of the formation rate — and
`driver_acdc_J.f90:359-363` keeps only `j_by_charge`. Both are free here.

Similarly, the net-flux matrix that underpins all mechanistic
interpretation of ACDC results exists only in the MATLAB path
(`dofluxes.m`). It is the same einsum as the RHS with the reduction not
taken, so the Fortran path could have had it at negligible cost.
