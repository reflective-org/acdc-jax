# Phase 5 — RHS, formation rate, fluxes

**Goal:** the ODE right-hand side as two einsums, plus the formation-rate
and net-flux diagnostics.

**Gate:** `dc/dt` matches the f2py `feval` to < 1e-12 relative on random and
physical states; `jacfwd` matches finite differences.

## 5.1 The RHS

```
f(i) = c(i)·( −Σ coef_quad(i,j,k)·c(j) − Σ coef_lin(p,q,i) )
     + Σ coef_quad(a,b,i)·c(a)·c(b)
     + Σ coef_quad(a,b,i)·mult·c(a)·c(b)      [boundary extra products]
     + Σ coef_lin(i,j,k)·c(k)
     + source(i)
```

(`acdc_equations_AN_ions_example.f90:91-123`.) In JAX: two `einsum`s and a
vector add. `isconst` species have their derivative zeroed — as a mask, not
a branch.

**Verify:** < 1e-12 vs f2py on ~20 log-uniform random states spanning
1e-6 … 1e14 m⁻³, plus a physically plausible state.

## 5.2 Exact Jacobian

Upstream ships an empty `jeval` and runs VODE with `mf=22`, paying `neqn`
extra RHS calls per Jacobian. Here it is `jax.jacfwd` — exact and free.
The RHS is quadratic, so the Jacobian is analytic too; use it as a test
oracle for `jacfwd`.

**Verify:** `jacfwd` vs central finite differences < 1e-6; `jacfwd` vs the
analytic quadratic Jacobian < 1e-12.

## 5.3 Formation rate

Replay `ind_quad_form(60..62, …)` to get `j_tot`, `j_by_charge(4)`,
`j_by_cluster`, `j_all(neqn,4)`
(`acdc_equations_AN_ions_example.f90:140-185`). Recombination products
(`j_by_charge(4)`) fold into the neutral channel.

**The driver computes `j_by_cluster` and `j_all` and throws them away**
(`driver_acdc_J.f90:359-363`). We return them: they are per-cluster,
per-charge-pair attribution of the formation rate and cost nothing.

**Verify:** `j_tot` and `j_by_charge` vs f2py `formation` < 1e-12.

## 5.4 Net-flux matrix

The flux matrix is the same einsum as the RHS **with the reduction not
taken** — so it is nearly free here, while upstream has it only in MATLAB
(`dofluxes.m`, generated inside the `if(!$lfortran)` block).

Reproduce the manual's §4.4 conventions exactly: `flux_2d` is
sign-canonicalised so each net flux appears once in the positive-net
direction, with the complement left at zero; diagonal (i+i) elements are
from the **reactants'** point of view, hence a factor 2; but `flux_out`'s
identical-cluster elements are from the **product's** point of view. The
manual explicitly warns about the mismatch.

**Verify:** row/column sums reconstruct `dc/dt`; `flux_out` summed
reproduces `j_tot`.

## 5.5 Conservation checks

Molecules in ≡ molecules out + accumulated, per species; charge balance
across the neutral/negative/positive partition. Not a reference comparison —
an independent invariant that catches index errors the reference shares.

**Verify:** conservation holds to < 1e-10 relative over a long integration.
