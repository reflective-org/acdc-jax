# Phase 3 — Enumeration to sparse index arrays

**Goal:** the upper-triangle pair loop and the decision ladder, producing
the same sparse index arrays the Fortran builds in
`initialize_parameters`.

**Gate:** index arrays identical to the emitted ones, entry for entry.

## 3.1 Pair enumeration and the decision ladder

`for i in 1..N: for j in i..N` (Perl `:2680-2686`), then in order
(Perl `:2694-2836`): forbidden by `--nst` → skip; non-standard reaction →
use its stoichiometry, no reverse; `neg+pos` → `n_flux_rec`; one partner is
the outgrown pool → coagulation onto formed particles; otherwise
`combine_labels` and, if out of set, the Phase 2 cascade.

**Verify:** the set of (i,j) pairs with nonzero coefficient matches the
2679 `coef_quad` and 476 `coef_lin` assignments.

## 3.2 The six index arrays

`ind_quad_loss(i)`, `ind_quad_form(k)`, `ind_lin_loss(k)`,
`ind_lin_form(i)`, `ind_quad_loss_extra(i)`, `ind_quad_form_extra(k)`, with
slot 0 holding the count. Semantics documented at
`acdc_equations_AN_ions_example.f90:219-236`.

Two things to reproduce exactly: **self-collisions are listed twice** with
`coef_quad(i,i,k) = 0.5·K(i,i)`, and `ind_quad_loss_extra` is built but
**never read** by `feval` — the extra-product source is applied from the
product's side via `ind_quad_form_extra`.

**Verify:** array-by-array equality against a dump from the f2py bridge.

## 3.3 Dense representation for the traced RHS

Convert the ragged index arrays into what JAX wants: a sparse COO or dense
`coef_quad[nclust, nclust, neq]` and `coef_lin[neq, neq, nclust]`. At 54
clusters the dense form is 54·54·63·8 B ≈ 1.5 MB — small enough that dense
is the cheap option; keep the COO path for larger sets.

**Verify:** round-trip ragged → dense → ragged is the identity.

## 3.4 Sources, constants, fitted

`isconst`, `source`, `fitted` from `sources_and_constants`, then the
generated override that makes the generic ions non-constant with
`source(53)=ipr_neg`, `source(54)=ipr_pos`
(`acdc_equations_AN_ions_example.f90:207-210`).

The `fitted` encoding: `fitted(1,0)` = number of constraints,
`fitted(n+1,0)` = dependent cluster, `fitted(n+1,1)` = partner count,
`fitted(n+1,2:)` = partners. Default is empty — upstream ships the fitted
branch commented out.

**Verify:** `isconst` pins `[1A]` and `[1N]` under `solve_ss`; ion sources
land on 53/54.
