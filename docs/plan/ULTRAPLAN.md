# ULTRAPLAN — porting ACDC to JAX

The whole arc, in one place. Each phase has its own file with tasks
numbered `N.M`; each task is one commit with its own verify step. Live
status is in [`PROGRESS.md`](PROGRESS.md).

## Goal

A standalone, differentiable, host-agnostic `acdc-jax` that:

1. reproduces Fortran ACDC's formation rates to the thresholds in
   [`../validation.md`](../validation.md),
2. takes cluster sets as **data** rather than generated code,
3. exposes `jax.grad` and `jax.vmap` through the whole solve.

Host-model integration (TOMAS, ClusterIn) is explicitly **out of scope**.

## Why

- **Gradients.** ACDC's dominant uncertainty is the input cluster free
  energies ΔH, ΔS. `jax.grad` makes ∂J/∂ΔG available, converting a manual
  sensitivity study into an inverse problem.
- **Batching.** Steady-state J over a (vapour, T, CS, IPR) grid is
  embarrassingly parallel; today it is a shell loop over a serial binary.
- **Cluster sets as data.** Upstream regenerates Fortran with an 11.7k-line
  Perl script per cluster set.
- **Exact Jacobian.** Upstream ships an empty `jeval` and runs VODE with
  `mf=22` — a finite-difference Jacobian costing `neqn` extra RHS calls per
  build.

## The shape of the problem

The generated Fortran's RHS is **already data-driven** — 58 lines looping
over integer index arrays (`acdc_equations_AN_ions_example.f90:68-125`).
The 10,669 lines are hardcoded *rate tables*: `get_coll` (5076),
`get_rate_coefs` (3226), `get_evap` (1283), `initialize_parameters` (807).

So the port **reproduces** the sparse index-array structure rather than
inventing one, and replaces only the numeric-literal emission — which JAX
traces natively. In particular, all the `sign()`/`max()` branchless trickery
in the emitted `get_coll` exists only because Perl could not evaluate a
temperature-dependent branch at generation time. Under JAX it collapses to
one `jnp.where`.

## Invariants (enforced in tests)

1. **float64 everywhere.** `jax_enable_x64` set in exactly one place.
2. **Setup is NumPy; only assembled arrays cross into traced code.**
   Cluster enumeration and the boundary cascade involve dict lookups,
   string labels and data-dependent `while` loops. They run once.
3. **Every upstream quirk is a `FidelityConfig` flag defaulting to the
   Fortran behaviour**, with a `docs/fidelity.md` row and a test at both
   settings.
4. **No Python branching on traced values**; double-where on every guarded
   division; mask before reducing, never multiply.
5. **`fortran/` stays byte-identical to upstream.** Build-time flags are
   fine; source edits go through `fortran/patches/`.
6. **Mass and charge bookkeeping is checked**, not assumed: total molecules
   in ≡ molecules out + molecules accumulated, per species.

## Phases

| Phase | Title | Gate |
|---|---|---|
| 0 | [Scaffold & harness](phase-0-harness.md) | reference builds; f2py imports; goldens captured |
| 1 | [Parser](phase-1-parser.md) | 54-cluster set + 117 energy entries reproduced |
| 2 | [Boundary cascade](phase-2-boundary.md) | **exact** match vs `--print_boundary`, 3 cluster sets |
| 3 | [Enumeration → index arrays](phase-3-enumeration.md) | index arrays identical to `initialize_parameters` |
| 4 | [Rates](phase-4-rates.md) | `K`, `E`, `cs` < 1e-12 rel vs emitted literals |
| 5 | [RHS, formation, fluxes](phase-5-rhs.md) | `dc/dt` < 1e-12 rel vs f2py `feval` |
| 6 | [Solve](phase-6-solve.md) | J < 1e-6 rel vs binary; rootfind vs integrate < 1e-8 |
| 7 | [Differentiability & batching](phase-7-autodiff.md) | grad vs FD < 1e-3; vmap ≡ serial |
| 8 | [Extended physics](phase-8-extended-physics.md) | each parameterization vs a Perl fixture |
| 9 | [Loop mode](phase-9-loop-mode.md) | matches a loop-mode fixture |
| 10 | [Pathways, figures, docs](phase-10-figures-docs.md) | figures regenerate from scratch |

Phases 0–7 are the core deliverable and are strictly sequential. Phases 8
and 9 are additive and can be reordered or dropped without stranding
anything. Phase 10 depends on 5 (fluxes) and 7 (gradients).

## Build order rationale

**Harness before implementation.** Phase 0 stands up the f2py bridge and
golden capture before any physics is written, so every subsequent phase has
something to check against on the same day it is written. This is the
`tomas-jax-condensation` "Phase 0 — Build the Comparison Harness First"
pattern.

**Boundary cascade early, and alone.** Phase 2 is the only part of the port
with no closed-form specification — it is defined operationally by ~250
lines of order-dependent Perl. It carries a large fraction of the mass flux
in narrow cluster sets, and getting it wrong produces a plausible-looking
wrong answer. It gets its own phase, its own module, and an exact-match
gate.

**Rates before RHS.** The rate tables are the largest body of reference
numbers available (2131 `K` assignments, 213 `E` formulas, 54 `cs` values),
so they are the strongest early signal that the physics is right.

**Port before jit.** Per the house rule, correctness is established eagerly
in float64 with no `jit` or `vmap`; those land in separate commits in
Phase 7. The eager driver stays permanently as the debugger.

## Known risks

| Risk | Mitigation |
|---|---|
| `check_boundary` has no spec (Perl `:10907-11256`) | Port line-by-line, not from the paper. Exact-match gate vs `--print_boundary` on three cluster sets. Own phase. |
| Loop mode is effectively a second model (~1500 lines) | Sequenced last (Phase 9). Droppable without stranding phases 0–8. |
| Wall-loss zoo is 6 chamber-specific empirical fits with no independent reference | Phase 8, one flag each, each against a Perl-generated fixture. Not needed unless CLOUD comparisons are wanted. |
| Committed example files were generated by the **2020** Perl, not 2024 | Both generators vendored. See `PROVENANCE.md`. |
| Steady-state root-find may find a non-physical root | Gated against the time-integration path at < 1e-8; integration stays the default. |
