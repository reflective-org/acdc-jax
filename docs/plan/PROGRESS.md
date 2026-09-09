# Progress

Live status. One line per task; each task is one commit. See
[`ULTRAPLAN.md`](ULTRAPLAN.md) for the arc and the per-phase files for
detail.

**Legend:** ✅ done · 🔨 in progress · ⬜ not started · ⏸ deferred

---

## Phase 0 — Scaffold & harness

- ✅ **0.1** Repository scaffold and vendored ACDC reference — `57bba17`
- ✅ **0.2** Planning documents and doc stubs — `22d7212`
- ✅ **0.3** Package skeleton and `config.py` — `e2e62b5`
- ✅ **0.4** f2py bridge to the reference — `9426f63`
- ✅ **0.5** Golden capture
- ✅ **0.6** Boundary-decision oracle

## Phase 1 — Parser ✅

Taken in dependency order: labels first, since everything else needs them.

- ✅ **1.3** Cluster labels and canonicalisation — `0e5c97f`
- ✅ **1.1** Molecule-property header — `0e5c97f`
- ✅ **1.2** Cluster-set body and range expansion — `0e5c97f`
- ✅ **1.4** Geometry (mass, volume, radius, mobility diameter) — `4ce0e2a`
- ✅ **1.5** Energy file (ΔH/ΔS/ΔG) — `13ffa6d`
- ✅ **1.6** Dipole/polarizability file — `13ffa6d`

## Phase 2 — Boundary cascade
- ⬜ **2.1** `combine_labels` protonation algebra
- ⬜ **2.2** Stage 1 — nucleated out?
- ⬜ **2.3** Stage 2 — per-species clamp
- ⬜ **2.4** Stage 3a — strength-guided stripping
- ⬜ **2.5** Stage 3b — fallback and give-up path
- ⬜ **2.6** Pruning rules

## Phase 3 — Enumeration
- ⬜ **3.1** Pair enumeration and decision ladder
- ⬜ **3.2** The six index arrays
- ⬜ **3.3** Dense representation for the traced RHS
- ⬜ **3.4** Sources, constants, fitted

## Phase 4 — Rates
- ⬜ **4.1** Hard-sphere collisions
- ⬜ **4.2** Ion–neutral enhancement (Su82)
- ⬜ **4.3** Su73 and constant
- ⬜ **4.4** Ion–ion recombination
- ⬜ **4.5** Evaporation by detailed balance
- ⬜ **4.6** Coagulation sink

## Phase 5 — RHS, formation, fluxes
- ⬜ **5.1** The RHS
- ⬜ **5.2** Exact Jacobian
- ⬜ **5.3** Formation rate
- ⬜ **5.4** Net-flux matrix
- ⬜ **5.5** Conservation checks

## Phase 6 — Solve
- ⬜ **6.1** Time integration
- ⬜ **6.2** Steady state by integration
- ⬜ **6.3** Steady state by root-find
- ⬜ **6.4** Fitted concentrations
- ⬜ **6.5** Replacing the hidden state

## Phase 7 — Differentiability & batching
- ⬜ **7.1** Gradient audit
- ⬜ **7.2** ∂J/∂ΔG
- ⬜ **7.3** ∂J/∂conditions
- ⬜ **7.4** vmap over condition grids
- ⬜ **7.5** jit

## Phase 8 — Extended physics
- ⬜ **8.1**–**8.9** see [phase-8](phase-8-extended-physics.md)

## Phase 9 — Loop mode
- ⏸ **9.1**–**9.5** decision revisited at Phase 8 close

## Phase 10 — Pathways, figures, docs
- ⬜ **10.1**–**10.5** see [phase-10](phase-10-figures-docs.md)

---

## Verified reference results

Anything established by actually running something, so it does not have to
be re-derived.

| What | Value | How |
|---|---|---|
| Reference builds | GNU Fortran 16.1.0, needs `-fallow-argument-mismatch` | `scripts/build_reference.sh` |
| AN+ions example J | **2.218 cm⁻³ s⁻¹** | `fortran/src/run` at [A]=1e7, [N]=1e9 cm⁻³, CS=1e-3 s⁻¹, T=280 K, IPR=3 cm⁻³ s⁻¹ |
| System size | 54 clusters, 63 equations | `acdc_system_AN_ions_example.f90:5-6` |
| Solver path | VODE (≤5000 eqs), steady-state assumption | `run` stdout |
| Cold-start J | **bit-identical across processes** | 3 fresh processes, same point |
| Warm-start J spread | **1.5e-6 relative** | same point re-solved after another in one process |
| Golden J grid | 60 points, J from 1.97e-8 to 2.76e5 cm⁻³ s⁻¹ | `validation/goldens/steadystate.npz` |
| `K` / `E` / `coef_quad` / `coef_lin` nonzeros | 2131 / 422 / 2679 / 474 | f2py bridge |
| Boundary oracle | **3034 decisions** across 3 cluster sets | `validation/goldens/boundary_*.json` |
| Cluster labels | all **52** match the KEY block exactly | Phase 1.2 |
| Mass / diameter / mob. diameter | all **52** match after `%.2f` rounding | Phase 1.4 |
| ΔH, ΔS | all **52** match the literals inlined in `get_evap` | Phase 1.5 |
| Regeneration fidelity | 0 structural diffs, max 9.8e-15 rel | replayed `run_perl.sh` vs committed |

---

## Changelog

### 2026-09-08 (Phase 1)

- **Phase 1 complete**, 191 tests. Cluster sets are now data: `.inp`,
  ΔH/ΔS and dipole tables parse straight into arrays, with every result
  checked against the generated Fortran.
- **F11 — a live unit bug in the reference's mobility diameter.** At
  Perl `:3672` `$mass1` is reassigned to g/mol; `:3681` then multiplies by
  `$mass_conv` *again*, so `sqrt(1 + 28.8·1.66e-27/98.08)` is **exactly
  1.0** and the intended Tammet mass correction vanishes. Emitted mobility
  diameters are just `d_mass + 0.3 nm` — visible in the committed Fortran
  itself, where every `get_mob_diameter` entry differs from `get_diameter`
  by 0.30 with no mass dependence. Reproduced by default (it feeds the
  size-bin classifier); `mobility_diameter="tammet"` fixes it, moving 1A
  from 0.85 to 0.97 nm.
- Declared-but-unused molecule types are dropped by the generator: the AN
  header declares five but no cluster contains dimethylamine, so
  `n_mol_types = 4`. Keeping all five would silently widen every
  composition vector downstream.
- Emitted geometry arrays go through `sprintf("%.2f")`, so full-precision
  comparison misses by ~1e-3 — outside anything resembling a tolerance
  failure and easy to misread as a formula error. Hence `geometry.emitted()`.

### 2026-09-08

- **0.1 complete.** Repository created at
  [`reflective-org/acdc-jax`](https://github.com/reflective-org/acdc-jax)
  (public, GPL-3.0). Fortran reference vendored from `tolenius/ACDC@870b82a`
  and confirmed to build and run.
- Found that upstream does not compile on gfortran ≥10: `formation` is
  called with 8 arguments at `driver_acdc_J.f90:359` and 5 at `:365`.
  The `:365` branch is dead (`small_set_mode` is a compile-time `.true.`).
  Resolved with a build flag rather than a patch, so `fortran/` stays
  byte-identical to upstream.
- **0.4 complete.** f2py bridge built. The Fortran declares no `intent`, so
  f2py silently returns untouched zero arrays; caught only because `K[0,0]`
  came back `0.0` instead of `3.35e-16`. Fixed with a hand-written `.pyf`.
- **0.5 complete.** Goldens captured (190 KB, plain `.npz`). While doing so,
  measured that the reference's steady-state J is **only defined to within
  `sstol`**: cold-start is bit-identical across processes, but re-solving
  the same point after a different one shifts J by 1.5e-6. The J acceptance
  gate was therefore relaxed from 1e-6 to **1e-5** — the original figure was
  below the reference's own self-consistency, so it could not have been met
  for reasons having nothing to do with the port. Goldens are now captured
  one fresh subprocess per grid point.
- A bug in the capture script swapped temperature and `cs_ref` via positional
  unpacking and produced a plausible-looking J field three orders too small.
  Now passed by keyword, with an assertion that the bundled example
  reproduces `2217995.192415948` before the grid is run.
- **0.6 complete.** Boundary oracle captured for all three cluster sets:
  AN_narrow (745 decisions), AN (1407), AD (882) — 3034 in total, each of
  which Phase 2 must reproduce exactly. Cross-checked by asserting the
  number of logged grow-out decisions equals the number of `-> out_`
  coefficient targets in the code the same run emitted.
- Confirmed the replayed `run_perl.sh` invocation regenerates the committed
  equation file with **zero structural differences** and a maximum relative
  difference of 9.8e-15 — 100x inside the rate gate. Recorded in
  `PROVENANCE.md`.
- A bug in the oracle test's composition helper required a leading digit, so
  bare stripped monomers (`2 A`, meaning two of molecule A) parsed as empty
  and molecule conservation appeared to fail on all 278 brought-back
  decisions. With that fixed, conservation now holds across all 1367
  brought-back decisions in the three sets — a real check on the parser.
- Corrected an early assumption worth recording: the generated `feval` is
  **not** an unrolled RHS. It is 58 lines looping over integer index arrays.
  The 10,669 lines are hardcoded *rate tables*. The port therefore
  reproduces the index-array structure rather than inventing one.
