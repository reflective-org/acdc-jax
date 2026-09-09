# Phase 9 — Loop mode (large cluster sets)

**Goal:** `small_set_mode = .false.` — clusters indexed by composition tuple
rather than enumerated, for systems too large to unroll.

**Gate:** matches a loop-mode fixture.

## Scope warning

Loop mode is effectively a **second model**, not a mode of the first:
composition-tuple indexing, Kelvin-equation evaporation instead of ΔG,
size-limited evaporation, an implicit growth compound, J-table
interpolation. ~1500 lines interleaved through the Perl. It exists mainly
for one-component validation studies.

It is sequenced last deliberately. If it proves disproportionate to its
value, it can be dropped without stranding Phases 0–8. Revisit the decision
at the Phase 8 close rather than committing now.

## 9.1 Composition-tuple indexing
`get_molecule_numbers` → the realizable compositions; `neq = nclust + (neq_max − nclust_max)`,
outflux index moves to `nclust + 1`.

## 9.2 Kelvin evaporation and evaporation size limit
Depends on Phase 8.3.

## 9.3 Size-bin grouping
The `nbins = 5` mobility-diameter classifier from
`acdc_simulation_setup.f90:192-246`. Bin 0 is a below-range catch-all; an
above-range cluster is a hard stop upstream. The ragged gathers become a
fixed `(nbins, nclust)` 0/1 matrix and a matmul.

**Note a latent upstream bug to fix rather than reproduce:**
`diameter_max_syst` is left uninitialized in loop mode
(`acdc_simulation_setup.f90:52` is commented out) and propagates to
`diameter_acdc`. Fix it, flag it, and document it — reproducing an
uninitialized read is not fidelity.

## 9.4 J-table interpolation
`--j_in`, `--j_in_function`, `--jlim`, `--loop_j_frac`.

## 9.5 Euler fallback
Upstream switches to hand-rolled Richardson-extrapolated Euler above 5000
equations because dense-LU VODE doesn't scale. With a sparse/Krylov implicit
solver in JAX this is probably unnecessary — but its `chmax`-based step cap
is a useful physical safeguard worth keeping as an option.
