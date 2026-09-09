# Phase 9 — Loop mode (large cluster sets)

**Goal:** `small_set_mode = .false.` — every composition up to a maximum
count per molecule type is a cluster, indexed by composition tuple rather
than listed in the input; no charges, no boundary rules, one outgoing flux.

**Gate:** K, E, losses, the right-hand side and J match f2py-compiled
loop-mode fixtures at 1e-12.

## What it actually is

The ultraplan feared a second ~1500-line model. Generating fixtures shows
the *emitted* loop-mode Fortran is 170–270 lines: an `i ≤ j` double loop
with `ij = product-or-outflux`, monomer-only Kelvin evaporation (or ΔG with
fissions), one `loss` vector, and `source(monomers) = coef`. It maps onto
the port's existing `Coefficients` layout exactly — collisions `(i, j,
K(j,i)·½ᵢ₌ⱼ) → ij`, evaporations `(ij → i + j, E)`, losses per slot,
sources — so `solve`, `sensitivity` and `formation_rate` are reused
unchanged. Loop mode here is a different *assembler*, not a second model.

## Oracle

`validation/reference/loop_bridge.py` f2py-compiles a generated loop-mode
equations file (plus a stub for `sources_and_constants` and
`shared_input`) so `get_coll`, `get_evap`, `get_losses`,
`get_masses_and_radii`, `feval` and `formation` can be called from Python.
`validation/capture_loop.py` generates each variant with the 2020 Perl,
builds the bridge and stores goldens (`loop_variant_*.npz`); tests need
neither Perl nor gfortran. Loop inputs live in `validation/fixtures/loop/`.

## 9.0 Oracle and plan ✅
## 9.1 Loop system ✅
Compositions `0 ≤ nₖ ≤ maxₖ`, not all zero, in the generator's nested-loop
order (`get_cluster_numbers`); one-component systems index by molecule
count. Mass `Σ nₖ mₖ` (g/mol), radius from `Σ nₖ mₖ/ρₖ`. Built as an
`AcdcSystem` (labels, compositions, zero charges, the usual flux slots) so
everything downstream applies. The input file is the ordinary cluster-set
header plus a single line with the largest composition; the header gains
optional `saturation vapor pressure` and `surface tension` rows.

## 9.2 Collision coefficients (was 8.1) ✅
`hard_spheres` — the loop form of kinetic gas theory — and `Dahneke`:
slip-corrected diffusivities, thermal speeds, `Kn = 2(Dᵢ+Dⱼ)/(√(vᵢ²+vⱼ²)(rᵢ+rⱼ))`,
`K = 4π(rᵢ+rⱼ)(Dᵢ+Dⱼ)(1+Kn)/(1+2Kn(1+Kn))`. A bare `--sticking_factor`
multiplies all of K.

## 9.3 Evaporation (was 8.3) ✅
`Kelvin`: monomer evaporation only (fissions disabled upstream),
`E = K·psat·x/kT · exp(2σ v_mon /(kT r))` with the mole fraction `x` of the
evaporating type in the parent; for a pair of two monomer types upstream
keeps the LARGER of the two channels' rates in the single symmetric `E`
entry. `--rlim_no_evap` stops evaporation above a radius. `DeltaG`: detailed
balance from a per-composition free energy, all fissions allowed.

## 9.4 Losses ✅
`exp_loss` (default coefficient 2.6e-3), `bg_loss`, wall losses and
dilution, each a loop over sizes. The wall-loss mobility-diameter mass
correction is *live* here (`sqrt(1+28.8/m)` with m in g/mol), unlike the
small-set path (F11). **F21:** with coagulation, wall loss and dilution all
on, the generator emits `loss = cs+ wl, dil` — invalid Fortran (the
`s/,/+/` at Perl `:7667` replaces one comma). Two losses at a time work.

## 9.5 Assembly ✅
`loop.assemble(...) -> Coefficients`: pairs `i ≤ j`, `K(i,i)` halved, the
product index or the `out_neu` slot, evaporation channels where `E > 0`,
losses on their slots, `source(monomers)`, monomers `isconst` under the
steady-state assumption. Dense `coef_quad`/`coef_lin` become optional so a
thousand-cluster system does not allocate `n²·neq`. Validated against the
bridge's `feval` on random states and `formation`.

## 9.6 Size bins ✅
`nbins = 5`, limits `(1.05, 1.28, 1.73, 2.59, 4.27, 6.36) nm` on mobility
diameter `2r + 0.3 nm`, monomers excluded, below-range → bin 0, above-range
is a hard stop upstream. Fixed `(nbins+1, nclust)` 0/1 matrix. The
uninitialised `diameter_max_syst` in `get_system_size` is fixed and
recorded, not reproduced.

## 9.7 Variable temperature, steady state, sensitivity ✅
K and E trace in T (upstream emits them as functions of `temperature`).
Steady state and dJ/dT on a loop system through the unchanged Phase 6–7
code.

## Found on the way
F21 (three losses emit invalid Fortran), F22 (one-component `feval` reads
`E(j,i)` out of bounds), F23 (two-component `get_masses_and_radii` does
not compile). All recorded in `docs/fidelity.md`; the oracle repairs F22
and F23 in its copy of the generated source.

## Review follow-up
`deltag_evaporation` masked only its result, so an off-grid pair's exponent
(its `safe` index points at cluster 0) could overflow and leave the `exp`
VJP computing `0 * inf` -- `jax.grad` returned NaN on a grid deep enough to
reach it. The exponent is masked too, and a test at a free-energy scale
that overflows fails without the fix. Also: `assemble(channels="monomer")`
for the Kelvin channel set (O(n·n_types) rather than O(n²)); the grid build
uses the generator's own `clust_from_indices` lookup instead of a pass over
pairs (n = 2000 went from 1.3 s to a fraction of that); `--exp_loss_exponent`
and `--exp_loss_ref_cluster` are honoured rather than hardcoded;
`LossSettings.coagulation=None` expresses a wall-loss-only run; the
variable-temperature goldens are captured at 250 K so their tests gate the
temperature dependence; the no-op `LOOP_FIDELITY` is gone (the wall losses'
mass correction is live on both paths -- F11 concerns the emitted metadata
table, not the wall-loss formula).

## Deferred, recorded
`--j_in`/`--j_in_function`/`--jlim`/`--loop_j_frac` (J-table input and
flux-over-size diagnostics), `--loop_cs` (sink cluster inside the set),
`--loop_restrict`/`--boundary` (need an external `restriction_criteria`
module), the implicit growth compound, MCMC coefficients.
