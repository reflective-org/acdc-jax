# Phase 8 — Extended physics

**Goal:** the option surface beyond the bundled example's defaults. Each is
a `FidelityConfig` option with a `docs/fidelity.md` row and a test against a
Perl-generated fixture.

**Gate:** each parameterization reproduced against its own fixture.

Additive — nothing in Phases 0–7 depends on this, and individual items can
be dropped.

## 8.1 ~~Dahneke transition-regime collisions~~ → moved to Phase 9

**Correction.** Dahneke collisions are a LOOP-MODE option
(`--loop_coll_coef Dahneke`, Perl `:8036`), and the generator dies if it is
combined with variable temperature. They are not available in the small-set
mode this port targets, so they belong with loop mode in Phase 9. The
manual lists the option under §2.1.1 "Loop mode in Fortran", which is where
the confusion came from.

The `bg_loss` sink in 8.2 also uses Dahneke *transition-regime coagulation*,
but onto a background population -- a different formula that IS available
in small-set mode.

## 8.0 Su73 and constant ion-neutral methods ✅ (was 4.3)

Validated against Perl-generated fixtures across 250-320 K: Su73 at 4.9e-15,
`constant_no_enhancement` at 2.2e-15. Found and recorded fidelity **F15**:
`--ion_coll_method constant` with `--variable_temp` silently drops the
documented factor of 10 and emits bare hard-sphere rates for all 587
ion-neutral pairs. Exposed as two explicitly-named options.

## 8.2 `bg_loss` coagulation sink
Losses onto a monodisperse background scavenger population with Dahneke
transition-regime rates — the physical model that `exp_loss` approximates.

## 8.3 ~~Kelvin evaporation~~ → moved to Phase 9

**Correction.** Like Dahneke collisions, Kelvin evaporation is a LOOP-MODE
option (`--loop_evap_coef Kelvin`, gated by `$lloop` at Perl `:600`, `:8517`,
`:8694`). Not available in small-set mode. Moves to Phase 9 with Dahneke.

## 8.4 Hydrates / RH
Pre-equilibrium Boltzmann weighting over water content, then hydrate
averaging of every rate constant. Three conventions must be copied exactly
or numbers won't match: the water-conserving evaporation partition sum
(Perl `:9640-9705`), the ion enhancement applied *inside* the evaporation
average (Perl `:9698`), and the escape hatch that **silently discards** a
distribution normalising to ≤ 0.99 and treats the cluster as dry
(Perl `:2637-2643`). Incompatible with variable temperature upstream; here
it need not be, but the default reproduces the restriction.

## 8.5 Wall losses
Six parameterizations: `CLOUD4_JA`, `CLOUD4_JK`, `CLOUD4_AK`,
`CLOUD4_simple`, `CLOUD3`, `ift`, plus a flow-tube diffusion model. Five are
chamber-specific empirical fits transcribed from an Excel file and three
papers — ~260 lines of magic numbers with no independent reference. Needed
only for chamber comparisons; drop if CLOUD work is not planned.

## 8.6 Dilution losses
Single rate, default `9.6e-5 s⁻¹`.

## 8.7 Sticking factors and ΔG scaling
Per-collision sticking coefficients and per-channel ΔG corrections from
files; the corrections go **inside** the exponent, in kcal/mol.

## 8.8 Non-standard reactions
Explicit product overrides for collisions where the product breaks up
(energy non-accommodation), with no reverse evaporation.

## 8.9 Charge balance ✅
The generator's `--charge_balance ±1` (Perl `:2135-2180`), not just the
MATLAB driver's projection. One generic ion is made algebraic: `isconst`,
its source dropped, and at the top of every `feval`/`formation` call it is
set to `c_other + Σ(same-sign clusters) − Σ(opposite-sign clusters)`, with
the other ion clamped from below at zero when that difference is negative.
Ported in `rhs.charge_balance_projection`, applied inside `rhs()` and
`formation_rate()` so it traces; `assemble(charge_balance=±1)` marks the
species. Validated against the emitted block in `acdc_equations_cb1.f90`
by a literal NumPy transcription (1e-13, summation order) and by asserting
charge neutrality of the result. Default 0 = the shipped Fortran.
