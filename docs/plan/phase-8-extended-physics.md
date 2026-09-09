# Phase 8 — Extended physics

**Goal:** the option surface beyond the bundled example's defaults. Each is
a `FidelityConfig` option with a `docs/fidelity.md` row and a test against a
Perl-generated fixture.

**Gate:** each parameterization reproduced against its own fixture.

Additive — nothing in Phases 0–7 depends on this, and individual items can
be dropped.

## 8.1 Dahneke transition-regime collisions
Kinetic-to-continuum transition, as an alternative to hard spheres.

## 8.2 `bg_loss` coagulation sink
Losses onto a monodisperse background scavenger population with Dahneke
transition-regime rates — the physical model that `exp_loss` approximates.

## 8.3 Kelvin evaporation
Evaporation from the Kelvin equation instead of ΔG data, for clusters
without quantum chemistry. Also the loop-mode default, so this is a
prerequisite for Phase 9.

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

## 8.9 Charge balance
MATLAB applies an explicit algebraic projection on the generic ions each
outer iteration (Perl `:10346-10377`); Fortran has **none**, relying on
symmetric ion sources. Arguably a correctness feature rather than a
diagnostic. Implement as a flag; default Fortran.
