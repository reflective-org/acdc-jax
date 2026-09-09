# Phase 4 — Rate constants

**Goal:** collision, evaporation and loss coefficients as traced functions
of temperature — replacing 9585 lines of emitted numeric literals.

**Gate:** `K`, `E`, `cs` match the emitted literals to < 1e-12 relative,
across a temperature sweep (250–320 K), not just at the reference T.

## 4.1 Hard-sphere collisions

`β = (8πk_BT)^½ · (1/m₁+1/m₂)^½ · (r₁+r₂)²` (Perl `:8785-8790`).
Free-molecular, **no** van der Waals or Fuchs correction — a commented-out
vdW enhancement at Perl `:8798` shows it was tried and abandoned.
√T factors out, so the emitted form is `K = A_ij·√T`.

**Verify:** 125 neutral–neutral prefactors reproduced; e.g.
`K(1,1) = 2.00300435981201e-17·√T`.

## 4.2 Ion–neutral enhancement (Su82, default)

`x = 100·μ_D / √(2·α·k_B·10²³·T)`; ratio `= (x+0.5090)²/10.526 + 0.9754`
for `x < 2`, else `0.4767x + 0.62`; × Langevin prefactor
`2π·4.8032e-16·√(α(1/m₁+1/m₂)/1e27)`; then `K = max(ionic, hard_sphere)`.

The emitted Fortran's `(.5d0 ± sign(.5d0, T−T0))` pairs are a branchless
Heaviside, and `T0_ij` is exactly the temperature where `x = 2` — verified:
`a=71.6423, T0=1283.15 → a/√T0 = 2.0000`. **Under JAX this whole apparatus
collapses to one `jnp.where`**, because JAX traces T natively. That is the
single clearest example of the port being simpler than its source.

**Verify:** 587 ion–neutral entries across the temperature sweep, including
either side of each pair's `T0`.

## 4.3 Su73 and constant

Su73: `9.5436e-29·√α + 6.4805e-27·μ_D/√T`, × `√(1/m₁+1/m₂)`, ÷ hard-sphere,
floored at 1, with the dipole locking coefficients applied to μ.
`constant`: factor 10 for everything. Both are `FidelityConfig` options.

**Verify:** against fixtures generated with `--ion_coll_method Su73` and
`constant`.

## 4.4 Ion–ion recombination

Flat `1.6e-12 m³/s`, no size dependence, every opposite-charge pair
(361 = 19×19 for the AN set).

**Verify:** all 361 entries equal the constant.

## 4.5 Evaporation by detailed balance

`γ(k→i+j) = β_ij · (p_ref/k_BT) · exp[(ΔG_k − ΔG_i − ΔG_j)/(k_BT)]`,
×0.5 when `i == j` (Perl `:9670-9683`).

**The ion enhancement is applied to γ as well** when exactly one partner is
neutral (Perl `:9698-9701`) — omitting it breaks detailed balance while
still producing finite, plausible numbers. This is the single easiest thing
to get wrong in the whole port; it gets its own test asserting that the
equilibrium distribution of a source-free, sink-free system is stationary.

**Verify:** 213 formulas across the sweep; equilibrium stationarity test.

## 4.6 Coagulation sink

`exp_loss`: `CS_i = cs_ref·(d_i/d_ref)^(−1.6)`, charged species ×`fcs`.
`--cs_only X,0` excludes the vapour monomers, which is why `cs(1)=cs(3)=0`.

**Verify:** the 54 size-dependence values; spot-checked analytically —
`2A → 2^(1/3)^(−1.6) = 0.690900` vs emitted `0.6909564`.
