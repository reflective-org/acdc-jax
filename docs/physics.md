# Physics

The equations the port implements, with citations into the vendored
reference. Line references are to `fortran/` in this repository.

Cluster concentrations are in **m⁻³** and all rates SI, matching the Fortran
interface. (The MATLAB path uses cm⁻³ at its interface; that convention is
not adopted here.)

## Birth–death equations

For each cluster `i`:

```
dc_i/dt = c_i·( −Σ_j β_ij·c_j − Σ γ_i→· − CS_i )      losses
        + Σ_{a+b→i} β_ab·c_a·c_b                       formation by collision
        + Σ_{a+b→i+extra} β_ab·mult·c_a·c_b            boundary extra products
        + Σ_{k→i+j} γ_k→ij·c_k                          formation by evaporation
        + S_i                                            external source
```

`acdc_equations_AN_ions_example.f90:91-123`. Self-collisions carry a factor
½ (`coef_quad(i,i,k) = 0.5·K(i,i)`) because the pair is enumerated twice.

## Collision rates

**Neutral–neutral, hard sphere** (Perl `:8785-8790`):

```
β_ij = (8π k_B T)^½ · (1/m_i + 1/m_j)^½ · (r_i + r_j)²
```

Free-molecular kinetic gas theory. No van der Waals enhancement, no Fuchs
transition-regime correction — a vdW form exists commented out at Perl
`:8798`. Since √T factors out, the reference emits `K = A_ij·√T`.

**Ion–neutral, Su & Chesnavich (1982)** — the default. Electrostatic
attraction between the ion and the neutral's dipole and induced dipole makes
these rates 1–10× the hard-sphere value, which matters most for the smallest
clusters. With `μ_D` in Debye and `α` in Å³:

```
x     = 100·μ_D / √(2·α·k_B·10²³·T)
ratio = (x + 0.5090)²/10.526 + 0.9754     for x < 2
      = 0.4767·x + 0.62                    for x ≥ 2
β_ij  = max( ratio · L_ij , β_ij^hard-sphere )
L_ij  = 2π·4.8032e-16·√( α·(1/m_i + 1/m_j) / 1e27 )
```

Perl `:8843-8888`. **Su & Bowers (1973)** is available as an alternative,
using the dipole *locking* coefficients from the dipole file header:
`9.5436e-29·√α + 6.4805e-27·μ_D/√T`, times `√(1/m_i+1/m_j)`, divided by the
hard-sphere rate, floored at 1 (Perl `:8816-8842`).

**Ion–ion recombination**: a flat `1.6e-12 m³ s⁻¹` for every
opposite-charge pair, with no size dependence at all (Perl `:372`).

Same-sign pairs are forbidden outright (Perl `:10771-10777`).

## Evaporation rates

By detailed balance against the *same* collision rate (Perl `:9670-9683`):

```
γ(k → i+j) = β_ij · (p_ref / k_B T) · exp[ (ΔG_k − ΔG_i − ΔG_j) / (k_B T) ]
```

halved when `i == j`. `p_ref` is the reference pressure from line 1 of the
energy file (1 atm here), so `p_ref/k_B T = 7.3389e27/T`.

The **ion enhancement applies to γ as well** when exactly one partner is
neutral (Perl `:9698-9701`). This is not optional: using the enhanced β
forward and the unenhanced β in detailed balance breaks equilibrium while
still producing finite, plausible numbers.

Free energies come from the energy file as `ΔG` (kcal/mol) or as `ΔH`
(kcal/mol) and `ΔS` (cal/mol/K) with `ΔG = ΔH − T·ΔS`. They are **formation**
free energies relative to the constituent monomers, so monomers are zero by
definition.

## External losses

**Coagulation sink**, `exp_loss` — the widely used approximation to loss
onto a background aerosol population:

```
CS_i = CS_ref · (d_i / d_ref)^m       m = −1.6 by default
```

Charged species are additionally multiplied by `fcs` (1.0 in the bundled
setup). The reference emits only the size dependence and multiplies by the
runtime `CS_ref` (`acdc_equations_AN_ions_example.f90:1011`).

`bg_loss` (explicit monodisperse scavengers, Dahneke transition regime),
wall losses (six chamber parameterizations), and dilution are Phase 8.

## Ion production and charging

Constant zeroth-order sources on the two generic charger ions only, at the
ion production rate (`acdc_equations_AN_ions_example.f90:207-210`). Charge
is transferred by collision: `neg + 1A → 1B`, `pos + 1N → 1N1P`. Charge is
carried structurally — a cluster is negative if it contains a molecule of
charge −1, positive if it contains the proton.

## Geometry

```
m_i = Σ n_k·m_k
V_i = Σ n_k·m_k/ρ_k              additive bulk liquid volumes
r_i = (3V_i / 4π)^⅓
d_mob = (2r_i + 0.3 nm)·√(1 + 28.8·m_conv/m_i)
```

The proton and the "missing proton" pseudo-species contribute **mass but not
volume** (Perl `:11498`, `:11566`).

## Formation rate

*J* is the flux of clusters growing out of the simulated system — the flux
into the `out_neu`/`out_neg`/`out_pos` counters, resolved by the charge of
the colliding pair, with recombination products folded into the neutral
channel (`driver_acdc_J.f90:359-363`).

Which products count as "out" is set by threshold rules in the cluster-set
file: a product is out when **every** molecule count meets or exceeds a
rule. Products that fall outside the set without meeting a rule are stripped
back into it — see [`plan/phase-2-boundary.md`](plan/phase-2-boundary.md),
the one part of the model with no closed-form specification.

## Steady state ≠ equilibrium

Worth stating because it is the most common misreading of ACDC output. At
**steady state**, for each cluster the sum of formation fluxes equals the
sum of loss fluxes. At **equilibrium**, each forward/backward pair balances
individually, so there is no net flux and nothing forms. Equilibrium is a
steady state; a steady state is generally not an equilibrium, and in
realistic conditions it never is — a nonzero *J* is precisely the
statement that it isn't. (Manual, Nomenclature, p. iv.)
