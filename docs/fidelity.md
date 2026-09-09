# Fidelity

Every place where this port could differ from upstream ACDC, and the flag
that controls it. **Defaults reproduce the Fortran**, always. Each row gets
a test at both settings.

The governing rule (`CLAUDE.md`): port from the code, not the comments, and
not the manual. Where the three disagree, the generated Fortran wins.

## Upstream quirks reproduced by default

| # | Quirk | Reference | Flag | Default |
|---|---|---|---|---|
| F1 | **Fitted-`c` projection.** Fortran assigns the residual to the monomer and rescales multiplicatively *only* if that would go negative. MATLAB **always** rescales. Different distributions from the same constraint. | `driver_acdc_J.f90:405-410` vs Perl `:10336-10344` | `fitted_projection` | `"fortran"` |
| F2 | **Charge balance.** MATLAB applies an explicit algebraic projection on the generic ions each outer iteration. Fortran has **none** — it relies on symmetric ion sources and lets the equations balance themselves. | Perl `:10346-10377` | `enforce_charge_balance` | `False` |
| F3 | **Steady-state tolerance disagrees three ways:** Fortran `sstol=1e-5`, MATLAB `difftol=1e-3`, manual documents `1e-5`. | `solution_settings.f90:20` vs Perl `:10041` vs manual §4.3.2 | `sstol` | `1e-5` |
| F4 | **Negative J is silently clamped** to `1e-100` rather than reported. | `get_acdc_J.f90:117` | `clamp_negative_j` | `True` |
| F5 | **Non-convergence is indistinguishable from J=0.** Upstream returns `j_out=0` with `ok=.true.` on steady-state failure. | `driver_acdc_J.f90:178-187` | — | we return an explicit status; the numeric result is unchanged |
| F6 | **Warm start.** `c` and `t_iter` are `save`d across plugin calls, so identical inputs give different outputs — **measured at 1.5e-6 relative in J**, while a cold start is bit-identical across processes. This is why the J gate is 1e-5, not 1e-6; see `docs/validation.md`. | `get_acdc_J.f90:31,42` | — | we thread state explicitly; pure by construction |
| F7 | **Hydrate distributions normalising to ≤0.99 are silently discarded** and the cluster treated as dry. | Perl `:2637-2643` | `hydrate_discard_threshold` | `0.99` |
| F8 | **ACDC's physical constants** differ from CODATA-2018 in the last digits (`k_B = 1.3806504e-23`, `N_A = 6.02214179e23`). | Perl `:924-928` | `constants` | `"acdc"` |
| F9 | **`ind_quad_loss_extra` is built but never read** by `feval`; the extra-product source is applied from the product's side. | `acdc_equations_*.f90:89-124` | — | reproduced (built, unused) |
| F10 | **No van der Waals or Fuchs correction** on hard-sphere collisions; a vdW enhancement exists commented-out upstream. | Perl `:8798` | — | reproduced |
| F15 | **`--ion_coll_method constant` is a no-op under variable temperature.** The manual documents a size-independent enhancement of 10. At fixed temperature upstream applies it; with `--variable_temp` it is silently dropped and every ion–neutral pair emits the bare hard-sphere rate — verified by generating the fixture (0 of 587 pairs enhanced, no `10.d0` anywhere in `get_coll`). Someone using it for a quick variable-temperature run gets collisions 5–17× too slow with no warning. Exposed as two explicitly-named options rather than one silent default. | Perl `:8267-8275` + generated fixture | `ion_collision_method` | `"su82"` |
| F14 | **The integrator's absolute tolerance is loosened.** The reference uses `atol = 1e-6` m⁻³ = 1e-12 cm⁻³; VODE's BDF tolerates that because it rescales internally, but Kvaerno5 exhausts a 20,000-step budget on the evaporation-dominated corner of the grid. This port uses 1.0 m⁻³ — still six orders below any meaningful concentration. J is measured identical from atol 1e-6 to 1e4, so nothing physical rides on it; `test_solve.py` asserts that insensitivity. | `solution_settings.f90:9` | `atol` argument | `1.0` m⁻³ |
| F13 | **The root-find path under-reports convergence.** Its relative residual `f/c` floors near 6e-5 because at steady state `f` is a small difference of large opposing fluxes, so optimistix never certifies even though J is correct to 2e-6 and matches the integration path to 2e-9. Gate on the inter-path agreement, not the flag. | — | — | known limitation |
| F12 | **The ion-neutral enhancement exceeds the manual's stated range.** The manual describes it as "a factor between one and ten"; measured on the reference's own `get_coll` at 280 K it runs 1.17–16.95, with 6% of ion-neutral pairs above 10. The manual is an approximate characterisation, not a bound the code enforces. Reproduced (there is nothing to fix); asserted in `test_rates.py` so a future clamp would fail. | manual §2.5.1 vs Perl `:8853-8888` | — | reproduced |
| F11 | **The mobility-diameter mass correction is defeated by a unit bug.** `$mass1` is reassigned to g/mol at `:3672`, then the correction multiplies by `$mass_conv` *again* at `:3681`, giving `28.8·1.66e-27/98.08 ≈ 5e-28`, so `sqrt(1+…)` is exactly `1.0`. The emitted mobility diameters are just `d_mass + 0.3 nm` — every `get_mob_diameter` entry differs from `get_diameter` by 0.30 with no mass dependence. Reproduced by default because these feed the size-bin classifier. | Perl `:3672,:3681` | `mobility_diameter` | `"fortran"` |

## Upstream bugs *not* reproduced

Reproducing an uninitialized read is not fidelity.

| # | Bug | Reference | What we do |
|---|---|---|---|
| B1 | `diameter_max_syst` is left uninitialized in loop mode and propagates to `diameter_acdc`. | `acdc_simulation_setup.f90:52` (commented out) | Compute it. Documented, tested, no flag. |

## Structural differences (not configurable)

These are the point of the port; they are not divergences to be flagged but
capabilities the reference lacks.

| Difference | Reference | Here |
|---|---|---|
| Jacobian | empty `jeval`, VODE `mf=22` finite differences, `neqn` extra RHS calls per build | `jax.jacfwd`, exact |
| Rate constants | 9585 lines of emitted numeric literals, frozen per outer call | traced functions of `T` |
| Temperature branches | `(.5±sign(.5,T−T0))` branchless Heaviside, because Perl could not evaluate the branch at generation time | one `jnp.where` |
| `j_by_cluster`, `j_all` | computed by `formation`, discarded by the driver | returned |
| Net-flux matrix | MATLAB only (`dofluxes.m`) | native, ~free from the RHS einsum |
| Rate recomputation gate | `ipar` mutated through the ODE solver as a hidden side channel | immutable params pytree |
| Cluster set changes | regenerate Fortran with Perl, recompile | parse a data file |
