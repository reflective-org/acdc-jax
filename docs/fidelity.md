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
| F6 | **Warm start.** `c` and `t_iter` are `save`d across plugin calls, so identical inputs give different outputs. | `get_acdc_J.f90:31,42` | — | we thread state explicitly; pure by construction |
| F7 | **Hydrate distributions normalising to ≤0.99 are silently discarded** and the cluster treated as dry. | Perl `:2637-2643` | `hydrate_discard_threshold` | `0.99` |
| F8 | **ACDC's physical constants** differ from CODATA-2018 in the last digits (`k_B = 1.3806504e-23`, `N_A = 6.02214179e23`). | Perl `:924-928` | `constants` | `"acdc"` |
| F9 | **`ind_quad_loss_extra` is built but never read** by `feval`; the extra-product source is applied from the product's side. | `acdc_equations_*.f90:89-124` | — | reproduced (built, unused) |
| F10 | **No van der Waals or Fuchs correction** on hard-sphere collisions; a vdW enhancement exists commented-out upstream. | Perl `:8798` | — | reproduced |

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
