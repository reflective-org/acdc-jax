# Phase 0 — Scaffold & comparison harness

**Goal:** stand up the repository and the Fortran comparison bridge *before*
any physics is written, so every later phase has something to check against
on the day it is written.

**Gate:** `uv run pytest` green · f2py module imports and returns a `dc/dt`
· goldens captured for the bundled AN+ions example.

---

## 0.1 Repository scaffold and vendored reference ✅

Package skeleton (`pyproject.toml`, hatchling, `src/` layout, uv), `LICENSE`
(GPL-3.0), `README.md`, `CLAUDE.md`, `COPYRIGHT.md`, `PROVENANCE.md`,
`CITATION.cff`, `.gitignore`.

Vendor `fortran/` from `tolenius/ACDC@870b82a`, mirroring the upstream
`ACDC_Fortran_standard/` layout so the Makefile works unmodified.

`scripts/build_reference.sh` wraps the build with
`-fallow-argument-mismatch`. Upstream does not compile on gfortran ≥10:
`driver_acdc_J.f90` calls the external `formation` with 8 arguments at
`:359` and 5 at `:365`. `small_set_mode` is a compile-time `.true.`, so the
second call is dead code, but gfortran type-checks both. A build flag rather
than a patch keeps `fortran/` byte-identical.

**Verify:** `./scripts/build_reference.sh` → J = 2.218 cm⁻³ s⁻¹ at
[A]=1e7, [N]=1e9 cm⁻³, CS=1e-3 s⁻¹, T=280 K, IPR=3 cm⁻³ s⁻¹.
*Done — GNU Fortran 16.1.0.*

## 0.2 Planning documents

`docs/plan/` — `ULTRAPLAN.md`, one file per phase, `PROGRESS.md`.
Plus the doc stubs the README links: `docs/physics.md`, `docs/fidelity.md`,
`docs/validation.md`, `docs/porting-notes.md`, `docs/KEY_DECISIONS.md`.

**Verify:** every link in `README.md` resolves.

## 0.3 Package skeleton and config

`src/acdc_jax/__init__.py` — the **only** place `jax_enable_x64` is set.
`src/acdc_jax/config.py` — all constants, no magic numbers downstream:

- Physical: `k_B = 1.3806504e-23`, `N_A = 6.02214179e23`,
  `KCAL_PER_MOL_TO_J = 4.184e3 / N_A`, `P_ATM = 101325.0`.
  These are ACDC's own values (Perl `:924-928`), which differ in the last
  digits from CODATA-2018 — matching the reference matters more than being
  current, so the modern values are available as a `FidelityConfig` flag.
- Solver: `RTOL=1e-5`, `ATOL=1e-6`, `CHMAX=1e-2`, `CHTOL=1e-6`,
  `NEGTOL=-1e-6`, `SSTOL=1e-5`, `SSTIMECH=600.0`, `SSTIMETOT=1200.0`
  (`fortran/src/solvers/solution_settings.f90`).
- Model: `RECOMB_COEFF=1.6e-12`, `MOB_DIAMETER_OFFSET=0.3e-9`,
  Su82 coefficients `(0.5090, 10.526, 0.9754, 0.4767, 0.62, 2.0)`.
- `FidelityConfig` — a frozen dataclass whose defaults reproduce the
  Fortran. Populated as quirks are encountered; starts nearly empty.

**Verify:** `test_config.py` asserts every solver constant equals the value
parsed out of `solution_settings.f90`, so a vendored-reference bump that
changes a tolerance fails loudly instead of silently.

## 0.4 f2py bridge to the reference

`validation/reference/` — build an importable extension exposing the
generated Fortran leaves directly, so comparison is in-process and
term-by-term rather than by scraping stdout:

- `feval(neqn, t, c, coef, ipar) -> f` — the RHS
- `get_coll(temperature) -> K(54,54)`
- `get_evap(K, temperature) -> E(54,54)`
- `get_losses() -> cs(54)`
- `formation(c, coef, ipar) -> j_tot, j_by_charge, j_by_cluster, j_all`
- the `acdc_system` metadata getters (`get_mass`, `get_diameter`,
  `get_mob_diameter`, `get_charging_state`, `cluster_names`, `get_bound`)

Two wrinkles to handle: `feval` mutates `ipar` as a hidden "recompute rates"
side channel, so the wrapper must control `ipar` explicitly and reset it per
call; and `feval` and `formation` hold *separate* `save`d copies of the
coefficient tensors, so both must be initialised.

**Verify:** `python -c "import acdc_ref; print(acdc_ref.feval(...))"`
returns a finite `(63,)` array; calling it twice with the same input gives
the same answer (proves the `ipar`/`save` state is under control).

## 0.5 Golden capture

`validation/capture_reference.py` — write goldens to
`validation/goldens/` as plain `.npz` (no Git LFS):

- `rates_T{250,280,300,320}.npz` — `K`, `E`, `cs` at several temperatures,
  which is what pins the temperature dependence rather than just one point
- `rhs_random.npz` — `dc/dt` for ~20 log-uniform random states spanning
  1e-6 … 1e14 m⁻³, plus a physically-plausible state
- `steadystate_grid.npz` — J over the (vapour, T, CS, IPR) grid, from the
  `run` binary
- `system_metadata.npz` — masses, diameters, charge states, names, bounds

Goldens are generated once on a pinned toolchain and committed, with the
gfortran version and flags recorded in `validation/goldens/MANIFEST.md`.

**Verify:** `pytest tests/test_goldens.py` loads every golden and asserts
shapes and finiteness. Re-running capture on the same toolchain reproduces
them bit-for-bit.

## 0.6 Boundary-decision oracle

`validation/capture_boundary.py` — run
`perl fortran/perl/acdc_2020_04_28.pl --print_boundary …` for the three
cluster sets and parse the decision log into a structured
`boundary_decisions_{AN_narrow,AN,AD}.json`:

```json
{"reactants": ["2A", "2A"], "kind": "brought_back",
 "products": [["2A", 1], ["1A", 2]]}
{"reactants": ["1A", "5A5N"], "kind": "out", "channel": "out_neu"}
```

This is the oracle for the Phase 2 exact-match gate. Capturing it now, while
the harness is fresh, means Phase 2 starts with its target already in hand.

**Verify:** the parsed decision count equals the number of `boundary`
comments in the generated `acdc_equations_*.f90`, cross-checking the parser
against an independent view of the same information.
