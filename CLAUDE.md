# CLAUDE.md — Project Instructions

## General working guidelines

These guidelines bias toward caution over speed. For trivial tasks, use
judgment.

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, and clarifying questions come before
implementation rather than after mistakes.

---

## Project-specific rules

`acdc-jax` is a faithful JAX port of ACDC (Atmospheric Cluster Dynamics
Code; Kupiainen-Määttä & Olenius; GPLv3). These rules are not style
preferences — each exists because breaking it produces a result that looks
plausible and is wrong.

### Precision

**float64 everywhere.** `jax_enable_x64` is set in exactly one place,
`src/acdc_jax/__init__.py`. Do not add redundant config calls, and never
disable it. Cluster concentrations span ~30 orders of magnitude and
evaporation rates are exponentials of ΔG/kT; float32 is a different
trajectory, not a noisier one. float32 is permitted only as an explicitly
labelled benchmark mode, never as a validated trajectory.

### Fidelity to the reference

**Port from the code, not the comments, and not the manual.** Where the
Fortran, the Perl, and the manual disagree, the generated Fortran is the
specification. Known three-way disagreements are catalogued in
`docs/fidelity.md` (e.g. the steady-state tolerance is `sstol=1e-5` in
Fortran, `difftol=1e-3` in MATLAB, and documented as `1e-5` in the manual).

**Every upstream quirk gets a `FidelityConfig` flag whose default
reproduces the Fortran**, plus a `docs/fidelity.md` entry and a test at both
settings. Getting a default backwards silently changes results. "Obviously a
bug, so I fixed it" is how a port stops being a port.

**`fortran/` is read-only.** It is a vendored copy of `tolenius/ACDC` — see
`PROVENANCE.md`. Divergence only via `fortran/patches/NNNN-*.patch`, each
with a rationale and a demonstration that reference output is unchanged.

### JAX

**No Python branching on traced values.** Use `jnp.where`.

**Every guarded division needs the double-where idiom:**

```python
safe_den = jnp.where(cond, den, 1.0)
out = jnp.where(cond, num / safe_den, 0.0)
```

A single `where` around a division still evaluates the unsafe branch and
gives NaN cotangents under reverse-mode AD.

**Mask before reducing, never multiply.** Use `jnp.where(mask, term, 0.0)`
ahead of a sum, not `mask * term` — rate coefficient arrays are evaluated
over the full `(nclust, nclust)` extent and `0.0 * inf = NaN`.

**Port first, `jit` second, in separate commits.** float64, no `jit`, no
`vmap`; validate against the reference; only then optimise. The eager driver
stays permanently — it is the debugger, because it can raise real
exceptions.

**Setup code is NumPy, not JAX.** Cluster enumeration and the boundary
cascade run once and involve dict lookups, string labels and data-dependent
`while` loops. They belong in plain NumPy outside `jit`. Only the assembled
`ClusterSystem` arrays cross into traced code.

### Numerical parity is a feature

Changes to the physics or solver math must re-run the validation harness and
keep the acceptance thresholds in `docs/validation.md`. The boundary-cascade
gate is **exact match** against `perl acdc.pl --print_boundary`, not a
tolerance — it is combinatorial, and any divergence is a different model.

**Never skip or hide failures** in tests, benchmarks, or validation; fail
loudly with informative errors.

### Repository conventions

- `uv` manages the virtualenv. `uv sync --extra dev`, `uv run pytest`.
  Never `pip install` into a system interpreter.
- All constants live in `src/acdc_jax/config.py`. No magic numbers in the
  physics modules.
- Planning docs live in `docs/plan/`: `ULTRAPLAN.md`, one `phase-N-*.md`
  per phase, and `PROGRESS.md` as the live status log. Each task `N.M` is
  one commit.
- **No Claude attribution** (no `Co-Authored-By` / `Generated with` lines)
  in commits or PRs.
