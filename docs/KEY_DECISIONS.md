# Key decisions (ADR log)

Decisions that shaped the port, with the reasoning that produced them, so
they are not silently relitigated.

## ADR-001 — Native Python parser, not a Perl-output reader

**Decision.** Parse `.inp`, ΔH/ΔS and dipole files directly into a
`ClusterSystem` pytree. No Perl at runtime, no code generation.

**Alternatives.** (a) Keep Perl as a build step and scrape the generated
`acdc_system_*.f90` tables. (b) Both — reader first as an oracle, then the
native parser.

**Why.** Scraping keeps a Perl dependency forever and cannot explore a new
cluster set without regenerating and recompiling — which forfeits the main
reason to port. `jax_micm` made the same call against MICM's generated
sparse Jacobian.

**Cost, accepted.** The cluster-enumeration and boundary logic must be
re-derived from an 11.7k-line Perl script. Mitigated by giving the boundary
cascade its own phase and an exact-match gate against `--print_boundary`,
which is upstream's own decision log. So option (b)'s oracle is retained
where it matters, without building a second code path.

## ADR-002 — Reproduce the index-array structure; replace only the literals

**Decision.** The traced RHS mirrors the Fortran's sparse index arrays
(`ind_quad_loss/form`, `ind_lin_loss/form`, `*_extra`).

**Why.** The generated `feval` is **already data-driven** — 58 lines looping
over integer index arrays. The 10,669 lines are hardcoded *rate tables*:
`get_coll` (5076), `get_rate_coefs` (3226), `get_evap` (1283),
`initialize_parameters` (807). The structure is a gift; only the numeric
emission needs replacing, and JAX traces temperature natively.

**Consequence.** Term-by-term comparison against the reference is possible,
which is what makes the < 1e-12 RHS gate achievable.

## ADR-003 — Steady state both ways, integration as the default

**Decision.** Implement the reference's time-integration convergence test as
the parity default *and* an `optimistix.Newton` root-find on `f(c)=0`, gated
against each other at < 1e-8.

**Why.** The manual's own definition of steady state (Nomenclature, p. iv)
is exactly `f(c) = 0`, and root-finding is both faster and cleanly
differentiable through the implicit function theorem. But it can converge to
a non-physical root, and it gives up bit-level comparability with the
reference's `sstol`/`sstimech`/`sstimetot` criteria. Keeping both means the
fast path is always checkable against the faithful one.

## ADR-004 — GPL-3.0

**Decision.** License `acdc-jax` under GPL-3.0 and vendor the reference
Fortran under `fortran/`.

**Why.** ACDC is GPLv3. This port is derived from reading its sources, and
we vendor them as the validation reference, so the combined work is GPLv3
regardless of how the boundary is drawn.

**Consequence, deliberately surfaced.** This differs from the sibling ports
(GLOMAP-JAX, MAM4-JAX are permissive) and constrains downstream linking: a
permissively-licensed host cannot statically incorporate this code. A
process-boundary or data-file interface — for example generating a
formation-rate lookup table offline — avoids the question. See
`COPYRIGHT.md`.

## ADR-005 — Build flag, not a patch, for the gfortran incompatibility

**Decision.** Build the reference with `-fallow-argument-mismatch` via
`scripts/build_reference.sh`. `fortran/patches/` stays empty.

**Why.** Upstream calls the external `formation` with 8 arguments at
`driver_acdc_J.f90:359` and 5 at `:365`; gfortran ≥10 rejects this. The
`:365` branch is dead code — `small_set_mode` is a compile-time `.true.`. A
build flag is not a source change, so `fortran/` stays byte-identical to
upstream and provenance stays trivially checkable.

## ADR-006 — Setup code is NumPy, not JAX

**Decision.** Cluster enumeration, label parsing and the boundary cascade
are plain NumPy, outside `jit`.

**Why.** They involve dict lookups, string labels, and data-dependent
`while` loops with no bounded trip count. They run **once** per cluster set.
Forcing them into traced code would be a large complexity cost for zero
performance gain — the cost is entirely in the ODE solve. Only the assembled
`ClusterSystem` arrays cross into traced code.

## ADR-007 — All four physics tiers in scope, loop mode last

**Decision.** Core defaults, alternative rate parameterizations, and
chamber/experiment losses are in scope (Phases 4, 8). Loop mode is in scope
but sequenced last (Phase 9), with the decision revisited at the Phase 8
close.

**Why.** Loop mode is effectively a second model — composition-tuple
indexing, Kelvin evaporation, size-limited evaporation, J-table
interpolation, ~1500 lines interleaved through the Perl. Sequencing it last
means Phases 0–8 ship and validate independently and it can be dropped
without stranding anything.
