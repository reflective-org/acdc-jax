# Phase 10 — Pathways, figures, documentation

**Goal:** the interpretive layer and the published artefacts.

**Gate:** `uv run python scripts/make_figures.py` regenerates every figure
from scratch; the mkdocs site builds.

## 10.1 Growth-pathway extraction

The MATLAB-only capability that the QuickGuide's entire setup methodology
depends on (`track_fluxes.m`, `plotflux.m`, `plotflux_out.m`):

1. significant exit channels from the flux matrix (`crit_out = 0.05`)
2. charge-filter each, attributing the pathway to the **growing** cluster
   rather than the condensing monomer
3. backward breadth-first traversal to the source, with a visited set
4. greedy argmax back-walk to extract the single dominant chain

Pure combinatorics on the Phase 5.4 flux matrix; runs once per solved state,
not per timestep. Plain NumPy — no reason for this to be traced.

**Verify:** the dominant route for the AN system at reference conditions
matches the published one.

## 10.2 Source back-solve
Monomer source terms inferred from the steady-state fluxes — an inverse
diagnostic with no Fortran analogue (manual p. 37).

## 10.3 ΔG surfaces
Reference→actual free-energy conversion,
`ΔG_act = ΔG_ref − k_BT(n_A·ln(p_A/p_ref) + n_B·ln(p_B/p_ref))`, with
charged-monomer reference-state renormalisation. The system-adequacy check
the QuickGuide prescribes. Plain NumPy.

## 10.4 Figures

`scripts/make_figures.py`, reproducible from a clean checkout:

1. **J vs [H₂SO₄]** with the Fortran overlay — the headline parity result
2. **Cluster distribution** at steady state, by charge
3. **∂J/∂ΔG sensitivity** — which clusters actually control the rate
4. **vmap scaling** — wall-clock vs batch size against the serial loop
5. **Growth pathway** in composition space, with the system boundary
6. **Evaporation-rate map** — the QuickGuide's system-adequacy check

Follow the house dataviz conventions; every figure carries its generating
command in the caption.

## 10.5 Documentation

`docs/physics.md` (equations with Perl/Fortran line citations),
`docs/api.md`, `docs/fidelity.md`, `docs/porting-notes.md`,
`docs/validation.md`, `docs/KEY_DECISIONS.md`. mkdocs-material, published.
