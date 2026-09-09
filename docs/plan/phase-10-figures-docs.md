# Phase 10 — Pathways, figures, documentation

**Goal:** the interpretive layer and the published artefacts.

**Gate:** `uv run python scripts/make_figures.py` regenerates every figure
from scratch; the mkdocs site builds.

## 10.1 Growth-pathway extraction ✅

The MATLAB-only capability that the QuickGuide's entire setup methodology
depends on (`track_fluxes.m`, `plotflux.m`, `plotflux_out.m`):

1. significant exit channels from the flux matrix (`crit_out = 0.05`)
2. charge-filter each, attributing the pathway to the **growing** cluster
   rather than the condensing monomer
3. backward breadth-first traversal to the source, with a visited set
4. greedy argmax back-walk to extract the single dominant chain

Pure combinatorics on the Phase 5.4 flux matrix; runs once per solved state,
not per timestep. Plain NumPy — no reason for this to be traced.

**Verify:** the dominant route for the AN system at the QuickGuide's
conditions (5·10⁶ cm⁻³ acid, 100 ppt NH₃, 280 K) exits through 5A5N →
6A5N, the same exit the QuickGuide's slide 6 shows, running along and
below the acid = base diagonal. Ported in `acdc_jax.pathways` on the
port's explicit reaction list (no label-combining search); boundary
collisions are attributed to the growing collider where MATLAB parks them
on a 'bound' node it then declines to track.

## 10.2 Source back-solve ✅
Monomer source terms inferred from the steady-state fluxes — an inverse
diagnostic with no Fortran analogue (manual §4.3.3 `Sources_out`, Perl
`:5731-5741`). `pathways.monomer_sources`: row sum minus column sum of the
gross flux matrix for neutral monomers, row sum for the charger ions.
Tested by re-inserting the sources with the monomers free and checking the
state stays stationary, and by recovering the ion production rate.

## 10.3 ΔG surfaces ✅
Reference→actual free-energy conversion,
`ΔG_act = ΔG_ref − k_BT(n_A·ln(p_A/p_ref) + n_B·ln(p_B/p_ref))`, with
charged-monomer reference-state renormalisation. The system-adequacy check
the QuickGuide prescribes. Plain NumPy. `free_energy.actual_free_energy`:
only neutral molecule types carry a partial-pressure term ("the ion is not
a molecule", `rates_and_deltags_ABe.m:483`); `composition_grid` lays any
per-cluster quantity on the (acid, base) chart with the bisulfate ion
folded into the acid count.

## 10.4 Figures ✅

`scripts/make_figures.py`, reproducible from a clean checkout:

1. **J vs [H₂SO₄]** with the Fortran overlay — the headline parity result
2. **Cluster distribution** at steady state, by charge
3. **∂J/∂ΔG sensitivity** — which clusters actually control the rate
4. **vmap scaling** — wall-clock vs batch size against the serial loop
5. **Growth pathway** in composition space, with the system boundary
6. **Evaporation-rate map** — the QuickGuide's system-adequacy check

Follow the house dataviz conventions; every figure carries its generating
command in the caption.

## 10.5 Documentation ✅

`docs/physics.md` (equations with Perl/Fortran line citations),
`docs/api.md`, `docs/fidelity.md`, `docs/porting-notes.md`,
`docs/validation.md`, `docs/KEY_DECISIONS.md`. mkdocs-material with
mkdocstrings (`mkdocs.yml`, `docs/api.md`); `uv run mkdocs build --strict`
is the gate. Publishing to GitHub Pages is a repository setting, left to
the owner.
