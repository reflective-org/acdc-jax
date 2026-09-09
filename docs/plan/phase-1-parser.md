# Phase 1 — Parser: cluster sets and thermodynamics as data

**Goal:** replace the Perl generator's input-reading half. `.inp` +
ΔH/ΔS + dipole/polarizability tables → plain NumPy structures.

**Gate:** reproduces the 54-cluster AN set and all 117 energy entries; the
composition matrix matches the KEY block of the generated Fortran
(`acdc_equations_AN_ions_example.f90:1-65`).

All of this is setup code: plain NumPy, dict lookups, string labels. It runs
once and never enters `jit`.

## 1.1 Molecule-property header

`clusterset.py` — parse the `.inp` header. Directives are identified by
**regex on the row label, not by column position** (Perl `:1097-1290`):
`name`, `charge`, `corresponding neutral molecule`, `corresponding
negative/positive ion`, `mass [g/mol]`, `density [kg/m**3]`, `acid
strength`, `base strength`, and the optional `saturation vapor pressure`,
`surface tension`, `can be lost`, `can evaporate`.

Two traps: a **negative mass marks the "missing proton" pseudo-species**
(Perl `:1216`), and `corresponding positive ion` may name a *cluster*
(`N` → `1N1P`), not a molecule.

**Verify:** all 5 molecule types of the AN file parsed with correct charge,
mass, density, and strengths; the proton is identified as `P`.

## 1.2 Cluster-set body and range expansion

Any line with exactly `n_mol_types` numeric columns is a cluster row.
Exactly one column may carry `lo-hi`, expanded into a run; the range column
need not be the first (Perl `:1451-1457`); a bare single cluster is
rewritten as `n-n`.

Also the `out neutral/negative/positive` rows — these are **inequality
thresholds, not clusters**: a product counts as grown out when *every*
molecule count meets or exceeds the rule, and multiple rules per charge are
OR'd (Perl `:10967-10972`).

**Verify:** 54 clusters in the documented order — 1–16 neutral, 17–34
negative, 35–52 positive, 53 `neg`, 54 `pos`; composition matrix equals the
KEY block.

## 1.3 Cluster labels and canonicalisation

`parse_label("2A1N") -> {A:2, N:1}` and the inverse, with canonical
molecule ordering so `1N2A` and `2A1N` are the same cluster
(Perl `get_cluster_number` `:11263`, `compare_clusters` `:11340`).

**Verify:** round-trip on all 54 labels plus the 117 energy-file labels;
permuted labels resolve to the same index.

## 1.4 Geometry

`m = Σ nᵢmᵢ`; `V = Σ nᵢmᵢ/ρᵢ` (additive bulk liquid volumes);
`r = (3V/4π)^⅓`; `d_mob = (d + 0.3 nm)·√(1 + 28.8·m_conv/m)`.
The proton and missing-proton contribute **mass but not volume**
(Perl `:11498`, `:11566`).

**Verify:** masses and diameters match `get_mass`/`get_diameter`/
`get_mob_diameter` from `acdc_system_AN_ions_example.f90` to < 1e-10
relative — noting the emitted values are rounded to 2 d.p., so compare
against the full-precision recomputation and check the rounding agrees.

## 1.5 Energy file

Line 1 = reference pressure (Pa), line 2 = reference temperature (K). Data
lines are `label ΔH ΔS` (kcal/mol, cal/mol/K) or `label ΔG` (kcal/mol).
`G = H − T·S`. Monomers default to zero and need an entry only if nonzero.

Hard errors, matching upstream: duplicate definition dies; a cluster in the
set with no energy and not a monomer dies with an accumulated message
(Perl `:2521`, `:2560-2590`).

**Verify:** 117 entries parsed; ΔG at 298.15 K reproduces the values inlined
in `get_evap`; a duplicate and a missing entry each raise.

## 1.6 Dipole/polarizability file

Line 1 and 2 are dipole **locking coefficients** for monomers and clusters
(used only by Su73). Data lines are `label μ[Debye] α[Å³]`. Only
electrically neutral clusters need entries; a missing one dies
(Perl `:8525-8538`).

**Verify:** 53 entries; every neutral cluster in the set has data.
