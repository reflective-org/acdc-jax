# Provenance of the vendored Fortran reference

`fortran/` is a copy of the Fortran distribution of **ACDC — Atmospheric
Cluster Dynamics Code**, taken from the upstream repository at a pinned
commit. It is the validation reference for this port and is **read-only**.

## Upstream

| | |
|---|---|
| Repository | <https://github.com/tolenius/ACDC> |
| Commit | `870b82a2a9e354189ba6e507680421bd13e6c5a7` |
| Commit date | 2026-06-12 |
| Commit subject | `Delete acdc_2022_10_05.pl` |
| Licence | GNU General Public License v3.0 |
| Code developer | Tinja Olenius |

## File-by-file origin

Every file under `fortran/` is byte-identical to its upstream counterpart
unless a patch in `fortran/patches/` says otherwise.

| Path in this repo | Path upstream |
|---|---|
| `fortran/src/driver_acdc_J.f90` | `ACDC_Fortran_standard/driver_acdc_J.f90` |
| `fortran/src/get_acdc_J.f90` | `ACDC_Fortran_standard/get_acdc_J.f90` |
| `fortran/src/run_acdc_J_example.f90` | `ACDC_Fortran_standard/run_acdc_J_example.f90` |
| `fortran/src/acdc_simulation_setup.f90` | `ACDC_Fortran_standard/acdc_simulation_setup.f90` |
| `fortran/src/acdc_system_AN_ions_example.f90` | `ACDC_Fortran_standard/acdc_system_AN_ions_example.f90` |
| `fortran/src/acdc_equations_AN_ions_example.f90` | `ACDC_Fortran_standard/acdc_equations_AN_ions_example.f90` |
| `fortran/src/Makefile` | `ACDC_Fortran_standard/Makefile` |
| `fortran/src/run_perl.sh` | `ACDC_Fortran_standard/run_perl.sh` |
| `fortran/solvers/dvode.f` | `ACDC_Fortran_standard/solvers/dvode.f` |
| `fortran/solvers/solution_settings.f90` | `ACDC_Fortran_standard/solvers/solution_settings.f90` |
| `fortran/solvers/dvode_acknowledgement.txt` | `ACDC_Fortran_standard/solvers/dvode_acknowledgement.txt` |
| `fortran/perl/acdc_2024_02_12.pl` | `acdc_2024_02_12.pl` (repository root) |
| `fortran/perl/acdc_2020_04_28.pl` | `ACDC_Fortran_standard/acdc_2020_04_28.pl` |
| `fortran/inputs/*` | `ACDC_Fortran_standard/Perl_input/*` |
| `fortran/cluster_sets/*` | `ACDC_Matlab_standard/ACDC_main/Cluster_set_files/B3LYP_RICC2/*` |

### Why two Perl versions are vendored

`acdc_2024_02_12.pl` is the current generator and the one the port targets.
`acdc_2020_04_28.pl` is kept because the committed example equation files
(`acdc_equations_AN_ions_example.f90`, `acdc_system_AN_ions_example.f90`)
were produced by *that* version, not the 2024 one. They differ in emitted
detail — for instance the 2024 generator emits an `ipar(4)` override block
that is absent from the committed example. Reproducing the committed files
byte-for-byte requires the 2020 generator; validating against current
upstream behaviour requires the 2024 one.

### Regeneration is reproducible to 1e-14

`fortran/src/run_perl.sh`'s invocation, replayed with the 2020 generator
(see `validation/capture_boundary.py` for the exact command line),
regenerates the committed `acdc_equations_AN_ions_example.f90` with:

| | |
|---|---|
| Line count | identical (10,669) |
| Structural differences | **none** — every cluster, reaction and index array entry matches |
| Differing lines | 71 of 10,669 (0.7%) |
| Max relative difference | **9.8e-15** |

The differences are last-digit rounding in the `sprintf('%.14e', ...)`
literals, presumably a Perl or libm version difference. That is 100x inside
the 1e-12 rate-constant acceptance gate, so the committed files and a fresh
regeneration are interchangeable as validation targets.

This also confirms the reconstructed command line is correct, which matters
because `run_perl.sh` builds it from shell variables rather than stating it.

### Not vendored

The MATLAB distribution (`ACDC_Matlab_standard/`, except the cluster-set
input files listed above) is not copied. Its capabilities that have no
Fortran equivalent — the net-flux matrix, growth-pathway tracking, ΔG
surfaces — are reimplemented natively in `src/acdc_jax/` rather than
ported from MATLAB. The two PDF manuals are not vendored either; they are
available from the upstream repository.

## Thermodynamic input data

`fortran/inputs/HS298.15K_example.txt` and `dip_pol_298.15K_example.txt`
contain B3LYP//RICC2 data from:

> Olenius, Kupiainen-Määttä, Ortega, Vehkamäki, Riipinen: *Free energy
> barrier in the growth of sulfuric acid–ammonia and sulfuric
> acid–dimethylamine clusters*, J. Chem. Phys. **139**, 084312 (2013).
> <https://doi.org/10.1063/1.4819024>

Upstream's own `Perl_input/README.txt` notes that newer quantum-chemical
data sets are recommended for quantitative work, and that cluster set files
must be revised when the thermochemistry input changes.
