# Vendored ACDC reference — READ ONLY

This directory is a pinned copy of the upstream ACDC Fortran distribution
and Perl equation generator. See `../PROVENANCE.md` for the exact commit and
file-by-file origin.

**Do not edit these files.** They are the specification the port is
validated against; editing them silently moves the target. Files are marked
read-only on checkout as a reminder.

If a change is genuinely required (a compiler incompatibility, a build
guard), add a patch under `patches/` as `NNNN-short-description.patch`, with
a header comment giving the rationale and a demonstration that reference
output is unchanged. Apply patches in the build, never in place.

## Building the reference

    make -C src

produces `src/run`, the steady-state example driver. It solves the bundled
sulfuric acid–ammonia system with ions (54 clusters, 63 equations).

## Regenerating the equation files

`src/run_perl.sh` regenerates `acdc_equations_*.f90` and `acdc_system_*.f90`
from `inputs/`. Note that the committed example files were produced by
`perl/acdc_2020_04_28.pl`, not the 2024 generator — see `../PROVENANCE.md`.
