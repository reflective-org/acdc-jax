# Copyright and attribution

This repository contains code from two sources. Both are distributed under
the GNU General Public License v3.0, the terms of which are in `LICENSE`.

## Vendored Fortran and Perl — `fortran/`

Copyright (c) Tinja Olenius and Oona Kupiainen-Määttä.

`fortran/` is a copy of the Fortran distribution and Perl equation generator
of ACDC, taken from [tolenius/ACDC](https://github.com/tolenius/ACDC) at
commit `870b82a`. See `PROVENANCE.md` for the exact upstream commit and the
file-by-file origin. It is read-only; divergence only through
`fortran/patches/`.

`fortran/solvers/dvode.f` is the VODE ODE solver from Lawrence Livermore
National Laboratory, retained with its original acknowledgement file:

> Brown, P. N., Byrne, G. D., and Hindmarsh, A. C.: *VODE, a
> variable-coefficient ODE solver*, SIAM J. Sci. Stat. Comput. **10**,
> 1038–1051 (1989). <https://doi.org/10.1137/0910062>

## New Python/JAX code — `src/`, `tests/`, `validation/`, `scripts/`, `benchmarks/`

Copyright (c) 2026 Reflective.

This is an independent reimplementation of the ACDC model in JAX. Because it
is derived from reading the GPLv3 sources vendored here, it is itself
licensed under GPL-3.0.

## Why this repository is GPL-3.0

The other JAX ports in this family (GLOMAP-JAX, MAM4-JAX) carry permissive
licences because their upstreams do. ACDC is GPLv3, so `acdc-jax` is too.
This constrains downstream linking: a permissively-licensed host model
cannot statically incorporate this code without itself becoming GPL. Plan
integrations accordingly — a process-boundary or data-file interface (for
example, generating a formation-rate lookup table offline) avoids the
question entirely.

## Citing

If you use this code, cite the original model as well as this port. See
`CITATION.cff`, and at minimum:

> Olenius, T., Kupiainen-Määttä, O., Ortega, I. K., Kurtén, T., Vehkamäki,
> H.: *Free energy barrier in the growth of sulfuric acid–ammonia and
> sulfuric acid–dimethylamine clusters*, J. Chem. Phys. **139**, 084312
> (2013). <https://doi.org/10.1063/1.4819024>
