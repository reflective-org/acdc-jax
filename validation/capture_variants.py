"""Capture rate-coefficient goldens for generator options the bundled example lacks.

The shipped example was generated with Su82 ion enhancement and no wall or
dilution losses. Every other option -- Su73, constant, later the losses and
hydrates -- needs its own generated fixture. Rather than build an f2py module
per variant, this regenerates each with Perl into a gitignored directory,
evaluates the emitted ``K(i,j)`` expressions across the temperature sweep
with :mod:`reference.emitted`, and commits the resulting matrices as small
``.npz`` goldens. Tests then need neither Perl nor gfortran.

The evaluator is itself checked against the f2py bridge on the shipped
example (``test_reference_bridge.py``); it reproduces it to 0.0 relative.

Usage::

    uv run python validation/capture_variants.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FORTRAN = REPO / "fortran"
GENERATED = HERE / "fixtures/generated"
GOLDENS = HERE / "goldens"

sys.path.insert(0, str(HERE / "reference"))
import emitted  # noqa: E402

GENERATOR = FORTRAN / "perl/acdc_2020_04_28.pl"
INPUT = FORTRAN / "src/Perl_input/input_ANnarrow_neutral_neg_pos.inp"
ENERGY = FORTRAN / "src/Perl_input/HS298.15K_example.txt"
DIPOLE = FORTRAN / "src/Perl_input/dip_pol_298.15K_example.txt"

TEMPERATURES = (250.0, 280.0, 298.15, 320.0)
NCLUST = 54

# Each variant: the extra generator flags that select it. The base
# invocation is run_perl.sh's, so everything else matches the shipped example.
VARIANTS: dict[str, list[str]] = {
    "su73": ["--ion_coll_method", "Su73"],
    "constant": ["--ion_coll_method", "constant"],
}

BASE_FLAGS = [
    "--fortran",
    "--save_outgoing",
    "--variable_cs",
    "--cs",
    "exp_loss",
    "--exp_loss_exponent",
    "-1.6",
    "--e",
    str(ENERGY),
    "--dip",
    str(DIPOLE),
    "--variable_temp",
    "--variable_ion_source",
    "--cs_only",
    "1A,0",
    "--cs_only",
    "1N,0",
    "--i",
    str(INPUT),
    "--exp_loss_ref_cluster",
    "1A",
]


def generate(name: str, flags: list[str]) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    cmd = ["perl", str(GENERATOR), *BASE_FLAGS, *flags, "--append", f"_{name}"]
    result = subprocess.run(cmd, cwd=GENERATED, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"generator failed for variant {name}")
    return GENERATED / f"acdc_equations_{name}.f90"


def main() -> int:
    if shutil.which("perl") is None:
        print("perl not found on PATH", file=sys.stderr)
        return 1
    GOLDENS.mkdir(exist_ok=True)

    for name, flags in VARIANTS.items():
        equations = generate(name, flags)
        data = {"temperatures": np.asarray(TEMPERATURES)}
        for t in TEMPERATURES:
            data[f"K_{t:g}"] = emitted.collision_matrix(equations, t, NCLUST)
        path = GOLDENS / f"rates_variant_{name}.npz"
        np.savez_compressed(path, **data)
        nonzero = np.count_nonzero(data["K_280"])
        print(f"wrote {path.relative_to(REPO)}  (K nonzero at 280 K: {nonzero})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
