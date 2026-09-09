"""Build the f2py bridge to the generated ACDC equation routines.

Produces an importable ``acdc_ref`` extension so the port can be compared to
the reference **in process and term by term**, rather than by scraping the
stdout of the `run` binary. A single end-to-end number agreeing can hide two
compensating errors; per-leaf comparison cannot.

Usage::

    uv run python validation/reference/build.py

The extension lands next to this file and is gitignored -- it is a build
artifact, and the goldens captured from it are what get committed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FORTRAN_SRC = REPO / "fortran/src"

# Order matters: modules must compile before the code that uses them.
# acdc_simulation_setup is needed only as a link dependency -- feval calls
# its sources_and_constants -- and is deliberately not exposed. f2py cannot
# wrap its get_bin_limits/group_size_bins anyway, because it fails to
# resolve the module parameter `nbins` into the C wrapper's array
# dimensions ("use of undeclared identifier 'nbins'").
SOURCES = [
    "acdc_system_AN_ions_example.f90",
    "acdc_simulation_setup.f90",
    "acdc_equations_AN_ions_example.f90",
    "solvers/solution_settings.f90",
    "solvers/dvode.f",
    "driver_acdc_J.f90",
    "get_acdc_J.f90",
]

# New acdc-jax code, not part of the vendored reference.
LOCAL_SOURCES = ["plugin_shim.f90"]

# driver_acdc_J.f90 calls the external `formation` with 8 arguments at :359
# and 5 at :365; gfortran >=10 rejects that. The :365 branch is dead code
# (small_set_mode is a compile-time .true.). Same reason as
# scripts/build_reference.sh -- a build flag, not a source change.
FFLAGS = "-O3 -fallow-argument-mismatch -std=legacy"

SIGNATURE = HERE / "acdc_ref.pyf"
MODULE = "acdc_ref"


def main() -> int:
    if not SIGNATURE.exists():
        print(f"missing signature file: {SIGNATURE}", file=sys.stderr)
        return 1

    # f2py writes its build tree into the working directory, so build in a
    # temporary one and copy only the extension back.
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in SOURCES:
            src = FORTRAN_SRC / name
            if not src.exists():
                print(f"missing reference source: {src}", file=sys.stderr)
                return 1
            # fortran/ is checked out read-only; copy2 would carry that over
            # and meson needs to be able to work with these.
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, work / name)
        for name in LOCAL_SOURCES:
            shutil.copyfile(HERE / name, work / name)
        shutil.copyfile(SIGNATURE, work / SIGNATURE.name)

        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            SIGNATURE.name,
            *SOURCES,
            *LOCAL_SOURCES,
            f"--f90flags={FFLAGS}",
            f"--f77flags={FFLAGS}",
        ]
        print("$", " ".join(cmd))
        result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stderr.write(result.stdout[-4000:])
            sys.stderr.write(result.stderr[-4000:])
            print("\nf2py build FAILED", file=sys.stderr)
            return result.returncode

        built = sorted(work.glob(f"{MODULE}*.so"))
        if not built:
            print(
                f"f2py reported success but produced no {MODULE}*.so", file=sys.stderr
            )
            return 1

        for old in HERE.glob(f"{MODULE}*.so"):
            old.unlink()
        for artifact in built:
            shutil.copy2(artifact, HERE / artifact.name)
            print(f"built {HERE / artifact.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
