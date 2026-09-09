"""Capture the steady-state J grid from the Fortran reference.

Each grid point runs in a **fresh subprocess**. That is not paranoia:
``acdc_plugin`` keeps the concentration vector in a ``save``d array
(``get_acdc_J.f90:31``) and warm-starts from it, so the answer depends on
what was solved before it in the same process.

Measured on this machine: cold-start J is bit-identical across processes
(2217995.192415948 every time for the bundled example), while re-solving the
same point after a different one in the same process shifts J by ~1.5e-6
relative. That is not solver noise -- it is the steady-state criterion
being a *relative-change* test (``sstol = 1e-5``) rather than an exact root,
so where you start determines where inside the tolerance band you stop.

The consequence for validation: the reference's steady-state J is only
*defined* to within sstol. Gating the port against it more tightly than
~1e-5 would be gating against an arbitrary point in that band. See
docs/validation.md.

Usage::

    uv run python validation/capture_steadystate.py
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLDENS = HERE / "goldens"

CM3 = 1e6  # cm^-3 -> m^-3

# Grid. [H2SO4] spans the atmospherically interesting range where J turns
# over from evaporation-limited to collision-limited; the rest are the
# conditions a host model would vary.
C_A = np.array([1e6, 3e6, 1e7, 3e7, 1e8]) * CM3  # m^-3
C_N = np.array([1e9, 1e10]) * CM3  # m^-3
TEMPERATURE = np.array([250.0, 280.0, 300.0])  # K
CS_REF = np.array([1e-3, 1e-2])  # 1/s
IPR = np.array([3.0]) * CM3  # 1/m^3/s

# Runs one point and prints the repr of the two doubles, so the parent gets
# full precision back. stdout is redirected because acdc_plugin prints a
# banner on its first call.
WORKER = """
import sys, os
sys.path.insert(0, {refdir!r})
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
import acdc_ref
j, d = acdc_ref.solve_j(*[float(x) for x in sys.argv[1:6]])
os.write(2, f"{{j!r}} {{d!r}}".encode())
"""


def solve_cold(
    c_a: float, c_n: float, cs: float, t: float, ipr: float
) -> tuple[float, float]:
    """Solve one point in a fresh process, so no warm-start state leaks in.

    Arguments are passed as ``repr(float(x))`` -- full round-trip precision,
    and ``float()`` first because NumPy 2 reprs a scalar as
    ``np.float64(1.0)``, which the child cannot parse back.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            WORKER.format(refdir=str(HERE / "reference")),
            *(repr(float(x)) for x in (c_a, c_n, cs, t, ipr)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    j_str, d_str = result.stderr.strip().split()
    return float(j_str), float(d_str)


# The bundled example (fortran/src/run_acdc_J_example.f90), whose answer the
# `run` binary prints. Any argument-order or unit slip in this script shows
# up here immediately, which is the point -- an earlier version of this file
# swapped temperature and cs_ref and produced a J field that looked
# perfectly plausible until checked against this number.
EXAMPLE_POINT = dict(c_a=1e7 * CM3, c_n=1e9 * CM3, cs=1e-3, t=280.0, ipr=3.0 * CM3)
EXAMPLE_J = 2217995.192415948  # m^-3 s^-1, i.e. 2.218 cm^-3 s^-1


def check_example() -> None:
    """Fail loudly before spending time on a grid that would be wrong."""
    j, _ = solve_cold(**EXAMPLE_POINT)
    if not np.isclose(j, EXAMPLE_J, rtol=1e-12):
        raise SystemExit(
            f"reference sanity check FAILED\n"
            f"  bundled example expected J = {EXAMPLE_J!r} m^-3 s^-1\n"
            f"  got                        {j!r}\n"
            f"  ({j / EXAMPLE_J:.6f}x). Check argument order and units."
        )
    print(f"sanity check: bundled example reproduces J = {j * 1e-6:.6f} cm^-3 s^-1")


def main() -> int:
    GOLDENS.mkdir(exist_ok=True)
    check_example()
    grid = list(itertools.product(C_A, C_N, TEMPERATURE, CS_REF, IPR))
    print(f"solving {len(grid)} points, one process each")

    conditions = np.array(grid)
    j = np.empty(len(grid))
    diameter = np.empty(len(grid))
    for i, (c_a, c_n, temperature, cs_ref, ipr) in enumerate(grid):
        # Unpacked and passed by keyword deliberately. Positional unpacking
        # here silently swapped temperature and cs_ref -- the grid is
        # ordered to match `condition_names`, which is not the argument
        # order of solve_j. It ran happily and produced a plausible-looking
        # J field three orders too small.
        j[i], diameter[i] = solve_cold(
            c_a=c_a, c_n=c_n, cs=cs_ref, t=temperature, ipr=ipr
        )
        if (i + 1) % 10 == 0 or i + 1 == len(grid):
            print(f"  {i + 1}/{len(grid)}")

    path = GOLDENS / "steadystate.npz"
    np.savez_compressed(
        path,
        conditions=conditions,
        condition_names=np.array(["c_A", "c_N", "temperature", "cs_ref", "ipr"]),
        j=j,
        diameter=diameter,
    )
    print(f"\nwrote {path.relative_to(REPO)}")
    print(f"J range: {j.min() * 1e-6:.3e} to {j.max() * 1e-6:.3e} cm^-3 s^-1")
    n_zero = int((j <= 1e-99).sum())
    if n_zero:
        print(f"  ({n_zero} points at the 1e-100 clamp -- see fidelity F4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
