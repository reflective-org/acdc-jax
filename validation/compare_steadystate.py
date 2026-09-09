"""Sweep the full steady-state grid against the Fortran goldens.

The headline validation of the port. The test suite checks a representative
slice to keep the suite fast; this walks all 60 points.

Usage::

    uv run python validation/compare_steadystate.py
    uv run python validation/compare_steadystate.py --method rootfind
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from acdc_jax import rates, rhs, solve
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set
from acdc_jax.reactions import enumerate_reactions
from acdc_jax.system import build_system
from acdc_jax.thermo import parse_dipole_file, parse_energy_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = HERE / "goldens"

GATE = 1e-5
"""Against the reference. See docs/validation.md for why not tighter."""


def build():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    boundary = BoundarySystem(cluster_set)
    reactions = enumerate_reactions(system, boundary)
    energies = parse_energy_file(
        INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
    )
    inputs = rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )
    return system, reactions, inputs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default="integrate", choices=["integrate", "rootfind"]
    )
    args = parser.parse_args(argv[1:])

    path = GOLDENS / "steadystate.npz"
    if not path.exists():
        print(
            "goldens not captured; run validation/capture_steadystate.py",
            file=sys.stderr,
        )
        return 1
    goldens = np.load(path)
    conditions, expected = goldens["conditions"], goldens["j"]

    system, reactions, inputs = build()

    print(f"{len(conditions)} points, method={args.method}, gate={GATE:g}")
    print(
        f"{'[A] cm^-3':>12} {'[N] cm^-3':>12} {'T':>6} {'CS':>8} "
        f"{'J port':>12} {'J ref':>12} {'rel':>10}"
    )

    worst = 0.0
    failures = 0
    started = time.time()

    for row, (c_a, c_n, temperature, cs, ipr) in enumerate(conditions):
        coefficients = rhs.assemble(
            system,
            reactions,
            inputs,
            float(temperature),
            float(cs),
            float(ipr),
            float(ipr),
        )
        c0 = solve.set_vapours(
            system, jnp.zeros(system.n_equations), {"1A": float(c_a), "1N": float(c_n)}
        )
        result = solve.solve_steady_state(
            system, coefficients, method=args.method, c0=c0
        )
        got = float(result.j_total)
        want = float(expected[row])
        relative = abs(got - want) / want
        worst = max(worst, relative)
        flag = "" if relative < GATE else "  <-- FAIL"
        if relative >= GATE:
            failures += 1
        print(
            f"{c_a * 1e-6:12.2e} {c_n * 1e-6:12.2e} {temperature:6.0f} {cs:8.0e} "
            f"{got * 1e-6:12.4e} {want * 1e-6:12.4e} {relative:10.2e}{flag}"
        )

    elapsed = time.time() - started
    print()
    print(
        f"worst relative difference: {worst:.3e}   ({elapsed:.0f}s, "
        f"{elapsed / len(conditions):.1f}s per point)"
    )
    print("PASS" if failures == 0 else f"FAIL: {failures} of {len(conditions)} points")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
