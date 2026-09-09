"""Replay the boundary oracle against the port. The Phase 2 gate.

Every decision the Perl generator logged under ``--print_boundary`` is
replayed through ``acdc_jax.boundary`` and compared **exactly**. Not a
tolerance: the cascade is combinatorial, so a decision either matches or it
does not, and a mismatch means a different model rather than a rounding
difference.

Usage::

    uv run python validation/compare_boundary.py            # all sets
    uv run python validation/compare_boundary.py AN_narrow  # one set
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from acdc_jax import labels
from acdc_jax.boundary import BoundarySystem
from acdc_jax.clusterset import parse_cluster_set

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLDENS = HERE / "goldens"

CLUSTER_SETS = {
    "AN_narrow": REPO / "fortran/src/Perl_input/input_ANnarrow_neutral_neg_pos.inp",
    "AN": REPO / "fortran/cluster_sets/input_AN_neutral_neg_pos.inp",
    "AD": REPO / "fortran/cluster_sets/input_AD_neutral_neg_pos.inp",
    "AN_narrow_nostrength": REPO / "validation/fixtures/input_ANnarrow_nostrength.inp",
}


def _counts(system: BoundarySystem, label: str) -> tuple[int, ...] | None:
    """Composition vector for a label, or None for a generic charger ion.

    Generic ions have no composition and cannot be fed to the cascade; the
    reference handles them before it is reached.
    """
    if labels.is_generic_ion(label):
        return None
    return tuple(labels.composition_vector(label, system.order))


def replay(name: str) -> tuple[int, int, list[str]]:
    """Replay one cluster set. Returns (checked, skipped, failures)."""
    oracle = json.loads((GOLDENS / f"boundary_{name}.json").read_text())
    system = BoundarySystem(parse_cluster_set(CLUSTER_SETS[name]))

    checked = skipped = 0
    failures: list[str] = []

    for decision in oracle["decisions"]:
        reactants = decision["reactants"]
        if any(labels.is_generic_ion(r) for r in reactants):
            # Charge transfer from a generic ion is decided before the
            # cascade runs; combine() has no composition to work with.
            skipped += 1
            continue

        raw = decision.get("raw_product")
        if raw is None:
            skipped += 1
            continue

        product = _counts(system, raw)
        if product is None:
            skipped += 1
            continue

        checked += 1
        expected_kind = decision["kind"]

        if system.is_member(product):
            if expected_kind != "unused":
                failures.append(
                    f"{'+'.join(reactants)} -> {raw}: in system, "
                    f"but oracle says {expected_kind}"
                )
            continue

        try:
            result = system.check_boundary(product)
        except ValueError as exc:
            failures.append(f"{'+'.join(reactants)} -> {raw}: raised {exc}")
            continue

        if expected_kind == "out":
            channel = {1: "out_neu", 2: "out_neg", 3: "out_pos"}.get(result.lout)
            if channel != decision["channel"]:
                failures.append(
                    f"{'+'.join(reactants)} -> {raw}: got {channel}, "
                    f"expected {decision['channel']}"
                )
        elif expected_kind == "brought_back":
            expected = _expected_products(decision)
            got = _actual_products(result, system)
            if got != expected:
                failures.append(
                    f"{'+'.join(reactants)} -> {raw}: got {got}, expected {expected}"
                )

    return checked, skipped, failures


def _expected_products(decision: dict) -> dict[str, int]:
    """The oracle's product bag: main cluster plus stripped monomers."""
    main, *stripped = decision["products"]
    out = {"__main__": main[0]}
    for label, mult in stripped:
        out[label] = out.get(label, 0) + mult
    return out


def _actual_products(result, system: BoundarySystem) -> dict[str, int]:
    out = {"__main__": result.label}
    for name, count in result.monomers.items():
        if count:
            out[name] = out.get(name, 0) + count
    return out


def main(argv: list[str]) -> int:
    names = argv[1:] or list(CLUSTER_SETS)
    total_failures = 0

    for name in names:
        if not (GOLDENS / f"boundary_{name}.json").exists():
            print(f"{name}: oracle not captured, skipping", file=sys.stderr)
            continue
        checked, skipped, failures = replay(name)
        status = "OK" if not failures else f"{len(failures)} MISMATCHES"
        print(f"{name:<10} checked={checked:<5} skipped={skipped:<4} {status}")
        for line in failures[:15]:
            print(f"    {line}")
        if len(failures) > 15:
            print(f"    ... and {len(failures) - 15} more")
        total_failures += len(failures)

    print()
    print("PASS" if total_failures == 0 else f"FAIL: {total_failures} mismatches")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
