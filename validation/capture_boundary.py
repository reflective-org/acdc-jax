"""Capture the Perl generator's boundary decisions as a structured oracle.

The boundary cascade decides, for every collision whose product falls
outside the enumerated cluster set, whether it counts as grown out
(contributing to J) or is stripped back into the set -- and if stripped,
into exactly what. It is ~250 lines of stateful, order-dependent Perl
(`:10907-11256` in the 2024 generator) with no closed-form specification:
the mapping is defined operationally by the algorithm.

`--print_boundary` makes the generator log every decision it takes. That log
is the exact-match target for Phase 2. Capturing it here, in Phase 0, means
Phase 2 starts with its gate already in hand.

Usage::

    uv run python validation/capture_boundary.py

Requires `perl` on PATH. Output: validation/goldens/boundary_<set>.json.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FORTRAN = REPO / "fortran"
GOLDENS = HERE / "goldens"

# The 2020 generator, not the 2024 one: the committed example equation files
# were produced by that version, so it is the one whose decisions must match
# the reference we validate against. See PROVENANCE.md.
GENERATOR = FORTRAN / "perl/acdc_2020_04_28.pl"

# Cluster sets to capture, with the vapour names each uses.
# input_ANnarrow is the reduced set the bundled example is built from;
# input_AN is the wider version; input_AD swaps ammonia for dimethylamine
# and so exercises a different acid/base strength ladder in the cascade.
CLUSTER_SETS = {
    "AN_narrow": (
        FORTRAN / "src/Perl_input/input_ANnarrow_neutral_neg_pos.inp",
        ("A", "N"),
    ),
    "AN": (FORTRAN / "cluster_sets/input_AN_neutral_neg_pos.inp", ("A", "N")),
    "AD": (FORTRAN / "cluster_sets/input_AD_neutral_neg_pos.inp", ("A", "D")),
}

ENERGY = FORTRAN / "src/Perl_input/HS298.15K_example.txt"
DIPOLE = FORTRAN / "src/Perl_input/dip_pol_298.15K_example.txt"

# Decision line formats, from the print statements in the generator.
#   Growing succesfully out: 1A + 5A5N -> 6A5N -> out_neu
#   Brought back to boundary: 2A + 2A -> 4A -> 2A + 2 A
#   Not using boundary collision 1A + 2A -> 1A + 2A
RE_OUT = re.compile(
    r"^Growing succesfully out: (\S+) \+ (\S+) -> (\S+) -> (out\S*)\s*$"
)
RE_BACK = re.compile(r"^Brought back to boundary: (\S+) \+ (\S+) -> (\S+) -> (.+?)\s*$")
RE_UNUSED = re.compile(r"^Not using boundary collision (\S+) \+ (\S+) -> (.+?)\s*$")


def _parse_products(tail: str) -> list[tuple[str, int]]:
    """Parse `3A1N + 2 A` into [("3A1N", 1), ("A", 2)].

    The main product carries no multiplicity; stripped monomers are printed
    as `<count> <molecule>`.
    """
    products: list[tuple[str, int]] = []
    for i, part in enumerate(p.strip() for p in tail.split("+")):
        if not part:
            continue
        bits = part.split()
        if i == 0:
            products.append((bits[0], 1))
        elif len(bits) == 2:
            products.append((bits[1], int(bits[0])))
        else:
            products.append((bits[0], 1))
    return products


def parse_log(text: str) -> list[dict]:
    decisions: list[dict] = []
    for line in text.splitlines():
        line = line.rstrip()
        if m := RE_OUT.match(line):
            i, j, product, channel = m.groups()
            decisions.append(
                {
                    "kind": "out",
                    "reactants": [i, j],
                    "raw_product": product,
                    "channel": channel,
                }
            )
        elif m := RE_BACK.match(line):
            i, j, product, tail = m.groups()
            decisions.append(
                {
                    "kind": "brought_back",
                    "reactants": [i, j],
                    "raw_product": product,
                    "products": _parse_products(tail),
                }
            )
        elif m := RE_UNUSED.match(line):
            i, j, tail = m.groups()
            decisions.append(
                {
                    "kind": "unused",
                    "reactants": [i, j],
                    "products": _parse_products(tail),
                }
            )
    return decisions


def run_generator(inp: Path, vapors: tuple[str, ...], work: Path) -> str:
    """Reproduce fortran/src/run_perl.sh's invocation, plus --print_boundary.

    Verified: this invocation regenerates the committed example equation file
    with zero structural differences and a maximum relative difference of
    9.8e-15 in the emitted literals (last-digit sprintf rounding, presumably
    a libm/Perl version difference). That is 100x inside the 1e-12 rate gate.
    """
    cmd = [
        "perl",
        str(GENERATOR),
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
        "--i",
        str(inp),
        "--exp_loss_ref_cluster",
        f"1{vapors[0]}",
        "--append",
        "_capture",
        "--print_boundary",
    ]
    for v in vapors:
        cmd += ["--cs_only", f"1{v},0"]

    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-3000:])
        sys.stderr.write(result.stderr[-3000:])
        raise SystemExit(f"generator failed for {inp.name}")
    return result.stdout


def cross_check(decisions: list[dict], work: Path) -> None:
    """The number of grow-out decisions must equal the number of `-> out_`
    coefficient targets in the code the same run emitted.

    An independent view of the same information: if the log parser drops or
    duplicates a decision, these disagree.
    """
    emitted = list(work.glob("acdc_equations_*.f90"))
    if not emitted:
        return
    text = emitted[0].read_text()
    n_out_targets = len(re.findall(r"-> out_(?:neu|neg|pos)", text))
    n_out_logged = sum(d["kind"] == "out" for d in decisions)
    if n_out_targets != n_out_logged:
        raise SystemExit(
            f"cross-check FAILED: {n_out_logged} grow-out decisions logged but "
            f"{n_out_targets} '-> out_' targets emitted"
        )


def main() -> int:
    if shutil.which("perl") is None:
        print("perl not found on PATH", file=sys.stderr)
        return 1
    GOLDENS.mkdir(exist_ok=True)

    for name, (inp, vapors) in CLUSTER_SETS.items():
        if not inp.exists():
            print(f"skip {name}: {inp} missing", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            log = run_generator(inp, vapors, work)
            decisions = parse_log(log)
            cross_check(decisions, work)

        counts: dict[str, int] = {}
        for d in decisions:
            counts[d["kind"]] = counts.get(d["kind"], 0) + 1

        path = GOLDENS / f"boundary_{name}.json"
        path.write_text(
            json.dumps(
                {
                    "cluster_set": inp.name,
                    "vapors": list(vapors),
                    "generator": GENERATOR.name,
                    "counts": counts,
                    "decisions": decisions,
                },
                indent=1,
            )
            + "\n"
        )
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(
            f"wrote {path.relative_to(REPO)}  ({len(decisions)} decisions: {summary})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
