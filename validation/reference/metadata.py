"""Cluster metadata parsed out of the generated Fortran.

The Phase 1 target. `acdc_system_*.f90` is pure data -- parameters and
subroutines that copy hardcoded literal arrays, with nothing computed -- so
reading it as text is simpler and more robust than wrapping it. f2py handles
its `character(len=11)` name arrays badly, and the numeric getters would
still need the module initialised.

Everything here is a validation target, never imported by `src/acdc_jax`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SYSTEM_F90 = REPO / "fortran/src/acdc_system_AN_ions_example.f90"
EQUATIONS_F90 = REPO / "fortran/src/acdc_equations_AN_ions_example.f90"


def _array_literal(text: str, name: str) -> list[str]:
    """Pull the elements out of `name = (/ a, b, & \n &c /)`.

    Fortran continues lines with a trailing `&` and resumes after a leading
    `&`, so both must be stripped before splitting on commas.
    """
    match = re.search(rf"{re.escape(name)}\s*=\s*\(/(.*?)/\)", text, re.DOTALL)
    if match is None:
        raise KeyError(f"array literal {name!r} not found")
    body = match.group(1).replace("&", " ").replace("\n", " ").replace("\t", " ")
    return [item.strip() for item in body.split(",") if item.strip()]


@lru_cache(maxsize=1)
def _system_text() -> str:
    return SYSTEM_F90.read_text()


@lru_cache(maxsize=1)
def cluster_names() -> list[str]:
    """The 54 cluster labels, in state-vector order.

    Read from the KEY comment block at the top of the equations file rather
    than from `cluster_names` in the system module: the KEY block is one
    entry per line and unambiguous, while the Fortran version is a wall of
    quoted `character(len=11)` assignments.

    Indices 53 and 54 are the generic charger ions, which the KEY block
    names in prose ("generic negative ion"); they are normalised to `neg`
    and `pos`, matching the labels the Perl generator uses in its own
    boundary log.
    """
    names: list[str] = []
    for line in EQUATIONS_F90.read_text().splitlines():
        m = re.match(r"^!\s*Cluster\s+(\d+):\s*(.+?)\s*$", line)
        if not m:
            continue
        index, label = int(m.group(1)), m.group(2)
        assert index == len(names) + 1, f"KEY block out of order at {index}"
        if label == "generic negative ion":
            label = "neg"
        elif label == "generic positive ion":
            label = "pos"
        names.append(label)
    return names


@lru_cache(maxsize=1)
def molecule_names() -> list[str]:
    """The molecule types actually used, in index order.

    Note this is **4** for the bundled system (A, B, N, P), not the 5
    declared in the .inp header: dimethylamine `D` is declared but appears
    in no cluster, and the generator drops unused types. A parser that keeps
    all five produces the wrong composition-matrix width.
    """
    pairs = re.findall(r"labels\((\d+)\)\(:\) = '([^']+)'", _system_text())
    return [label for _, label in sorted(pairs, key=lambda p: int(p[0]))]


def _int_param(name: str) -> int:
    m = re.search(rf"parameter\s*::\s*{name}\s*=\s*(\d+)", _system_text())
    if m is None:
        raise KeyError(name)
    return int(m.group(1))


def sizes() -> dict[str, int]:
    return {
        "nclust": _int_param("nclust"),
        "neq": _int_param("neq"),
        "n_mol_types": _int_param("n_mol_types"),
        "n_charges": _int_param("n_charges"),
        "n_monomer_types": _int_param("n_monomer_types"),
    }


def mass() -> np.ndarray:
    """Cluster masses, g/mol. Shape (54,)."""
    return np.array([float(x) for x in _array_literal(_system_text(), "mass")])


def diameter() -> np.ndarray:
    """Mass-equivalent diameters, nm, as emitted -- rounded to 2 decimals.

    The rounding matters: comparing a full-precision computation against
    these directly fails at ~1e-3. Compare `round(computed, 2)` instead.
    """
    return np.array([float(x) for x in _array_literal(_system_text(), "diameter")])


def mob_diameter() -> np.ndarray:
    """Mobility diameters, nm, as emitted (2 decimals)."""
    return np.array([float(x) for x in _array_literal(_system_text(), "mob_diameter")])


def charging_state() -> np.ndarray:
    """1 = neutral, 2 = negative, 3 = positive. Shape (54,).

    Note this is a 1-based code, not a signed charge. The `formation`
    routine in the equations file uses a *different*, signed convention
    (0/-1/+1) for the same information.
    """
    return np.array([int(x) for x in _array_literal(_system_text(), "charging_state")])


def n_a_in_clusters() -> np.ndarray:
    """Number of A (sulfuric acid) molecules per cluster. Shape (54,)."""
    return np.array([int(x) for x in _array_literal(_system_text(), "n_A")])


def monomer_indices() -> np.ndarray:
    """1-based Fortran indices of the monomers: 1A, 1N, 1B, 1N1P."""
    return np.array([int(x) for x in _array_literal(_system_text(), "n_monomers")])


def bound_clusters() -> tuple[np.ndarray, np.ndarray]:
    """Boundary clusters and their compositions.

    Returns (indices, compositions) with shapes (12,) and (12, 4). These sit
    on the edge of the enumerated composition box; collisions from them that
    would leave it are the ones the Phase 2 cascade handles.
    """
    text = _system_text()
    indices = np.array([int(x) for x in _array_literal(text, "bound_clusters")])
    rows = re.findall(r"nmols_bound\((\d+),:\)\s*=\s*\(/([^)]+)/\)", text)
    comps = np.zeros((len(rows), 4), dtype=int)
    for row, values in rows:
        comps[int(row) - 1] = [int(v) for v in values.split(",")]
    return indices, comps


def out_thresholds() -> dict[str, np.ndarray]:
    """The grow-out composition thresholds, per charge state.

    A collision product counts as grown out when **every** molecule count
    meets or exceeds the threshold for its charge.
    """
    text = _system_text()
    out = {}
    for key, name in [
        ("neutral", "nmols_out_neutral"),
        ("negative", "nmols_out_negative"),
        ("positive", "nmols_out_positive"),
    ]:
        # Emitted as `nmols_out_neutral(1, 4) = reshape((/6, 0, 5, 0/),(/1, 4/))`
        # -- a parameter with a reshape, not a plain assignment.
        m = re.search(rf"{name}\([^)]*\)\s*=\s*reshape\(\(/([^)]+)/\)", text)
        if m is None:
            raise KeyError(name)
        out[key] = np.array([int(v) for v in m.group(1).split(",")])
    return out


__all__ = [
    "bound_clusters",
    "charging_state",
    "cluster_names",
    "diameter",
    "mass",
    "mob_diameter",
    "molecule_names",
    "monomer_indices",
    "n_a_in_clusters",
    "out_thresholds",
    "sizes",
]


# --------------------------------------------------------------------------
# Thermodynamic data recovered from the emitted evaporation expressions
# --------------------------------------------------------------------------

# One evaporation assignment plus its trailing `! parent -> d1 + d2` comment.
_EVAP_BLOCK = re.compile(
    # The comment may carry a trailing note -- charged evaporations are
    # annotated ", including ion enhancement" -- so the last product is
    # matched up to a comma rather than to end of line.
    r"E\((\d+),(\d+)\)\s*=\s*(.*?)!\s*([^\s,]+)\s*->\s*([^\s,]+)\s*\+\s*([^\s,]+)",
    re.DOTALL | re.MULTILINE,
)

# A free-energy slot inside the exp() argument: either an (H, S) group or a
# bare `0.d0` standing for a zero-reference monomer. Matching both in one
# alternation keeps the three slots positional, which is what lets them be
# paired with (parent, daughter, daughter) from the comment.
#
# Note the `d0` sits INSIDE the parentheses -- `(-71.024601d0)/1.d3`, not
# `(-71.024601)d0/1.d3`. Getting that backwards matches nothing and yields a
# table of zeros rather than an error.
_SLOT = re.compile(
    r"\((-?[\d.]+)d0/temperature-\(?(-?[\d.]+)d0\)?/1\.d3\)"
    r"|(?<![\d.])(0\.d0)"
)


def _normalise(body: str) -> str:
    """Strip Fortran line continuations so the expression is one string."""
    return body.replace("&", "").replace("\n", "").replace("\t", "").replace(" ", "")


@lru_cache(maxsize=1)
def energy_data() -> dict[str, tuple[float, float]]:
    """Recover per-cluster (Delta-H, Delta-S) from ``get_evap``.

    The generator inlines the raw energy-file values into every evaporation
    expression as ``(H/temperature-(S)/1.d3)`` groups, in the order parent,
    daughter, daughter, matched to labels by the trailing
    ``! 2A1N -> 1A1N + 1A`` comment. Zero-reference monomers appear as a
    bare ``0.d0`` instead.

    Cross-checks itself: a cluster appearing in many expressions must carry
    identical values in all of them, so a mis-parse is caught here rather
    than surfacing later as a rate discrepancy.
    """
    text = EQUATIONS_F90.read_text()
    found: dict[str, tuple[float, float]] = {}

    for _i, _j, body, parent, d1, d2 in _EVAP_BLOCK.findall(text):
        slots = _SLOT.findall(_normalise(body))
        if len(slots) != 3:
            raise AssertionError(
                f"expected 3 free-energy slots for {parent} -> {d1} + {d2}, "
                f"found {len(slots)}"
            )
        for label, (h, s, zero) in zip((parent, d1, d2), slots, strict=True):
            values = (0.0, 0.0) if zero else (float(h), float(s))
            previous = found.get(label)
            if previous is not None and previous != values:
                raise AssertionError(
                    f"inconsistent energy data for {label}: {previous} vs {values}"
                )
            found[label] = values
    return found


def energy_labels() -> set[str]:
    """Cluster labels for which the emitted code carries energy data."""
    return set(energy_data())
