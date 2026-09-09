"""Evaluate rate coefficients straight from generated Fortran text.

For validating options the bundled example was not generated with -- Su73,
constant, wall losses, hydrates -- each needs its own generated fixture. Wrapping
every variant with f2py would mean a build per option; the emitted `K(i,j) =`
lines are simple enough to evaluate directly instead.

Handles the forms the generator emits for collision coefficients:

    K(i,j) = 2.003d-17*sqrt(temperature)
    K(i,j) = max((A+B/sqrt(temperature)),C*sqrt(temperature))
    K(i,j) = max(((.5d0+sign(.5d0,temperature-T0))*(...)+...)*L, C*sqrt(temperature))
    K(i,j) = 1.6d-12
    K(i,j) = K(j,i)
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_ASSIGN = re.compile(r"^\s*K\((\d+),(\d+)\)\s*=\s*(.+?)\s*(?:!.*)?$")


def _fortran_to_python(expr: str) -> str:
    expr = expr.replace("&", "").replace("\n", "").replace("\t", "")
    # d-exponents -> e-exponents, including bare `.5d0`
    expr = re.sub(r"(\d)d([-+]?\d)", r"\1e\2", expr)
    expr = re.sub(r"(\d\.)d([-+]?\d)", r"\1e\2", expr)
    expr = expr.replace("**", "^").replace("^", "**")
    return expr


def _sign(a: float, b: float) -> float:
    """Fortran SIGN(a, b): |a| with the sign of b."""
    return math.copysign(abs(a), b)


def collision_matrix(path: str | Path, temperature: float, nclust: int) -> np.ndarray:
    """Evaluate get_coll from a generated equations file at one temperature."""
    text = Path(path).read_text()
    start = text.index("subroutine get_coll")
    end = text.index("end subroutine get_coll")
    body = text[start:end]

    # Re-join continuation lines so each assignment is one logical line.
    joined = re.sub(r"&\s*\n\s*&?", "", body)

    k = np.zeros((nclust, nclust))
    env = {"sqrt": math.sqrt, "max": max, "sign": _sign, "temperature": temperature}
    copies: list[tuple[int, int, int, int]] = []

    for line in joined.splitlines():
        m = _ASSIGN.match(line)
        if not m:
            continue
        i, j, rhs = int(m.group(1)) - 1, int(m.group(2)) - 1, m.group(3)
        copy = re.fullmatch(r"K\((\d+),(\d+)\)", rhs.strip())
        if copy:
            copies.append((i, j, int(copy.group(1)) - 1, int(copy.group(2)) - 1))
            continue
        k[i, j] = eval(_fortran_to_python(rhs), {"__builtins__": {}}, env)  # noqa: S307

    for i, j, si, sj in copies:
        k[i, j] = k[si, sj]
    return k
