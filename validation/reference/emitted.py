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


_EVAP_ASSIGN = re.compile(r"^\s*E\((\d+),(\d+)\)\s*=\s*(.+?)\s*(?:!.*)?$")


def evaporation_matrix(path: str | Path, temperature: float, nclust: int) -> np.ndarray:
    """Evaluate get_evap from a generated equations file at one temperature.

    The emitted ``E(i,j)`` expressions reference ``K(i,j)``, so get_coll is
    evaluated first and exposed to the expressions as a function. Zero where
    no channel is emitted.
    """
    k = collision_matrix(path, temperature, nclust)
    text = Path(path).read_text()
    start = text.index("subroutine get_evap")
    end = text.index("end subroutine get_evap")
    joined = re.sub(r"&\s*\n\s*&?", "", text[start:end])

    e = np.zeros((nclust, nclust))
    env = {
        "sqrt": math.sqrt,
        "exp": math.exp,
        "max": max,
        "sign": _sign,
        "temperature": temperature,
        "K": lambda i, j: k[i - 1, j - 1],
    }
    copies: list[tuple[int, int, int, int]] = []
    for line in joined.splitlines():
        m = _EVAP_ASSIGN.match(line)
        if not m:
            continue
        i, j, rhs = int(m.group(1)) - 1, int(m.group(2)) - 1, m.group(3)
        copy = re.fullmatch(r"E\((\d+),(\d+)\)", rhs.strip())
        if copy:
            copies.append((i, j, int(copy.group(1)) - 1, int(copy.group(2)) - 1))
            continue
        e[i, j] = eval(_fortran_to_python(rhs), {"__builtins__": {}}, env)  # noqa: S307
    for i, j, si, sj in copies:
        e[i, j] = e[si, sj]
    return e


def collision_triples(path: str | Path) -> set[tuple[int, int, int]]:
    """Every ``coef_quad(i,j,k) = K(...)`` assignment as 0-based ``(i, j, k)``.

    The reaction graph itself, independent of rate values: which pairs
    collide and what each produces. Used to validate --nst.
    """
    text = Path(path).read_text()
    # Self-collisions are emitted as `0.5d0*K(i,i)`; everything else as `K(i,j)`.
    pattern = re.compile(
        r"^\s*coef_quad\((\d+),(\d+),(\d+)\)\s*=\s*(?:0\.5d0\*)?K\(", re.M
    )
    return {(int(a) - 1, int(b) - 1, int(c) - 1) for a, b, c in pattern.findall(text)}


def evaporation_pairs(path: str | Path) -> set[tuple[int, int]]:
    """Every daughter pair with an emitted ``E(i,j) =`` line, 0-based, i>=j and i<j."""
    text = Path(path).read_text()
    start = text.index("subroutine get_evap")
    end = text.index("end subroutine get_evap")
    pattern = re.compile(r"^\s*E\((\d+),(\d+)\)\s*=", re.M)
    return {(int(a) - 1, int(b) - 1) for a, b in pattern.findall(text[start:end])}


def formation_extras(path: str | Path) -> set[tuple[int, int, int, int]]:
    """``ind_quad_form_extra(k, 0:m) = (/ n, i,j,mult, ... /)`` as 0-based
    ``(i, j, k, mult)`` -- the secondary products of a collision (boundary
    monomers, non-standard extras) with their multiplicities."""
    text = Path(path).read_text()
    joined = re.sub(r"&\s*\n\s*&?", "", text)
    pattern = re.compile(
        r"^\s*ind_quad_form_extra\((\d+),0:\d+\)\s*=\s*\(/(.*?)/\)", re.M
    )
    out: set[tuple[int, int, int, int]] = set()
    for k, body in pattern.findall(joined):
        values = [int(v) for v in body.replace(",", " ").split()]
        count, rest = values[0], values[1:]
        assert len(rest) == 3 * count, (k, body)
        for n in range(count):
            i, j, mult = rest[3 * n : 3 * n + 3]
            out.add((i - 1, j - 1, int(k) - 1, mult))
    return out


def loss_vector(path: str | Path, nclust: int, name: str = "cs") -> np.ndarray:
    """Evaluate a loss vector from a generated equations file.

    Handles both emitted shapes:

    - per-cluster literals, ``wl(3) = 1.39d-03``, for the size-dependent
      parameterizations;
    - a single scalar, ``wl = 2.3d-02`` or ``dil = 9.6d-05``, which upstream
      then applies to every cluster (``coef_lin(58,58,k) = dil`` for all k).
      Broadcast to a full vector here so callers see one shape.

    Every entry is a plain literal: the loss routines carry no temperature
    dependence in any generated variant, because the generator refuses to
    combine bg_loss or wall losses with --variable_temp.
    """
    text = Path(path).read_text()
    start = text.index("subroutine get_losses")
    end = text.index("end subroutine get_losses")
    body = text[start:end]

    vector = re.compile(rf"^\s*{name}\((\d+)\)\s*=\s*([-+0-9.dDeE]+)\s*(?:!.*)?$")
    scalar = re.compile(rf"^\s*{name}\s*=\s*([-+0-9.dDeE]+)\s*(?:!.*)?$")

    out = np.zeros(nclust)
    for line in body.splitlines():
        if m := vector.match(line):
            out[int(m.group(1)) - 1] = float(_fortran_to_python(m.group(2)))
        elif m := scalar.match(line):
            out[:] = float(_fortran_to_python(m.group(1)))
    return out
