"""Typed access to the f2py-wrapped ACDC Fortran reference.

Import this rather than ``acdc_ref`` directly: it gives a build hint instead
of an ImportError when the extension is missing, keeps the ``ipar``
protocol in one place, and returns arrays in a sane orientation.

Never imported by ``src/acdc_jax`` -- the port does not depend on the
reference at runtime, only the tests and the golden capture do.

The reference system is the bundled sulfuric acid--ammonia example with
ions: 54 clusters, 63 equations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

NCLUST = 54
"""Real clusters plus the two generic charger ions."""

NEQ = 63
"""State-vector length: NCLUST plus 9 flux accumulators."""

N_MOL_TYPES = 4
"""A = H2SO4, B = HSO4-, N = NH3, P = proton."""

# State-vector layout (acdc_equations_AN_ions_example.f90:1-65). Fortran is
# 1-based; these are 0-based Python indices.
IDX_NEUTRAL = slice(0, 16)
IDX_NEGATIVE = slice(16, 34)
IDX_POSITIVE = slice(34, 52)
IDX_NEG_ION = 52
IDX_POS_ION = 53
IDX_COAG = 55
IDX_REC = 58
IDX_OUT_NEU = 59
IDX_OUT_NEG = 60
IDX_OUT_POS = 61
# 54 (source), 56 (wall), 57 (dilution), 62 (boundary) are unused in this
# system -- no coefficient anywhere writes to them.


def _load():
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    try:
        import acdc_ref
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise ImportError(
            "The Fortran reference extension is not built. Run:\n"
            "    uv run python validation/reference/build.py"
        ) from exc
    return acdc_ref


_ref = _load()


def _fresh_ipar() -> np.ndarray:
    """A zeroed ipar, which forces the rate tables to be rebuilt.

    ``feval`` and ``formation`` mutate ``ipar`` to record that they have
    initialised their (separate, ``save``d) coefficient tensors. Upstream
    zeroes elements 1 and 3 before every driver call precisely so that
    changed ambient conditions take effect -- ``get_acdc_J.f90:96-97``.
    Passing a fresh array every time makes each call here independent.
    """
    return np.zeros(4, dtype=np.int32)


def make_coef(
    temperature: float, cs_ref: float, ipr_neg: float, ipr_pos: float | None = None
) -> np.ndarray:
    """Assemble the 4-element ambient-conditions vector.

    The layout depends on whether the system was generated with
    ``--variable_temp``. This one was, so it is
    ``(T, cs_ref, ipr_neg, ipr_pos)`` -- ``driver_acdc_J.f90:70``. Without
    variable temperature it would be ``(cs_ref, ipr_neg, ipr_pos, 0)``.

    Args:
        temperature: K
        cs_ref: reference coagulation sink, 1/s
        ipr_neg: negative ion production rate, 1/m^3/s
        ipr_pos: positive ion production rate; defaults to ipr_neg, which is
            what upstream does and is why the Fortran path needs no explicit
            charge-balance projection.
    """
    if ipr_pos is None:
        ipr_pos = ipr_neg
    return np.array([temperature, cs_ref, ipr_neg, ipr_pos], dtype=np.float64)


def get_coll(temperature: float) -> np.ndarray:
    """Collision coefficients K, m^3/s. Shape (54, 54).

    Zero for forbidden pairs -- same-sign ions, and products that cannot be
    represented. Symmetric.
    """
    return np.asarray(_ref.get_coll(temperature))


def get_evap(temperature: float, k: np.ndarray | None = None) -> np.ndarray:
    """Evaporation coefficients E, 1/s. Shape (54, 54).

    Detailed balance is taken against ``k``, so passing a K that did not come
    from the same temperature produces a silently inconsistent E. Defaults to
    computing K at the same temperature, which is the only correct usage.
    """
    if k is None:
        k = get_coll(temperature)
    return np.asarray(_ref.get_evap(k, temperature))


def get_losses() -> np.ndarray:
    """Coagulation-sink SIZE DEPENDENCE, dimensionless. Shape (54,).

    Not a rate. The caller multiplies by cs_ref (``get_rate_coefs:1011``).
    Entries for the neutral vapour monomers are zero because the system was
    generated with ``--cs_only 1A,0 --cs_only 1N,0``.
    """
    return np.asarray(_ref.get_losses())


def get_fcr() -> float:
    """Ion enhancement factor for losses. 1.0 in this system."""
    return float(_ref.get_fcr())


def get_rate_coefs(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The assembled coefficient tensors ``(coef_quad, coef_lin)``.

    Shapes (54, 54, 63) and (63, 63, 54). This is K, E and cs scattered into
    reaction slots with the 0.5 self-collision factors applied -- the thing
    the port's rate assembly must reproduce entry for entry.
    """
    cq, cl = _ref.get_rate_coefs(coef)
    return np.asarray(cq), np.asarray(cl)


def feval(c: np.ndarray, coef: np.ndarray, t: float = 0.0) -> np.ndarray:
    """dc/dt for the cluster birth-death system. Shape (63,).

    Args:
        c: concentrations, m^-3, shape (63,)
        coef: from :func:`make_coef`
        t: time; the system is autonomous, so this is ignored downstream

    Species marked ``isconst`` (here the neutral vapour monomers, under the
    steady-state assumption) have their derivative left at zero.
    """
    c = np.ascontiguousarray(c, dtype=np.float64)
    if c.shape != (NEQ,):
        raise ValueError(f"c must have shape ({NEQ},), got {c.shape}")
    f, _ipar = _ref.feval(t, c, coef, _fresh_ipar())
    return np.asarray(f)


def formation(c: np.ndarray, coef: np.ndarray) -> dict[str, np.ndarray | float]:
    """Formation rate and its attribution.

    Returns ``j_tot`` (1/m^3/s), ``j_by_charge`` (4: neutral-neutral,
    neutral-negative, neutral-positive, recombination), ``j_by_cluster``
    (63) and ``j_all`` (63, 4).

    The driver keeps only ``j_by_charge`` and folds the recombination channel
    into the neutral one (``driver_acdc_J.f90:359-363``); the per-cluster
    attribution is computed and discarded. It is returned here because the
    port keeps it.
    """
    c = np.ascontiguousarray(c, dtype=np.float64)
    if c.shape != (NEQ,):
        raise ValueError(f"c must have shape ({NEQ},), got {c.shape}")
    j_tot, j_by_charge, j_by_cluster, j_all, _ipar = _ref.formation(
        c, _fresh_ipar(), coef
    )
    return {
        "j_tot": float(j_tot),
        "j_by_charge": np.asarray(j_by_charge),
        "j_by_cluster": np.asarray(j_by_cluster),
        "j_all": np.asarray(j_all),
    }


__all__ = [
    "IDX_COAG",
    "IDX_NEGATIVE",
    "IDX_NEG_ION",
    "IDX_NEUTRAL",
    "IDX_OUT_NEG",
    "IDX_OUT_NEU",
    "IDX_OUT_POS",
    "IDX_POSITIVE",
    "IDX_POS_ION",
    "IDX_REC",
    "NCLUST",
    "NEQ",
    "N_MOL_TYPES",
    "feval",
    "formation",
    "get_coll",
    "get_evap",
    "get_fcr",
    "get_losses",
    "get_rate_coefs",
    "make_coef",
]
