"""The emitted-expression evaluator must agree with the compiled reference.

Every Phase 8 golden is produced by ``validation/reference/emitted.py``,
which evaluates the generated Fortran's ``K(i,j) = ...`` / ``E(i,j) = ...``
text with Python. It is a regex-and-eval tool and would fail QUIETLY -- an
unmatched line is simply left at zero. So it is checked here against the
f2py bridge on the shipped example at four temperatures, for both matrices
and the loss vector. Requires the compiled bridge; skipped without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
sys.path.insert(0, str(REPO / "validation/reference"))

try:
    import reference
except ImportError as exc:  # the compiled bridge is absent
    # `pytest.importorskip` does not cover this: the package imports and
    # then raises because its .so is missing, which pytest treats as an
    # error rather than a missing module. CI without gfortran found it.
    pytest.skip(
        f"Fortran bridge not built ({exc}); "
        "run: uv run python validation/reference/build.py",
        allow_module_level=True,
    )
import emitted  # noqa: E402

EQUATIONS = REPO / "fortran/src/acdc_equations_AN_ions_example.f90"
TEMPERATURES = (250.0, 280.0, 298.15, 320.0)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_collision_matrix_equals_the_bridge(temperature: float) -> None:
    got = emitted.collision_matrix(EQUATIONS, temperature, 54)
    want = reference.get_coll(temperature)
    assert np.count_nonzero(got) == 2131
    np.testing.assert_array_equal(got != 0, want != 0)
    mask = want != 0
    assert np.max(np.abs(got - want)[mask] / np.abs(want)[mask]) < 1e-14


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_evaporation_matrix_equals_the_bridge(temperature: float) -> None:
    got = emitted.evaporation_matrix(EQUATIONS, temperature, 54)
    want = reference.get_evap(temperature)
    assert np.count_nonzero(got) == 422
    np.testing.assert_array_equal(got != 0, want != 0)
    mask = want != 0
    assert np.max(np.abs(got - want)[mask] / np.abs(want)[mask]) < 1e-13


def test_loss_vector_equals_the_bridge() -> None:
    """The shipped example's cs is the exp_loss shape (variable cs)."""
    got = emitted.loss_vector(EQUATIONS, 54, name="cs")
    want = reference.get_losses()
    assert np.count_nonzero(got) == 52
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0)


def test_reaction_graph_equals_the_bridge_layout() -> None:
    """Every coef_quad triple the text names must be nonzero in the bridge's
    assembled tensor and vice versa."""
    coef = reference.make_coef(temperature=280.0, cs_ref=1e-3, ipr_neg=3e6)
    quad, _ = reference.get_rate_coefs(coef)
    from_text = emitted.collision_triples(EQUATIONS)
    from_bridge = {
        tuple(int(x) for x in idx) for idx in zip(*np.nonzero(quad), strict=True)
    }
    assert from_text == from_bridge
