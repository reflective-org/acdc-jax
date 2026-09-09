"""The committed goldens must be loadable, well-formed and physical.

These do not test the port -- there is no port yet. They check that the
captured reference data is usable as a validation target, so that a later
phase failing means the port is wrong rather than the goldens being
corrupt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDENS = Path(__file__).resolve().parents[1] / "validation/goldens"

NCLUST, NEQ = 54, 63
TEMPERATURES = (250.0, 280.0, 298.15, 320.0)


def _load(name: str):
    path = GOLDENS / name
    if not path.exists():
        pytest.skip(f"{name} not captured: uv run python validation/capture_*.py")
    return np.load(path)


@pytest.fixture(scope="module")
def rates():
    return _load("rates.npz")


@pytest.fixture(scope="module")
def rhs():
    return _load("rhs.npz")


@pytest.fixture(scope="module")
def steadystate():
    return _load("steadystate.npz")


class TestRates:
    def test_temperature_sweep_present(self, rates) -> None:
        """A single temperature would let the piecewise ion-neutral branch
        sit on the wrong side for most pairs and still pass."""
        np.testing.assert_allclose(rates["temperatures"], TEMPERATURES)
        for t in TEMPERATURES:
            assert f"K_{t:g}" in rates
            assert f"E_{t:g}" in rates

    @pytest.mark.parametrize("t", TEMPERATURES)
    def test_shapes_finite_nonnegative(self, rates, t: float) -> None:
        for key in (f"K_{t:g}", f"E_{t:g}"):
            arr = rates[key]
            assert arr.shape == (NCLUST, NCLUST)
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)

    @pytest.mark.parametrize("t", TEMPERATURES)
    def test_symmetric(self, rates, t: float) -> None:
        for key in (f"K_{t:g}", f"E_{t:g}"):
            np.testing.assert_array_equal(rates[key], rates[key].T)

    def test_sparsity_is_stable_across_temperature(self, rates) -> None:
        """Which pairs are allowed is combinatorial, so it must not depend
        on temperature -- only the values may."""
        patterns = {(rates[f"K_{t:g}"] != 0).tobytes() for t in TEMPERATURES}
        assert len(patterns) == 1

    def test_expected_nonzero_counts(self, rates) -> None:
        assert np.count_nonzero(rates["K_280"]) == 2131
        assert np.count_nonzero(rates["E_280"]) == 422

    def test_losses(self, rates) -> None:
        cs = rates["cs"]
        assert cs.shape == (NCLUST,)
        assert np.all(cs >= 0.0)
        assert cs[0] == 0.0 and cs[2] == 0.0  # vapour monomers excluded
        assert rates["fcr"] == 1.0


class TestRHS:
    def test_states_span_many_orders(self, rhs) -> None:
        states = rhs["states"]
        assert states.shape[1] == NEQ
        populated = states[states > 0]
        assert populated.max() / populated.min() > 1e15

    @pytest.mark.parametrize("t", TEMPERATURES)
    def test_dcdt_finite(self, rhs, t: float) -> None:
        dcdt = rhs[f"dcdt_{t:g}"]
        assert dcdt.shape == rhs["states"].shape
        assert np.all(np.isfinite(dcdt))

    def test_constant_monomers_are_pinned(self, rhs) -> None:
        dcdt = rhs["dcdt_280"]
        assert np.all(dcdt[:, 0] == 0.0)  # 1A
        assert np.all(dcdt[:, 2] == 0.0)  # 1N

    def test_empty_state_gives_only_ion_sources(self, rhs) -> None:
        """First structured state is all zeros; only the two zeroth-order
        ion production terms can be nonzero."""
        f = rhs["dcdt_280"][0]
        np.testing.assert_array_equal(np.flatnonzero(f), [52, 53])

    def test_flux_counters_never_decrease(self, rhs) -> None:
        """Indices 59-61 are monotone accumulators: they appear only as
        products, never as reactants, so their derivative cannot be
        negative for a non-negative state."""
        states, dcdt = rhs["states"], rhs["dcdt_280"]
        physical = np.all(states >= 0.0, axis=1)
        assert np.all(dcdt[physical][:, 59:62] >= 0.0)


class TestSteadyState:
    def test_grid_shape(self, steadystate) -> None:
        cond, j = steadystate["conditions"], steadystate["j"]
        assert cond.shape == (60, 5)
        assert j.shape == (60,)
        assert list(steadystate["condition_names"]) == [
            "c_A",
            "c_N",
            "temperature",
            "cs_ref",
            "ipr",
        ]

    def test_all_positive_and_finite(self, steadystate) -> None:
        j = steadystate["j"]
        assert np.all(np.isfinite(j))
        assert np.all(j > 0.0)

    def test_reproduces_the_bundled_example(self, steadystate) -> None:
        """The one number the `run` binary prints, bit for bit.

        Cold-start J is deterministic across processes; this is exact
        equality, not a tolerance.
        """
        cond, j = steadystate["conditions"], steadystate["j"]
        target = np.array([1e13, 1e15, 280.0, 1e-3, 3e6])
        row = np.flatnonzero((cond == target).all(axis=1))
        assert row.size == 1
        assert j[row[0]] == 2217995.192415948

    def test_j_increases_with_sulfuric_acid(self, steadystate) -> None:
        """The one monotonicity the model must have: more vapour, more
        particles, everything else held fixed."""
        cond, j = steadystate["conditions"], steadystate["j"]
        others = cond[:, 1:]
        for row in np.unique(others, axis=0):
            sel = (others == row).all(axis=1)
            c_a, j_sel = cond[sel, 0], j[sel]
            order = np.argsort(c_a)
            assert np.all(np.diff(j_sel[order]) > 0.0), f"non-monotonic at {row}"

    def test_j_decreases_with_coagulation_sink(self, steadystate) -> None:
        """A larger sink scavenges clusters before they grow out."""
        cond, j = steadystate["conditions"], steadystate["j"]
        keep = np.delete(np.arange(5), 3)
        others = cond[:, keep]
        for row in np.unique(others, axis=0):
            sel = (others == row).all(axis=1)
            order = np.argsort(cond[sel, 3])
            assert np.all(np.diff(j[sel][order]) < 0.0), f"non-monotonic at {row}"

    def test_spans_a_wide_dynamic_range(self, steadystate) -> None:
        """A validation grid that only covers one regime is not a gate.

        13 orders of magnitude means the port has to be right where
        evaporation dominates and where collisions do.
        """
        j = steadystate["j"]
        assert j.max() / j.min() > 1e10


def test_manifest_records_the_toolchain() -> None:
    path = GOLDENS / "MANIFEST.md"
    if not path.exists():
        pytest.skip("goldens not captured")
    text = path.read_text()
    assert "GNU Fortran" in text
    assert "fallow-argument-mismatch" in text
