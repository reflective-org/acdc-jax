"""Loop mode against f2py-compiled loop-mode fixtures. Phase 9.

Each golden (``validation/goldens/loop_variant_*.npz``, from
``validation/capture_loop.py``) holds K, E, the loss vector, masses and
radii, the composition indices, ``feval`` on three random states and
``formation``, for one generator invocation at 280 K. Gate 1e-12.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from acdc_jax import config, loop, losses, rhs
from acdc_jax.clusterset import parse_cluster_set

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "validation/fixtures/loop"
GOLDENS = REPO / "validation/goldens"
T = 280.0
GATE = 1e-12
BG_CONCENTRATION = 1.0e9  # what capture_loop.py compiled into the stub module


def _golden(name: str):
    path = GOLDENS / f"loop_variant_{name}.npz"
    if not path.exists():
        pytest.skip(
            f"{path.name} not captured: uv run python validation/capture_loop.py"
        )
    return np.load(path)


def _worst(got, expected) -> float:
    got, expected = np.asarray(got), np.asarray(expected)
    mask = expected != 0
    if not mask.any():
        return 0.0
    return float((np.abs(got - expected)[mask] / np.abs(expected)[mask]).max())


def _synthetic_gibbs(counts: np.ndarray) -> np.ndarray:
    """The bridge's stub cluster_energies: g = 5 n^(2/3) - 3 n kcal/mol."""
    n = counts.sum(axis=1)
    return 5.0 * n ** (2.0 / 3.0) - 3.0 * n


@pytest.fixture(scope="module")
def a20():
    return loop.build_loop_system(parse_cluster_set(INPUTS / "A20.inp"))


@pytest.fixture(scope="module")
def an66():
    return loop.build_loop_system(parse_cluster_set(INPUTS / "AN66.inp"))


class TestSystem:
    def test_one_component_is_indexed_by_count(self, a20) -> None:
        assert a20.n_clusters == 20
        assert a20.system.labels[:3] == ("1A", "2A", "3A")
        assert list(a20.monomers) == [0]
        assert a20.product_index[0, 0] == 1  # 1A + 1A -> 2A
        assert a20.product_index[9, 10] == -1  # 10A + 11A leaves the grid

    def test_two_component_order_matches_get_cluster_numbers(self, an66) -> None:
        golden = _golden("loopAN66")
        assert an66.n_clusters == 48
        np.testing.assert_array_equal(an66.counts, golden["indices"])
        assert an66.system.labels[0] == "1N"  # (0,1) comes before (1,0)
        assert [an66.system.labels[i] for i in an66.monomers] == ["1A", "1N"]

    @pytest.mark.parametrize(
        "name, fixture", [("loopA20k", "a20"), ("loopAN66", "an66")]
    )
    def test_masses_and_radii(self, name: str, fixture: str, request) -> None:
        system = request.getfixturevalue(fixture)
        golden = _golden(name)
        np.testing.assert_allclose(system.mass_g, golden["mass_g"], rtol=1e-14)
        np.testing.assert_allclose(system.radius, golden["radius"], rtol=1e-14)

    def test_header_rows_parsed(self, an66) -> None:
        assert an66.molecules[0].psat == 5e-11
        assert an66.molecules[1].psat == 1e4
        assert an66.molecules[0].surface_tension == 0.05

    def test_rejects_a_cluster_list(self) -> None:
        cs = parse_cluster_set(
            REPO / "fortran/src/Perl_input/input_ANnarrow_neutral_neg_pos.inp"
        )
        with pytest.raises(ValueError, match="exactly one composition row"):
            loop.build_loop_system(cs)


class TestCollisions:
    @pytest.mark.parametrize(
        "name, fixture, method",
        [
            ("loopA20k", "a20", "hard_spheres"),
            ("loopA20d", "a20", "dahneke"),
            ("loopAN66", "an66", "hard_spheres"),
            ("loopAN66d", "an66", "dahneke"),
        ],
    )
    def test_matches_fixture(self, name, fixture, method, request) -> None:
        system = request.getfixturevalue(fixture)
        golden = _golden(name)
        k = loop.collision_coefficients(system, T, method)
        assert _worst(k, golden["K"]) < GATE

    def test_variable_temperature_form(self, a20) -> None:
        golden = _golden("loopA20vt")
        k = loop.collision_coefficients(a20, float(golden["temperature"]))
        assert _worst(k, golden["K"]) < GATE

    def test_dahneke_traces_in_temperature(self, a20) -> None:
        g = jax.grad(lambda t: loop.dahneke(a20, t).sum())(T)
        assert np.isfinite(float(g)) and float(g) != 0


class TestEvaporation:
    def test_kelvin_one_component(self, a20) -> None:
        golden = _golden("loopA20k")
        k = loop.collision_coefficients(a20, T)
        e = np.asarray(loop.kelvin_evaporation(a20, k, T))
        # The one-component Fortran fills column 1 only (F22); the port keeps
        # the symmetric matrix. Compare E(j,i), i <= j -- what feval reads.
        np.testing.assert_array_equal(np.tril(e) != 0, np.tril(golden["E"]) != 0)
        assert _worst(np.tril(e), np.tril(golden["E"])) < GATE
        np.testing.assert_array_equal(e, e.T)
        # only the monomer column/row is populated: no fissions
        assert np.count_nonzero(e[1:, 1:]) == 0

    def test_kelvin_size_limit(self, a20) -> None:
        golden = _golden("loopA20lim")
        k = loop.collision_coefficients(a20, T)
        e = np.asarray(loop.kelvin_evaporation(a20, k, T, rlim_no_evap=0.5e-9))
        np.testing.assert_array_equal(np.tril(e) != 0, np.tril(golden["E"]) != 0)
        assert _worst(np.tril(e), np.tril(golden["E"])) < GATE
        # 0.5 nm falls inside the grid: the larger parents stop evaporating
        assert 0 < np.count_nonzero(e[:, 0]) < 19

    def test_kelvin_two_components(self, an66) -> None:
        golden = _golden("loopAN66")
        k = loop.collision_coefficients(an66, T)
        e = loop.kelvin_evaporation(an66, k, T)
        np.testing.assert_array_equal(np.asarray(e) != 0, golden["E"] != 0)
        assert _worst(e, golden["E"]) < GATE

    def test_kelvin_variable_temperature(self, a20) -> None:
        golden = _golden("loopA20vt")
        t = float(golden["temperature"])
        k = loop.collision_coefficients(a20, t)
        assert _worst(loop.kelvin_evaporation(a20, k, t), golden["E"]) < GATE

    def test_deltag(self, a20) -> None:
        golden = _golden("loopA20dg")
        k = loop.collision_coefficients(a20, T)
        e = loop.deltag_evaporation(a20, k, T, _synthetic_gibbs)
        np.testing.assert_array_equal(np.asarray(e) != 0, golden["E"] != 0)
        assert _worst(e, golden["E"]) < GATE
        # fissions ARE allowed in this mode
        assert np.count_nonzero(np.asarray(e)[1:, 1:]) > 0


class TestLosses:
    @pytest.mark.parametrize(
        "name, fixture, settings",
        [
            ("loopA20cs", "a20", losses.LossSettings()),
            (
                "loopA20bg",
                "a20",
                losses.LossSettings(
                    coagulation="bg_loss", bg_concentration=BG_CONCENTRATION
                ),
            ),
            ("loopA20wl", "a20", losses.LossSettings(wall="cloud4_ja")),
            ("loopA20wljk", "a20", losses.LossSettings(wall="cloud4_jk")),
            (
                "loopA20dil",
                "a20",
                losses.LossSettings(dilution=config.DILUTION_DEFAULT),
            ),
            ("loopA20cswl", "a20", losses.LossSettings(wall="cloud4_ja")),
            ("loopAN66cs", "an66", losses.LossSettings(wall="cloud4_ja")),
        ],
    )
    def test_total_loss_matches_fixture(self, name, fixture, settings, request) -> None:
        """Upstream sums everything into one `loss`; compare the sum of the
        port's per-slot vectors. Variants with only a wall loss or only
        dilution have no coagulation term upstream, so drop `coag` there."""
        system = request.getfixturevalue(fixture)
        golden = _golden(name)
        vectors = loop.first_order_losses(system, settings, T)
        if name in ("loopA20wl", "loopA20wljk", "loopA20dil"):
            vectors.pop("coag")
        total = sum(vectors.values())
        assert _worst(total, golden["loss"]) < GATE

    def test_mass_correction_is_live_in_loop_mode(self, a20) -> None:
        """Perl :7024 `sqrt(1+28.8/m)`: unlike the small-set F11 path, the
        mobility diameter here depends on mass."""
        golden = _golden("loopA20wl")
        wl = np.asarray(golden["loss"])
        d_mob = wl[0] / wl[1]  # ratio of 1.66e-12/d for 1A vs 2A
        d_mass_only = (2 * a20.radius[1] + 0.3e-9) / (2 * a20.radius[0] + 0.3e-9)
        assert d_mob != pytest.approx(d_mass_only, rel=1e-3)


def _feval(system, k, e, golden, loss_vectors=None, source=None):
    """Port right-hand side and J on the golden's states."""
    if source is None:
        source = golden["source"]
    co = loop.assemble(system, k, e, source, loss_vectors, constant_monomers=False)
    n = system.n_clusters
    out = []
    js = []
    for c in golden["states"]:
        full = np.zeros(system.system.n_equations)
        full[:n] = c
        out.append(np.asarray(rhs.rhs(co, full))[:n])
        js.append(float(rhs.formation_rate(system.system, co, full)["j_tot"]))
    return np.array(out), np.array(js), co


def _check_rhs(system, k, e, golden, loss_vectors=None, source=None) -> None:
    """Two gates. The STRUCTURE of the right-hand side -- which pairs, which
    product, the halving, the outflux, the losses -- is checked with the
    fixture's own K and E at 1e-12. The port's own K and E are separately
    gated at 1e-12, but f is a small difference of large opposing fluxes on
    these random states, so a 1e-13 wobble in E shows up as ~1e-8 in f;
    that combined check is held at 1e-6."""
    k_ref, e_ref = jnp.asarray(golden["K"]), jnp.asarray(golden["E"])
    f, j, _ = _feval(system, k_ref, e_ref, golden, loss_vectors, source)
    assert _worst(f, golden["feval"]) < GATE
    assert _worst(j, golden["formation"]) < GATE
    f, j, _ = _feval(system, k, e, golden, loss_vectors, source)
    assert _worst(f, golden["feval"]) < 1e-6
    assert _worst(j, golden["formation"]) < GATE


class TestRightHandSide:
    @pytest.mark.parametrize(
        "name, fixture, settings",
        [
            ("loopA20k", "a20", None),
            ("loopA20d", "a20", None),
            ("loopA20lim", "a20", None),
            ("loopA20cs", "a20", losses.LossSettings()),
            ("loopA20cswl", "a20", losses.LossSettings(wall="cloud4_ja")),
            ("loopAN66", "an66", None),
            ("loopAN66cs", "an66", losses.LossSettings(wall="cloud4_ja")),
        ],
    )
    def test_feval_and_formation(self, name, fixture, settings, request) -> None:
        system = request.getfixturevalue(fixture)
        golden = _golden(name)
        method = "dahneke" if name.endswith("d") else "hard_spheres"
        k = loop.collision_coefficients(system, T, method)
        rlim = 0.5e-9 if name.endswith("lim") else None
        e = loop.kelvin_evaporation(system, k, T, rlim)
        vectors = loop.first_order_losses(system, settings, T) if settings else None
        _check_rhs(system, k, e, golden, vectors)

    def test_deltag_rhs(self, a20) -> None:
        golden = _golden("loopA20dg")
        k = loop.collision_coefficients(a20, T)
        e = loop.deltag_evaporation(a20, k, T, _synthetic_gibbs)
        _check_rhs(a20, k, e, golden)

    def test_variable_temperature_rhs(self, a20) -> None:
        """Under --variable_temp coef(1) is the temperature and the monomer
        source stays at the stub's zero."""
        golden = _golden("loopA20vt")
        t = float(golden["temperature"])
        k = loop.collision_coefficients(a20, t)
        e = loop.kelvin_evaporation(a20, k, t)
        _check_rhs(a20, k, e, golden, source=np.zeros(1))

    def test_outflux_is_booked(self, a20) -> None:
        k = loop.collision_coefficients(a20, T)
        e = loop.kelvin_evaporation(a20, k, T)
        co = loop.assemble(a20, k, e, np.array([1e6]))
        c = np.zeros(a20.system.n_equations)
        c[: a20.n_clusters] = 1e12
        f = np.asarray(rhs.rhs(co, c))
        out = a20.system.flux_index["out_neu"]
        assert f[out] > 0
        assert f[out] == pytest.approx(
            float(rhs.formation_rate(a20.system, co, c)["j_tot"]), rel=1e-12
        )

    def test_constant_monomers(self, a20) -> None:
        k = loop.collision_coefficients(a20, T)
        e = loop.kelvin_evaporation(a20, k, T)
        co = loop.assemble(a20, k, e, np.array([1e6]), constant_monomers=True)
        c = np.full(a20.system.n_equations, 1e12)
        assert float(rhs.rhs(co, c)[0]) == 0.0
        assert co.coef_quad is None


class TestSizeBins:
    def test_matrix_shape_and_monomer_exclusion(self, a20) -> None:
        m = loop.size_bin_matrix(a20)
        assert m.shape == (config.NBINS + 1, a20.n_clusters)
        assert m[:, 0].sum() == 0  # the monomer is in no bin
        # every other cluster in exactly one bin
        np.testing.assert_array_equal(m[:, 1:].sum(axis=0), 1.0)

    def test_assignment_follows_the_edges(self, a20) -> None:
        m = loop.size_bin_matrix(a20)
        d_mob = 2.0 * a20.radius + config.MOB_DIAMETER_OFFSET
        edges = np.asarray(config.BIN_LIMITS_NM) * 1e-9
        for i in range(1, a20.n_clusters):
            b = int(np.argmax(m[:, i]))
            if b == 0:
                assert d_mob[i] < edges[0]
            else:
                assert edges[b - 1] <= d_mob[i] < edges[b]
        # 2A (0.998 nm) is below the first edge; 3A (1.099 nm) is in bin 1
        assert m[0, a20.system.index("2A")] == 1.0
        assert m[1, a20.system.index("3A")] == 1.0

    def test_binned_concentrations(self, an66) -> None:
        m = loop.size_bin_matrix(an66)
        c = np.arange(1.0, an66.n_clusters + 1)
        binned = m @ c
        monomers = an66.monomers
        assert binned.sum() == pytest.approx(c.sum() - c[monomers].sum())

    def test_above_range_is_an_error(self) -> None:
        from acdc_jax.clusterset import ClusterSetFile

        cs = parse_cluster_set(INPUTS / "A20.inp")
        big = ClusterSetFile(
            path=cs.path,
            molecules=cs.molecules,
            compositions=((2000, 0, 0, 0, 0),),
            out_rules=cs.out_rules,
        )
        with pytest.raises(ValueError, match="above the last size-bin edge"):
            loop.size_bin_matrix(loop.build_loop_system(big))


class TestSteadyState:
    """9.7: the Phase 6-7 machinery on a loop system, untouched."""

    @pytest.fixture(scope="class")
    def steady(self, a20):
        from acdc_jax import solve

        def coefficients(t):
            k = loop.collision_coefficients(a20, t)
            e = loop.kelvin_evaporation(a20, k, t)
            vec = loop.first_order_losses(a20, losses.LossSettings(), t)
            return loop.assemble(
                a20, k, e, np.array([0.0]), vec, constant_monomers=True
            )

        c0 = jnp.zeros(a20.system.n_equations).at[0].set(1e14)
        result = solve.solve_steady_state(a20.system, coefficients(T), c0=c0)
        return a20, coefficients, c0, result

    def test_converges_to_a_positive_formation_rate(self, steady) -> None:
        a20, _, _, result = steady
        assert result.converged
        assert float(result.j_total) > 0
        c = np.asarray(result.concentrations)
        assert c[0] == 1e14  # the monomer was held constant
        assert np.all(c[: a20.n_clusters] >= 0)

    def test_formation_rate_is_the_outflux(self, steady) -> None:
        a20, coefficients, _, result = steady
        j = rhs.formation_rate(a20.system, coefficients(T), result.concentrations)
        assert float(j["j_tot"]) == pytest.approx(float(result.j_total), rel=1e-10)

    def test_temperature_sensitivity_by_finite_difference(self, steady) -> None:
        """With the fixture's fixed psat the Kelvin exponent shrinks as T
        rises, so evaporation FALLS and J at fixed vapour rises."""
        from acdc_jax import solve

        a20, coefficients, c0, result = steady
        warm = solve.solve_steady_state(a20.system, coefficients(T + 5.0), c0=c0)
        k_cold = loop.collision_coefficients(a20, T)
        k_warm = loop.collision_coefficients(a20, T + 5.0)
        e_cold = loop.kelvin_evaporation(a20, k_cold, T)[1, 0]
        e_warm = loop.kelvin_evaporation(a20, k_warm, T + 5.0)[1, 0]
        assert float(e_warm) < float(e_cold)
        assert float(warm.j_total) > float(result.j_total)

    def test_rhs_differentiates_in_temperature(self, steady) -> None:
        a20, coefficients, _, result = steady
        c = result.concentrations

        def j_of(t):
            return rhs.formation_rate(a20.system, coefficients(t), c)["j_tot"]

        g = float(jax.grad(j_of)(T))
        assert np.isfinite(g)
