"""config.py must agree with the vendored reference.

These tests re-parse the Fortran and Perl sources rather than hardcoding a
second copy of each number. The point is that bumping the vendored reference
to a new upstream commit fails loudly here if a tolerance or a constant
moved, instead of silently changing every result downstream.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from acdc_jax import config

REPO = Path(__file__).resolve().parents[1]
SOLUTION_SETTINGS = REPO / "fortran/src/solvers/solution_settings.f90"
PERL = REPO / "fortran/perl/acdc_2024_02_12.pl"
SYSTEM_F90 = REPO / "fortran/src/acdc_system_AN_ions_example.f90"


def _fortran_params(path: Path) -> dict[str, float]:
    """Parse `real(...), parameter :: name = value` lines, skipping comments.

    Fortran double literals use `d` for the exponent (`1.d-5`, `6.d2`), which
    Python does not accept, so the exponent marker is normalised.
    """
    params: dict[str, float] = {}
    # Require at least one digit: a bare `[\d.]+` also matches the `.` of
    # `.true.`/`.false.`, which appear as parameters in the generated system
    # module and are not numbers.
    pattern = re.compile(
        r"parameter\s*::\s*(\w+)\s*=\s*"
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[dDeE][-+]?\d+)?)\s*(?:,|!|$)"
    )
    for line in path.read_text().splitlines():
        code = line.split("!", 1)[0]  # strip trailing comment
        if code.lstrip().startswith("!"):
            continue
        for name, value in pattern.findall(code):
            params[name.lower()] = float(re.sub(r"[dD]", "e", value))
    return params


@pytest.fixture(scope="module")
def solver_params() -> dict[str, float]:
    return _fortran_params(SOLUTION_SETTINGS)


@pytest.mark.parametrize(
    ("fortran_name", "config_name"),
    [
        ("rtol_solver", "RTOL_SOLVER"),
        ("atol_solver", "ATOL_SOLVER"),
        ("chmax", "CHMAX"),
        ("chtol", "CHTOL"),
        ("negtol", "NEGTOL"),
        ("sstol", "SSTOL"),
        ("sstimech", "SSTIMECH"),
        ("sstimetot", "SSTIMETOT"),
    ],
)
def test_solver_tolerance_matches_reference(
    solver_params: dict[str, float], fortran_name: str, config_name: str
) -> None:
    assert solver_params[fortran_name] == getattr(config, config_name)


def test_all_solver_parameters_are_captured(solver_params: dict[str, float]) -> None:
    """Guard against upstream adding a tolerance we then silently ignore.

    The commented-out `rtol_solver = 1.d-3` alternative is excluded by the
    comment-stripping in the parser, so this is exactly the eight live ones.
    """
    assert len(solver_params) == 8


def _perl_scalar(name: str) -> float:
    """Extract `$name = <number>;` from the generator."""
    match = re.search(
        rf"\${re.escape(name)}\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*;",
        PERL.read_text(),
    )
    assert match is not None, f"${name} not found in {PERL.name}"
    return float(match.group(1))


@pytest.mark.parametrize(
    ("perl_name", "config_name"),
    [
        ("boltz", "K_B"),
        ("Na", "N_A"),
        ("pi", "PI"),
        ("recomb_coeff", "RECOMB_COEFF"),
    ],
)
def test_physical_constant_matches_generator(perl_name: str, config_name: str) -> None:
    assert _perl_scalar(perl_name) == getattr(config, config_name)


def test_acdc_pi_is_not_math_pi() -> None:
    """ACDC truncates pi, and we match it deliberately (fidelity F8).

    If this ever starts failing because someone "fixed" PI to math.pi, the
    hard-sphere collision rates shift by ~1e-14 relative -- small, but it
    eats 1% of the 1e-12 rate-gate budget for no reason.
    """
    import math

    assert config.PI != math.pi
    assert abs(config.PI - math.pi) / math.pi < 1e-13


def test_derived_constants() -> None:
    assert config.MASS_CONV == 1.0 / config.N_A / 1000.0
    assert config.KCAL_PER_MOL_TO_J == 4.184 * 1000.0 / config.N_A


def test_cs_defaults_match_generated_system() -> None:
    """The generated system module carries the exp_loss defaults."""
    params = _fortran_params(SYSTEM_F90)
    assert params["cs_exponent_default"] == config.CS_EXPONENT_DEFAULT
    assert params["cs_coefficient_default"] == config.CS_COEFFICIENT_DEFAULT


def test_charger_ion_properties_match_generator() -> None:
    text = PERL.read_text()
    for pattern, expected in [
        (r"\$mass_nion\s*=\s*32\.00\*", config.MASS_NEG_ION),
        (r"\$dens_nion\s*=\s*1141\.0", config.DENS_NEG_ION),
        (r"\$mass_pion\s*=\s*19\.02\*", config.MASS_POS_ION),
        (r"\$dens_pion\s*=\s*997\.0", config.DENS_POS_ION),
    ]:
        assert re.search(pattern, text), f"{pattern} not found; config has {expected}"


def test_float64_is_enabled() -> None:
    """The single most important invariant in the package."""
    import jax
    import jax.numpy as jnp

    assert jax.config.jax_enable_x64 is True
    assert jnp.zeros(1).dtype == jnp.float64


class TestFidelityConfig:
    def test_defaults_reproduce_fortran(self) -> None:
        cfg = config.DEFAULT
        assert cfg.constants == "acdc"
        assert cfg.fitted_projection == "fortran"
        assert cfg.charge_balance == 0
        assert cfg.clamp_negative_j is True

    def test_is_frozen(self) -> None:
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.DEFAULT.constants = "codata2018"  # type: ignore[misc]

    def test_resolve_constants_acdc(self) -> None:
        assert config.DEFAULT.resolve_constants()["K_B"] == config.K_B

    def test_resolve_constants_codata(self) -> None:
        import math

        resolved = config.FidelityConfig(constants="codata2018").resolve_constants()
        assert resolved["K_B"] == 1.380649e-23
        assert resolved["PI"] == math.pi

    def test_codata_differs_enough_to_matter(self) -> None:
        """Justifies making the constant set an explicit choice.

        k_B shifts by ~1e-6 relative, which reaches the collision rate as
        ~5e-7 (beta goes as sqrt(k_B T)) -- five orders above the 1e-12 rate
        gate, so selecting codata2018 will fail validation. That is intended:
        it should be a deliberate, visible act, not a silent default.
        """
        acdc = config.DEFAULT.resolve_constants()
        codata = config.FidelityConfig(constants="codata2018").resolve_constants()
        rel = abs(codata["K_B"] - acdc["K_B"]) / acdc["K_B"]
        assert 1e-7 < rel < 1e-5
