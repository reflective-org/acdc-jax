"""Constants and fidelity configuration.

Every constant used anywhere in the port lives here. No magic numbers in the
physics modules — a value that appears inline is a value nobody can check
against the reference.

Each entry cites where it comes from in ``fortran/``. Paths are relative to
the repository root:

- ``fortran/perl/acdc_2024_02_12.pl`` — the equation generator (``P:``)
- ``fortran/src/solvers/solution_settings.f90`` — solver tolerances (``S:``)
- ``fortran/src/acdc_system_AN_ions_example.f90`` — generated metadata
"""

from __future__ import annotations

import dataclasses
from typing import Literal

# ---------------------------------------------------------------------------
# Physical constants — ACDC's own values (P:924-928)
# ---------------------------------------------------------------------------
# These differ from CODATA-2018 in the last digits. Matching the reference
# matters more than being current: the rate-constant acceptance gate is
# 1e-12 relative, and these differences are of that order. Modern values are
# selectable via FidelityConfig.constants for anyone who wants them.

K_B = 1.3806504e-23
"""Boltzmann constant, J/K (P:924). CODATA-2018: 1.380649e-23."""

N_A = 6.02214179e23
"""Avogadro constant, 1/mol (P:926). CODATA-2018: 6.02214076e23."""

PI = 3.14159265359
"""ACDC's truncated pi (P:925).

Deliberately not ``math.pi``. The truncation is 6.6e-14 relative, which
propagates to ~1.1e-14 in the hard-sphere collision rate — about 1% of the
1e-12 rate gate. Small, but free to match exactly, and unexplained
near-threshold failures are expensive to debug.
"""

MASS_CONV = 1.0 / N_A / 1000.0
"""g/mol -> kg (P:927)."""

KCAL_PER_MOL_TO_J = 4.184 * 1000.0 / N_A
"""kcal/mol -> J per molecule (P:928)."""

P_ATM = 101325.0
"""Standard pressure, Pa. The reference pressure in the bundled energy file."""

# CODATA-2018 alternatives, for FidelityConfig.constants == "codata2018".
_CODATA2018 = {"K_B": 1.380649e-23, "N_A": 6.02214076e23, "PI": 3.141592653589793}

# ---------------------------------------------------------------------------
# Solver tolerances (S:5-23)
# ---------------------------------------------------------------------------
# test_config.py re-parses solution_settings.f90 and asserts these match, so
# a vendored-reference bump that changes a tolerance fails loudly.

RTOL_SOLVER = 1e-5
"""Relative tolerance: VODE rtol, and the Euler per-step local error tol (S:6)."""

ATOL_SOLVER = 1e-6
"""Absolute tolerance, m^-3 (S:9). 1e-6 m^-3 = 1e-12 cm^-3, i.e. "zero"."""

ATOL_INTEGRATION = 1.0
"""Absolute tolerance actually used by this port's integrator, m^-3.

DELIBERATELY DIFFERENT from ATOL_SOLVER, which is the reference's literal
value. See docs/fidelity.md F14.

1e-6 m^-3 is 1e-12 cm^-3 -- a millionth of a molecule per cubic metre. VODE
tolerates being asked for that because its BDF implementation rescales
internally; diffrax's Kvaerno5 does not, and on the evaporation-dominated
corner of the condition grid (lowest vapour, lowest temperature) it exhausts
a 20,000-step budget rather than converging.

The formation rate is insensitive to this across eight orders of magnitude:
measured J identical from atol = 1e-6 to 1e4, and the same to 1e-7 relative
against the Fortran either way. 1.0 m^-3 is still six orders below any
physically meaningful concentration. `test_solve.py` asserts the
insensitivity so the justification is checked rather than merely claimed.
"""

CHMAX = 1e-2
"""Max relative concentration change per Euler step (S:12)."""

CHTOL = 1e-6
"""Concentration floor, m^-3, for relative-change denominators (S:15)."""

NEGTOL = -1e-6
"""Most negative concentration accepted, m^-3 (S:18)."""

SSTOL = 1e-5
"""Steady state: max relative change in concentrations (S:21).

Note the three-way disagreement recorded in docs/fidelity.md F3: Fortran
uses 1e-5, the generated MATLAB driver uses difftol=1e-3, and the manual
documents 1e-5.
"""

SSTIMECH = 6e2
"""Steady state: minimum interval between convergence checks, s (S:22)."""

SSTIMETOT = 1.2e3
"""Steady state: minimum total simulated time before SS may be declared, s (S:23)."""

SS_MAX_TIME = 1e8
"""Cap on time the driver may use reaching steady state, s.

Not in solution_settings; hardcoded at ``get_acdc_J.f90:101``, where it
overrides whatever simulation time the caller passed.
"""

SOLVER_EQ_LIMIT = 5000
"""Above this many equations the reference abandons VODE for Euler.

``get_acdc_J.f90:79``. The reason is cost, not physics: VODE here is dense
LU with ``work(22+9*neqn+2*neqn**2)``. Recorded because it explains loop
mode's existence, not because this port needs the same threshold.
"""

# ---------------------------------------------------------------------------
# Collision and evaporation model constants
# ---------------------------------------------------------------------------

RECOMB_COEFF = 1.6e-12
"""Ion-ion recombination coefficient, m^3/s (P:371).

Flat: every opposite-charge pair, no size dependence whatsoever.
"""

# Su & Chesnavich (1982) ion-neutral capture, the default (P:8853-8859).
#   x     = 100*mu_D / sqrt(2*alpha*k_B*1e23*T)      mu_D in Debye, alpha in A^3
#   ratio = (x + SU82_A)^2 / SU82_B + SU82_C          for x <  SU82_X_SWITCH
#         = SU82_D*x + SU82_E                         for x >= SU82_X_SWITCH
#   beta  = max(ratio * L, beta_hard_sphere)
#   L     = 2*pi*SU82_LANGEVIN * sqrt(alpha*(1/m1 + 1/m2) / 1e27)
SU82_A = 0.5090
SU82_B = 10.526
SU82_C = 0.9754
SU82_D = 0.4767
SU82_E = 0.62
SU82_X_SWITCH = 2.0
SU82_LANGEVIN = 4.8032e-16

# Su & Bowers (1973), the alternative (P:8825).
#   fcr = (SU73_POL*sqrt(alpha) + SU73_DIP*mu_D/sqrt(T)) * sqrt(1/m1 + 1/m2)
#         / beta_hard_sphere,  floored at 1
SU73_POL = 9.5436e-29
SU73_DIP = 6.4805e-27

ION_ENHANCEMENT_CONSTANT = 10.0
"""Size-independent enhancement for ``--ion_coll_method constant`` (P:8889)."""

# ---------------------------------------------------------------------------
# Loss constants
# ---------------------------------------------------------------------------

CS_EXPONENT_DEFAULT = -1.6
"""exp_loss size exponent m in CS = CS_ref*(d/d_ref)^m.

``acdc_system_AN_ions_example.f90:14``; generator default at P:437.
"""

CS_COEFFICIENT_DEFAULT = 2.6e-3
"""exp_loss reference coagulation sink, 1/s (acdc_system_AN_ions_example.f90:15)."""

FCS_DEFAULT = 1.0
"""Ion enhancement factor for the coagulation sink (P:'--fcs' default)."""

FWL_DEFAULT = 3.3
"""Ion enhancement factor for wall losses (P:'--fwl' default). Phase 8."""

DILUTION_DEFAULT = 9.6e-5
"""Default dilution rate, 1/s (P:'--dil_value'). Phase 8."""

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

MOB_DIAMETER_OFFSET = 0.3e-9
"""Mass -> mobility diameter offset, m.

d_mob = (d_mass + 0.3nm) * sqrt(1 + MOB_MASS_N2/m). P:3961.
"""

MOB_MASS_N2 = 28.8
"""Carrier-gas mass in the mobility-diameter correction, g/mol (P:3961)."""

# Generic charger-ion properties (P:930-943).
MASS_NEG_ION = 32.00
"""Generic negative charger ion mass, g/mol -- an O2 molecule."""

DENS_NEG_ION = 1141.0
"""Generic negative charger ion density, kg/m^3."""

MASS_POS_ION = 19.02
"""Generic positive charger ion mass, g/mol -- a charged water molecule."""

DENS_POS_ION = 997.0
"""Generic positive charger ion density, kg/m^3."""

MASS_NEG_ION_NITRATE = 125.0
"""Negative ion mass under --nitrate, g/mol -- a nitrate dimer."""

DENS_NEG_ION_NITRATE = 1512.9
"""Negative ion density under --nitrate, kg/m^3."""

# ---------------------------------------------------------------------------
# Fidelity configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FidelityConfig:
    """Switches for places where the port could differ from upstream.

    **Every default reproduces the Fortran.** Each field has a row in
    ``docs/fidelity.md`` and a test at both settings. Getting a default
    backwards silently changes results; that is why this exists rather than
    a scattering of keyword arguments.

    Populated as quirks are encountered, so it grows through the phases.
    """

    constants: Literal["acdc", "codata2018"] = "acdc"
    """Which physical constants to use. See F8.

    ``"acdc"`` reproduces the reference bit-for-bit. ``"codata2018"`` is
    physically current and shifts collision rates by ~1e-8 relative --
    enough to fail the rate gate, which is the point of making it explicit.
    """

    fitted_projection: Literal["fortran", "matlab"] = "fortran"
    """How a fitted-concentration constraint is projected. See F1.

    ``"fortran"``: assign the residual to the monomer, rescaling
    multiplicatively only if that would go negative
    (``driver_acdc_J.f90:405-410``). ``"matlab"``: always rescale
    multiplicatively (P:10336-10344). Same constraint, different
    distributions.
    """

    enforce_charge_balance: bool = False
    """Project the generic ions onto charge neutrality each outer iteration.

    See F2. MATLAB does this (P:10346-10377); Fortran does not, relying on
    symmetric ion sources. Arguably a correctness feature rather than a
    diagnostic, which is why it is available -- but the default is Fortran.
    """

    clamp_negative_j: bool = True
    """Replace negative formation rates with 1e-100 rather than reporting.

    See F4. ``get_acdc_J.f90:117``.
    """

    ion_collision_method: Literal[
        "su82", "su73", "constant", "constant_no_enhancement"
    ] = "su82"
    """Ion-neutral collision parameterization. See F15.

    ``"su82"`` (Su & Chesnavich 1982) is the reference's default.
    ``"su73"`` (Su & Bowers 1973) uses the dipole locking coefficients from
    the dipole file header, which Su82 reads but ignores.

    The two ``constant`` values exist because upstream's behaviour depends on
    whether temperature is a runtime variable:

    - ``"constant"`` applies the documented size-independent factor of 10,
      which is what upstream does at FIXED temperature.
    - ``"constant_no_enhancement"`` reproduces upstream's VARIABLE-temperature
      path, where the factor is silently dropped and ion-neutral collisions
      come out at the bare hard-sphere rate. Verified by generating that
      fixture: zero of 587 ion-neutral pairs carry any enhancement.

    Split into two names rather than one flag because either choice would
    otherwise be silent, and 'constant' quietly meaning 'no enhancement' is
    a worse trap than an extra option.
    """

    mobility_diameter: Literal["fortran", "tammet"] = "fortran"
    """How the mobility diameter is computed. See F11.

    ``"fortran"``: ``d_mass + 0.3 nm``, which is what the reference actually
    emits -- its intended mass-diffusion correction is defeated by a unit
    bug (``$mass1`` is in g/mol but the formula multiplies by ``$mass_conv``
    again, so the square root is exactly 1.0). ``"tammet"`` applies the
    correction as intended, moving 1A from 0.85 to 0.97 nm.
    """

    hydrate_discard_threshold: float = 0.99
    """Discard a hydrate distribution normalising at or below this and treat
    the cluster as dry. See F7, P:2637-2643. Phase 8.
    """

    def resolve_constants(self) -> dict[str, float]:
        """Return the constant set this config selects."""
        if self.constants == "acdc":
            return {"K_B": K_B, "N_A": N_A, "PI": PI}
        return dict(_CODATA2018)


DEFAULT = FidelityConfig()
"""The reference-reproducing configuration. Use this unless you mean not to."""
