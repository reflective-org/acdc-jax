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

# bg_loss: coagulation onto a monodisperse background population
# (Perl :325-327 for the defaults, :6668-6699 for the physics). The
# generator dies if this is combined with variable temperature (:6665); JAX
# has no such limit, so here the sink follows temperature.

BG_CONCENTRATION_DEFAULT = 1e3 * 1e6
"""Background particle number concentration, m^-3 (generator default 1e3 cm^-3)."""

BG_DIAMETER_DEFAULT = 100e-9
"""Background particle diameter, m (generator default 100 nm)."""

BG_DENSITY_DEFAULT = 1000.0
"""Background particle density, kg/m^3."""

AIR_VISCOSITY_A = 2.5277e-7
AIR_VISCOSITY_B = 0.75302
"""Dynamic viscosity of air as ``A * T**B``, Pa s -- the DMAN fit ACDC
inherited (Perl :6668). Not Sutherland's form; matching the reference wins."""

AIR_MOLAR_MASS = 0.0289
"""kg/mol, in the mean-free-path expression (Perl :6670)."""

# Phillips (1975) slip correction to the Brownian diffusivity, as a rational
# function of lambda/r:  (5 + 4x + 6x^2 + 18x^3) / (5 - x + (8 + pi) x^2).
# The reference labels it "S&P Eq. 9.73" (Perl :6672, :6692).
PHILLIPS_SLIP_NUMERATOR = (5.0, 4.0, 6.0, 18.0)
PHILLIPS_SLIP_DENOMINATOR = (5.0, 1.0, 8.0)

# Wall losses (Perl :7100-7375). Six chamber/flow-tube parameterizations.
# All of them apply to EVERY cluster including the vapour monomers -- unlike
# the coagulation sink, there is no --cs_only exclusion for walls -- and
# charged clusters are multiplied by FWL_DEFAULT.

WL_IFT = 2.3e-2
"""IfT-LFT flow-tube wall loss, 1/s, flat (Berndt & Richters 2011; Perl :7373)."""

# CLOUD4 (Almeida et al. 2013): 1.66e-12 / d_mob, with d_mob = (d + 0.3 nm)
# * sqrt(1 + m_N2/m). Here the mass correction is applied CORRECTLY --
# $mass1 is the raw kg value -- unlike the metadata emission in F11.
WL_CLOUD4_JA = 1.66e-12
"""m/s, Perl :7208."""

WL_CLOUD3 = 1.310e-12
"""m/s, same form as CLOUD4_JA with the CLOUD3 coefficient (Perl :7251)."""

WL_CLOUD4_SIMPLE = 1.0e-12
"""m/s, over (d + 0.3 nm) with NO mass correction (Kurten et al. 2015; Perl :7230)."""

# CLOUD4_JK (Jasper Kirkby's Excel fit; Perl :7101-7118). wl = 0.774*sqrt(D)
# with D the slip-corrected Brownian diffusivity of the mobility diameter.
WL_JK_PREFACTOR = 0.774
WL_JK_VISCOSITY = (1.708e-5, 273.15, 1.5, 393.396, 120.246)
"""Sutherland-type air viscosity: 1.708e-5 (T/273.15)^1.5 * 393.396/(T+120.246)."""
WL_JK_SLIP = (2.0e-9, 101300.0, 0.752e-6, 6.32, 2.01, 0.1095e9)
"""Slip correction 1 + 2e-9/(P d 0.752e-6) (6.32 + 2.01 exp(-0.1095e9 P d 0.752e-6)),
at P = 101300 Pa, as written."""

# CLOUD4_AK (Andreas Kurten; Perl :7155-7175). wl = 0.77*sqrt(D) with D from
# the GEOMETRIC diameter (the generator's own comment: "for some reason").
WL_AK_PREFACTOR = 0.77
WL_AK_VISCOSITY = (11.798, 0.629976, -1.81158e-04, 1e-7)
"""Polynomial air viscosity: (11.798 + 0.629976 T - 1.81158e-4 T^2) * 1e-7."""
WL_AK_LAMBDA = (100000.0, 0.37e-9)
"""Mean free path k_B T / (sqrt2 * P * pi * d_mol^2) at P = 1e5 Pa, d_mol = 0.37 nm."""
WL_AK_SLIP = (1.142, 0.558, 0.999)
"""Cunningham-type slip: 1 + Kn (1.142 + 0.558 exp(-0.999/Kn))."""

# diffusion (flow tube of Hanson & Eisele 2000; Perl :7283-7335). Loss
# relative to the acid monomer's diffusivity in N2, scaled by the laminar
# diffusion-limited factor 3.65/R^2 (Brown 1978).
WL_DIFFUSION_TUBE_RADIUS = 0.049 / 2
"""m, Hanson & Eisele 2000 default; --flowtube_radius overrides (given in cm)."""
WL_DIFFUSION_TUBE_PRESSURE = 133.322 * 620
"""Pa, 620 Torr default; --flowtube_pressure overrides."""
WL_DIFFUSION_LAMINAR = 3.65
"""Brown (1978) tubular-reactor factor: D -> wall loss as 3.65 D / R^2."""
MASS_N2 = 28.01
"""g/mol."""
N2_SUTHERLAND = (17.9e-6, 300.0, 111.0)
"""N2 viscosity: 17.9e-6 (300+111)/(T+111) (T/300)^1.5 (Crane 1988 / CRC)."""

WEXLER = (
    -2991.2729,
    -6017.0128,
    18.87643854,
    -0.028354721,
    0.17838301e-4,
    -0.84150417e-9,
    0.44412543e-12,
    2.858487,
)
"""Wexler (1976) water saturation-pressure fit, Pa: exp(a0/T^2 + a1/T + a2 +
a3 T + a4 T^2 + a5 T^3 + a6 T^4 + a7 ln T). Perl :858. An older CNT-book
form sits commented out beside it upstream."""

MASS_WATER = 18.02
"""g/mol (Perl :854)."""
DENS_WATER = 997.0
"""kg/m^3 (Perl :855). A hydrate's volume is the dry volume plus n water
volumes at this bulk density."""

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

    charge_balance: int = 0
    """The generator's ``--charge_balance`` flag: 0, +1 or -1.

    ``0`` (default, the shipped example): both generic ions are sourced at
    the ion production rate and the equations balance themselves.

    ``+1``: the positive ion is not integrated. Before every RHS and
    formation evaluation it is SET to ``c(neg) + sum(negative clusters) -
    sum(positive clusters)``, i.e. whatever balances the net charge, floored
    at zero with the excess pushed onto the negative ion. ``-1`` is the
    mirror. Emitted at the top of feval/formation (fixture
    acdc_equations_cb1.f90:93-99), with the fitted ion marked isconst.

    Distinct from F2, which is the MATLAB driver's per-outer-iteration
    projection; this one is a generator option and part of the Fortran path.
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

    sticking_on_evaporation: Literal["upstream", "detailed_balance"] = "upstream"
    """How a --sticking_factor reaches the evaporation rate. See F19.

    Upstream emits ``E(i,j) = s*<prefactor>*exp(...)*K(i,j)`` where ``K(i,j)``
    already carries ``s``, so evaporation is scaled by s^2 and collision by s,
    breaking detailed balance by a factor s. ``"upstream"`` reproduces that;
    ``"detailed_balance"`` applies the factor once, through K only.
    """

    nonstandard_main_coefficient: Literal["fortran", "literal"] = "fortran"
    """What happens to the coefficient of the FIRST product of a --nst line.

    See F20. The generator folds it into ``coef_quad_form``, which only the
    MATLAB emitter reads; the Fortran forms the main product once per
    collision whatever the coefficient says (``1N 2A1N 2 1A1N`` forms one
    1A1N and loses the other). ``"fortran"`` reproduces that; ``"literal"``
    honours the coefficient. Extra products (third column onward) keep
    their coefficients on both paths.
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
