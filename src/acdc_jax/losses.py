"""External loss processes beyond the exp_loss sink in rates.py.

Phase 8. Each is a first-order loss per cluster, 1/s, that the right-hand
side books to the corresponding flux slot exactly as it books the exp_loss
coagulation sink.

Traced: functions of temperature where the physics has one. Upstream cannot
do that for ``bg_loss`` -- air viscosity and mean free path are
temperature-dependent and the generator dies rather than emit them
symbolically (Perl ``:6665``). JAX traces them like anything else, so here
the sink follows temperature. Validated at fixed temperature against a
fixture generated at that temperature, which is the only comparison
upstream can offer.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax.numpy as jnp

from acdc_jax import config, rates
from acdc_jax.rates import RateInputs


def air_viscosity(temperature) -> jnp.ndarray:
    """Dynamic viscosity of air, Pa s.

    The power-law fit ACDC inherited from DMAN (Perl ``:6668``), not the
    Sutherland form. Kept as-is: the reference's sinks are computed from it.
    """
    return config.AIR_VISCOSITY_A * temperature**config.AIR_VISCOSITY_B


def air_mean_free_path(temperature) -> jnp.ndarray:
    """Mean free path of air at standard pressure, m (S&P Eq. 9.6)."""
    thermal = jnp.sqrt(
        8.0
        * config.AIR_MOLAR_MASS
        / (config.PI * config.N_A * config.K_B * temperature)
    )
    return 2.0 * air_viscosity(temperature) / (config.P_ATM * thermal)


def slip_corrected_diffusivity(radius, temperature) -> jnp.ndarray:
    """Brownian diffusivity with the Phillips (1975) slip correction, m^2/s.

    ``D = k_B T / (3 pi mu) / (2 r) * P(lambda/r)`` where ``P`` is the
    rational function in ``PHILLIPS_SLIP``. Reduces to Stokes-Einstein in the
    continuum limit (``lambda/r -> 0``, ``P -> 1``) and grows as ``1/r^2``
    in the free-molecular limit. This is what the reference labels
    "S&P Eq. 9.73" (Perl ``:6672``).
    """
    knudsen_like = air_mean_free_path(temperature) / radius
    a5, a4, a6, a18 = config.PHILLIPS_SLIP_NUMERATOR
    b5, b1, b8 = config.PHILLIPS_SLIP_DENOMINATOR
    numerator = a5 + a4 * knudsen_like + a6 * knudsen_like**2 + a18 * knudsen_like**3
    denominator = b5 - b1 * knudsen_like + (b8 + config.PI) * knudsen_like**2
    stokes_einstein = (
        config.K_B * temperature / (3.0 * config.PI * air_viscosity(temperature))
    )
    return stokes_einstein / (2.0 * radius) * numerator / denominator


def thermal_speed(mass, temperature) -> jnp.ndarray:
    """``sqrt(8 k_B T / (pi m))``, m/s."""
    return jnp.sqrt(8.0 * config.K_B * temperature / (config.PI * mass))


def background_coagulation_sink(
    inputs: RateInputs,
    temperature,
    bg_concentration: float = config.BG_CONCENTRATION_DEFAULT,
    bg_diameter: float = config.BG_DIAMETER_DEFAULT,
    bg_density: float = config.BG_DENSITY_DEFAULT,
) -> jnp.ndarray:
    """``bg_loss``: collision frequency with a monodisperse background, 1/s.

    Fuchs-form Brownian coagulation kernel with Dahneke's transition-regime
    interpolation (Perl ``:6690-6699``)::

        Kn   = 2 (D_i + D_bg) / ( sqrt(c_i^2 + c_bg^2) (r_i + r_bg) )
        beta = 4 pi (r_i + r_bg) (D_i + D_bg) (1 + Kn) / (1 + 2 Kn (1 + Kn))
        CS_i = beta * N_bg

    This is the physical process that ``exp_loss`` approximates with a power
    law. Shape ``(nclust,)``.

    Args:
        bg_concentration: number concentration of background particles, m^-3
        bg_diameter: their diameter, m
        bg_density: their density, kg/m^3
    """
    radius = jnp.asarray(inputs.radius)
    mass = jnp.asarray(inputs.mass)

    r_bg = bg_diameter / 2.0
    mass_bg = bg_density * config.PI / 6.0 * bg_diameter**3

    d_i = slip_corrected_diffusivity(radius, temperature)
    d_bg = slip_corrected_diffusivity(r_bg, temperature)
    c_i = thermal_speed(mass, temperature)
    c_bg = thermal_speed(mass_bg, temperature)

    d_sum = d_i + d_bg
    r_sum = radius + r_bg
    knudsen = 2.0 * d_sum / (jnp.sqrt(c_i**2 + c_bg**2) * r_sum)
    kernel = (
        4.0
        * config.PI
        * r_sum
        * d_sum
        * (1.0 + knudsen)
        / (1.0 + 2.0 * knudsen * (1.0 + knudsen))
    )
    sink = kernel * bg_concentration

    # --cs_only X,0: the same exclusion the exp_loss shape carries, read from
    # the inputs so the two sinks can never disagree about who is scavenged.
    sink = jnp.where(jnp.asarray(inputs.cs_excluded), 0.0, sink)
    return sink


# ---------------------------------------------------------------------------
# Wall losses. Six parameterizations, each a 1/s loss per cluster. Unlike
# the coagulation sink there is no exclusion for the vapour monomers: every
# cluster hits the wall (Perl :7100-7375, and confirmed in the generated
# fixtures, where wl(1) for 1A is nonzero in every variant).
# ---------------------------------------------------------------------------


def mobility_diameter(inputs: RateInputs) -> jnp.ndarray:
    """``(d + 0.3 nm) sqrt(1 + m_N2/m)`` in metres -- the CORRECT form.

    The same expression the metadata emission gets wrong (fidelity F11).
    Here the wall-loss code uses the raw kg mass, so the Tammet correction
    is live (Perl :7208, :7112, :7251).
    """
    diameter = 2.0 * jnp.asarray(inputs.radius)
    mass_kg = jnp.asarray(inputs.mass)
    m_n2 = config.MOB_MASS_N2 * config.MASS_CONV
    return (diameter + config.MOB_DIAMETER_OFFSET) * jnp.sqrt(1.0 + m_n2 / mass_kg)


def wall_loss_ift(inputs: RateInputs) -> jnp.ndarray:
    """IfT-LFT flow tube: one flat rate for every cluster (Perl :7373)."""
    return jnp.full(inputs.radius.shape, config.WL_IFT)


def wall_loss_cloud4_ja(inputs: RateInputs) -> jnp.ndarray:
    """CLOUD4, Almeida et al. 2013: ``1.66e-12 / d_mob`` (Perl :7208)."""
    return config.WL_CLOUD4_JA / mobility_diameter(inputs)


def wall_loss_cloud3(inputs: RateInputs) -> jnp.ndarray:
    """CLOUD3: the CLOUD4_JA form with 1.31e-12 (Perl :7251)."""
    return config.WL_CLOUD3 / mobility_diameter(inputs)


def wall_loss_cloud4_simple(inputs: RateInputs) -> jnp.ndarray:
    """Simplified CLOUD4, Kurten et al. 2015: ``1e-12 / (d + 0.3 nm)``.

    Deliberately NO mass correction -- the generator's comment says so
    (Perl :7229).
    """
    diameter = 2.0 * jnp.asarray(inputs.radius)
    return config.WL_CLOUD4_SIMPLE / (diameter + config.MOB_DIAMETER_OFFSET)


def wall_loss_cloud4_jk(inputs: RateInputs, temperature) -> jnp.ndarray:
    """CLOUD4 from Jasper Kirkby's Excel fit: ``0.774 sqrt(D)`` (Perl :7101-7118).

    ``D`` is the Brownian diffusivity of the mobility diameter with a slip
    correction whose constants are transcribed exactly as written -- including
    a hard-coded 101300 Pa and a 0.752e-6 factor whose provenance the source
    does not record. Upstream dies on this with variable temperature; here it
    traces.
    """
    d = mobility_diameter(inputs)
    mu0, t0, expo, c1, c2 = config.WL_JK_VISCOSITY
    viscosity = mu0 * (temperature / t0) ** expo * c1 / (temperature + c2)
    a, pressure, b, c, e, f = config.WL_JK_SLIP
    scale = pressure * d * b
    slip = 1.0 + a / scale * (c + e * jnp.exp(-f * scale))
    diffusivity = slip * config.K_B * temperature / (3.0 * config.PI * viscosity * d)
    return config.WL_JK_PREFACTOR * jnp.sqrt(diffusivity)


def wall_loss_cloud4_ak(inputs: RateInputs, temperature) -> jnp.ndarray:
    """CLOUD4 from Andreas Kurten: ``0.77 sqrt(D)`` (Perl :7155-7175).

    Uses the GEOMETRIC diameter, not the mobility diameter -- the generator's
    own comment flags this ("for some reason"). Its own polynomial air
    viscosity and a hard-sphere mean free path at 1e5 Pa with a 0.37 nm
    molecular diameter, then a Cunningham-type slip correction.
    """
    d = 2.0 * jnp.asarray(inputs.radius)
    a, b, c, scale = config.WL_AK_VISCOSITY
    viscosity = (a + b * temperature + c * temperature**2) * scale
    pressure, d_mol = config.WL_AK_LAMBDA
    lam = config.K_B * temperature / (jnp.sqrt(2.0) * pressure * config.PI * d_mol**2)
    knudsen = lam / d
    s1, s2, s3 = config.WL_AK_SLIP
    slip = 1.0 + knudsen * (s1 + s2 * jnp.exp(-s3 / knudsen))
    diffusivity = slip * config.K_B * temperature / (3.0 * config.PI * viscosity * d)
    return config.WL_AK_PREFACTOR * jnp.sqrt(diffusivity)


def wall_loss_diffusion(
    inputs: RateInputs,
    temperature,
    tube_radius: float = config.WL_DIFFUSION_TUBE_RADIUS,
    tube_pressure: float = config.WL_DIFFUSION_TUBE_PRESSURE,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Flow-tube diffusion loss in N2 (Perl :7283-7335).

    Kinetic-theory diffusivity ``D = 3/8 kT/P (r + r_N2)^-2 sqrt(kT/2pi
    (1/m + 1/m_N2))`` (Present 1958, eq. 8-87) at the tube pressure, times
    Brown's laminar factor ``3.65/R^2``. The N2 radius comes from its
    viscosity (Present eq. 11-67) via Sutherland's formula.

    Upstream writes this as the acid monomer's absolute loss times each
    cluster's diffusivity relative to the acid's. The acid reference
    cancels exactly -- the result has no dependence on which cluster is
    called the reference -- so it is written directly here.
    """
    radius = jnp.asarray(inputs.radius)
    mass = jnp.asarray(inputs.mass)
    mass_n2 = config.MASS_N2 * config.MASS_CONV

    mu0, t0, c = config.N2_SUTHERLAND
    viscosity_n2 = mu0 * (t0 + c) / (temperature + c) * (temperature / t0) ** 1.5
    radius_n2 = (
        0.5
        * (5.0 / 16.0 / viscosity_n2) ** 0.5
        * (mass_n2 * config.K_B * temperature / config.PI) ** 0.25
    )

    kt = config.K_B * temperature
    diffusivity = (
        3.0
        / 8.0
        * kt
        / tube_pressure
        / (radius + radius_n2) ** 2
        * jnp.sqrt(kt / (2.0 * config.PI) * (1.0 / mass + 1.0 / mass_n2))
    )
    wl = diffusivity * config.WL_DIFFUSION_LAMINAR / tube_radius**2

    # F16: the 2020 generator's branch loops over the real clusters only
    # (Perl :7325), so the generic charger ions get NO diffusion wall loss
    # where every other loss includes them; the fixture has 52 nonzero
    # entries. The 2024 generator includes them.
    if fidelity.diffusion_wall_loss_generic_ions == "excluded_2020":
        wl = jnp.where(jnp.asarray(inputs.is_generic_ion), 0.0, wl)
    return wl


def wall_loss(
    method: str,
    inputs: RateInputs,
    temperature,
    fwl: float = config.FWL_DEFAULT,
    fidelity: config.FidelityConfig = config.DEFAULT,
    tube_radius: float = config.WL_DIFFUSION_TUBE_RADIUS,
    tube_pressure: float = config.WL_DIFFUSION_TUBE_PRESSURE,
) -> jnp.ndarray:
    """Per-cluster wall loss for a named parameterization, 1/s, with the
    ion enhancement applied to charged clusters.

    Charged clusters are multiplied by ``fwl`` (3.3 by default) and neutrals
    are not -- confirmed in every generated fixture, where
    ``coef_lin(57,57,k)`` reads ``fwl*wl(k)`` for ions and ``wl(k)`` for
    neutrals. ``tube_radius``/``tube_pressure`` (``--flowtube_radius``,
    ``--flowtube_pressure``) only matter to ``diffusion``.
    """
    if method == "ift":
        wl = wall_loss_ift(inputs)
    elif method == "cloud4_ja":
        wl = wall_loss_cloud4_ja(inputs)
    elif method == "cloud3":
        wl = wall_loss_cloud3(inputs)
    elif method == "cloud4_simple":
        wl = wall_loss_cloud4_simple(inputs)
    elif method == "cloud4_jk":
        wl = wall_loss_cloud4_jk(inputs, temperature)
    elif method == "cloud4_ak":
        wl = wall_loss_cloud4_ak(inputs, temperature)
    elif method == "diffusion":
        wl = wall_loss_diffusion(
            inputs, temperature, tube_radius, tube_pressure, fidelity
        )
    else:
        raise ValueError(f"unknown wall-loss method {method!r}")

    charged = jnp.asarray(inputs.charge) != 0
    return jnp.where(charged, fwl * wl, wl)


def dilution(inputs: RateInputs, rate: float = config.DILUTION_DEFAULT) -> jnp.ndarray:
    """Flat dilution loss for every cluster, 1/s (Perl :335, :558).

    A scalar upstream -- ``coef_lin(58,58,k) = dil`` for all k, generic ions
    included, with no ion enhancement -- broadcast here to a vector so the
    right-hand side treats it like any other first-order loss.
    """
    return jnp.full(inputs.radius.shape, rate)


@dataclasses.dataclass(frozen=True)
class LossSettings:
    """Which first-order losses a run includes, and their parameters.

    Mirrors the generator's ``--cs``, ``--use_wl --wl`` and ``--use_dilution``
    options. The default is the shipped example: the exp_loss coagulation
    sink alone.
    """

    coagulation: Literal["exp_loss", "bg_loss"] | None = "exp_loss"
    """None for no coagulation sink at all -- upstream's default, and what a
    `--use_wl`-only run has."""
    cs_exponent: float = config.CS_EXPONENT_DEFAULT
    """`--exp_loss_exponent`."""
    cs_reference: str | None = None
    """`--exp_loss_ref_cluster`; None means the reference the inputs were
    built with (the small-set path) or the first monomer (loop mode)."""
    bg_concentration: float = config.BG_CONCENTRATION_DEFAULT
    bg_diameter: float = config.BG_DIAMETER_DEFAULT
    bg_density: float = config.BG_DENSITY_DEFAULT
    wall: str | None = None
    """A ``wall_loss`` method name, or None for no wall loss."""
    fwl: float = config.FWL_DEFAULT
    tube_radius: float = config.WL_DIFFUSION_TUBE_RADIUS
    tube_pressure: float = config.WL_DIFFUSION_TUBE_PRESSURE
    dilution: float | None = None
    """Dilution rate, 1/s, or None for none."""


def first_order_losses(
    settings: LossSettings,
    inputs: RateInputs,
    temperature,
    cs_ref,
    fcs: float = config.FCS_DEFAULT,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> dict[str, jnp.ndarray]:
    """Every first-order loss the settings select, keyed by its flux slot.

    ``coag`` is either the exp_loss sink scaled by ``cs_ref`` or the
    bg_loss sink scaled by the background concentration (``cs_ref`` is then
    unused, as upstream's ``--variable_cs`` is meaningless for bg_loss), and
    is absent when ``coagulation`` is None; ``wall`` and ``dilution`` appear
    only when selected.
    """
    out: dict[str, jnp.ndarray] = {}
    if settings.coagulation is None:
        pass
    elif settings.coagulation == "exp_loss":
        out["coag"] = rates.coagulation_sink(inputs, cs_ref, fcs)
    elif settings.coagulation == "bg_loss":
        sink = background_coagulation_sink(
            inputs,
            temperature,
            settings.bg_concentration,
            settings.bg_diameter,
            settings.bg_density,
        )
        charged = jnp.asarray(inputs.charge) != 0
        out["coag"] = sink * jnp.where(charged, fcs, 1.0)
    else:  # pragma: no cover - Literal keeps this unreachable
        raise ValueError(f"unknown coagulation sink {settings.coagulation!r}")
    if settings.wall is not None:
        out["wall"] = wall_loss(
            settings.wall,
            inputs,
            temperature,
            settings.fwl,
            fidelity,
            settings.tube_radius,
            settings.tube_pressure,
        )
    if settings.dilution is not None:
        out["dilution"] = dilution(inputs, settings.dilution)
    return out


__all__ = [
    "LossSettings",
    "air_mean_free_path",
    "first_order_losses",
    "air_viscosity",
    "background_coagulation_sink",
    "dilution",
    "mobility_diameter",
    "slip_corrected_diffusivity",
    "thermal_speed",
    "wall_loss",
    "wall_loss_cloud3",
    "wall_loss_cloud4_ak",
    "wall_loss_cloud4_ja",
    "wall_loss_cloud4_jk",
    "wall_loss_cloud4_simple",
    "wall_loss_diffusion",
    "wall_loss_ift",
]
