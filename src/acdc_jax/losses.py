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

import jax.numpy as jnp

from acdc_jax import config
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
    excluded: jnp.ndarray | None = None,
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
        excluded: optional bool mask of clusters with no sink -- the
            ``--cs_only X,0`` mechanism that removes the vapour monomers
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

    if excluded is not None:
        sink = jnp.where(jnp.asarray(excluded), 0.0, sink)
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
    acid_index: int,
    tube_radius: float = config.WL_DIFFUSION_TUBE_RADIUS,
    tube_pressure: float = config.WL_DIFFUSION_TUBE_PRESSURE,
    generic_ion_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flow-tube diffusion loss, relative to sulfuric acid in N2 (Perl :7283-7335).

    Kinetic-theory diffusivity ``D ~ (r + r_N2)^-2 sqrt(1/m + 1/m_N2)``, every
    cluster scaled relative to the acid monomer, whose absolute loss is set by
    its diffusivity at the tube pressure times Brown's laminar factor
    ``3.65/R^2``. The N2 radius comes from its viscosity (Present 1958,
    eq. 11-67) via Sutherland's formula.
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

    r_acid = radius[acid_index]
    m_acid = mass[acid_index]
    kt = config.K_B * temperature
    # Acid diffusivity in N2 at 1 atm (Present 1958, eq. 8-87), then to the
    # tube pressure, then to a wall loss.
    d_acid = (
        3.0
        / 8.0
        / config.P_ATM
        * kt
        / (r_acid + radius_n2) ** 2
        * jnp.sqrt(kt / (2.0 * config.PI) * (1.0 / m_acid + 1.0 / mass_n2))
    )
    wl_acid = (
        d_acid
        / (tube_pressure / config.P_ATM)
        * (config.WL_DIFFUSION_LAMINAR / tube_radius**2)
    )

    d0_factor = 1.0 / (r_acid + radius_n2) ** 2 * jnp.sqrt(1.0 / m_acid + 1.0 / mass_n2)
    d_factor = 1.0 / (radius + radius_n2) ** 2 * jnp.sqrt(1.0 / mass + 1.0 / mass_n2)
    wl = d_factor / d0_factor * wl_acid

    # The generic charger ions get NO diffusion wall loss. This branch alone
    # loops `for iclus = 1 .. $max_cluster` (Perl :7325) -- the real clusters
    # -- where every other loss branch loops to `$max_cluster_number`, which
    # includes the two generic ions. An inconsistency in the reference,
    # reproduced: the fixture has exactly 52 nonzero entries, zero at the
    # generic-ion slots. Fidelity F16.
    if generic_ion_mask is not None:
        wl = jnp.where(jnp.asarray(generic_ion_mask), 0.0, wl)
    return wl


def wall_loss(
    method: str,
    inputs: RateInputs,
    temperature,
    acid_index: int | None = None,
    fwl: float = config.FWL_DEFAULT,
    generic_ion_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Per-cluster wall loss for a named parameterization, 1/s, with the
    ion enhancement applied to charged clusters.

    Charged clusters are multiplied by ``fwl`` (3.3 by default) and neutrals
    are not -- confirmed in every generated fixture, where
    ``coef_lin(57,57,k)`` reads ``fwl*wl(k)`` for ions and ``wl(k)`` for
    neutrals.
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
        if acid_index is None:
            raise ValueError("the diffusion wall loss is relative to the acid monomer")
        wl = wall_loss_diffusion(
            inputs, temperature, acid_index, generic_ion_mask=generic_ion_mask
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


__all__ = [
    "air_mean_free_path",
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
