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


__all__ = [
    "air_mean_free_path",
    "air_viscosity",
    "background_coagulation_sink",
    "slip_corrected_diffusivity",
    "thermal_speed",
]
