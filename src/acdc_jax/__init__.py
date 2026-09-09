"""acdc-jax — differentiable atmospheric molecular-cluster dynamics.

A faithful JAX port of ACDC (Atmospheric Cluster Dynamics Code;
Kupiainen-Määttä & Olenius), validated against the vendored Fortran
reference under ``fortran/``.

This module is the **only** place ``jax_enable_x64`` is set. Cluster
concentrations span ~30 orders of magnitude and evaporation rates are
exponentials of ΔG/kT, so float32 is a different trajectory rather than a
noisier one. See ``CLAUDE.md``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from acdc_jax import config as config  # noqa: E402

__all__ = ["config"]
__version__ = "0.1.0"
