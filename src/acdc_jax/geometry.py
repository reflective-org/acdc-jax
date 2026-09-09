"""Cluster mass, volume, radius and diameters.

Mirrors the Perl generator's `calculate_mass` (`:11509`) and
`calculate_volume_and_radius` (`:11547`), plus the emission block at
`:3670-3684` that turns them into the arrays in `acdc_system_*.f90`.

Plain NumPy: this is setup, computed once per cluster set.
"""

from __future__ import annotations

import numpy as np

from acdc_jax import config
from acdc_jax.clusterset import Molecule


def _volume_contributors(molecules: tuple[Molecule, ...]) -> np.ndarray:
    """Mask of molecules that contribute volume.

    The proton and the missing-proton pseudo-species contribute **mass but
    not volume** (Perl `:11498`, `:11566`) -- they are charge bookkeeping,
    not matter with a density. In the input file they are marked by having
    no density (`-`).
    """
    return np.array(
        [
            m.density is not None and not m.is_proton and not m.is_missing_proton
            for m in molecules
        ]
    )


def cluster_mass(
    compositions: np.ndarray, molecules: tuple[Molecule, ...]
) -> np.ndarray:
    """Cluster masses in g/mol. Shape (nclust,).

    Every molecule contributes, including the proton.

    Args:
        compositions: (nclust, n_molecules) integer counts
        molecules: in the same column order as `compositions`
    """
    masses = np.array([m.mass for m in molecules], dtype=np.float64)
    return np.asarray(compositions, dtype=np.float64) @ masses


def cluster_volume(
    compositions: np.ndarray, molecules: tuple[Molecule, ...]
) -> np.ndarray:
    """Cluster volumes in m^3. Shape (nclust,).

    Additive bulk liquid volumes: V = sum(n_i * m_i / rho_i). This is a
    crude model for a sub-nanometre cluster -- it assumes each molecule
    occupies the volume it would in the bulk liquid -- but it is what the
    reference does and what the collision cross-sections are built from.
    """
    contributes = _volume_contributors(molecules)
    per_molecule = np.zeros(len(molecules), dtype=np.float64)
    for i, m in enumerate(molecules):
        if contributes[i]:
            per_molecule[i] = m.mass * config.MASS_CONV / m.density
    return np.asarray(compositions, dtype=np.float64) @ per_molecule


def cluster_radius(
    compositions: np.ndarray, molecules: tuple[Molecule, ...]
) -> np.ndarray:
    """Mass-equivalent radii in m: r = (3V / 4pi)^(1/3)."""
    volume = cluster_volume(compositions, molecules)
    return (3.0 * volume / (4.0 * config.PI)) ** (1.0 / 3.0)


def cluster_diameter(
    compositions: np.ndarray, molecules: tuple[Molecule, ...]
) -> np.ndarray:
    """Mass-equivalent diameters in nm."""
    return 2e9 * cluster_radius(compositions, molecules)


def mobility_diameter(
    compositions: np.ndarray,
    molecules: tuple[Molecule, ...],
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> np.ndarray:
    """Mobility diameters in nm.

    The reference intends the standard mass-diffusion correction,

        d_mob = (d_mass + 0.3 nm) * sqrt(1 + m_carrier / m_cluster)

    but does not compute it. At Perl `:3672` ``$mass1`` is reassigned to the
    mass in **g/mol**::

        $mass1 = sprintf("%.2f", $mass1/$mass_conv);

    and then at `:3681` the correction multiplies by ``$mass_conv`` again::

        (2.0e9*$radius+0.3)*sqrt(1+28.8*$mass_conv/$mass1)

    so the term is ``28.8 * 1.66e-27 / 98.08`` ~ 5e-28 for sulfuric acid and
    the square root is **exactly 1.0** in double precision. The emitted
    mobility diameters are therefore just ``d_mass + 0.3 nm``: every entry
    of `get_mob_diameter` in the generated system module differs from
    `get_diameter` by 0.30, with no mass dependence at all.

    Reproduced by default, because these diameters feed the size-bin
    classifier and any downstream comparison. `fidelity.mobility_diameter =
    "tammet"` applies the correction as intended, which shifts 1A from
    0.85 nm to 0.97 nm.
    """
    diameter = cluster_diameter(compositions, molecules)
    if fidelity.mobility_diameter == "fortran":
        return diameter + config.MOB_DIAMETER_OFFSET * 1e9

    mass = cluster_mass(compositions, molecules)
    correction = np.sqrt(1.0 + config.MOB_MASS_N2 / mass)
    return (diameter + config.MOB_DIAMETER_OFFSET * 1e9) * correction


def emitted(values: np.ndarray) -> np.ndarray:
    """Round as the generator does when writing the arrays.

    Masses and diameters are emitted through ``sprintf("%.2f", ...)``
    (Perl `:3672`, `:3677`, `:3681`), so comparing a full-precision
    computation against `acdc_system_*.f90` fails at ~1e-3 unless the same
    rounding is applied first.
    """
    return np.round(values, 2)
