"""Free-energy surfaces at the actual vapour pressures.

The energy file holds *reference* formation free energies, at the file's
reference pressure (1 atm) for every constituent. The setup QuickGuide's
adequacy check wants the *actual* surface at the studied vapour
concentrations,

    dG_act = dG_ref - k_B T sum_m n_m ln(p_m / p_ref),

with ``p_m = c_m k_B T`` the vapour's partial pressure (MATLAB
``rates_and_deltags_ABe.m:516``). Only neutral molecules count: "the ion is
not a molecule" (`:483`), so the charge-carrying species -- the bisulfate
ion, the proton -- contribute no partial-pressure term. Plain NumPy.
"""

from __future__ import annotations

import numpy as np

from acdc_jax import config, rates
from acdc_jax.system import AcdcSystem


def actual_free_energy(
    system: AcdcSystem,
    inputs: rates.RateInputs,
    temperature: float,
    vapours: dict[str, float],
    reference_pressure: float = config.P_ATM,
) -> np.ndarray:
    """dG_act per cluster, kcal/mol, at the given vapour concentrations.

    Args:
        vapours: neutral monomer label -> concentration, 1/m^3, e.g.
            ``{"1A": 5e12, "1N": 2.6e15}``. Every neutral molecule type in
            the system must be given; a type absent from a cluster simply
            contributes nothing to it.

    Clusters without energy data (the monomers, whose reference is zero)
    come back as their reference value, zero.
    """
    kt_kcal = config.K_B * temperature / config.KCAL_PER_MOL_TO_J
    reference = np.asarray(rates.gibbs_at(inputs, temperature))
    compositions = np.asarray(system.compositions)
    actual = reference.copy()
    for t, molecule in enumerate(system.order):
        if system.charges_of_molecule[t] != 0 or system.molecule_is_pseudo[t]:
            continue
        label = f"1{molecule}"
        if label not in vapours:
            raise ValueError(f"no vapour concentration given for {label}")
        pressure = vapours[label] * config.K_B * temperature
        actual = actual - kt_kcal * compositions[:, t] * np.log(
            pressure / reference_pressure
        )
    return actual


def total_evaporation_rate(
    n_clusters: int, evaporation_k: np.ndarray, evaporation_rate: np.ndarray
) -> np.ndarray:
    """Sum of every evaporation channel leaving each cluster, 1/s: the
    QuickGuide's "overall evaporation" map."""
    total = np.zeros(n_clusters)
    np.add.at(total, np.asarray(evaporation_k), np.asarray(evaporation_rate))
    return total


def composition_grid(
    system: AcdcSystem,
    values: np.ndarray,
    row_molecule: str,
    column_molecule: str,
    charge: int = 0,
) -> np.ndarray:
    """Lay a per-cluster quantity on a (row molecule, column molecule) grid.

    For the two-component charts of the QuickGuide: rows count
    ``row_molecule``, columns ``column_molecule``, one charging state at a
    time; the charged forms of a molecule (the bisulfate ion for the acid,
    the protonated base) are folded into their neutral parent's count.
    Cells with no cluster are NaN.
    """
    counts = np.asarray(system.compositions)
    row_t = [t for t, name in enumerate(system.order) if name == row_molecule]
    col_t = [t for t, name in enumerate(system.order) if name == column_molecule]
    if not row_t or not col_t:
        raise ValueError("row/column molecule not in the system")
    rows = counts[:, row_t[0]].copy()
    cols = counts[:, col_t[0]].copy()
    for t, name in enumerate(system.order):
        if system.charges_of_molecule[t] == 0 or system.molecule_is_pseudo[t]:
            continue
        # a charged molecule type: fold into whichever axis it belongs to by
        # convention -- acids' ions into the acid axis, bases' into the base
        rows = rows + (
            counts[:, t] if name in _ION_PARENTS.get(row_molecule, ()) else 0
        )
        cols = cols + (
            counts[:, t] if name in _ION_PARENTS.get(column_molecule, ()) else 0
        )
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    for i in range(system.n_clusters):
        if system.charges[i] != charge:
            continue
        if i in (system.generic_neg, system.generic_pos):
            continue
        grid[rows[i], cols[i]] = values[i]
    return grid


_ION_PARENTS = {"A": ("B",), "N": ()}
"""Which charged molecule types count on which neutral axis. The AN example
declares B (bisulfate) as A's negative ion; the proton is a pseudo-species
and counts nowhere."""


__all__ = ["actual_free_energy", "composition_grid", "total_evaporation_rate"]
