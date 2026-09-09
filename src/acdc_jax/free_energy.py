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

    Two conventions copied from ``rates_and_deltags_ABe.m``: only neutral
    molecule types carry a pressure term ("the ion is not a molecule",
    `:483`), and in a ONE-component system the count is ``n - 1`` (`:518`)
    -- that chart plots the free energy relative to one monomer at its
    actual pressure -- where a two-component system uses the full counts
    (`:516`). When the positive ion is the proton pseudo-species, the
    protonated base is "the ion" too and its host molecule counts one less
    on positive clusters (`l_ch_nonmol_AB`, `:496-507`). Clusters without
    energy data keep their reference value; the generic charger ions, whose
    pseudo-compositions carry negative entries, are left at zero.
    """
    kt_kcal = config.K_B * temperature / config.KCAL_PER_MOL_TO_J
    reference = np.asarray(rates.gibbs_at(inputs, temperature))
    compositions = np.asarray(system.compositions)
    neutral_types = [
        t
        for t in range(len(system.order))
        if system.charges_of_molecule[t] == 0 and not system.molecule_is_pseudo[t]
    ]
    charges = np.asarray(system.charges)
    pseudo = np.asarray(system.molecule_is_pseudo, dtype=bool)
    carries_proton = (charges > 0) & (compositions[:, pseudo].sum(axis=1) > 0)
    actual = reference.copy()
    for t in neutral_types:
        label = f"1{system.order[t]}"
        if label not in vapours:
            raise ValueError(f"no vapour concentration given for {label}")
        counts = compositions[:, t].astype(float)
        if len(neutral_types) == 1:
            counts = np.maximum(counts - 1.0, 0.0)
        if system.order[t] == system.proton_host:
            counts = np.where(carries_proton, np.maximum(counts - 1.0, 0.0), counts)
        pressure = vapours[label] * config.K_B * temperature
        actual = actual - kt_kcal * counts * np.log(pressure / reference_pressure)
    for i in (system.generic_neg, system.generic_pos):
        if i >= 0:
            actual[i] = reference[i]
    return actual


def total_evaporation_rate(
    n_clusters: int, evaporation_k: np.ndarray, evaporation_rate: np.ndarray
) -> np.ndarray:
    """Sum of every evaporation channel leaving each cluster, 1/s: the
    QuickGuide's "overall evaporation" map."""
    total = np.zeros(n_clusters)
    np.add.at(total, np.asarray(evaporation_k), np.asarray(evaporation_rate))
    return total


def folded_count(system: AcdcSystem, counts: np.ndarray, molecule: str) -> np.ndarray:
    """Molecules of ``molecule`` per composition row, with the ions that
    derive from it (the header's `corresponding neutral molecule`, e.g. the
    bisulfate ion for the acid) folded in -- the QuickGuide's chart axes.

    ``counts`` is ``(..., n_types)`` in ``system.order``.
    """
    if molecule not in system.order:
        raise ValueError(f"{molecule!r} is not a molecule of the system")
    counts = np.asarray(counts)
    total = counts[..., system.order.index(molecule)].copy()
    for t, parent in enumerate(system.molecule_parent):
        if parent == molecule:
            total = total + counts[..., t]
    return total


def composition_grid(
    system: AcdcSystem,
    values: np.ndarray,
    row_molecule: str,
    column_molecule: str,
    charge: int = 0,
) -> np.ndarray:
    """Lay a per-cluster quantity on a (row molecule, column molecule) grid.

    One charging state at a time; ions fold onto their parent's axis via
    :func:`folded_count`. Cells with no cluster are NaN.
    """
    counts = np.asarray(system.compositions)
    rows = folded_count(system, counts, row_molecule)
    cols = folded_count(system, counts, column_molecule)
    grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    for i in range(system.n_clusters):
        if system.charges[i] != charge or i in (system.generic_neg, system.generic_pos):
            continue
        grid[rows[i], cols[i]] = values[i]
    return grid


__all__ = [
    "actual_free_energy",
    "composition_grid",
    "folded_count",
    "total_evaporation_rate",
]
