"""Hydrate averaging -- the ``--rh`` option.

Water is not a species in ACDC. Each dry cluster is instead taken to be in
instantaneous equilibrium with its hydrates, weighted by a Boltzmann
distribution over the hydrate formation free energies, and every rate
coefficient is averaged over that distribution:

* collisions (Perl :8117-8296): ``K_eff(i,j) = sum_a sum_b d_i[a] d_j[b]
  K(i_a, j_b)``, with the ion-neutral enhancement evaluated per hydrate
  pair from the hydrate's own dipole and polarizability;
* evaporations (:8998-9060): ``E_eff(k->i+j) = sum_c d_k[c] sum_{a+b=c}
  E(k_c -> i_a + j_b)`` -- the parent's distribution only, and water is
  conserved across the channel (a hydrate can only split into hydrates
  whose waters add up to its own);
* the coagulation sink (:6577-6595, :6826-6850): ``cs_eff(i) = sum_a d_i[a]
  cs(i_a)``, with special ``--cs_only`` values used as given.

The port does this by EXPANSION: every (cluster, water count) pair becomes
a species in a second :class:`~acdc_jax.rates.RateInputs`, the ordinary
rate formulas run on that, and a weight matrix contracts the result back
to the dry clusters. That reuses every validated formula unchanged, and it
makes the whole thing a traced function of temperature -- which upstream
refuses to allow (``--rh`` with ``--variable_temp`` dies, Perl :818).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from acdc_jax import config, rates
from acdc_jax.reactions import ReactionSet
from acdc_jax.system import AcdcSystem
from acdc_jax.thermo import DipoleTable, EnergyTable

WATER = "W"


def water_saturation_pressure(temperature):
    """Saturation vapour pressure of water, Pa -- the Wexler (1976) fit ACDC
    uses (Perl :858). Traces."""
    t = temperature
    a = config.WEXLER
    return jnp.exp(
        a[0] * t**-2
        + a[1] / t
        + a[2]
        + a[3] * t
        + a[4] * t**2
        + a[5] * t**3
        + a[6] * t**4
        + a[7] * jnp.log(t)
    )


@dataclasses.dataclass(frozen=True)
class HydrateModel:
    """Static bookkeeping for hydrate averaging of one system.

    Species ``s`` is the ``waters[s]``-hydrate of dry cluster ``owner[s]``;
    the dry cluster itself is a species with zero waters, so every cluster
    has at least one. Evaporation channels are pre-expanded into
    water-conserving hydrate combinations.
    """

    inputs: rates.RateInputs
    """The dry inputs this model was built from."""
    expanded: rates.RateInputs
    """One entry per species."""
    owner: np.ndarray
    waters: np.ndarray
    n_clusters: int
    rh_percent: float
    reference_pressure: float
    """The energy file's, used both for the water activity and for E."""

    channel_parent: np.ndarray
    channel_i: np.ndarray
    channel_j: np.ndarray
    """The dry evaporation channels, in the order the caller supplied."""
    combo_channel: np.ndarray
    combo_parent: np.ndarray
    combo_i: np.ndarray
    combo_j: np.ndarray
    """Hydrate combinations: species indices, grouped by dry channel."""

    @property
    def n_species(self) -> int:
        return len(self.owner)

    @property
    def has_hydrates(self) -> np.ndarray:
        counts = np.bincount(self.owner, minlength=self.n_clusters)
        return counts > 1


def build_hydrate_model(
    system: AcdcSystem,
    energies: EnergyTable,
    dipoles: DipoleTable,
    inputs: rates.RateInputs,
    reactions: ReactionSet,
    rh_percent: float,
    cs_exponent: float = config.CS_EXPONENT_DEFAULT,
) -> HydrateModel:
    """Expand a dry system into hydrate species (Perl :2302-2360, :2401-2450).

    A cluster's hydrates are the energy-file rows ``<label><n>W``; the dry
    cluster is species 0 of its own. Mass adds ``n`` waters, volume adds
    ``n`` bulk-water volumes (the ``l_bulk_density`` path, the only one
    without a radius file). A hydrate's dipole and polarizability come from
    the dipole file's matching row; a neutral hydrate without one, in a
    system with ions, is an error rather than a silent zero (upstream
    counts it as ``missing_fcr`` and later dies).

    Args:
        cs_exponent: the ``exp_loss`` exponent, needed to size the
            hydrates' sink relative to the dry reference cluster; the dry
            ``cs_shape`` already carries the reference and the
            ``--cs_only`` zeros, which are inherited by owner.
    """
    n = system.n_clusters
    owner: list[int] = []
    waters: list[int] = []
    for i, label in enumerate(system.labels):
        owner.append(i)
        waters.append(0)
        # Monomers usually have no energy row of their own (their G is the
        # zero reference) but may still have hydrates; the dry species then
        # carries the zero from `inputs`.
        w = 1
        while f"{label}{w}{WATER}" in energies.delta_h:
            owner.append(i)
            waters.append(w)
            w += 1
        # Upstream tolerates a gap (it warns and leaves the entry undefined);
        # a gap here just ends the ladder, which is the same distribution.
    owner_arr = np.array(owner)
    waters_arr = np.array(waters)
    species = len(owner_arr)
    labels_h = [
        system.labels[o] + (f"{w}{WATER}" if w else "")
        for o, w in zip(owner_arr, waters_arr, strict=True)
    ]

    mass = inputs.mass[owner_arr] + waters_arr * config.MASS_WATER * config.MASS_CONV
    water_volume = config.MASS_WATER * config.MASS_CONV / config.DENS_WATER
    volume = (
        4.0 / 3.0 * config.PI * inputs.radius[owner_arr] ** 3
        + waters_arr * water_volume
    )
    radius = (3.0 * volume / (4.0 * config.PI)) ** (1.0 / 3.0)
    diameter_nm = 2e9 * radius

    charge = inputs.charge[owner_arr]
    has_ions = bool(np.any(inputs.charge != 0))
    dipole = inputs.dipole[owner_arr].copy()
    polarizability = inputs.polarizability[owner_arr].copy()
    for s in range(species):
        if waters_arr[s] == 0:
            continue
        label = labels_h[s]
        if label in dipoles.dipole:
            dipole[s] = dipoles.dipole[label]
            polarizability[s] = dipoles.polarizability[label]
        elif has_ions and charge[s] == 0:
            raise ValueError(
                f"no dipole moment/polarizability for hydrate {label}; the "
                "ion-neutral enhancement cannot be averaged without it"
            )
    locking = np.array(
        [
            dipoles.monomer_locking if system.is_monomer(o) else dipoles.cluster_locking
            for o in owner_arr
        ]
    )
    dipole_locked = dipole * locking

    delta_h = inputs.delta_h[owner_arr].copy()
    delta_s = inputs.delta_s[owner_arr].copy()
    for s in range(species):
        if waters_arr[s]:
            delta_h[s] = energies.delta_h[labels_h[s]]
            delta_s[s] = energies.delta_s[labels_h[s]]

    pair_kind, neutral_partner = rates._classify_pairs(charge)

    # Sink: the dry shape carries (d_i/d_ref)^m and the --cs_only zeros;
    # a hydrate rescales its owner's by its own diameter.
    cs_shape = (
        inputs.cs_shape[owner_arr]
        * (diameter_nm / inputs.diameter_nm[owner_arr]) ** cs_exponent
    )

    expanded = rates.RateInputs(
        mass=mass,
        radius=radius,
        diameter_nm=diameter_nm,
        dipole=dipole,
        polarizability=polarizability,
        dipole_locked=dipole_locked,
        charge=charge,
        pair_kind=pair_kind,
        neutral_partner=neutral_partner,
        delta_g_kcal=np.stack([delta_h, delta_s], axis=1),
        delta_h=delta_h,
        delta_s=delta_s,
        cs_shape=cs_shape,
        cs_excluded=inputs.cs_excluded[owner_arr],
        is_generic_ion=inputs.is_generic_ion[owner_arr],
        sticking=inputs.sticking[np.ix_(owner_arr, owner_arr)],
        evap_scale_kcal=inputs.evap_scale_kcal[np.ix_(owner_arr, owner_arr)],
        has_energy_data=inputs.has_energy_data[owner_arr],
        valid_pairs=inputs.valid_pairs[np.ix_(owner_arr, owner_arr)],
    )

    # Water-conserving evaporation combinations (Perl :9003-9020).
    by_cluster: dict[int, dict[int, int]] = {i: {} for i in range(n)}
    for s in range(species):
        by_cluster[int(owner_arr[s])][int(waters_arr[s])] = s

    parents = np.array([e.k for e in reactions.evaporations], dtype=int)
    di = np.array([e.i for e in reactions.evaporations], dtype=int)
    dj = np.array([e.j for e in reactions.evaporations], dtype=int)
    combo = {"channel": [], "parent": [], "i": [], "j": []}
    for c, (k, i, j) in enumerate(zip(parents, di, dj, strict=True)):
        for kw, ks in by_cluster[k].items():
            found = False
            for iw, i_s in by_cluster[i].items():
                jw = kw - iw
                if jw not in by_cluster[j]:
                    continue
                if i == j and jw < iw:
                    continue  # the same evaporation, already counted
                found = True
                combo["channel"].append(c)
                combo["parent"].append(ks)
                combo["i"].append(i_s)
                combo["j"].append(by_cluster[j][jw])
            if not found:
                raise ValueError(
                    f"no evaporation products for {labels_h[ks]} -> "
                    f"{system.labels[i]} + {system.labels[j]} hydrates"
                )

    return HydrateModel(
        inputs=inputs,
        expanded=expanded,
        owner=owner_arr,
        waters=waters_arr,
        n_clusters=n,
        rh_percent=rh_percent,
        reference_pressure=energies.pressure,
        channel_parent=parents,
        channel_i=di,
        channel_j=dj,
        combo_channel=np.array(combo["channel"], dtype=int),
        combo_parent=np.array(combo["parent"], dtype=int),
        combo_i=np.array(combo["i"], dtype=int),
        combo_j=np.array(combo["j"], dtype=int),
    )


def hydrate_weights(
    model: HydrateModel,
    temperature,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Normalised hydrate distributions as a (n_clusters, n_species) matrix.

    Row ``i`` holds ``d_i`` on that cluster's species and zeros elsewhere
    (Perl :2401-2450). ``w_a = (RH/100 * p_sat(T) / p_ref)^a *
    exp(-(G_a - G_dry) / kT)``, normalised per cluster.

    The generator then drops any distribution whose normalised sum is not
    above ``hydrate_discard_threshold`` and treats that cluster as dry. A
    sum of normalised weights is 1 by construction, so in practice this
    catches only a non-finite distribution (an overflowing exponent): NaN
    fails the comparison and the cluster silently reverts to dry. Same
    here.
    """
    owner = jnp.asarray(model.owner)
    waters = jnp.asarray(model.waters)
    n = model.n_clusters

    gibbs = rates.gibbs_at(model.expanded, temperature)  # kcal/mol, per species
    dry_gibbs = rates.gibbs_at(model.inputs, temperature)[owner]
    relative = (gibbs - dry_gibbs) * config.KCAL_PER_MOL_TO_J
    activity = model.rh_percent / 100.0 * water_saturation_pressure(temperature)
    activity = activity / model.reference_pressure
    unnormalised = activity**waters * jnp.exp(-relative / (config.K_B * temperature))

    total = jax.ops.segment_sum(unnormalised, owner, num_segments=n)
    normalised = unnormalised / total[owner]
    check = jax.ops.segment_sum(normalised, owner, num_segments=n)
    keep = check > fidelity.hydrate_discard_threshold
    weights = jnp.where(keep[owner], normalised, (waters == 0).astype(normalised.dtype))

    return (
        jnp.zeros((n, model.n_species))
        .at[owner, jnp.arange(model.n_species)]
        .set(weights)
    )


def collision_coefficients(
    model: HydrateModel,
    temperature,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Hydrate-averaged K, m^3/s, shape (n_clusters, n_clusters)."""
    w = hydrate_weights(model, temperature, fidelity)
    k = rates.collision_coefficients(model.expanded, temperature, fidelity)
    return w @ k @ w.T


def evaporation(
    model: HydrateModel,
    temperature,
    reference_pressure: float | None = None,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Hydrate-averaged evaporation rates, 1/s, one per dry channel.

    Ordered as the channels the model was built from. Each hydrate
    combination's rate uses the hydrates' own free energies and the
    hydrate-pair collision coefficient (including its ion enhancement,
    Perl :9052), weighted by the PARENT's distribution.
    """
    if reference_pressure is None:
        reference_pressure = model.reference_pressure
    w = hydrate_weights(model, temperature, fidelity)
    k = rates.collision_coefficients(model.expanded, temperature, fidelity)
    per_combo = rates.evaporation_for_pairs(
        model.expanded,
        k,
        model.combo_parent,
        model.combo_i,
        model.combo_j,
        temperature,
        reference_pressure,
        fidelity,
    )
    parent_weight = w[model.owner[model.combo_parent], model.combo_parent]
    return jax.ops.segment_sum(
        parent_weight * per_combo,
        jnp.asarray(model.combo_channel),
        num_segments=len(model.channel_parent),
    )


def average_vector(
    model: HydrateModel,
    per_species: jnp.ndarray,
    temperature,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Contract any per-species first-order rate to the dry clusters.

    This is how upstream averages bg_loss (Perl :6640-6700) and the wall
    losses (:7443-7460) too: evaluate the formula on every hydrate, weight
    by the distribution.
    """
    return hydrate_weights(model, temperature, fidelity) @ per_species


def coagulation_sink(
    model: HydrateModel,
    cs_ref,
    fcs: float = config.FCS_DEFAULT,
    temperature=None,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Hydrate-averaged coagulation sink, 1/s, shape (n_clusters,).

    Needs the temperature only for the distribution; upstream evaluates
    both at its single fixed temperature.
    """
    w = hydrate_weights(model, temperature, fidelity)
    return w @ rates.coagulation_sink(model.expanded, cs_ref, fcs)


__all__ = [
    "HydrateModel",
    "average_vector",
    "build_hydrate_model",
    "coagulation_sink",
    "collision_coefficients",
    "evaporation",
    "hydrate_weights",
    "water_saturation_pressure",
]
