"""Loop mode: every composition up to a maximum count is a cluster.

The generator's ``--loop`` (``small_set_mode = .false.``) trades the small,
hand-listed cluster set for a dense grid of compositions ``0 <= n_k <= max_k``,
no charges, no boundary rules and one outgoing flux. The emitted Fortran is
an ``i <= j`` double loop with ``ij = product-or-outflux``, monomer-only
Kelvin evaporation (or Delta-G with fissions), a single ``loss`` vector and
``source(monomers) = coef``.

That maps exactly onto :class:`acdc_jax.rhs.Coefficients`, so this module
is a second *assembler*: it builds an :class:`~acdc_jax.system.AcdcSystem`
for the composition grid, computes K, E and the losses in the loop-mode
forms, and hands back a ``Coefficients`` that ``solve``, ``sensitivity``
and ``formation_rate`` consume unchanged.

References are to ``acdc_2020_04_28.pl``: system enumeration
``get_cluster_numbers`` :6308-6379, geometry :6228-6276, K :7997-8110,
E :8590-8860, ``feval`` :4548-4720, ``formation`` :5341-5819.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np

from acdc_jax import config, labels, losses, rates, rhs
from acdc_jax.clusterset import ClusterSetFile, Molecule
from acdc_jax.system import FLUX_SLOTS, AcdcSystem


@dataclasses.dataclass(frozen=True)
class LoopSystem:
    """A composition grid and its geometry."""

    system: AcdcSystem
    """Labels, compositions, zero charges and the usual flux slots."""
    molecules: tuple[Molecule, ...]
    """The used molecule types, in the header's order."""
    max_counts: tuple[int, ...]
    counts: np.ndarray
    """(nclust, n_types) int."""
    mass_g: np.ndarray
    """g/mol, ``sum_k n_k m_k`` (the loop-mode formulas keep g/mol)."""
    radius: np.ndarray
    """m, from the summed bulk molecular volumes."""
    monomers: np.ndarray
    """(n_types,) cluster index of each pure monomer."""
    product_index: np.ndarray
    """(nclust, nclust) int: the cluster ``i + j`` forms, or -1 when it
    leaves the grid (booked to ``out_neu``)."""

    @property
    def n_clusters(self) -> int:
        return self.system.n_clusters

    @property
    def n_types(self) -> int:
        return len(self.molecules)


def build_loop_system(cluster_set: ClusterSetFile) -> LoopSystem:
    """Enumerate the grid the way ``get_cluster_numbers`` does.

    The input file carries the ordinary header plus ONE composition row --
    the largest cluster (Perl :1889 dies on more). Used molecule types are
    those with a positive count, in header order. Nested loops, first type
    outermost, each ``0..max``, skipping the empty composition; a
    one-component grid is therefore indexed by molecule count.
    """
    if len(cluster_set.compositions) != 1:
        raise ValueError(
            "loop mode wants exactly one composition row, the largest cluster; "
            f"got {len(cluster_set.compositions)}"
        )
    largest = cluster_set.compositions[0]
    by_name = {m.name: m for m in cluster_set.molecules}
    used = [
        (name, count)
        for name, count in zip(cluster_set.molecule_names, largest, strict=True)
        if count > 0
    ]
    if not used:
        raise ValueError("loop mode: the largest cluster has no molecules")
    order = tuple(name for name, _ in used)
    max_counts = tuple(count for _, count in used)
    molecules = tuple(by_name[name] for name in order)
    for m in molecules:
        if m.charge != 0:
            raise ValueError(f"loop mode has no charges; {m.name} carries {m.charge}")
        if m.density is None:
            raise ValueError(f"loop mode needs a density for {m.name}")

    grid = np.stack(
        np.meshgrid(*[np.arange(c + 1) for c in max_counts], indexing="ij"), axis=-1
    ).reshape(-1, len(max_counts))
    counts = grid[grid.sum(axis=1) > 0]  # C order == first type outermost
    n = len(counts)
    index_of = {tuple(int(x) for x in row): i for i, row in enumerate(counts)}

    species_labels = tuple(
        labels.format_label(dict(zip(order, row, strict=True)), order) for row in counts
    )
    flux_index = {name: n + i for i, name in enumerate(FLUX_SLOTS)}
    system = AcdcSystem(
        order=order,
        labels=species_labels,
        compositions=tuple(tuple(int(x) for x in row) for row in counts),
        charges=tuple(0 for _ in range(n)),
        n_clusters=n,
        n_equations=n + len(FLUX_SLOTS),
        flux_index=flux_index,
        has_generic_neg=False,
        has_generic_pos=False,
        charges_of_molecule=tuple(0 for _ in order),
        molecule_is_pseudo=tuple(False for _ in order),
    )

    mol_mass = np.array([m.mass for m in molecules])
    mol_volume = np.array([m.mass * config.MASS_CONV / m.density for m in molecules])
    mass_g = counts @ mol_mass
    # get_masses_and_radii declares pi = 4*atan(1), the exact value, where
    # the rate formulas carry the generator's 12-digit literal (config.PI).
    radius = (3.0 / (4.0 * math.pi) * (counts @ mol_volume)) ** (1.0 / 3.0)

    monomers = np.array(
        [
            index_of[tuple(int(k == t) for k in range(len(order)))]
            for t in range(len(order))
        ]
    )
    product = np.full((n, n), -1, dtype=int)
    for i in range(n):
        for j in range(i, n):
            k = index_of.get(tuple(int(x) for x in counts[i] + counts[j]))
            if k is not None:
                product[i, j] = product[j, i] = k

    return LoopSystem(
        system=system,
        molecules=molecules,
        max_counts=max_counts,
        counts=counts,
        mass_g=mass_g,
        radius=radius,
        monomers=monomers,
        product_index=product,
    )


# ---------------------------------------------------------------------------
# Collision coefficients
# ---------------------------------------------------------------------------


def hard_spheres(loop: LoopSystem, temperature) -> jnp.ndarray:
    """Kinetic gas theory in the loop form (Perl :8023-8035):
    ``sqrt(8 pi k_B T / u (1/m_i + 1/m_j)) (r_i + r_j)^2`` with m in g/mol."""
    m = jnp.asarray(loop.mass_g)
    r = jnp.asarray(loop.radius)
    factor = 8.0 * config.PI * config.K_B / config.MASS_CONV * temperature
    inverse_mass = 1.0 / m[:, None] + 1.0 / m[None, :]
    return jnp.sqrt(factor * inverse_mass) * (r[:, None] + r[None, :]) ** 2


def dahneke(loop: LoopSystem, temperature) -> jnp.ndarray:
    """Dahneke (1983) transition-regime kernel (Perl :8036-8100; Seinfeld &
    Pandis ch. 9, 12, 13): slip-corrected diffusivities, thermal speeds,
    ``Kn = 2 (D_i + D_j) / (sqrt(v_i^2 + v_j^2) (r_i + r_j))`` and
    ``K = 4 pi (r_i + r_j)(D_i + D_j)(1 + Kn) / (1 + 2 Kn (1 + Kn))``.

    Upstream refuses this with --variable_temp; here it traces.
    """
    r = jnp.asarray(loop.radius)
    m = jnp.asarray(loop.mass_g) * config.MASS_CONV
    d = losses.slip_corrected_diffusivity(r, temperature)
    v = losses.thermal_speed(m, temperature)
    r_sum = r[:, None] + r[None, :]
    d_sum = d[:, None] + d[None, :]
    speed = jnp.sqrt(v[:, None] ** 2 + v[None, :] ** 2)
    kn = 2.0 * d_sum / (speed * r_sum)
    return 4.0 * config.PI * r_sum * d_sum * (1.0 + kn) / (1.0 + 2.0 * kn * (1.0 + kn))


def collision_coefficients(
    loop: LoopSystem,
    temperature,
    method: Literal["hard_spheres", "dahneke"] = "hard_spheres",
    sticking_factor: float = 1.0,
) -> jnp.ndarray:
    """K, m^3/s, shape (nclust, nclust), NOT yet halved on the diagonal
    (feval does that, Perl :4575). A bare ``--sticking_factor`` multiplies
    the whole matrix (:8107)."""
    if method == "hard_spheres":
        k = hard_spheres(loop, temperature)
    elif method == "dahneke":
        k = dahneke(loop, temperature)
    else:
        raise ValueError(f"unknown loop collision method {method!r}")
    return sticking_factor * k


# ---------------------------------------------------------------------------
# Evaporation
# ---------------------------------------------------------------------------


def kelvin_evaporation(
    loop: LoopSystem,
    collision: jnp.ndarray,
    temperature,
    rlim_no_evap: float | None = None,
) -> jnp.ndarray:
    """Monomer evaporation from the Kelvin equation (Perl :8694-8860).

    ``E(mon_t, j) = K(mon_t, j) * psat_t * x_t / kT * exp(2 sigma v_t / (kT r_ij))``
    where ``ij = mon_t + j`` must be on the grid, ``x_t`` is the mole
    fraction of type ``t`` in ``ij`` (one for a single component), ``v_t``
    the bulk molecular volume, ``sigma`` the surface tension of the FIRST
    component (upstream warns and uses it for all). Halved for two identical
    monomers. When ``j`` is itself a monomer of another type both channels
    write the same symmetric entry and upstream keeps the LARGER (:8833-8835).
    ``rlim_no_evap`` (m): no evaporation from parents at or above that
    radius. Everything else is zero: fissions are disabled in this mode.

    Returns the full symmetric (nclust, nclust) matrix; the one-component
    Fortran stores only column 1 (see F22).
    """
    n = loop.n_clusters
    kt = config.K_B * temperature
    sigma = loop.molecules[0].surface_tension
    if sigma is None or any(m.psat is None for m in loop.molecules):
        raise ValueError(
            "Kelvin evaporation needs `saturation vapor pressure` and "
            "`surface tension` rows in the cluster-set header"
        )
    r = jnp.asarray(loop.radius)
    counts = jnp.asarray(loop.counts, dtype=float)
    total = counts.sum(axis=1)

    e = jnp.zeros((n, n))
    for t, molecule in enumerate(loop.molecules):
        i = int(loop.monomers[t])
        product = loop.product_index[i]
        valid = product >= 0
        safe = np.where(valid, product, 0)
        r_ij = r[safe]
        x = counts[safe, t] / total[safe]
        monvol = molecule.mass * config.MASS_CONV / molecule.density
        row = (
            collision[i, :]
            * molecule.psat
            * x
            / kt
            * jnp.exp(2.0 * sigma * monvol / (kt * r_ij))
        )
        row = jnp.where(jnp.asarray(valid), row, 0.0)
        if rlim_no_evap is not None:
            row = jnp.where(r_ij >= rlim_no_evap, 0.0, row)
        self_pair = jnp.arange(n) == i
        row = jnp.where(self_pair, 0.5 * row, row)
        # a previous type's channel into the same entry wins if larger
        row = jnp.where(~self_pair & (e[:, i] > row), e[:, i], row)
        e = e.at[i, :].set(row).at[:, i].set(row)
    return e


def deltag_evaporation(
    loop: LoopSystem,
    collision: jnp.ndarray,
    temperature,
    gibbs_kcal: Callable[[np.ndarray], np.ndarray],
    reference_pressure: float = config.P_ATM,
    rlim_no_evap: float | None = None,
) -> jnp.ndarray:
    """Detailed-balance evaporation from a per-composition free energy
    (Perl :8600-8690, the user-supplied ``cluster_energies`` routine).

    ``E(i,j) = K(i,j) p_ref/kT exp((G_ij - G_i - G_j) / kT)`` for every pair
    whose product is on the grid -- fissions included -- halved for i == j.
    ``gibbs_kcal`` maps a (nclust, n_types) count array to kcal/mol.
    """
    n = loop.n_clusters
    kt = config.K_B * temperature
    g = jnp.asarray(gibbs_kcal(loop.counts)) * config.KCAL_PER_MOL_TO_J
    product = loop.product_index
    valid = product >= 0
    safe = np.where(valid, product, 0)
    delta = g[safe] - g[:, None] - g[None, :]
    e = collision * reference_pressure / kt * jnp.exp(delta / kt)
    e = jnp.where(jnp.asarray(valid), e, 0.0)
    if rlim_no_evap is not None:
        e = jnp.where(jnp.asarray(loop.radius)[safe] >= rlim_no_evap, 0.0, e)
    return jnp.where(jnp.eye(n, dtype=bool), 0.5 * e, e)


# ---------------------------------------------------------------------------
# Losses and assembly
# ---------------------------------------------------------------------------


def rate_inputs(loop: LoopSystem) -> rates.RateInputs:
    """A :class:`~acdc_jax.rates.RateInputs` for the grid, so the small-set
    loss formulas apply. Only the geometric fields carry information; the
    grid has no charges, dipoles or energies. ``cs_shape`` is relative to the
    first monomer, upstream's ``r_ref`` (Perl :6577)."""
    n = loop.n_clusters
    diameter_nm = 2e9 * loop.radius
    zeros = np.zeros(n)
    pair_kind, partner = rates._classify_pairs(np.zeros(n, dtype=int))
    reference = loop.radius[loop.monomers[0]]
    return rates.RateInputs(
        mass=loop.mass_g * config.MASS_CONV,
        radius=loop.radius,
        diameter_nm=diameter_nm,
        dipole=zeros,
        polarizability=zeros,
        dipole_locked=zeros,
        charge=np.zeros(n, dtype=int),
        pair_kind=pair_kind,
        neutral_partner=partner,
        delta_g_kcal=np.zeros((n, 2)),
        delta_h=zeros,
        delta_s=zeros,
        cs_shape=(loop.radius / reference) ** config.CS_EXPONENT_DEFAULT,
        cs_excluded=np.zeros(n, dtype=bool),
        is_generic_ion=np.zeros(n, dtype=bool),
        sticking=np.ones((n, n)),
        evap_scale_kcal=np.zeros((n, n)),
        has_energy_data=np.zeros(n, dtype=bool),
        valid_pairs=np.ones((n, n), dtype=bool),
    )


LOOP_FIDELITY = dataclasses.replace(config.DEFAULT, mobility_diameter="tammet")
"""Loop-mode wall losses use the LIVE mobility-diameter mass correction
(``sqrt(1 + 28.8/m)``, Perl :7024), unlike the small-set path where a unit
slip defeats it (F11)."""


def first_order_losses(
    loop: LoopSystem,
    settings: losses.LossSettings,
    temperature,
    cs_ref=config.CS_COEFFICIENT_DEFAULT,
) -> dict[str, jnp.ndarray]:
    """The loop-mode ``get_losses`` (Perl :6433-7670), keyed by flux slot.

    Same formulas as the small-set losses evaluated on the grid's sizes; the
    generator sums them into one ``loss`` vector, the port keeps them apart.
    """
    return losses.first_order_losses(
        settings, rate_inputs(loop), temperature, cs_ref, 1.0, LOOP_FIDELITY
    )


def assemble(
    loop: LoopSystem,
    collision: jnp.ndarray,
    evaporation: jnp.ndarray | None,
    monomer_sources,
    loss_vectors: dict[str, jnp.ndarray] | None = None,
    constant_monomers: bool = False,
) -> rhs.Coefficients:
    """The loop-mode right-hand side as a :class:`~acdc_jax.rhs.Coefficients`.

    Pairs ``i <= j`` (Perl :4569-4600): rate ``K(i,j)``, halved on the
    diagonal; product the grid cluster or the ``out_neu`` slot. Evaporation
    channels wherever ``E(j,i) > 0`` (the product is then on the grid).
    ``monomer_sources`` (n_types,) go to the monomers -- ``source(n_monomers)
    = coef``. ``constant_monomers`` is the driver's steady-state setting
    (``isconst(n_monomers) = .true.``, ``acdc_simulation_setup.f90:106``).
    No dense tensors.
    """
    system = loop.system
    n, neq = system.n_clusters, system.n_equations
    iu, ju = np.triu_indices(n)
    product = loop.product_index[iu, ju]
    out = system.flux_index["out_neu"]
    product_index = np.where(product >= 0, product, out)
    rate = collision[iu, ju] * jnp.where(iu == ju, 0.5, 1.0)

    if evaporation is None:
        evap_k = evap_i = evap_j = np.zeros(0, dtype=int)
        evap_rate = jnp.zeros(0)
    else:
        # feval reads E(j,i) for i <= j (Perl :4593), column-major for speed;
        # the one-component Fortran fills only column 1 (F22), so read the
        # same element rather than assuming symmetry.
        e_pairs = np.asarray(evaporation)[ju, iu]
        channel = (e_pairs > 0) & (product >= 0)
        evap_k = product[channel]
        evap_i = iu[channel]
        evap_j = ju[channel]
        evap_rate = evaporation[ju[channel], iu[channel]]

    source = jnp.zeros(neq).at[loop.monomers].set(jnp.asarray(monomer_sources))
    isconst = np.zeros(neq, dtype=bool)
    if constant_monomers:
        isconst[loop.monomers] = True

    loss_vectors = loss_vectors or {}
    return rhs.Coefficients(
        n_clusters=n,
        coef_quad=None,
        coef_lin=None,
        multiplicity=None,
        source=source,
        isconst=isconst,
        collision_i=iu,
        collision_j=ju,
        collision_rate=rate,
        product_index=product_index,
        product_owner=np.arange(len(iu)),
        product_multiplicity=jnp.ones(len(iu)),
        evaporation_k=evap_k,
        evaporation_i=evap_i,
        evaporation_j=evap_j,
        evaporation_rate=evap_rate,
        losses=tuple(loss_vectors.values()),
        loss_slots=tuple(system.flux_index[name] for name in loss_vectors),
    )


__all__ = [
    "LOOP_FIDELITY",
    "LoopSystem",
    "assemble",
    "build_loop_system",
    "collision_coefficients",
    "dahneke",
    "deltag_evaporation",
    "first_order_losses",
    "hard_spheres",
    "kelvin_evaporation",
    "rate_inputs",
]
