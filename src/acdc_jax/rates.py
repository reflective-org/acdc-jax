"""Collision, evaporation and loss coefficients.

The first traced module: everything here is a function of temperature and
is differentiable. It replaces 9585 lines of emitted numeric literals --
``get_coll`` (5076), ``get_rate_coefs`` (3226) and ``get_evap`` (1283) --
with closed-form expressions.

The static inputs (masses, radii, dipole moments, polarizabilities, charges)
are assembled once in NumPy by :func:`build_rate_inputs`; only arrays cross
into the traced functions.

Per the house rule, this is written eagerly in float64 with no ``jit`` and
no ``vmap``. Those come in Phase 7, in separate commits.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from acdc_jax import config, geometry, rules
from acdc_jax.clusterset import ClusterSetFile
from acdc_jax.system import AcdcSystem
from acdc_jax.thermo import DipoleTable, EnergyTable

# Pair classification, computed once. Which formula applies to a collision
# depends only on the charges of the two partners.
PAIR_FORBIDDEN = 0
"""Same-sign ions: they never collide."""
PAIR_NEUTRAL = 1
"""Both neutral: hard sphere."""
PAIR_ION_NEUTRAL = 2
"""One charged, one neutral: hard sphere with electrostatic enhancement."""
PAIR_RECOMBINATION = 3
"""Opposite-sign ions: a flat constant."""


@dataclasses.dataclass(frozen=True)
class RateInputs:
    """Static per-cluster quantities the rate formulas need.

    All SI: masses in kg, radii in m. Dipole moments stay in Debye and
    polarizabilities in Angstrom^3, because the Su parameterizations are
    published in those units and the reference's numeric constants absorb
    the conversions.
    """

    mass: np.ndarray
    """kg, shape (nclust,)."""
    radius: np.ndarray
    """m, shape (nclust,)."""
    diameter_nm: np.ndarray
    """Mass-equivalent diameter in nm, for the coagulation sink."""
    dipole: np.ndarray
    """Debye. Zero where the cluster is charged (it is never the polarised
    partner) or has no data."""
    polarizability: np.ndarray
    """Angstrom^3."""
    dipole_locked: np.ndarray
    """Debye, multiplied by the Su73 dipole LOCKING coefficient.

    Su73 damps the dipole by a factor from the dipole file's header --
    a separate coefficient for monomers and for clusters (Perl :7880-7886).
    Su82 reads those two header lines and ignores them, so this column is
    used only by Su73.
    """
    charge: np.ndarray
    """-1, 0 or +1, shape (nclust,)."""
    pair_kind: np.ndarray
    """(nclust, nclust) int, one of the PAIR_* constants."""
    neutral_partner: np.ndarray
    """(nclust, nclust) int: for an ion-neutral pair, the index of the
    NEUTRAL partner, whose dipole and polarizability set the enhancement.
    Self-index elsewhere."""
    delta_g_kcal: np.ndarray
    """Formation enthalpy and entropy folded per cluster; see
    :func:`gibbs_at`. Stored as (delta_h, delta_s) columns."""
    delta_h: np.ndarray
    delta_s: np.ndarray
    cs_shape: np.ndarray
    """Coagulation-sink size dependence, dimensionless, shape (nclust,)."""
    sticking: np.ndarray
    """(nclust, nclust) sticking factor multiplying K, 1.0 by default.

    From the --sticking_factor rule list (:mod:`acdc_jax.rules`). Already
    rounded to the generator's %.4e literal. Under upstream's emission the
    same factor also multiplies E -- see F19 and
    ``FidelityConfig.sticking_on_evaporation``.
    """
    evap_scale_kcal: np.ndarray
    """(nclust, nclust) kcal/mol ADDED to the reaction free energy of the
    evaporation k -> i + j, indexed by the daughters. Zero by default. From
    the --scale_evap rule list; the emitted Fortran shows it as an extra
    ``+scale/temperature`` term beside H/T - S/1e3, i.e. inside the exponent.
    K is untouched.
    """
    cs_excluded: np.ndarray
    """(nclust,) bool: clusters given a special zero sink (``--cs_only X,0``).
    Every sink formulation honours it; ``cs_shape`` already carries the zero
    for the exp_loss form."""
    is_generic_ion: np.ndarray
    """(nclust,) bool: the two generic charger ions."""
    has_energy_data: np.ndarray
    """(nclust,) bool: clusters with tabulated quantum-chemical energies.

    The parameters a sensitivity study may legitimately vary. Monomers and
    the generic charger ions are excluded because their formation free
    energy is **zero by definition** -- formation energies are relative to
    the free monomers, so a monomer is the reference, not a measurement.
    Differentiating with respect to them shifts the whole reference state
    and produces a large, meaningless derivative: with them included, 1A and
    1N dominate dJ/dH for the bundled system, which reads as a physical
    result and is not one.
    """
    valid_pairs: np.ndarray
    """(nclust, nclust) bool: pairs with an enumerated collision.

    Charge alone does not decide this. A pair can be charge-compatible and
    still have no collision -- `neg + 1N` has no acid to convert, so its
    product would need a negative molecule count -- and the reference's
    `get_coll` leaves those at exactly zero. It also zeroes the no-op
    boundary collisions that the enumeration prunes. Without this mask the
    coefficient matrix has 63 spurious nonzeros for the bundled system.
    """


def build_rate_inputs(
    system: AcdcSystem,
    cluster_set: ClusterSetFile,
    energies: EnergyTable,
    dipoles: DipoleTable,
    reactions=None,
    sticking_rules: tuple[rules.Rule, ...] = (),
    evap_scale_rules: tuple[rules.Rule, ...] = (),
    cs_reference_label: str = "1A",
    cs_exponent: float = config.CS_EXPONENT_DEFAULT,
    cs_excluded: tuple[str, ...] = ("1A", "1N"),
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> RateInputs:
    """Assemble the static inputs. NumPy, runs once."""
    n = system.n_clusters
    by_name = {m.name: m for m in cluster_set.molecules}
    molecules = tuple(by_name[name] for name in system.order)

    compositions = np.array(system.compositions)

    # The generic charger ions carry a pseudo-composition with a NEGATIVE
    # entry (`neg` is 1B-1A), which would give a negative volume and a NaN
    # radius. Zero those rows before the geometry, then substitute the
    # generator's hardcoded ion properties (Perl :930-943).
    geometry_rows = compositions.copy()
    for index in (system.generic_neg, system.generic_pos):
        if index >= 0:
            geometry_rows[index] = 0

    mass_g = geometry.cluster_mass(geometry_rows, molecules)
    radius = geometry.cluster_radius(geometry_rows, molecules)
    diameter_nm = geometry.cluster_diameter(geometry_rows, molecules)

    for index, (mass_value, density) in (
        (system.generic_neg, (config.MASS_NEG_ION, config.DENS_NEG_ION)),
        (system.generic_pos, (config.MASS_POS_ION, config.DENS_POS_ION)),
    ):
        if index < 0:
            continue
        mass_g[index] = mass_value
        volume = mass_value * config.MASS_CONV / density
        radius[index] = (3.0 * volume / (4.0 * config.PI)) ** (1.0 / 3.0)
        diameter_nm[index] = 2e9 * radius[index]

    charge = np.array(system.charges)

    dipole = np.zeros(n)
    polarizability = np.zeros(n)
    dipole_locked = np.zeros(n)
    for i, label in enumerate(system.labels):
        if label in dipoles.dipole:
            dipole[i] = dipoles.dipole[label]
            polarizability[i] = dipoles.polarizability[label]
            locking = (
                dipoles.monomer_locking
                if system.is_monomer(i)
                else dipoles.cluster_locking
            )
            dipole_locked[i] = dipoles.dipole[label] * locking

    delta_h = np.array([energies.delta_h.get(label, 0.0) for label in system.labels])
    delta_s = np.array([energies.delta_s.get(label, 0.0) for label in system.labels])
    has_energy_data = np.array(
        [label in energies.delta_h for label in system.labels], dtype=bool
    )

    pair_kind, neutral_partner = _classify_pairs(charge)

    cs_shape = _coagulation_sink_shape(
        system, diameter_nm, cs_reference_label, cs_exponent, cs_excluded
    )

    cs_excluded_mask = np.array([label in cs_excluded for label in system.labels])
    is_generic_ion = np.zeros(n, dtype=bool)
    for index in (system.generic_neg, system.generic_pos):
        if index >= 0:
            is_generic_ion[index] = True

    valid_pairs = np.zeros((n, n), dtype=bool)
    if reactions is None:
        # No reaction list supplied: fall back to charge compatibility. Gives
        # the spurious nonzeros described on `valid_pairs`, so this is only
        # for inspecting the formulas in isolation.
        valid_pairs = pair_kind != PAIR_FORBIDDEN
    else:
        for collision in reactions.collisions:
            valid_pairs[collision.i, collision.j] = True
            valid_pairs[collision.j, collision.i] = True

    return RateInputs(
        sticking=rules.sticking_matrix(system, sticking_rules),
        evap_scale_kcal=rules.evap_scale_matrix(system, evap_scale_rules),
        mass=mass_g * config.MASS_CONV,
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
        cs_excluded=cs_excluded_mask,
        is_generic_ion=is_generic_ion,
        has_energy_data=has_energy_data,
        valid_pairs=valid_pairs,
    )


def _classify_pairs(charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(charge)
    kind = np.full((n, n), PAIR_FORBIDDEN, dtype=np.int32)
    partner = np.tile(np.arange(n), (n, 1))

    qi = charge[:, None]
    qj = charge[None, :]

    kind[(qi == 0) & (qj == 0)] = PAIR_NEUTRAL
    kind[qi * qj < 0] = PAIR_RECOMBINATION

    ion_neutral = ((qi != 0) & (qj == 0)) | ((qi == 0) & (qj != 0))
    kind[ion_neutral] = PAIR_ION_NEUTRAL

    # For an ion-neutral pair the enhancement depends on the NEUTRAL
    # partner's dipole and polarizability: the ion is the polariser, the
    # neutral is what gets polarised.
    j_is_neutral = np.tile(charge == 0, (n, 1))
    partner = np.where(j_is_neutral, np.arange(n)[None, :], np.arange(n)[:, None])
    return kind, partner


def _coagulation_sink_shape(
    system: AcdcSystem,
    diameter_nm: np.ndarray,
    reference_label: str,
    exponent: float,
    excluded: tuple[str, ...],
) -> np.ndarray:
    """``(d_i / d_ref)^m``, dimensionless.

    The reference emits only this size dependence and multiplies by the
    runtime ``cs_ref`` (``get_rate_coefs:1011``). Clusters named in
    ``excluded`` get exactly zero -- that is how ``--cs_only 1A,0`` removes
    the vapour monomers from scavenging.
    """
    reference_diameter = diameter_nm[system.labels.index(reference_label)]
    shape = (diameter_nm / reference_diameter) ** exponent
    for label in excluded:
        if label in system.labels:
            shape[system.labels.index(label)] = 0.0
    return shape


# ---------------------------------------------------------------------------
# Traced rate formulas
# ---------------------------------------------------------------------------


def hard_sphere(inputs: RateInputs, temperature) -> jnp.ndarray:
    """Kinetic gas-theory collision rate, m^3/s. Shape (nclust, nclust).

    ``beta = sqrt(8 pi k_B T) * sqrt(1/m_i + 1/m_j) * (r_i + r_j)^2``
    (Perl :8785-8790). Free-molecular: no van der Waals enhancement and no
    Fuchs transition-regime correction. A vdW form exists commented out
    upstream (:8798) and was evidently abandoned.
    """
    mass = jnp.asarray(inputs.mass)
    radius = jnp.asarray(inputs.radius)
    reduced = jnp.sqrt(1.0 / mass[:, None] + 1.0 / mass[None, :])
    cross_section = (radius[:, None] + radius[None, :]) ** 2
    return (
        jnp.sqrt(8.0 * config.PI * config.K_B * temperature) * reduced * cross_section
    )


def su82_enhancement(inputs: RateInputs, temperature) -> jnp.ndarray:
    """Su & Chesnavich (1982) ion-neutral capture ratio. Dimensionless.

    The reduced dipole parameter and the piecewise capture-rate ratio
    (Perl :8853-8859), with ``mu`` in Debye and ``alpha`` in Angstrom^3::

        x     = 100 mu / sqrt(2 alpha k_B 1e23 T)
        ratio = (x + 0.5090)^2 / 10.526 + 0.9754     for x < 2
              = 0.4767 x + 0.62                       otherwise

    The emitted Fortran writes this as a pair of ``sign()``-based Heaviside
    terms with a per-pair switch temperature baked in, because Perl could
    not evaluate a temperature-dependent branch at code-generation time.
    JAX traces temperature natively, so it is one ``jnp.where`` and the
    switch temperature never appears.
    """
    partner = inputs.neutral_partner
    dipole = jnp.asarray(inputs.dipole)[partner]
    polarizability = jnp.asarray(inputs.polarizability)[partner]

    # Guard the division: a pair with no polarizability data is not an
    # ion-neutral pair and its value is masked out downstream, but the
    # reciprocal must not produce a NaN that poisons the gradient.
    safe_alpha = jnp.where(polarizability > 0, polarizability, 1.0)
    x = 100.0 * dipole / jnp.sqrt(2.0 * safe_alpha * config.K_B * 1e23 * temperature)
    x = jnp.where(polarizability > 0, x, 0.0)

    low = (x + config.SU82_A) ** 2 / config.SU82_B + config.SU82_C
    high = config.SU82_D * x + config.SU82_E
    return jnp.where(x < config.SU82_X_SWITCH, low, high)


def langevin_prefactor(inputs: RateInputs) -> jnp.ndarray:
    """``2 pi * 4.8032e-16 * sqrt(alpha (1/m_i + 1/m_j) / 1e27)``, m^3/s.

    Temperature-independent (Perl :8859).
    """
    mass = jnp.asarray(inputs.mass)
    polarizability = jnp.asarray(inputs.polarizability)[inputs.neutral_partner]
    reduced = 1.0 / mass[:, None] + 1.0 / mass[None, :]
    return (
        2.0
        * config.PI
        * config.SU82_LANGEVIN
        * jnp.sqrt(polarizability * reduced / 1e27)
    )


def su73_ionic_rate(inputs: RateInputs, temperature) -> jnp.ndarray:
    """Su & Bowers (1973) ion-neutral capture rate, m^3/s.

    Unlike Su82 this returns an ABSOLUTE rate rather than a ratio -- the
    published constants absorb the Langevin prefactor (Perl :8199-8213)::

        rate = (9.5436e-29 sqrt(alpha) + 6.4805e-27 mu_locked / sqrt(T))
               * sqrt(1/m_i + 1/m_j)

    ``mu_locked`` is the dipole moment damped by the locking coefficient,
    which is the parameterization's way of accounting for the dipole not
    staying aligned with the field during the encounter.
    """
    partner = inputs.neutral_partner
    polarizability = jnp.asarray(inputs.polarizability)[partner]
    dipole = jnp.asarray(inputs.dipole_locked)[partner]
    mass = jnp.asarray(inputs.mass)
    reduced = jnp.sqrt(1.0 / mass[:, None] + 1.0 / mass[None, :])

    return (
        config.SU73_POL * jnp.sqrt(polarizability)
        + config.SU73_DIP * dipole / jnp.sqrt(temperature)
    ) * reduced


def collision_coefficients(
    inputs: RateInputs,
    temperature,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Collision coefficients K, m^3/s. Shape (nclust, nclust).

    Dispatches on the pair classification: hard sphere for neutral pairs,
    ``max(ionic, hard sphere)`` for ion-neutral, a flat constant for
    recombination, and exactly zero for same-sign pairs.

    The ``max`` is the reference's, not a safety clamp: the Su
    parameterization can fall below the geometric rate for a large, weakly
    polar neutral, and the collision cannot be slower than hard spheres
    (Perl :8886-8888).
    """
    beta = hard_sphere(inputs, temperature)
    method = fidelity.ion_collision_method

    if method == "su82":
        enhanced = jnp.maximum(
            su82_enhancement(inputs, temperature) * langevin_prefactor(inputs), beta
        )
    elif method == "su73":
        enhanced = jnp.maximum(su73_ionic_rate(inputs, temperature), beta)
    elif method == "constant":
        enhanced = config.ION_ENHANCEMENT_CONSTANT * beta
    elif method == "constant_no_enhancement":
        # Reproduces upstream's variable-temperature path, where the
        # documented factor of 10 is silently dropped. See fidelity F15.
        enhanced = beta
    else:  # pragma: no cover - Literal keeps this unreachable
        raise ValueError(f"unknown ion collision method {method!r}")

    kind = jnp.asarray(inputs.pair_kind)
    k = jnp.where(kind == PAIR_NEUTRAL, beta, 0.0)
    k = jnp.where(kind == PAIR_ION_NEUTRAL, enhanced, k)
    k = jnp.where(kind == PAIR_RECOMBINATION, config.RECOMB_COEFF, k)
    # Sticking factors sit outside everything else, including the max():
    # `K(53,1) = 2.0000d+00*max((...`. Ones where no rule applies.
    k = k * jnp.asarray(inputs.sticking)
    return jnp.where(jnp.asarray(inputs.valid_pairs), k, 0.0)


def gibbs_at(inputs: RateInputs, temperature) -> jnp.ndarray:
    """Formation free energies at a temperature, kcal/mol. Shape (nclust,).

    ``G = H - T S / 1000``, the 1000 converting entropy from cal to kcal.
    """
    return jnp.asarray(inputs.delta_h) - temperature * jnp.asarray(inputs.delta_s) / 1e3


def evaporation_for_pairs(
    inputs: RateInputs,
    collision: jnp.ndarray,
    parents: np.ndarray,
    daughters_i: np.ndarray,
    daughters_j: np.ndarray,
    temperature,
    reference_pressure: float = config.P_ATM,
    fidelity: config.FidelityConfig = config.DEFAULT,
) -> jnp.ndarray:
    """Evaporation rates for a list of channels ``k -> i + j``, 1/s.

    Args:
        parents: parent cluster index per evaporation channel.
        daughters_i: first daughter index, parallel to ``parents``.
        daughters_j: second daughter index, parallel to ``parents``.

    Returns:
        Shape ``(n_channels,)``.
    """
    gibbs = gibbs_at(inputs, temperature)
    delta = gibbs[parents] - gibbs[daughters_i] - gibbs[daughters_j]
    # --scale_evap: a per-channel Delta-G correction, kcal/mol, added inside
    # the exponent. Zero unless rules were supplied.
    delta = delta + jnp.asarray(inputs.evap_scale_kcal)[daughters_i, daughters_j]

    # kcal/mol -> J per molecule, then divided by k_B T. The reference folds
    # this into the single constant 5.03218937158374e2 = KCAL_PER_MOL_TO_J/k_B.
    exponent = delta * config.KCAL_PER_MOL_TO_J / (config.K_B * temperature)

    number_density = reference_pressure / (config.K_B * temperature)
    beta = collision[daughters_i, daughters_j]

    symmetric = jnp.asarray(daughters_i == daughters_j)
    factor = jnp.where(symmetric, 0.5, 1.0)

    rate = factor * number_density * jnp.exp(exponent) * beta

    # F19: upstream prepends the sticking literal to the E expression, which
    # already references the K that carries it, so E scales as s^2 while K
    # scales as s. Reproduced by default; "detailed_balance" applies it once
    # (through K only) so E/K stays exp(dG/kT)/n_ref.
    if fidelity.sticking_on_evaporation == "upstream":
        rate = rate * jnp.asarray(inputs.sticking)[daughters_i, daughters_j]
    return rate


def coagulation_sink(
    inputs: RateInputs,
    cs_ref,
    fcs: float = config.FCS_DEFAULT,
) -> jnp.ndarray:
    """Per-cluster coagulation sink, 1/s. Shape (nclust,).

    ``CS_i = cs_ref * (d_i/d_ref)^m``, with charged species additionally
    scaled by ``fcs`` (``get_rate_coefs:4169-4222``). ``fcs`` is 1.0 in the
    bundled setup, so the ion enhancement is invisible there -- but the
    branch exists and is applied only to charged clusters.
    """
    shape = jnp.asarray(inputs.cs_shape)
    charged = jnp.asarray(inputs.charge) != 0
    return cs_ref * shape * jnp.where(charged, fcs, 1.0)
