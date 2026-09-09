"""Assembling a cluster set into the full simulation system.

Takes the parsed `.inp` clusters and adds everything the generator appends:
the two generic charger ions, and the flux accounting slots that make up the
tail of the state vector.

For the bundled sulfuric acid--ammonia system this turns 52 declared
clusters into 54 species and a 63-long state vector.

Plain Python: built once, then handed to the traced code as arrays.
"""

from __future__ import annotations

import dataclasses

from acdc_jax import config, labels
from acdc_jax.clusterset import ClusterSetFile

# Flux accumulator slots, in the order the generator appends them
# (acdc_equations_AN_ions_example.f90:55-63). They are not clusters: they
# have no composition and no rate coefficients of their own, they simply
# collect fluxes.
FLUX_SLOTS = (
    "source",
    "coag",
    "wall",
    "dilution",
    "rec",
    "out_neu",
    "out_neg",
    "out_pos",
    "bound",
)


@dataclasses.dataclass(frozen=True)
class AcdcSystem:
    """A cluster set assembled into a simulation system."""

    order: tuple[str, ...]
    """Molecule names, in index order -- the used ones only."""
    labels: tuple[str, ...]
    """Species labels: the declared clusters, then the generic charger ions."""
    compositions: tuple[tuple[int, ...], ...]
    """One row per species. The generic ions carry PSEUDO-compositions with
    negative entries; see `generic_ion_composition`."""
    charges: tuple[int, ...]
    n_clusters: int
    """Number of real species, including the two generic ions."""
    n_equations: int
    """State-vector length: `n_clusters` plus the flux slots."""
    flux_index: dict[str, int]
    """Slot name -> 0-based index into the state vector."""
    has_generic_neg: bool
    has_generic_pos: bool

    def index(self, label: str) -> int:
        return self.labels.index(label)

    @property
    def generic_neg(self) -> int:
        """0-based index of the generic negative ion, or -1."""
        return self.labels.index("neg") if self.has_generic_neg else -1

    @property
    def generic_pos(self) -> int:
        return self.labels.index("pos") if self.has_generic_pos else -1

    def is_monomer(self, i: int) -> bool:
        """A single molecule, ignoring the proton it may be carrying.

        `1N1P` (protonated ammonia) is a monomer even though it has two
        entries: the proton is charge bookkeeping, not a second molecule.
        """
        counts = self.compositions[i]
        total = sum(
            n
            for j, n in enumerate(counts)
            if self.charges_of_molecule[j] == 0 and n > 0
        )
        return total == 1

    charges_of_molecule: tuple[int, ...] = ()


def generic_ion_composition(
    cluster_set: ClusterSetFile, order: tuple[str, ...], negative: bool
) -> tuple[int, ...]:
    """Pseudo-composition of a generic charger ion.

    The generator does not treat the charger ions as opaque. It gives each
    one a composition so that colliding with one is ordinary vector addition
    (Perl ``determine_cluster_composition``, :10965-10969):

    - the positive ion becomes ``1P`` -- literally one proton;
    - the negative ion becomes ``1B-1A``, i.e. **plus one bisulfate, minus
      one sulfuric acid**.

    The negative entry is the whole trick. Adding ``neg`` to ``1A`` gives
    ``1B``: charge transfer falls out of the same summation as any other
    collision. And adding it to ``1N``, which has no acid to convert, gives
    a negative count, which is exactly how the reference detects that the
    charge has nowhere to go and rejects the collision.
    """
    counts = [0] * len(order)
    by_name = {m.name: m for m in cluster_set.molecules}

    if not negative:
        proton = next((m for m in cluster_set.molecules if m.is_proton), None)
        if proton is None or proton.name not in order:
            raise ValueError("no proton declared; cannot build the positive ion")
        counts[order.index(proton.name)] = 1
        return tuple(counts)

    # The negative ion is described either by an explicit missing-proton
    # pseudo-species, or -- as in these files -- by a charged molecule and
    # the neutral it derives from (Perl :1646-1652).
    missing = next((m for m in cluster_set.molecules if m.is_missing_proton), None)
    if missing is not None and missing.name in order:
        counts[order.index(missing.name)] = 1
        return tuple(counts)

    for molecule in cluster_set.molecules:
        if molecule.charge != -1 or molecule.corr_neutral is None:
            continue
        if molecule.name not in order or molecule.corr_neutral not in order:
            continue
        counts[order.index(molecule.name)] = 1
        counts[order.index(molecule.corr_neutral)] = -1
        return tuple(counts)

    raise ValueError("cannot construct the generic negative ion composition")


def build_system(
    cluster_set: ClusterSetFile,
    include_generic_neg: bool = True,
    include_generic_pos: bool = True,
) -> AcdcSystem:
    """Assemble a parsed cluster set into a full simulation system."""
    order = cluster_set.used_molecule_names
    keep = [cluster_set.molecule_names.index(n) for n in order]
    by_name = {m.name: m for m in cluster_set.molecules}
    molecule_charges = tuple(by_name[n].charge for n in order)

    species_labels = list(cluster_set.labels())
    compositions = [tuple(row[i] for i in keep) for row in cluster_set.compositions]

    if include_generic_neg:
        species_labels.append(labels.GENERIC_NEG)
        compositions.append(generic_ion_composition(cluster_set, order, negative=True))
    if include_generic_pos:
        species_labels.append(labels.GENERIC_POS)
        compositions.append(generic_ion_composition(cluster_set, order, negative=False))

    n_clusters = len(species_labels)
    flux_index = {name: n_clusters + i for i, name in enumerate(FLUX_SLOTS)}

    charges = tuple(_charge_of(counts, molecule_charges) for counts in compositions)

    return AcdcSystem(
        order=order,
        labels=tuple(species_labels),
        compositions=tuple(compositions),
        charges=charges,
        n_clusters=n_clusters,
        n_equations=n_clusters + len(FLUX_SLOTS),
        flux_index=flux_index,
        has_generic_neg=include_generic_neg,
        has_generic_pos=include_generic_pos,
        charges_of_molecule=molecule_charges,
    )


def _charge_of(counts: tuple[int, ...], molecule_charges: tuple[int, ...]) -> int:
    """Charge of a composition: the first charged molecule present.

    Matches Perl :10289-10305 -- scan in molecule order, stop at the first
    charged type with a positive count. Not an algebraic sum, which matters
    only for the pseudo-compositions where a count can be negative.
    """
    for i, n in enumerate(counts):
        if n > 0 and molecule_charges[i] == -1:
            return -n
        if n > 0 and molecule_charges[i] == 1:
            return n
    return 0


def reference_number_density(pressure: float, temperature: float) -> float:
    """p_ref / (k_B T), m^-3."""
    return pressure / (config.K_B * temperature)
