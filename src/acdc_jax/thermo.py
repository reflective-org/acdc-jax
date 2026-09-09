"""Cluster formation free energies, and dipole/polarizability data.

Parses the two quantum-chemistry input files the Perl generator reads
(`:2452-2540` and `:8459-8524`). Both are whitespace-separated tables with a
short numeric preamble.

These are **formation** free energies -- of the cluster relative to its
constituent free monomers -- so a monomer is zero by definition and only
needs an entry if it is not.

Plain Python: setup only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from acdc_jax import config, labels


@dataclasses.dataclass(frozen=True)
class EnergyTable:
    """Cluster formation enthalpies and entropies.

    ``delta_h`` is kcal/mol and ``delta_s`` is cal/(mol K) -- the units of
    the file, kept rather than converted so that comparisons against the
    values the generator inlines into ``get_evap`` are exact.
    """

    pressure: float
    """Reference pressure, Pa. Line 1 of the file."""
    temperature: float
    """Reference temperature, K. Line 2. The temperature the data is FOR,
    not necessarily the one the simulation runs at."""
    delta_h: dict[str, float]
    delta_s: dict[str, float]
    from_gibbs: frozenset[str]
    """Labels given as a single Delta-G rather than as H and S.

    These cannot be re-evaluated at another temperature: the generator dies
    if `--temperature` differs from the file's and any entry is G-only
    (Perl :2540).
    """

    def gibbs(self, label: str, temperature: float | None = None) -> float:
        """Formation free energy in kcal/mol.

        ``G = H - T*S/1000``, the 1000 converting the entropy from
        cal/(mol K) to kcal/(mol K).
        """
        if temperature is None:
            temperature = self.temperature
        if label in self.from_gibbs and temperature != self.temperature:
            raise ValueError(
                f"{label!r} was given as a single Delta-G at "
                f"{self.temperature} K and cannot be re-evaluated at "
                f"{temperature} K; supply H and S instead"
            )
        h = self.delta_h.get(label, 0.0)
        s = self.delta_s.get(label, 0.0)
        return h - temperature * s / 1000.0

    def gibbs_array(
        self,
        cluster_labels: list[str] | tuple[str, ...],
        temperature: float | None = None,
    ) -> np.ndarray:
        """Formation free energies for a list of clusters, kcal/mol.

        Clusters with no entry are zero -- correct for monomers, and for the
        generic charger ions, which have no formation energy at all.
        """
        return np.array([self.gibbs(label, temperature) for label in cluster_labels])


def parse_energy_file(
    path: str | Path, molecule_order: tuple[str, ...] | None = None
) -> EnergyTable:
    """Parse a ``HS<T>.txt`` / ``G<T>.txt`` free-energy file.

    Format: line 1 is the reference pressure in Pa, line 2 the reference
    temperature in K. Remaining lines are blank, ``#`` comments, or
    ``<label> <H> <S>`` (kcal/mol, cal/mol/K) or ``<label> <G>`` (kcal/mol).

    Args:
        molecule_order: if given, labels are canonicalised into this order,
            so a file written as ``1N2A`` matches a cluster set that calls
            it ``2A1N``.

    Raises:
        ValueError: on a duplicate entry (Perl :2521), a malformed line, or
            a preamble that is not two numbers.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    preamble: list[float] = []
    delta_h: dict[str, float] = {}
    delta_s: dict[str, float] = {}
    from_gibbs: set[str] = set()

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        if len(preamble) < 2:
            try:
                preamble.append(float(line))
            except ValueError as exc:
                raise ValueError(
                    f"{path}: expected pressure (Pa) then temperature (K) "
                    f"as the first two values, got {line!r}"
                ) from exc
            continue

        fields = line.split()
        label = fields[0]
        if molecule_order is not None:
            label = labels.canonical_hydrate(label, molecule_order)
        if label in delta_h:
            raise ValueError(f"{path}: duplicate entry for cluster {label!r}")

        if len(fields) == 3:
            delta_h[label] = float(fields[1])
            delta_s[label] = float(fields[2])
        elif len(fields) == 2:
            delta_h[label] = float(fields[1])
            delta_s[label] = 0.0
            from_gibbs.add(label)
        else:
            raise ValueError(
                f"{path}: expected `<label> <H> <S>` or `<label> <G>`, got {line!r}"
            )

    if len(preamble) < 2:
        raise ValueError(f"{path}: missing pressure and/or temperature preamble")
    pressure, temperature = preamble
    if pressure <= 0:
        raise ValueError(f"{path}: reference pressure must be positive, got {pressure}")

    return EnergyTable(
        pressure=pressure,
        temperature=temperature,
        delta_h=delta_h,
        delta_s=delta_s,
        from_gibbs=frozenset(from_gibbs),
    )


def check_energies_available(
    table: EnergyTable,
    cluster_labels: list[str] | tuple[str, ...],
    monomer_labels: set[str] | frozenset[str],
) -> None:
    """Raise if a cluster that needs a free energy has none.

    Monomers and generic charger ions are exempt: their formation energy is
    zero by definition of the reference state. Upstream accumulates the
    missing ones and dies with the full list (Perl :2560-2590), which is
    more useful than failing on the first, so this does the same.
    """
    missing = [
        label
        for label in cluster_labels
        if label not in table.delta_h
        and label not in monomer_labels
        and not labels.is_generic_ion(label)
    ]
    if missing:
        raise ValueError(
            f"no free energy for {len(missing)} cluster(s): {', '.join(missing)}"
        )


@dataclasses.dataclass(frozen=True)
class DipoleTable:
    """Dipole moments and polarizabilities of the electrically neutral species.

    Needed for the ion-neutral collision enhancement: the ion polarises the
    neutral, so the capture rate depends on the neutral's dipole moment and
    polarizability, not on the ion's.
    """

    monomer_locking: float
    """Dipole locking coefficient for monomers, in [0, 1]. Su73 only."""
    cluster_locking: float
    """Dipole locking coefficient for clusters, in [0, 1]. Su73 only."""
    dipole: dict[str, float]
    """Debye."""
    polarizability: dict[str, float]
    """Angstrom^3."""


def parse_dipole_file(
    path: str | Path, molecule_order: tuple[str, ...] | None = None
) -> DipoleTable:
    """Parse a ``dip_pol<T>.txt`` file.

    Lines 1 and 2 are the dipole locking coefficients for monomers and for
    clusters; the rest are ``<label> <dipole/D> <polarizability/A^3>``.

    The locking coefficients are read even under Su82, which does not use
    them -- the file format carries them unconditionally.
    """
    path = Path(path)
    preamble: list[float] = []
    dipole: dict[str, float] = {}
    polarizability: dict[str, float] = {}

    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        if len(preamble) < 2:
            preamble.append(float(line))
            continue

        fields = line.split()
        if len(fields) != 3:
            raise ValueError(
                f"{path}: expected `<label> <dipole> <polarizability>`, got {line!r}"
            )
        label = fields[0]
        if molecule_order is not None:
            label = labels.canonical_hydrate(label, molecule_order)
        if label in dipole:
            raise ValueError(f"{path}: duplicate entry for cluster {label!r}")
        dipole[label] = float(fields[1])
        polarizability[label] = float(fields[2])

    if len(preamble) < 2:
        raise ValueError(f"{path}: missing the two dipole locking coefficients")
    for value in preamble:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{path}: dipole locking coefficient {value} outside [0, 1]"
            )

    return DipoleTable(
        monomer_locking=preamble[0],
        cluster_locking=preamble[1],
        dipole=dipole,
        polarizability=polarizability,
    )


def check_dipoles_available(
    table: DipoleTable,
    neutral_labels: list[str] | tuple[str, ...],
) -> None:
    """Raise if a neutral cluster has no dipole data (Perl :8525-8538).

    Only neutrals need it: in an ion-neutral collision the ion is the
    polarising partner and the neutral is the one being polarised.
    """
    missing = [label for label in neutral_labels if label not in table.dipole]
    if missing:
        raise ValueError(
            f"no dipole/polarizability data for {len(missing)} neutral "
            f"cluster(s): {', '.join(missing)}"
        )


def reference_number_density(pressure: float, temperature: float) -> float:
    """p_ref / (k_B T), m^-3 -- the prefactor in the evaporation rate.

    At 1 atm this is the 7.33893243358348e27/T that appears in every
    ``get_evap`` expression.
    """
    return pressure / (config.K_B * temperature)
