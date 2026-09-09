"""Cluster labels: parsing, canonicalisation, formatting.

A cluster is named by concatenating ``<count><molecule>`` in a canonical
molecule order: ``2A1N`` is two sulfuric acid and one ammonia. The two
generic charger ions are the exception -- they are bare words, ``neg`` and
``pos``, because they carry a charge and a mass but no composition.

The same cluster can be written several ways (``1N2A`` is ``2A1N``), so
everything here goes through a canonical form. Getting this wrong does not
raise: it silently creates two entries for one cluster.

Pure Python; runs once at setup, never inside ``jit``.
"""

from __future__ import annotations

import re

GENERIC_NEG = "neg"
GENERIC_POS = "pos"
GENERIC_IONS = frozenset({GENERIC_NEG, GENERIC_POS})

# `<optional count><molecule>`. The count is optional because the Perl
# generator prints stripped monomers as a bare molecule name -- `2 A` in a
# boundary log means two of molecule `A`, not one cluster `2A`. A molecule
# name is an upper-case letter optionally followed by a lower-case one.
_TERM = re.compile(r"(\d*)([A-Z][a-z]?)")
_VALID = re.compile(r"^(?:\d*[A-Z][a-z]?)+$")

Composition = dict[str, int]


def is_generic_ion(label: str) -> bool:
    return label in GENERIC_IONS


def parse(label: str) -> Composition:
    """``4A1B3N`` -> ``{'A': 4, 'B': 1, 'N': 3}``.

    A missing count means one. Generic ions parse to an empty composition:
    they have no molecules, which is exactly right for mass bookkeeping of
    the *molecular* content, but callers that need their mass must look it
    up separately.

    Raises:
        ValueError: if the label is not a well-formed composition.
    """
    if is_generic_ion(label):
        return {}
    if not _VALID.match(label):
        raise ValueError(f"malformed cluster label: {label!r}")
    out: Composition = {}
    for count, molecule in _TERM.findall(label):
        out[molecule] = out.get(molecule, 0) + (int(count) if count else 1)
    return out


def format_label(composition: Composition, order: list[str] | tuple[str, ...]) -> str:
    """Canonical label for a composition, in the given molecule order.

    Molecules with a zero count are omitted, matching the generator: the
    cluster of two acids is ``2A``, never ``2A0N``.
    """
    parts = [f"{composition[m]}{m}" for m in order if composition.get(m, 0) > 0]
    return "".join(parts)


def canonical(label: str, order: list[str] | tuple[str, ...]) -> str:
    """Rewrite a label into canonical molecule order.

    ``canonical("1N2A", ["A", "B", "N", "P"]) == "2A1N"``. Generic ions pass
    through unchanged.
    """
    if is_generic_ion(label):
        return label
    composition = parse(label)
    unknown = set(composition) - set(order)
    if unknown:
        raise ValueError(
            f"label {label!r} contains molecules not in the system: {sorted(unknown)}"
        )
    return format_label(composition, order)


def total_molecules(label: str) -> int:
    """Total molecule count, used to tell monomers from clusters.

    The proton and the missing-proton pseudo-species are *not* excluded
    here; callers that need the physical molecule count must exclude them
    explicitly. `1N1P` -- protonated ammonia -- is a monomer with a total of
    two by this measure.
    """
    return sum(parse(label).values())


def composition_vector(label: str, order: list[str] | tuple[str, ...]) -> list[int]:
    """Composition as a dense vector in the given molecule order."""
    composition = parse(label)
    unknown = set(composition) - set(order)
    if unknown:
        raise ValueError(
            f"label {label!r} contains molecules not in the system: {sorted(unknown)}"
        )
    return [composition.get(m, 0) for m in order]
