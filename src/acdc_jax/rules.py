"""Rule lists: per-collision sticking factors and per-channel Delta-G corrections.

Two generator options share one shape -- a list of ``(value, [target,
[partner]])`` rules scanned in order with the LAST match winning:

* ``--sticking_factor`` / ``--sticking_factor_ion_neutral`` /
  ``--sticking_factor_file_name`` (Perl 10.2 :1290-1340, 10.7 :8308-8358)
  multiply a collision coefficient.
* ``--scale_evap_factor`` / ``--scale_evap_file_name`` (Perl :8878-8905,
  :8976-8991) add a kcal/mol correction to the reaction free energy of an
  evaporation channel, inside the exponent.

The matching rules differ in small ways that are copied exactly here; each
function's docstring says which Perl branch it mirrors. Everything is NumPy
and runs once at setup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acdc_jax import labels as label_module
from acdc_jax.system import AcdcSystem

Rule = tuple

ION_NEUTRAL = "ion-neutral"
"""Target keyword selecting every collision between one ion and one neutral."""


def parse_rule_file(path: str | Path) -> tuple[Rule, ...]:
    """Read a sticking or scaling rule file: ``value [label [label]]`` per line.

    Lines starting with ``#`` are comments. The value must be a positive
    number (the generator's ``is_pos_number``); zero or negative raises.
    """
    rules: list[Rule] = []
    for raw in Path(path).read_text().splitlines():
        if raw.startswith("#") or not raw.strip():
            continue
        columns = raw.split()
        value = float(columns[0])
        if value <= 0:
            raise ValueError(f"rule value must be positive: {raw!r}")
        rules.append((value, *columns[1:3]))
    return tuple(rules)


def sticking_rules(
    factor: float = 1.0,
    ion_neutral_factor: float = 1.0,
    file_rules: tuple[Rule, ...] = (),
) -> tuple[Rule, ...]:
    """Assemble the rule list exactly as the generator orders it (Perl :1293-1310).

    A bare ``--sticking_factor`` comes first, ``--sticking_factor_ion_neutral``
    second as an ``ion-neutral`` rule, then the file's rows. Because later
    rules override earlier ones, a file row can always win.
    """
    rules: list[Rule] = []
    if ion_neutral_factor != 1.0:
        if factor == 1.0:
            rules = [(ion_neutral_factor, ION_NEUTRAL)]
        else:
            rules = [(factor,), (ion_neutral_factor, ION_NEUTRAL)]
    elif factor:
        rules = [(factor,)]
    return (*rules, *file_rules)


def evap_scale_rules(
    factor: float = 0.0, file_rules: tuple[Rule, ...] = ()
) -> tuple[Rule, ...]:
    """``--scale_evap_factor`` first (if nonzero), then the file's rows."""
    rules: list[Rule] = [(factor,)] if factor != 0.0 else []
    return (*rules, *file_rules)


def _same_cluster(label_a: str, label_b: str, order: tuple[str, ...]) -> bool:
    """The generator's ``compare_clusters``: equal composition in any order.

    Generic ion labels compare as strings.
    """
    if label_module.is_generic_ion(label_a) or label_module.is_generic_ion(label_b):
        return label_a == label_b
    try:
        return label_module.canonical(label_a, order) == label_module.canonical(
            label_b, order
        )
    except ValueError:
        return False


def _contains_molecule(label: str, molecule: str) -> bool:
    """Perl :8316: ``\\d+NAME\\d+`` or ``\\d+NAME$`` -- the cluster has >= 1 of it."""
    return label_module.parse(label).get(molecule, 0) > 0


def sticking_matrix(system: AcdcSystem, rules: tuple[Rule, ...]) -> np.ndarray:
    """Per-pair sticking factor, shape (nclust, nclust), 1.0 where none applies.

    Mirrors Perl :8308-8358. For each pair the rules are scanned in order:

    * ``(v,)`` or ``(v, molecule)``: neutral-neutral collisions only; with a
      molecule name, only pairs where either cluster contains it.
    * ``(v, "ion-neutral")``: exactly one of the pair is charged.
    * ``(v, cluster)``: either member has that composition.
    * ``(v, cluster, partner)``: one member is ``cluster`` and the OTHER is
      ``partner``.

    The last matching rule wins. A factor that is not exactly 1 is rounded
    the way the generator prints it, ``%.4e``, because that rounded literal
    is what the Fortran multiplies by.
    """
    n = system.n_clusters
    order = system.order
    molecule_names = set(order)
    factor = np.ones((n, n))
    if not rules:
        return factor

    charged = np.asarray(system.charges) != 0
    for i in range(n):
        for j in range(i, n):
            li, lj = system.labels[i], system.labels[j]
            neutral_pair = not charged[i] and not charged[j]
            ion_neutral = charged[i] != charged[j]
            value = 1.0
            for rule in rules:
                if len(rule) == 1 or rule[1] in molecule_names:
                    if neutral_pair:
                        if len(rule) == 1:
                            value = rule[0]
                        elif _contains_molecule(li, rule[1]) or _contains_molecule(
                            lj, rule[1]
                        ):
                            value = rule[0]
                elif len(rule) == 2 and rule[1] == ION_NEUTRAL:
                    if ion_neutral:
                        value = rule[0]
                elif _same_cluster(li, rule[1], order) or _same_cluster(
                    lj, rule[1], order
                ):
                    if len(rule) == 2:
                        value = rule[0]
                    else:
                        other = lj if _same_cluster(li, rule[1], order) else li
                        if _same_cluster(other, rule[2], order):
                            value = rule[0]
            if value != 1.0:
                value = float(f"{value:.4e}")
            factor[i, j] = factor[j, i] = value
    return factor


def evap_scale_matrix(system: AcdcSystem, rules: tuple[Rule, ...]) -> np.ndarray:
    """Per-daughter-pair Delta-G correction, kcal/mol, shape (nclust, nclust).

    Mirrors Perl :8976-8991. Indexed by the two daughters ``(i, j)`` of the
    channel ``k -> i + j``; the parent is the in-system cluster with the
    summed composition, which is unique, so the pair identifies the channel.

    * ``(v,)``: every channel.
    * ``(v, cluster)``: channels where either daughter is ``cluster``.
    * ``(v, cluster, parent)``: additionally the parent must be ``parent``.

    Last match wins; the emitted literal is ``%.14e`` so no rounding is
    modelled. Pairs whose sum is not in the system never evaporate and are
    left at zero.
    """
    n = system.n_clusters
    order = system.order
    scale = np.zeros((n, n))
    if not rules:
        return scale

    by_composition = {tuple(c): k for k, c in enumerate(system.compositions)}
    compositions = np.asarray(system.compositions)
    for i in range(n):
        for j in range(i, n):
            parent = by_composition.get(tuple(compositions[i] + compositions[j]))
            if parent is None:
                continue
            li, lj, lk = system.labels[i], system.labels[j], system.labels[parent]
            value = 0.0
            for rule in rules:
                if len(rule) == 1:
                    value = rule[0]
                elif _same_cluster(li, rule[1], order) or _same_cluster(
                    lj, rule[1], order
                ):
                    if len(rule) == 2 or _same_cluster(lk, rule[2], order):
                        value = rule[0]
            scale[i, j] = scale[j, i] = value
    return scale


__all__ = [
    "ION_NEUTRAL",
    "Rule",
    "evap_scale_matrix",
    "evap_scale_rules",
    "parse_rule_file",
    "sticking_matrix",
    "sticking_rules",
]
