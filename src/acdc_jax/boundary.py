"""Collision products, and the boundary cascade.

This is the one part of ACDC with no closed-form specification. When a
collision product falls outside the enumerated cluster set, it is either
counted as having grown out of the system -- contributing to the formation
rate J -- or "brought back" by stripping molecules off it until it lands on
something the set contains. Which molecules come off, and in what order, is
defined *operationally* by ~250 lines of stateful Perl
(``acdc_2020_04_28.pl:10263-10618``).

So this module is a deliberate **line-by-line port, not a reimplementation**.
It is written to be diffed against the Perl, not to be elegant: the control
flow, the loop labels, the order of the strength ladder and the running
maximum in the fallback are all preserved even where they look accidental.
Boundary reactions carry a large share of the mass flux in narrow cluster
sets, so an error here yields a plausible wrong answer rather than an
obvious one.

Validated by exact match against ``perl … --print_boundary`` on three
cluster sets (3034 decisions). Plain Python: setup only.
"""

from __future__ import annotations

import dataclasses

from acdc_jax import labels
from acdc_jax.clusterset import ClusterSetFile, Molecule


@dataclasses.dataclass(frozen=True)
class Composition:
    """A molecule-count vector over the system's molecule order."""

    counts: tuple[int, ...]

    def label(self, order: tuple[str, ...]) -> str:
        return labels.format_label(
            {name: n for name, n in zip(order, self.counts, strict=True)}, order
        )


@dataclasses.dataclass(frozen=True)
class CombineResult:
    """Outcome of colliding two clusters (Perl ``combine_labels``)."""

    label: str | None
    """Product label, or None if the collision cannot happen."""
    valid_coll: bool
    valid_evap: bool
    """Whether the product may evaporate back into these two pieces."""
    in_system: bool
    """Whether the product is one of the enumerated clusters."""


@dataclasses.dataclass(frozen=True)
class BoundaryResult:
    """Outcome of the cascade (Perl ``check_boundary``)."""

    label: str
    """Where the product ended up: itself if it grew out, otherwise the
    in-system cluster it was stripped back to."""
    lout: int
    """0 = brought back; 1 = out as neutral, 2 = negative, 3 = positive."""
    monomers: dict[str, int]
    """Molecules stripped off, by molecule name. Empty if it grew out."""


class BoundarySystem:
    """The cluster set plus everything the cascade needs to consult.

    Built once per cluster set. Holds the membership test, the per-charge
    composition maxima, and the grow-out rules.
    """

    def __init__(self, cluster_set: ClusterSetFile) -> None:
        self.order = cluster_set.used_molecule_names
        self.n_mol_types = len(self.order)

        by_name = {m.name: m for m in cluster_set.molecules}
        self.molecules: tuple[Molecule, ...] = tuple(by_name[n] for n in self.order)

        keep = [cluster_set.molecule_names.index(n) for n in self.order]
        self.compositions = tuple(
            tuple(row[i] for i in keep) for row in cluster_set.compositions
        )

        # Membership test. `check_validity` (Perl :10798) is exactly "does
        # this composition have a cluster number", so a set lookup.
        self._members = set(self.compositions)

        self.proton = next((i for i, m in enumerate(self.molecules) if m.is_proton), -1)
        self.missing_proton = next(
            (i for i, m in enumerate(self.molecules) if m.is_missing_proton), -1
        )

        self.charge = tuple(m.charge for m in self.molecules)
        self.acid_strength = tuple(m.acid_strength for m in self.molecules)
        self.base_strength = tuple(m.base_strength for m in self.molecules)

        # Default true for every molecule (Perl :919); the input file may
        # override it with a `can be lost` row, which these files do not use.
        self.can_be_lost = tuple(True for _ in self.molecules)

        self.acid_max = max((s for s in self.acid_strength), default=0)
        self.base_max = max((s for s in self.base_strength), default=0)

        # Set when ANY neutral molecule carries an acid or base strength
        # (Perl :1539). Gates the strength-guided stripping stage entirely.
        self.strength_neutral = any(
            m.charge == 0 and (m.acid_strength > 0 or m.base_strength > 0)
            for m in self.molecules
        )

        self.n_max_type = self._max_per_charge(0)
        self.n_max_type_neg = self._max_per_charge(-1)
        self.n_max_type_pos = self._max_per_charge(+1)

        self.rules = {
            charge: tuple(
                tuple(rule[i] for i in keep)
                for rule in cluster_set.out_rules.get(name, ())
            )
            for charge, name in ((0, "neutral"), (-1, "negative"), (1, "positive"))
        }

    # -- helpers ---------------------------------------------------------

    def _max_per_charge(self, charge: int) -> tuple[int, ...]:
        """Largest count of each molecule type over clusters of one charge.

        Perl :1355-1396. Per charge class, not global: using the global
        maximum would let a negative cluster keep more acid than any
        negative cluster in the set actually has, and the error is silent.
        """
        maxima = [0] * self.n_mol_types
        for comp in self.compositions:
            if self.cluster_charge(comp) != charge:
                continue
            for i, n in enumerate(comp):
                maxima[i] = max(maxima[i], n)
        return tuple(maxima)

    def cluster_charge(self, counts: tuple[int, ...]) -> int:
        """Charge of a composition.

        Follows the reference exactly: scan molecules in order and stop at
        the FIRST charged one present, taking its signed count
        (Perl :10289-10305). A composition containing both a negative
        molecule and a proton therefore takes whichever comes first in the
        molecule order rather than the algebraic sum -- which cannot arise
        for a physical cluster, but the traversal order is what defines the
        answer if it ever did.
        """
        for i, n in enumerate(counts):
            if n > 0 and self.charge[i] == -1:
                return -n
            if n > 0 and self.charge[i] == 1:
                return n
        return 0

    def is_member(self, counts: tuple[int, ...]) -> bool:
        return counts in self._members

    def label_of(self, counts: tuple[int, ...]) -> str:
        return Composition(counts).label(self.order)

    def n_max_for(self, charge: int) -> tuple[int, ...]:
        if charge < 0:
            return self.n_max_type_neg
        if charge > 0:
            return self.n_max_type_pos
        return self.n_max_type

    def _forms_ion_with(self, imol: int, ion_index: int, positive: bool) -> bool:
        """Whether molecule `imol` plus the (missing) proton is a named ion.

        Perl compares `corr_positive_mol[imol]` against the literal
        `"1<name>1<proton label>"` via `compare_clusters`, i.e. by
        composition rather than by string. Used by the fallback stage to
        work out how many of a molecule are tied up carrying the charge.
        """
        molecule = self.molecules[imol]
        corr = molecule.corr_positive if positive else molecule.corr_negative
        if corr is None or ion_index < 0:
            return False
        expected = {molecule.name: 1, self.molecules[ion_index].name: 1}
        try:
            return labels.parse(corr) == expected
        except ValueError:
            return False

    # -- combine_labels --------------------------------------------------

    def combine(
        self, counts_i: tuple[int, ...], counts_j: tuple[int, ...]
    ) -> CombineResult:
        """Collide two clusters (Perl ``combine_labels``, :10022-10261).

        Returns the product composition's label together with whether the
        collision may happen at all, whether the product may evaporate back
        into these two pieces, and whether it is inside the system.
        """
        charge_i = self.cluster_charge(counts_i)
        charge_j = self.cluster_charge(counts_j)

        # Same-sign ions never collide (Perl :10130).
        if (charge_i < 0 and charge_j < 0) or (charge_i > 0 and charge_j > 0):
            return CombineResult(None, False, False, False)

        if charge_i * charge_j < 0:
            combined = self._recombine(counts_i, counts_j, charge_i)
            if combined is None:
                return CombineResult(None, False, False, False)
            # Recombination products can never evaporate back into an
            # ion pair (Perl :10082).
            valid_evap = False
        else:
            combined = tuple(a + b for a, b in zip(counts_i, counts_j, strict=True))
            if any(n < 0 for n in combined):
                return CombineResult(None, False, False, False)
            if not self._protonation_is_supportable(combined):
                return CombineResult(None, False, False, False)
            valid_evap = True

        in_system = self.is_member(combined)
        if not in_system:
            valid_evap = False

        return CombineResult(
            label=self.label_of(combined),
            valid_coll=True,
            valid_evap=valid_evap,
            in_system=in_system,
        )

    def _recombine(
        self, counts_i: tuple[int, ...], counts_j: tuple[int, ...], charge_i: int
    ) -> tuple[int, ...] | None:
        """Neutralise an ion pair before summing (Perl :10077-10125).

        The negative ion reverts to its corresponding neutral molecule (or,
        if it is described by a missing proton, that count is simply
        zeroed), and the positive cluster loses one proton. Only then are
        the compositions added. Summing first would conserve charge but
        invent molecules.
        """
        neg = list(counts_i if charge_i < 0 else counts_j)
        pos = list(counts_j if charge_i < 0 else counts_i)

        wanted = next(
            (i for i, n in enumerate(neg) if n > 0 and self.charge[i] < 0), -1
        )
        if wanted < 0:
            raise ValueError(
                f"no charged molecule in the negative cluster "
                f"{self.label_of(tuple(neg))!r}"
            )

        if self.missing_proton >= 0 and neg[self.missing_proton] == 1:
            neg[self.missing_proton] = 0
        else:
            corr_name = self.molecules[wanted].corr_neutral
            if corr_name is None or corr_name not in self.order:
                return None
            neg[wanted] -= 1
            neg[self.order.index(corr_name)] += 1

        if self.proton >= 0:
            pos[self.proton] -= 1

        combined = tuple(a + b for a, b in zip(neg, pos, strict=True))
        if any(n < 0 for n in combined):
            return None
        return combined

    def _protonation_is_supportable(self, counts: tuple[int, ...]) -> bool:
        """A (missing) proton needs somewhere to sit (Perl :10160-10189).

        A product containing a proton must also contain a neutral molecule
        that has a corresponding positive ion -- something able to accept
        it. Otherwise the composition is nameable but not chemical.
        """
        for index, positive in ((self.proton, True), (self.missing_proton, False)):
            if index < 0 or counts[index] <= 0:
                continue
            found = False
            for i, n in enumerate(counts):
                if n <= 0 or self.charge[i] != 0:
                    continue
                corr = (
                    self.molecules[i].corr_positive
                    if positive
                    else self.molecules[i].corr_negative
                )
                if corr is not None:
                    found = True
            if not found:
                return False
        return True

    # -- check_boundary --------------------------------------------------

    def check_boundary(self, counts: tuple[int, ...]) -> BoundaryResult:
        """Decide the fate of an out-of-system product (Perl :10263-10618).

        Three stages, in order: did it grow out; clamp any molecule type
        that exceeds the per-charge maximum; then strip molecules one at a
        time, first by acid/base strength and then by sheer numbers.

        Raises:
            ValueError: if the composition is already in the system (the
                reference dies here too -- it means the caller should not
                have asked), or if nothing can bring it back.
        """
        if self.is_member(counts):
            raise ValueError(
                f"{self.label_of(counts)} is inside the system; "
                "check_boundary should not have been called"
            )

        monomers = dict.fromkeys(self.order, 0)
        working = list(counts)
        charge_clus = self.cluster_charge(counts)

        lout = self._grew_out(counts, charge_clus)
        if lout:
            # Nothing was stripped, so report an empty bag rather than the
            # zero-filled working dict -- same convention as the
            # brought-back path below.
            return BoundaryResult(self.label_of(counts), lout, {})

        valid = self._clamp(working, monomers, charge_clus)

        if not valid and self.strength_neutral:
            valid = self._strip_by_strength(working, monomers)

        if not valid:
            valid = self._strip_by_count(working, monomers, charge_clus)

        if not valid:
            raise ValueError(
                f"cannot bring {self.label_of(counts)} back to the boundary"
            )

        return BoundaryResult(
            self.label_of(tuple(working)),
            0,
            {k: v for k, v in monomers.items() if v},
        )

    def _grew_out(self, counts: tuple[int, ...], charge_clus: int) -> int:
        """Stage 1: does the product satisfy a grow-out rule?

        A rule fires only when EVERY molecule count meets or exceeds it;
        multiple rules for a charge are OR'd (Perl :10309-10327).
        """
        rules = self.rules.get(
            -1 if charge_clus < 0 else (1 if charge_clus > 0 else 0), ()
        )
        channel = 2 if charge_clus < 0 else (3 if charge_clus > 0 else 1)
        for rule in rules:
            if all(n >= threshold for n, threshold in zip(counts, rule, strict=True)):
                return channel
        return 0

    def _clamp(
        self, working: list[int], monomers: dict[str, int], charge_clus: int
    ) -> bool:
        """Stage 2: cap each molecule type at the per-charge maximum.

        Perl :10331-10348. Anything above the largest count that appears in
        any cluster of this charge cannot be in the set, so shed it first.
        """
        maxima = self.n_max_for(charge_clus)
        for i, limit in enumerate(maxima):
            if working[i] > limit:
                monomers[self.order[i]] += working[i] - limit
                working[i] = limit
        return self.is_member(tuple(working))

    def _strip_by_strength(self, working: list[int], monomers: dict[str, int]) -> bool:
        """Stage 3a: remove the weakest acid or base, whichever is in excess.

        Perl :10352-10524. The loop alternates between removing acids and
        removing bases whenever the balance flips mid-removal, and walks up
        the strength ladder from 0 -- so a molecule of strength 2 is only
        considered after strengths 0 and 1 have been exhausted.

        Preserved as written, including the `last ACID_MOL_LOOP` structure
        and the fact that the ladder starts at strength 0, which matches no
        molecule in these files and simply costs an iteration.
        """
        n_acid = n_acid_other = n_base = n_base_other = n_neutral = 0
        for i, n in enumerate(working):
            if n <= 0:
                continue
            if self.charge[i] == 0:
                n_neutral += n
            if self.acid_strength[i] > 0:
                if self.can_be_lost[i] or i == self.proton:
                    n_acid += n
                else:
                    n_acid_other += n
                if i == self.proton:
                    # A proton makes an acid out of what would be a base.
                    n_base_other -= 1
                    n_neutral -= 1
            if self.base_strength[i] > 0:
                if self.can_be_lost[i] or i == self.missing_proton:
                    n_base += n
                else:
                    n_base_other += n
                if i == self.missing_proton:
                    n_acid_other -= 1
                    n_neutral -= 1

        if n_acid_other < 0:
            n_acid += n_acid_other
            n_acid_other = 0
        if n_base_other < 0:
            n_base += n_base_other
            n_base_other = 0

        remove_acids = (n_acid + n_acid_other) > (n_base + n_base_other)
        acid_min = base_min = 0
        diff = 0
        valid = False

        while not valid and n_neutral > diff:
            if remove_acids:
                if acid_min > self.acid_max:
                    break
                for i in range(self.n_mol_types):
                    if not self.can_be_lost[i]:
                        continue
                    if not (
                        working[i] > 0
                        and self.acid_strength[i] == acid_min
                        and self.charge[i] == 0
                    ):
                        continue
                    done = False
                    while working[i] > 0:
                        working[i] -= 1
                        n_neutral -= 1
                        monomers[self.order[i]] += 1
                        if self.acid_strength[i] > 0:
                            n_acid -= 1
                        if self.base_strength[i] > 0:
                            n_base -= 1
                        if (n_acid + n_acid_other) <= (n_base + n_base_other + diff):
                            remove_acids = False
                        valid = self.is_member(tuple(working))
                        if valid or not remove_acids:
                            done = True
                            break
                    if done:
                        break
                if remove_acids:
                    acid_min += 1
                else:
                    base_min = 0
            else:
                if base_min > self.base_max:
                    break
                for i in range(self.n_mol_types):
                    if not self.can_be_lost[i]:
                        continue
                    if not (
                        working[i] > 0
                        and self.base_strength[i] == base_min
                        and self.charge[i] == 0
                    ):
                        continue
                    done = False
                    while working[i] > 0:
                        working[i] -= 1
                        n_neutral -= 1
                        monomers[self.order[i]] += 1
                        if self.base_strength[i] > 0:
                            n_base -= 1
                        if self.acid_strength[i] > 0:
                            n_acid -= 1
                        if (n_acid + n_acid_other + diff) > (n_base + n_base_other):
                            remove_acids = True
                        valid = self.is_member(tuple(working))
                        if valid or remove_acids:
                            done = True
                            break
                    if done:
                        break
                if not remove_acids:
                    base_min += 1
                else:
                    acid_min = 0

        return valid

    def _strip_by_count(
        self, working: list[int], monomers: dict[str, int], charge_clus: int
    ) -> bool:
        """Stage 3b: remove the most numerous removable neutral molecule.

        Perl :10528-10598. Two things preserved deliberately:

        - ``removable_max`` is a RUNNING maximum, initialised once outside
          the loop and never reset. After a removal it is decremented, and
          the rescan only raises it. Resetting it each pass changes which
          molecule is chosen next.

        - For a charged cluster, molecules that are tied up carrying the
          charge are subtracted from the removable count, and that
          non-removability is transferred to other chargeable species.
        """
        removable_max = 0
        imol_max = -1
        valid = False

        while not valid:
            if removable_max > 0:
                removable_max -= 1
                working[imol_max] -= 1
                monomers[self.order[imol_max]] += 1
                valid = self.is_member(tuple(working))

            if valid:
                break

            for i in range(self.n_mol_types):
                if not self.can_be_lost[i] or self.charge[i] != 0:
                    continue
                if working[i] <= 0:
                    continue
                removable = working[i]

                ion_index, positive = (
                    (self.missing_proton, False)
                    if charge_clus < 0
                    else (self.proton, True)
                )
                if (
                    charge_clus != 0
                    and ion_index >= 0
                    and self._forms_ion_with(i, ion_index, positive)
                    and working[ion_index] > 0
                ):
                    nonremovable = working[ion_index]
                    removable -= nonremovable
                    for j in range(self.n_mol_types):
                        # Faithful to the Perl, which tests `charge[imol]`
                        # here rather than `charge[jmol]` -- almost
                        # certainly a typo, but since imol is known neutral
                        # at this point the test never fires and the loop
                        # runs over every other molecule type. Reproducing
                        # the typo keeps the traversal identical.
                        if j == i or self.charge[i] != 0:
                            continue
                        if working[j] > 0 and self._forms_ion_with(
                            j, ion_index, positive
                        ):
                            shift = min(working[j], nonremovable)
                            removable += shift
                            nonremovable -= shift
                        if nonremovable == 0:
                            break

                if removable > removable_max:
                    removable_max = removable
                    imol_max = i

            if removable_max == 0:
                break

        return valid
