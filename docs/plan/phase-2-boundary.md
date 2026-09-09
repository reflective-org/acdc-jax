# Phase 2 — The boundary cascade

**Goal:** decide, for every collision whose product falls outside the
enumerated cluster set, whether it counts as grown out (contributing to J)
or is stripped back into the system — and if stripped, into exactly what.

**Gate: exact match** against `perl … --print_boundary` for
`input_ANnarrow_neutral_neg_pos.inp`, `input_AN_neutral_neg_pos.inp` and
`input_AD_neutral_neg_pos.inp`. Not a tolerance. This is combinatorial; any
divergence is a different model.

## Why this is its own phase

Perl `:10907-11256` is ~250 lines of stateful, order-dependent removal:
acid/base-strength-guided stripping with `last ACID_MOL_LOOP` control flow,
an outer `while` that flips strategy mid-run, protonation-transfer
accounting, and a commented-out relaxation attempt. **There is no
closed-form specification** — the mapping is defined operationally by the
algorithm. Boundary reactions carry a large fraction of the mass flux in
narrow cluster sets, so an error here produces a plausible wrong answer
rather than an obvious one.

**Port it line-by-line, not from the paper.** Plain NumPy/Python, its own
module, no cleverness, comments citing Perl line numbers.

## 2.1 `combine_labels` — protonation algebra

Perl `:10666-10832`. Sum compositions, then *chemically neutralise* on
recombination: the negative ion becomes its `corresponding neutral
molecule` (or its missing-proton count is zeroed), the positive cluster
loses one proton. If any count goes negative the collision is dropped. A
proton in the product requires a neutral species with a `corresponding
positive ion` to sit on; a missing proton likewise.

Returns `(product_label, valid_coll, valid_evap, valid_in_system)`.
`valid_evap` starts true only for neutral+neutral and neutral+ion — **never
for recombination** (Perl `:10861-10864`).

**Verify:** every allowed/forbidden pair matches the emitted `coef_quad`
sparsity pattern; same-sign ion pairs and charger+charger are rejected.

## 2.2 Stage 1 — did it nucleate out?

Select the rule set by the product's charge, fire if *every* molecule count
meets or exceeds a rule, OR across rules. Result routes to
`out_neu`/`out_neg`/`out_pos` (indices 60/61/62).

**Verify:** the set of out-routed reactions matches
`ind_quad_form(60..62, …)` — 164, 157 and 117 pairs respectively.

## 2.3 Stage 2 — per-species clamp

For each molecule type exceeding `n_max_type[_neg|_pos]`, strip the excess
into a monomer bag. The maxima are **per charge**, accumulated during `.inp`
parsing; getting them wrong changes results silently rather than erroring.

**Verify:** `n_max_type*` equals `ij_ind_max` = `(5,1,5,1)` for the AN set.

## 2.4 Stage 3a — strength-guided stripping

Perl `:10997-11149`. Count acids and bases (proton counts as making an acid,
missing proton as making a base, `l_can_be_lost_bound` gates removability).
Remove the **weakest** neutral species of whichever class is in excess, one
molecule at a time, re-checking membership after each removal and switching
between the acid and base loops whenever the balance flips.

**Verify:** decision-by-decision match against `--print_boundary`.

## 2.5 Stage 3b — fallback and the give-up path

Perl `:11154-11235`. Repeatedly remove one of the **most numerous removable
neutral** species, with protonation-transfer accounting: molecules tied up
holding the proton/missing proton are subtracted from the removable count,
and that non-removability transfers to other chargeable species. If still
stuck, upstream dies with `Can't bring back to boundary` — reproduce that as
an exception, do not silently drop the reaction.

**Verify:** exact match on all three cluster sets; the give-up path is
exercised by a deliberately-too-narrow synthetic set.

## 2.6 Pruning rules

`--disable_boundary` drops boundary collisions that don't lead out
(Perl `:2773-2775`). By default (`disable_useless_collisions=1`,
Perl `:243`) a monomer+cluster boundary collision whose product set equals
its reactant set is dropped as a no-op (Perl `:2777-2787`).

**Verify:** with pruning on, the reaction count matches the emitted file;
with `--all_collisions` semantics, the no-ops reappear.
