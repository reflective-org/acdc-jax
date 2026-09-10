"""Parsing the cluster-set input file (``.inp``).

Replaces the input-reading half of the Perl generator (``:1013-2168``). The
file has two blocks: a header of per-molecule-type properties, and a body
of cluster definitions with ranges.

Plain Python and NumPy. This runs once per cluster set and involves dict
lookups, string labels and validation that must raise -- it never enters
``jit``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from acdc_jax import labels

# Header rows are identified by a regex on the row LABEL, not by position
# (Perl :1097-1290). Order in the file is free, and unrecognised rows are
# ignored rather than being an error.
_DIRECTIVES: list[tuple[str, re.Pattern[str]]] = [
    # `corresponding neutral molecule` must be tried before `charge`-style
    # short patterns, and `name` last, because several row labels contain
    # the word "molecule" or a substring of another directive.
    ("corr_neutral", re.compile(r"corresponding neutral molecule", re.I)),
    ("corr_negative", re.compile(r"corresponding negative ion", re.I)),
    ("corr_positive", re.compile(r"corresponding positive ion", re.I)),
    ("mass", re.compile(r"mass \[", re.I)),
    ("density", re.compile(r"density \[", re.I)),
    ("psat", re.compile(r"saturation vapor pressure \[", re.I)),
    ("surface_tension", re.compile(r"surface tension \[", re.I)),
    ("acid_strength", re.compile(r"acid", re.I)),
    ("base_strength", re.compile(r"base", re.I)),
    ("charge", re.compile(r"charge", re.I)),
    ("name", re.compile(r"name", re.I)),
]

_OUT_ROW = re.compile(r"^out\s+(neutral|negative|positive)\b", re.I)
_RANGE = re.compile(r"^(\d+)-(\d+)$")

_NONE = {"-", ""}


@dataclasses.dataclass(frozen=True)
class Molecule:
    """One molecule type declared in the header."""

    name: str
    charge: int
    """-1, 0 or +1. Anything else is rejected (Perl :1123)."""
    mass: float
    """g/mol. A NEGATIVE mass marks the 'missing proton' pseudo-species."""
    density: float | None
    """kg/m^3. None for the proton, which has no volume."""
    corr_neutral: str | None
    """For an ion, the neutral molecule it derives from: B -> A."""
    corr_negative: str | None
    """For a neutral, the negative ion it becomes: A -> B."""
    corr_positive: str | None
    """For a neutral, the positive ion it becomes. May name a CLUSTER
    (`N` -> `1N1P`), not a molecule."""
    acid_strength: int
    """-1 = not an acid; 1, 2, 3... = increasingly strong."""
    base_strength: int
    """-1 = not a base; 1, 2, 3... = increasingly strong."""
    psat: float | None = None
    """Saturation vapour pressure, Pa (optional `saturation vapor pressure`
    row, Perl :1088). Used only by loop mode's Kelvin evaporation."""
    surface_tension: float | None = None
    """N/m (optional `surface tension` row, Perl :1100). Loop mode, Kelvin."""

    @property
    def is_proton(self) -> bool:
        return self.charge == 1

    @property
    def is_missing_proton(self) -> bool:
        """Marked by a negative mass in the header (Perl :1216)."""
        return self.mass < 0


@dataclasses.dataclass(frozen=True)
class ClusterSetFile:
    """The parsed contents of a ``.inp`` file.

    ``molecules`` holds every type declared in the header, including any
    that no cluster uses. ``used_molecule_names`` is the subset that
    actually appears -- that, not the declared list, sets the width of the
    composition matrix.
    """

    path: Path
    molecules: tuple[Molecule, ...]
    compositions: tuple[tuple[int, ...], ...]
    """One row per cluster, in declared-molecule order, in file order."""
    out_rules: dict[str, tuple[tuple[int, ...], ...]]
    """Grow-out thresholds per charge state. A product is out when EVERY
    molecule count meets or exceeds a rule; multiple rules are OR'd."""

    @property
    def molecule_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.molecules)

    @property
    def used_molecule_names(self) -> tuple[str, ...]:
        """Molecule types appearing in at least one cluster, in order.

        The generator drops declared-but-unused types. The bundled AN file
        declares five (A, B, N, D, P) but no cluster contains dimethylamine,
        so the generated system has ``n_mol_types = 4``. Keeping all five
        would silently widen every composition vector and every index array.
        """
        used = [
            any(row[i] for row in self.compositions) for i in range(len(self.molecules))
        ]
        return tuple(m.name for m, u in zip(self.molecules, used, strict=True) if u)

    def labels(self) -> tuple[str, ...]:
        """Canonical label per cluster, over the used molecules only."""
        order = self.used_molecule_names
        keep = [self.molecule_names.index(n) for n in order]
        return tuple(
            labels.format_label({order[k]: row[i] for k, i in enumerate(keep)}, order)
            for row in self.compositions
        )


def _cells(line: str) -> list[str]:
    """Split a row into cells.

    The file is nominally tab-separated but is not consistently so -- the
    `density` row in the bundled file uses spaces for its first separator.
    Splitting on any run of whitespace handles both, and row labels are
    matched by regex on the whole line rather than by taking cell 0, so a
    label containing spaces ("corresponding neutral molecule:") is safe.
    """
    return line.split()


def _strip_label(line: str) -> list[str]:
    """Return the value cells of a header row, dropping the label.

    The label ends at the colon; everything after it is data.
    """
    _, _, rest = line.partition(":")
    return _cells(rest)


def _optional(value: str) -> str | None:
    return None if value in _NONE else value


def parse_cluster_set(path: str | Path) -> ClusterSetFile:
    """Parse a ``.inp`` cluster-set file.

    Raises:
        ValueError: on a malformed header, an unknown charge, a row whose
            cell count disagrees with the declared molecule count, or a
            cluster row with more than one range.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    n_types: int | None = None
    header: dict[str, list[str]] = {}
    compositions: list[tuple[int, ...]] = []
    out_rules: dict[str, list[tuple[int, ...]]] = {}

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        if line.lstrip().startswith("#"):
            # The comment row naming the compounds fixes the column count
            # (Perl :1057-1060). Later rows are checked against it.
            if n_types is None:
                cells = _cells(line.lstrip("#"))
                if cells:
                    n_types = len(cells)
            continue

        if m := _OUT_ROW.match(line):
            charge_name = m.group(1).lower()
            values = tuple(int(c) for c in _cells(line)[2:])
            out_rules.setdefault(charge_name, []).append(values)
            continue

        if ":" in line:
            for key, pattern in _DIRECTIVES:
                if key in header or not pattern.search(line.split(":", 1)[0]):
                    continue
                header[key] = _strip_label(line)
                break
            continue

        cells = _cells(line)
        if n_types is not None and len(cells) != n_types:
            continue
        compositions.extend(_expand_ranges(cells, path, raw))

    if "name" not in header:
        raise ValueError(f"{path}: no `name:` row found")

    molecules = _build_molecules(header, path)
    _validate(molecules, compositions, out_rules, path)

    return ClusterSetFile(
        path=path,
        molecules=molecules,
        compositions=tuple(compositions),
        out_rules={k: tuple(v) for k, v in out_rules.items()},
    )


def _expand_ranges(cells: list[str], path: Path, raw: str) -> list[tuple[int, ...]]:
    """Expand a cluster row, which may contain one ``lo-hi`` range.

    Exactly one column may carry a range, and it need not be the first
    (Perl :1451-1457). A row with no range is a single cluster.
    """
    range_col = None
    for i, cell in enumerate(cells):
        if _RANGE.match(cell):
            if range_col is not None:
                raise ValueError(
                    f"{path}: more than one range in cluster row: {raw.strip()!r}"
                )
            range_col = i

    if range_col is None:
        return [tuple(int(c) for c in cells)]

    lo, hi = (int(g) for g in _RANGE.match(cells[range_col]).groups())
    fixed = [0 if i == range_col else int(c) for i, c in enumerate(cells)]
    rows = []
    for n in range(lo, hi + 1):
        row = list(fixed)
        row[range_col] = n
        rows.append(tuple(row))
    return rows


def _float_or_none(cell: str) -> float | None:
    value = _optional(cell)
    return None if value is None else float(value)


def _build_molecules(header: dict[str, list[str]], path: Path) -> tuple[Molecule, ...]:
    names = header["name"]
    n = len(names)

    def column(key: str, default: str = "-") -> list[str]:
        values = header.get(key, [default] * n)
        if len(values) != n:
            raise ValueError(
                f"{path}: `{key}` row has {len(values)} cells, expected {n}"
            )
        return values

    charges = column("charge", "0")
    masses = column("mass", "0")
    densities = column("density")
    corr_neutral = column("corr_neutral")
    corr_negative = column("corr_negative")
    corr_positive = column("corr_positive")
    acid = column("acid_strength", "-1")
    base = column("base_strength", "-1")
    psat = column("psat")
    surface_tension = column("surface_tension")

    molecules = []
    for i, name in enumerate(names):
        charge = int(charges[i])
        if charge not in (-1, 0, 1):
            raise ValueError(
                f"{path}: molecule {name!r} has charge {charge}; "
                "only -1, 0 and +1 are allowed"
            )
        density = _optional(densities[i])
        molecules.append(
            Molecule(
                name=name,
                charge=charge,
                mass=float(masses[i]),
                density=float(density) if density is not None else None,
                corr_neutral=_optional(corr_neutral[i]),
                corr_negative=_optional(corr_negative[i]),
                corr_positive=_optional(corr_positive[i]),
                acid_strength=int(acid[i]),
                base_strength=int(base[i]),
                psat=_float_or_none(psat[i]),
                surface_tension=_float_or_none(surface_tension[i]),
            )
        )
    return tuple(molecules)


def _validate(
    molecules: tuple[Molecule, ...],
    compositions: list[tuple[int, ...]],
    out_rules: dict[str, list[tuple[int, ...]]],
    path: Path,
) -> None:
    n = len(molecules)
    for row in compositions:
        if len(row) != n:
            raise ValueError(
                f"{path}: cluster row has {len(row)} entries, expected {n}"
            )
    for charge_name, rules in out_rules.items():
        for rule in rules:
            if len(rule) != n:
                raise ValueError(
                    f"{path}: `out {charge_name}` has {len(rule)} entries, expected {n}"
                )
    if not compositions:
        raise ValueError(f"{path}: no cluster definitions found")

    seen: set[tuple[int, ...]] = set()
    for row in compositions:
        if row in seen:
            raise ValueError(f"{path}: duplicate cluster definition {row}")
        seen.add(row)
