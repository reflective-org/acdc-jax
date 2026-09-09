"""Capture rate-coefficient goldens for generator options the bundled example lacks.

The shipped example was generated with Su82 ion enhancement and no wall or
dilution losses. Every other option -- Su73, constant, later the losses and
hydrates -- needs its own generated fixture. Rather than build an f2py module
per variant, this regenerates each with Perl into a gitignored directory,
evaluates the emitted ``K(i,j)`` expressions across the temperature sweep
with :mod:`reference.emitted`, and commits the resulting matrices as small
``.npz`` goldens. Tests then need neither Perl nor gfortran.

The evaluator is itself checked against the f2py bridge on the shipped
example (``test_reference_bridge.py``); it reproduces it to 0.0 relative.

Usage::

    uv run python validation/capture_variants.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FORTRAN = REPO / "fortran"
GENERATED = HERE / "fixtures/generated"
GOLDENS = HERE / "goldens"

sys.path.insert(0, str(HERE / "reference"))
import emitted  # noqa: E402

GENERATOR = FORTRAN / "perl/acdc_2020_04_28.pl"
INPUT = FORTRAN / "src/Perl_input/input_ANnarrow_neutral_neg_pos.inp"
ENERGY = FORTRAN / "src/Perl_input/HS298.15K_example.txt"
DIPOLE = FORTRAN / "src/Perl_input/dip_pol_298.15K_example.txt"

TEMPERATURES = (250.0, 280.0, 298.15, 320.0)
NCLUST = 54

# Each variant: the extra generator flags that select it, and what to
# capture. The base invocation is run_perl.sh's, so everything else matches
# the shipped example.
VARIANTS: dict[str, dict] = {
    "su73": {"flags": ["--ion_coll_method", "Su73"], "capture": "K"},
    "constant": {"flags": ["--ion_coll_method", "constant"], "capture": "K"},
    # Phase 8.7. Sticking factors touch K AND E (E twice over, F19), and the
    # Delta-G scaling touches E only, so these capture both matrices.
    "stick05": {"flags": ["--sticking_factor", "0.5"], "capture": "KE"},
    "stickion2": {"flags": ["--sticking_factor_ion_neutral", "2"], "capture": "KE"},
    "scaleevap1": {"flags": ["--scale_evap_factor", "1"], "capture": "KE"},
}

# Variants the generator REFUSES to combine with --variable_temp. These are
# generated at one fixed temperature and capture the loss vector, which is
# then plain numbers rather than expressions.
FIXED_T = 280.0
FIXED_T_VARIANTS: dict[str, dict] = {
    # `constant` at FIXED temperature is the only path where upstream applies
    # its documented factor of 10 (F15); captured as literal K values.
    "constant_fixed": {"flags": ["--ion_coll_method", "constant"], "capture": "K"},
    "bgloss": {"flags": ["--cs", "bg_loss"], "capture": "cs"},
    "dilution": {"flags": ["--use_dilution"], "capture": "dil"},
    # Six wall-loss parameterizations. Every one applies to all 54 clusters
    # including the vapour monomers; the emitted loss is the BARE wl(k) --
    # the ion factor fwl is applied in get_rate_coefs, not in get_losses.
    **{
        f"wl_{name}": {"flags": ["--use_wl", "--wl", name], "capture": "wl"}
        for name in (
            "CLOUD4_JA",
            "CLOUD4_JK",
            "CLOUD4_AK",
            "CLOUD4_simple",
            "CLOUD3",
            "ift",
            "diffusion",
        )
    },
}

# Reaction-graph variants: --nst changes WHICH reactions exist, not their
# rates, so the golden is the set of coef_quad triples, the extra-product
# multiplicities and the emitted evaporation pairs.
NST_DIR = HERE / "fixtures/nst"
GRAPH_VARIANTS: dict[str, dict] = {
    "base": {"flags": []},
    "nst_override": {"flags": ["--nst", str(NST_DIR / "override.txt")]},
    "nst_duplicate": {"flags": ["--nst", str(NST_DIR / "duplicate_product.txt")]},
    "nst_clusters": {"flags": ["--nst", str(NST_DIR / "clusters.txt")]},
}

# Hydrate averaging: fixed temperature (upstream refuses --variable_temp),
# the bundled thermodynamics plus four synthetic monohydrate rows in each
# input file, RH 20%. Captures K, E and the sink, all literals.
HYDRATE_DIR = HERE / "fixtures/hydrates"
HYDRATE_FLAGS = [
    "--fortran",
    "--save_outgoing",
    "--temperature",
    f"{FIXED_T:g}",
    "--e",
    str(HYDRATE_DIR / "HS298.15K_with_hydrates.txt"),
    "--dip",
    str(HYDRATE_DIR / "dip_pol_298.15K_with_hydrates.txt"),
    "--variable_ion_source",
    "--cs",
    "exp_loss",
    "--exp_loss_exponent",
    "-1.6",
    "--exp_loss_ref_cluster",
    "1A",
    "--cs_only",
    "1A,0",
    "--cs_only",
    "1N,0",
    "--i",
    str(INPUT),
    "--rh",
    "20",
]

FIXED_T_BASE_FLAGS = [
    "--fortran",
    "--save_outgoing",
    "--temperature",
    f"{FIXED_T:g}",
    "--e",
    str(ENERGY),
    "--dip",
    str(DIPOLE),
    "--variable_ion_source",
    "--cs_only",
    "1A,0",
    "--cs_only",
    "1N,0",
    "--i",
    str(INPUT),
]

BASE_FLAGS = [
    "--fortran",
    "--save_outgoing",
    "--variable_cs",
    "--cs",
    "exp_loss",
    "--exp_loss_exponent",
    "-1.6",
    "--e",
    str(ENERGY),
    "--dip",
    str(DIPOLE),
    "--variable_temp",
    "--variable_ion_source",
    "--cs_only",
    "1A,0",
    "--cs_only",
    "1N,0",
    "--i",
    str(INPUT),
    "--exp_loss_ref_cluster",
    "1A",
]


def generate(name: str, flags: list[str], base: list[str] = BASE_FLAGS) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    cmd = ["perl", str(GENERATOR), *base, *flags, "--append", f"_{name}"]
    result = subprocess.run(cmd, cwd=GENERATED, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"generator failed for variant {name}")
    return GENERATED / f"acdc_equations_{name}.f90"


GENERATOR_2024 = FORTRAN / "perl/acdc_2024_02_12.pl"

# Variants where the 2024 generator is KNOWN to emit something different from
# the 2020 one that produced the shipped example. Anything else differing is
# an error: the goldens are 2020 output and the fidelity table must say so.
EXPECTED_2024_DIFFERENCES = {
    "wl_diffusion": "generic ions get a diffusion wall loss in 2024 (F16)",
}


def cross_check_2024() -> None:
    """Regenerate every variant with the 2024 generator and compare.

    The goldens come from acdc_2020_04_28.pl because that is what produced
    the shipped example the whole port is gated on. This pass shows where
    current upstream differs, so nothing 2020-specific is mistaken for
    physics. Rates are compared at 280 K; the reaction graph exactly.
    """
    if not GENERATOR_2024.exists():
        print("2024 generator not vendored; skipping cross-check")
        return
    print("\ncross-checking against acdc_2024_02_12.pl:")
    variants = [
        (name, spec["flags"], BASE_FLAGS, "rates") for name, spec in VARIANTS.items()
    ] + [
        (name, spec["flags"], FIXED_T_BASE_FLAGS, spec["capture"])
        for name, spec in FIXED_T_VARIANTS.items()
    ]
    for name, flags, base, kind in variants:
        old = GENERATED / f"acdc_equations_{name}.f90"
        cmd = ["perl", str(GENERATOR_2024), *base, *flags, "--append", f"_{name}_2024"]
        result = subprocess.run(cmd, cwd=GENERATED, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"2024 generator failed for variant {name}")
        new = GENERATED / f"acdc_equations_{name}_2024.f90"
        if kind in ("rates", "K"):
            a = emitted.collision_matrix(old, FIXED_T, NCLUST)
            b = emitted.collision_matrix(new, FIXED_T, NCLUST)
            if kind == "rates":
                a_e = emitted.evaporation_matrix(old, FIXED_T, NCLUST)
                b_e = emitted.evaporation_matrix(new, FIXED_T, NCLUST)
                a, b = (
                    np.concatenate([a.ravel(), a_e.ravel()]),
                    np.concatenate([b.ravel(), b_e.ravel()]),
                )
        else:
            a = emitted.loss_vector(old, NCLUST, name=kind)
            b = emitted.loss_vector(new, NCLUST, name=kind)
        same_pattern = np.array_equal(a != 0, b != 0)
        mask = a != 0
        worst = (
            float(np.max(np.abs(a - b)[mask] / np.abs(a)[mask])) if mask.any() else 0.0
        )
        differs = not same_pattern or worst > 1e-12
        note = EXPECTED_2024_DIFFERENCES.get(name)
        status = "identical" if not differs else f"DIFFERS ({note or 'UNEXPECTED'})"
        print(f"  {name:16s} {status}")
        if differs and note is None:
            raise SystemExit(f"unexpected 2020/2024 difference in variant {name}")


def main() -> int:
    if shutil.which("perl") is None:
        print("perl not found on PATH", file=sys.stderr)
        return 1
    GOLDENS.mkdir(exist_ok=True)

    for name, spec in VARIANTS.items():
        equations = generate(name, spec["flags"])
        data = {"temperatures": np.asarray(TEMPERATURES)}
        for t in TEMPERATURES:
            data[f"K_{t:g}"] = emitted.collision_matrix(equations, t, NCLUST)
            if "E" in spec["capture"]:
                data[f"E_{t:g}"] = emitted.evaporation_matrix(equations, t, NCLUST)
        path = GOLDENS / f"rates_variant_{name}.npz"
        np.savez_compressed(path, **data)
        nonzero = np.count_nonzero(data["K_280"])
        if nonzero == 0:
            raise SystemExit(f"variant {name}: get_coll evaluated to all zeros")
        print(f"wrote {path.relative_to(REPO)}  (K nonzero at 280 K: {nonzero})")

    for name, spec in GRAPH_VARIANTS.items():
        equations = generate(name, spec["flags"])
        triples = np.array(sorted(emitted.collision_triples(equations)), dtype=np.int32)
        extras = np.array(sorted(emitted.formation_extras(equations)), dtype=np.int32)
        pairs = np.array(sorted(emitted.evaporation_pairs(equations)), dtype=np.int32)
        path = GOLDENS / f"reactions_variant_{name}.npz"
        np.savez_compressed(path, triples=triples, extras=extras, evaporations=pairs)
        print(
            f"wrote {path.relative_to(REPO)}  "
            f"({len(triples)} quad terms, {len(extras)} extras, {len(pairs)} E)"
        )

    equations = generate("hydr", [], base=HYDRATE_FLAGS)
    path = GOLDENS / "rates_variant_hydr.npz"
    np.savez_compressed(
        path,
        temperature=FIXED_T,
        rh=20.0,
        K=emitted.collision_matrix(equations, FIXED_T, NCLUST),
        E=emitted.evaporation_matrix(equations, FIXED_T, NCLUST),
        cs=emitted.loss_vector(equations, NCLUST, name="cs"),
    )
    print(
        f"wrote {path.relative_to(REPO)}  (hydrate-averaged K, E, cs at {FIXED_T:g} K)"
    )

    for name, spec in FIXED_T_VARIANTS.items():
        equations = generate(name, spec["flags"], base=FIXED_T_BASE_FLAGS)
        if spec["capture"] == "K":
            k = emitted.collision_matrix(equations, FIXED_T, NCLUST)
            if np.count_nonzero(k) == 0:
                raise SystemExit(f"variant {name}: get_coll evaluated to all zeros")
            path = GOLDENS / f"rates_variant_{name}.npz"
            np.savez_compressed(path, temperature=FIXED_T, K=k)
            print(f"wrote {path.relative_to(REPO)}  (K nonzero: {np.count_nonzero(k)})")
            continue
        vector = emitted.loss_vector(equations, NCLUST, name=spec["capture"])
        if np.count_nonzero(vector) == 0:
            raise SystemExit(
                f"variant {name}: {spec['capture']} evaluated to all zeros"
            )
        path = GOLDENS / f"losses_variant_{name}.npz"
        np.savez_compressed(path, temperature=FIXED_T, **{spec["capture"]: vector})
        print(
            f"wrote {path.relative_to(REPO)}  "
            f"({spec['capture']} nonzero: {np.count_nonzero(vector)})"
        )

    cross_check_2024()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
