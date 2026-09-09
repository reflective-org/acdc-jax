"""Capture loop-mode goldens: generate with Perl, compile with f2py, store.

Every variant is a loop-mode generator invocation on one of the input files
under ``validation/fixtures/loop/``. The generated Fortran is compiled by
:mod:`reference.loop_bridge` and every routine is evaluated: K, E, the loss
vector, masses and radii, the composition indices, ``feval`` on fixed random
states, and ``formation``. Tests then need neither Perl nor gfortran.

Usage::

    uv run python validation/capture_loop.py
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
LOOP_INPUTS = HERE / "fixtures/loop"

sys.path.insert(0, str(HERE / "reference"))
import loop_bridge  # noqa: E402

GENERATOR = FORTRAN / "perl/acdc_2020_04_28.pl"
T = 280.0
T_VARIABLE = 250.0
"""The --variable_temp variants are EVALUATED here, away from the fixed-T
value, so a test on them actually gates the temperature dependence rather
than re-checking the 280 K numbers."""
BG_CONCENTRATION = 1.0e9
"""Background number concentration, 1/m^3, supplied to the bg_loss stub
module (upstream reads it from `shared_input`)."""

CS_FLAGS = [
    "--cs",
    "exp_loss",
    "--exp_loss_exponent",
    "-1.6",
    "--exp_loss_ref_cluster",
    "1A",
]
WL_FLAGS = ["--use_wl", "--wl", "CLOUD4_JA"]

# name -> (input file, extra flags). All at 280 K unless --variable_temp.
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "loopA20k": ("A20.inp", []),
    "loopA20d": ("A20.inp", ["--loop_coll_coef", "Dahneke"]),
    "loopA20lim": ("A20.inp", ["--rlim_no_evap", "0.5"]),  # nm; cuts at ~6A
    "loopA20cs": ("A20.inp", CS_FLAGS),
    "loopA20bg": ("A20.inp", ["--cs", "bg_loss"]),
    "loopA20wl": ("A20.inp", WL_FLAGS),
    "loopA20wljk": ("A20.inp", ["--use_wl", "--wl", "CLOUD4_JK"]),
    "loopA20dil": ("A20.inp", ["--use_dilution"]),
    "loopA20cswl": ("A20.inp", [*CS_FLAGS, *WL_FLAGS]),
    "loopA20vt": ("A20.inp", ["--variable_temp"]),
    "loopA20dg": ("A20.inp", ["--loop_evap_coef", "DeltaG"]),
    "loopAN66": ("AN66.inp", []),
    "loopAN66d": ("AN66.inp", ["--loop_coll_coef", "Dahneke"]),
    "loopAN66cs": ("AN66.inp", [*CS_FLAGS, *WL_FLAGS]),
    "loopAN66vt": ("AN66.inp", ["--variable_temp"]),
}

N_STATES = 3
STATE_SEED = 2024


def generate(name: str, input_file: str, flags: list[str]) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    cmd = [
        "perl",
        str(GENERATOR),
        "--fortran",
        "--loop",
        "--i",
        str(LOOP_INPUTS / input_file),
        *([] if "--variable_temp" in flags else ["--temperature", f"{T:g}"]),
        *flags,
        "--append",
        f"_{name}",
    ]
    result = subprocess.run(cmd, cwd=GENERATED, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"generator failed for {name}")
    return GENERATED / f"acdc_equations_{name}.f90"


def nclust_of(equations: Path) -> int:
    for line in equations.read_text().splitlines():
        if "integer, parameter :: nclust =" in line:
            return int(line.split("=")[1].split(",")[0])
    raise SystemExit(f"no nclust in {equations}")


def capture(name: str, input_file: str, flags: list[str]) -> None:
    equations = generate(name, input_file, flags)
    n = nclust_of(equations)
    variable_temp = "--variable_temp" in flags
    mod = loop_bridge.build(name, bg_concentration=BG_CONCENTRATION)
    text = equations.read_text()
    n_types = 2 if "indices(nclust,2)" in text else 1
    temperature = T_VARIABLE if variable_temp else T

    data: dict[str, np.ndarray | float] = {"temperature": temperature, "nclust": n}

    if variable_temp:
        k = mod.get_coll(n, temperature)
        e = mod.get_evap(k, temperature)  # nrates from K's shape, via f2py
    else:
        k = mod.get_coll(n)
        e = mod.get_evap(n)
    if np.count_nonzero(k) == 0 or np.count_nonzero(e) == 0:
        raise SystemExit(f"{name}: K or E evaluated to all zeros")
    data["K"] = k
    data["E"] = e
    if "subroutine get_losses" in text:
        data["loss"] = mod.get_losses(n)

    if n_types == 1:
        data["E"] = np.pad(e, ((0, 0), (0, n - e.shape[1])))  # widened, see F22
        mr = np.array([mod.get_masses_and_radii(i) for i in range(1, n + 1)])
        data["indices"] = np.arange(1, n + 1)[:, None]
    else:
        _, indices = mod.get_molecule_numbers()
        data["indices"] = indices
        mr = np.array([mod.get_masses_and_radii(row) for row in indices])
    data["mass_g"], data["radius"] = mr[:, 0], mr[:, 1]

    rng = np.random.default_rng(STATE_SEED)
    states = 10.0 ** rng.uniform(6, 14, size=(N_STATES, n))
    source = 1.0e6 if n_types == 1 else np.array([1.0e6, 2.0e6])
    rhs = []
    js = []
    for c in states:
        ipar = np.zeros(4, dtype=np.int32)
        # Under --variable_temp coef(1) is the temperature and the monomer
        # source is NOT overridden (it stays at the stub's zero); otherwise
        # coef is the monomer source(s).
        coef = np.array([temperature]) if variable_temp else source
        rhs.append(mod.feval(0.0, c, coef, ipar))
        ipar = np.zeros(4, dtype=np.int32)
        js.append(
            mod.formation(c=c, coef=coef, ipar=ipar)
            if variable_temp
            else mod.formation(c=c, ipar=ipar)
        )
    data["states"] = states
    data["source"] = np.atleast_1d(source)
    data["feval"] = np.array(rhs)
    data["formation"] = np.array(js)

    path = GOLDENS / f"loop_variant_{name}.npz"
    np.savez_compressed(path, **data)
    extras = " loss" if "loss" in data else ""
    print(
        f"wrote {path.relative_to(REPO)}  "
        f"(n={n}, K nz {np.count_nonzero(k)}, E nz {np.count_nonzero(e)}{extras})"
    )


def main() -> int:
    if shutil.which("perl") is None:
        print("perl not found", file=sys.stderr)
        return 1
    GOLDENS.mkdir(exist_ok=True)
    for name, (input_file, flags) in VARIANTS.items():
        capture(name, input_file, flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
