"""f2py-compile a generated LOOP-MODE equations file so it can be called.

The small-set bridge (``build.py``) wraps the shipped example with a
hand-written signature. Loop-mode fixtures are generated on demand with
different sizes and options, so this builds a bridge per fixture instead:
the equations file is copied with f2py ``intent`` directives inserted after
each subroutine header (without them every output array comes back as
zeros -- the same trap the small-set bridge fell into), and two tiny stub
modules stand in for the driver pieces the routines `use`:

* ``acdc_simulation_setup.sources_and_constants`` -> zero sources, nothing
  constant, nothing fitted (the port sets these itself);
* ``shared_input.bg_concentration`` -> the bg_loss background number
  concentration.

Usage::

    from loop_bridge import build
    mod = build("loopA20k", bg_concentration=1e9)
    K = mod.get_coll(20)
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
GENERATED = HERE.parent / "fixtures/generated"
OUT = HERE / "loop"

FFLAGS = "-O2 -fallow-argument-mismatch -std=legacy"

# intent directives per routine; f2py reads them from comments in the body.
DIRECTIVES = {
    "get_coll": ["!f2py intent(out) :: K", "!f2py intent(in) :: nrates"],
    "get_evap": ["!f2py intent(out) :: E", "!f2py intent(in) :: nrates"],
    "get_losses": ["!f2py intent(out) :: loss", "!f2py intent(in) :: nrates"],
    "get_masses_and_radii": ["!f2py intent(out) :: m, r"],
    "get_molecule_numbers": ["!f2py intent(out) :: nclust_inc, indices"],
    "get_cluster_numbers": ["!f2py intent(out) :: clust_from_indices"],
    "feval": ["!f2py intent(out) :: f_out", "!f2py intent(inout) :: ipar"],
    "formation": ["!f2py intent(out) :: j_out", "!f2py intent(inout) :: ipar"],
}

STUBS = """\
module acdc_simulation_setup
implicit none
contains
subroutine sources_and_constants(neqn,source,isconst,fitted)
    implicit none
    integer :: neqn, fitted(neqn,0:neqn)
    real(kind(1.d0)) :: source(neqn)
    logical :: isconst(neqn)
    source = 0.d0
    isconst = .false.
    fitted = 0
end subroutine sources_and_constants
end module acdc_simulation_setup

module shared_input
implicit none
real(kind(1.d0)), parameter :: bg_concentration = {bg:.14e}
end module shared_input

! The DeltaG loop variant calls a USER-SUPPLIED cluster_energies(g,p,nmol,ind):
! free energy g (kcal/mol) and reference pressure p (Pa) of the cluster with
! molecule counts ind(1:nmol). A liquid-drop-like synthetic form so the port
! can supply the same function: g = 5 n^(2/3) - 3 n.
subroutine cluster_energies(g,p,nmol,ind)
    implicit none
    integer :: nmol, ind(*), n
    real(kind(1.d0)) :: g, p
    n = sum(ind(1:nmol))
    g = 5.d0*real(n,kind(1.d0))**(2.d0/3.d0) - 3.d0*real(n,kind(1.d0))
    p = 101325.d0
end subroutine cluster_energies
"""


def _f2py_kinds(text: str) -> str:
    """f2py cannot evaluate ``kind(1.d0)`` and silently wraps such arrays as
    single precision (values come back as garbage). Spell the kind out."""
    return re.sub(r"real\s*\(\s*kind\s*\(\s*1\.d0\s*\)\s*\)", "real(8)", text)


def _with_directives(text: str) -> str:
    """Insert the f2py intent comments right after each `subroutine` header
    and its optional `use` lines (directives must precede declarations)."""
    text = _f2py_kinds(text)
    out: list[str] = []
    pending: list[str] | None = None
    for line in text.splitlines():
        if pending is not None and not line.strip().lower().startswith("use "):
            out.extend(f"\t{d}" for d in pending)
            pending = None
        m = re.match(r"^\s*subroutine\s+(\w+)\s*\(", line)
        if m:
            pending = DIRECTIVES.get(m.group(1), [])
        out.append(line)
    return "\n".join(out) + "\n"


def build(
    name: str,
    bg_concentration: float = 1e9,
    force: bool = False,
    fflags: str = FFLAGS,
    suffix: str = "",
    widen_e: bool = True,
) -> ModuleType:
    """Compile ``acdc_equations_<name>.f90`` (+ its system file) into an
    importable module named ``acdc_loop_<name>`` and return it."""
    module = f"acdc_loop_{name}{suffix}"
    OUT.mkdir(exist_ok=True)
    existing = sorted(OUT.glob(f"{module}*.so"))
    equations = GENERATED / f"acdc_equations_{name}.f90"
    system = GENERATED / f"acdc_system_{name}.f90"
    stale = not existing or (existing[0].stat().st_mtime < equations.stat().st_mtime)
    if force or stale:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "stubs.f90").write_text(
                _f2py_kinds(STUBS.format(bg=bg_concentration))
            )
            (work / "system.f90").write_text(_f2py_kinds(system.read_text()))
            source_text = equations.read_text()
            # F23: the multi-component get_masses_and_radii ends its sums
            # with a dangling `+&` continuation before the closing paren,
            # which does not compile. Close the sum properly.
            source_text = re.sub(r"\+&\s*\n\s*&\)", ")", source_text)
            if widen_e:
                # F22: one-component loop mode declares E(nclust,1) but
                # feval reads E(j,i) for every i <= j. Widening the array
                # turns those out-of-bounds reads into the zeros the code
                # evidently intends (get_evap fills column 1 only).
                source_text = source_text.replace("E(nclust,1)", "E(nclust,nclust)")
                source_text = source_text.replace("E(nrates,1)", "E(nrates,nrates)")
            (work / "equations.f90").write_text(_with_directives(source_text))
            cmd = [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                "-m",
                module,
                "stubs.f90",
                "system.f90",
                "equations.f90",
                f"--f90flags={fflags}",
            ]
            result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "f2py build failed for "
                    f"{name}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
                )
            for old in OUT.glob(f"{module}*.so"):
                old.unlink()
            for built in work.glob(f"{module}*.so"):
                shutil.copy2(built, OUT / built.name)
        existing = sorted(OUT.glob(f"{module}*.so"))
    spec = importlib.util.spec_from_file_location(module, existing[0])
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


__all__ = ["build"]
