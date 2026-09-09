#!/usr/bin/env bash
# Build the vendored ACDC Fortran reference.
#
# Why this wrapper exists instead of a plain `make -C fortran/src`:
#
# driver_acdc_J.f90 calls the external `formation` with two different
# signatures — 8 arguments at :359 (small_set_mode) and 5 at :365 (loop
# mode). `small_set_mode` is a compile-time `.true.` parameter, so the
# second call is dead code, but gfortran still type-checks both calls to
# the same external in one scope and since gfortran 10 this is an error
# rather than a warning. Verified on GNU Fortran 16.1.0 (Homebrew GCC).
#
# `-fallow-argument-mismatch` downgrades it back to a warning. This is a
# build-time flag, not a source change, which is why `fortran/` stays
# byte-identical to upstream and `fortran/patches/` stays empty.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../fortran/src"

FCFLAGS="-O3 -fcheck=bounds -finit-local-zero -fallow-argument-mismatch"

echo "Building ACDC reference with: $FCFLAGS"
make -C "$SRC" clean >/dev/null 2>&1 || true
make -C "$SRC" FCFLAGS="$FCFLAGS"

echo
echo "Built: $SRC/run"
echo "Smoke test:"
(cd "$SRC" && ./run)
