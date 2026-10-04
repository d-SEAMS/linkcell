#!/bin/sh
# Build and time the Kokkos certified walk. OpenMP is the backend this
# script expects. A Cuda Kokkos install uses the same sources.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
PREFIX=${KOKKOS_ROOT:-/tmp/kokkos-install}
BUILD=${LC_KOKKOS_BUILD:-/tmp/linkcell-kokkos}
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/13${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S "$ROOT/src/kokkos" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ROOT="$PREFIX"
cmake --build "$BUILD" -j "$(nproc)"
N=${LC_KOKKOS_N:-262144}
REPS=${LC_KOKKOS_REPS:-5}
for t in 1 2 4 8; do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=true LC_KOKKOS_N=$N LC_KOKKOS_REPS=$REPS \
    "$BUILD/lc_kokkos_bench" --kokkos-num-threads=$t
done
