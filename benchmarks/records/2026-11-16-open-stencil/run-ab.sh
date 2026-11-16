#!/bin/sh
# Interleaved serial A/B of the open-cell stencil change.
#
#   run-ab.sh OUT_DIR NAME=TREE [NAME=TREE ...]
#
# Each TREE is a checkout whose release examples are already built:
#   cargo build --locked --release --example open_band_bench \
#     --example open_cell_bench --example knn_bench --example pairs_scale
# ROUNDS (default 6) rounds run every variant once, in an order that
# rotates each round. Every process is pinned to CPU ${CPU:-5} with one
# rayon and one OpenMP thread. Raw output goes to OUT_DIR, one file per
# bench, variant, and round.
set -eu
OUT=$1
shift
ROUNDS=${ROUNDS:-6}
CPU=${CPU:-5}
mkdir -p "$OUT"
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
run() {
  taskset -c "$CPU" "$@"
}
round=1
while [ "$round" -le "$ROUNDS" ]; do
  order=$*
  skip=$(( (round - 1) % $# ))
  while [ "$skip" -gt 0 ]; do
    first=${order%% *}
    order="${order#* } $first"
    skip=$((skip - 1))
  done
  for pair in $order; do
    name=${pair%%=*}
    bin=${pair#*=}/target/release/examples
    tag=$name.r$round
    OPEN_BAND_REPS=11 run "$bin/open_band_bench" > "$OUT/open_band.$tag.log"
    OPEN_REPS=11 run "$bin/open_cell_bench" > "$OUT/open_cell.$tag.log"
    for n in 4096 110592; do
      KNN_BENCH_N=$n KNN_BENCH_REPS=11 run "$bin/knn_bench" > "$OUT/knn.n$n.$tag.log"
    done
    for n in 256 1024 4096; do
      for rep in 1 2; do
        PAIRS_N=$n PAIRS_REPS=50 run "$bin/pairs_scale" > "$OUT/pairs.n$n.$tag.$rep.log"
      done
    done
  done
  round=$((round + 1))
done
