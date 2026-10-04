#!/bin/sh
# Time one fixed knearest problem at 1, 2, 4, and 8 rayon threads.
# Build first: cargo build --release --example knn_scale
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN=${KNN_SCALE_BIN:-$ROOT/target/release/examples/knn_scale}
N=${KNN_SCALE_N:-262144}
REPS=${KNN_SCALE_REPS:-6}
SHAPE=${KNN_SCALE_SHAPE:-cubic}
for t in 1 2 4 8; do
  RAYON_NUM_THREADS=$t KNN_SCALE_N=$N KNN_SCALE_REPS=$REPS KNN_SCALE_SHAPE=$SHAPE \
    "$BIN"
done
