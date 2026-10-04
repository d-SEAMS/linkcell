#!/bin/sh
# Strong-scaling sweep with POP metrics.
# Build first: cargo build --release --example knn_scale
# LINKCELL_POP records useful time per worker. The hierarchy is
# load balance, communication efficiency, parallel efficiency,
# computation scaling against 1 thread, and global efficiency.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN=${KNN_SCALE_BIN:-$ROOT/target/release/examples/knn_scale}
N=${KNN_SCALE_N:-262144}
REPS=${KNN_SCALE_REPS:-6}
SHAPE=${KNN_SCALE_SHAPE:-cubic}
LOG=$(mktemp)
for t in 1 2 4 8; do
  RAYON_NUM_THREADS=$t LINKCELL_POP=1 KNN_SCALE_N=$N KNN_SCALE_REPS=$REPS \
    KNN_SCALE_SHAPE=$SHAPE "$BIN" | tee -a "$LOG"
done
python3 "$ROOT/scripts/pop-report.py" < "$LOG"
rm -f "$LOG"
