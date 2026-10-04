#!/bin/sh
# Cutoff-list sweep with the POP3 hierarchy.
# Build first: cargo build --release --example pairs_scale
# The probe always records useful time per worker. Computation scaling
# is against the 1-thread row of the same atom count.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN=${PAIRS_SCALE_BIN:-$ROOT/target/release/examples/pairs_scale}
REPS=${PAIRS_REPS:-8}
LOG=$(mktemp)
for n in 256 1024 4096; do
  for t in 1 2 4 8; do
    RAYON_NUM_THREADS=$t PAIRS_N=$n PAIRS_REPS=$REPS "$BIN" | tee -a "$LOG"
  done
done
python3 "$ROOT/scripts/pop-report.py" < "$LOG"
rm -f "$LOG"
