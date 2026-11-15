#!/bin/sh
# Open-cell list against the padded periodic box.
# Build first: cargo build --release --example open_cell_bench
# OPEN_REPS (default 8) is the repetition count. The example prints
# the median, the interquartile range, and the core count.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
BIN=${OPEN_CELL_BIN:-$ROOT/target/release/examples/open_cell_bench}
REPS=${OPEN_REPS:-8}
OPEN_REPS=$REPS "$BIN"
