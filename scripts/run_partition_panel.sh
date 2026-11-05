#!/bin/sh
# Production panel. The log opens with HEAD, TREE, and COMMAND.
# One-thread native methods run beside one-rank PT-Scotch.
# The eight-thread ladder runs beside eight-rank PT-Scotch.
set -eu
unset LINKCELL_ONLY_METHOD
unset LINKCELL_ASSIGN_SUM
out=${1:-benchmarks/work/panel.txt}
mkdir -p benchmarks/work
{
  echo "HEAD $(git rev-parse HEAD)"
  echo "TREE $(git rev-parse 'HEAD^{tree}')"
  echo "COMMAND scripts/run_partition_panel.sh $out"
  echo "NPROC $(nproc)"
  echo "HOST_BEGIN"
  nproc
  lscpu
  echo "HOST_END"
  echo "gpu_query command nvidia-smi"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "gpu_status unsupported"
  else
    echo "gpu_status unsupported"
    echo "gpu_reason nvidia-smi is absent on this host"
    echo "gpu_followup rerun the curve split on a host with a CUDA device"
  fi
  awk '
    /^#define SCOTCH_VERSION / { v = $3 }
    /^#define SCOTCH_RELEASE / { r = $3 }
    /^#define SCOTCH_PATCHLEVEL / { p = $3 }
    END { print "scotch_header", v "." r "." p }
  ' /usr/include/scotch/scotch.h
  echo "scotch_strategy SCOTCH_STRATQUALITY|SCOTCH_STRATBALANCE"
  set +e
  PTSCOTCH_RANKS=1 RAYON_NUM_THREADS=1 ./target/release/examples/partition_panel 8 12
  ec=$?
  if [ "$ec" -ne 0 ]; then
    echo "EXIT $ec"
    exit "$ec"
  fi
  PTSCOTCH_RANKS=8 RAYON_NUM_THREADS=8 ./target/release/examples/partition_panel 8 12 ladder
  ec=$?
  if [ "$ec" -ne 0 ]; then
    echo "EXIT $ec"
    exit "$ec"
  fi
  strong_log=$(mktemp)
  for threads in 1 2 4 8; do
    chunk=$(RAYON_NUM_THREADS="$threads" ./target/release/examples/partition_panel 8 12 scale strong 1520875)
    ec=$?
    printf '%s\n' "$chunk"
    printf '%s\n' "$chunk" >> "$strong_log"
    if [ "$ec" -ne 0 ]; then
      echo "EXIT $ec"
      exit "$ec"
    fi
    RAYON_NUM_THREADS="$threads" ./target/release/examples/partition_panel 8 12 scale weak 190109
    ec=$?
    if [ "$ec" -ne 0 ]; then
      echo "EXIT $ec"
      exit "$ec"
    fi
  done
  if ! python3 - "$strong_log" << 'PY'
import sys
from collections import defaultdict
text = open(sys.argv[1], encoding="utf-8").read()
rows = defaultdict(dict)
for line in text.splitlines():
    if not line.startswith("scale_row ") or " kind strong " not in line:
        continue
    bits = line.split()
    threads = bits[bits.index("threads") + 1]
    method = bits[bits.index("method") + 1]
    digest = bits[bits.index("assign_hash") + 1]
    rows[method][threads] = digest
want = ["1", "2", "4", "8"]
if not rows:
    sys.exit("assign_hash missing from the strong log")
for method, found in sorted(rows.items()):
    missing = [t for t in want if t not in found]
    if missing:
        sys.exit(f"assign_hash missing {method} threads {missing}")
    base = found["1"]
    for threads in want:
        if found[threads] != base:
            sys.exit(f"assign_hash mismatch {method} threads {threads} {found[threads]} != {base}")
    print(f"assign_hash_equal method {method} threads 1 2 4 8 hash {base}")
PY
  then
    echo "EXIT 1"
    exit 1
  fi
  rm -f "$strong_log"
  echo "EXIT 0"
} > "$out" 2>&1
ec=$?
cat "$out"
exit "$ec"
