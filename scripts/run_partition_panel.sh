#!/bin/sh
# Serial panel. The log opens with HEAD, TREE, and COMMAND.
set -eu
unset LINKCELL_ONLY_METHOD
unset LINKCELL_ASSIGN_SUM
out=${1:-benchmarks/work/panel.txt}
mkdir -p benchmarks/work
{
  echo "HEAD $(git rev-parse HEAD)"
  echo "TREE $(git rev-parse 'HEAD^{tree}')"
  echo "COMMAND scripts/run_partition_panel.sh $out"
  set +e
  RAYON_NUM_THREADS=1 ./target/release/examples/partition_panel 8 12
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
    n=$((190109 * threads))
    RAYON_NUM_THREADS="$threads" ./target/release/examples/partition_panel 8 12 scale weak "$n"
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
