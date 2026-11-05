#!/bin/sh
# Serial panel. The log opens with HEAD, TREE, and COMMAND.
set -eu
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
  for threads in 1 2 4 8; do
    RAYON_NUM_THREADS="$threads" ./target/release/examples/partition_panel 8 12 scale strong 1520875
    ec=$?
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
  echo "EXIT 0"
} > "$out" 2>&1
ec=$?
cat "$out"
exit "$ec"
