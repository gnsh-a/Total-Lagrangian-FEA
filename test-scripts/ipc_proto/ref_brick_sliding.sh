#!/usr/bin/env bash
# Reference DEME runs of lib_bin/ptest/brick_sliding (built in the main checkout) for all slopes.
# Runs from a per-commit folder with a data/ link so main stays clean:
#   test-scripts/ipc_proto/ref_brick_sliding.sh [main_dir] [ref_root]
set -u
MAIN=${1:-$HOME/work/fea/main}
ROOT=${2:-$HOME/work/fea/ref-runs}
REV=$(cat "$MAIN/.git/refs/heads/main" 2>/dev/null | cut -c1-7)
DIR="$ROOT/$REV"
BIN="$MAIN/bazel-bin/lib_bin/ptest/brick_sliding"
mkdir -p "$DIR"
ln -sfn "$MAIN/data" "$DIR/data"
cd "$DIR" || exit 1
echo "main $REV, binary $BIN" > brick_sliding_runs.txt
for s in 01 02 03p3 03p4 03p1 03 03p2 04; do
  echo "slope $s" | tee -a brick_sliding_runs.txt
  "$BIN" --slope="$s" --constraint_type=0 --export_interval=0 --rigid_equiv_csv > "brick_sliding_s$s.log" 2>&1
  echo "  exit $? ($(tail -1 "brick_sliding_s$s.log"))" | tee -a brick_sliding_runs.txt
done
