#!/usr/bin/env bash
# Reference DEME runs of lib_bin/ptest/brick_sliding (built in the main checkout).
# Runs from a per-commit folder with a data/ link so main stays clean:
#   test-scripts/ipc_proto/ref_brick_sliding.sh [main_dir] [ref_root]
# Env: MU=0.2 sets mu_s = mu_k = MU (default: the demo's 0.25/0.2), EXPORT=5 writes a .vtu
# every 5 steps (default 0 = none), SLOPES="01 04" picks slopes (default all 8).
set -u
MAIN=${1:-$HOME/work/fea/main}
ROOT=${2:-$HOME/work/fea/ref-runs}
REV=$(cut -c1-7 "$MAIN/.git/refs/heads/main")
DIR="$ROOT/$REV"
BIN="$MAIN/bazel-bin/lib_bin/ptest/brick_sliding"
SLOPES=${SLOPES:-"01 02 03p3 03p4 03p1 03 03p2 04"}
EXTRA=()
TAG=""
if [ -n "${MU:-}" ]; then
  EXTRA+=(--mu_s="$MU" --mu_k="$MU" --out_suffix="mu$MU")
  TAG="_mu$MU"
fi
mkdir -p "$DIR"
ln -sfn "$MAIN/data" "$DIR/data"
cd "$DIR" || exit 1
echo "main $REV, binary $BIN, mu ${MU:-default}" >> "brick_sliding_runs$TAG.txt"
# A run normally takes ~45 s; DEME occasionally deadlocks (process asleep, no CPU), so each
# run gets a timeout and up to two retries.
for s in $SLOPES; do
  for try in 1 2 3; do
    echo "slope $s (try $try)" | tee -a "brick_sliding_runs$TAG.txt"
    timeout "${TIMEOUT:-150}" "$BIN" --slope="$s" --constraint_type=0 --export_interval="${EXPORT:-0}" \
      --rigid_equiv_csv "${EXTRA[@]}" > "brick_sliding$TAG""_s$s.log" 2>&1
    rc=$?
    echo "  exit $rc ($(tail -1 "brick_sliding$TAG""_s$s.log"))" | tee -a "brick_sliding_runs$TAG.txt"
    [ $rc -eq 0 ] && break
  done
done
