#!/usr/bin/env bash
# One DEME brick_sliding variant on all 8 slopes, mu_s = mu_k = MU (default 0.2), with the
# brick_sliding built in this worktree (adds --dt, --gravity_ramp, --eta_damp, --mass_from_fe).
#   test-scripts/ipc_proto/deme_variant.sh <name> [extra brick_sliding args...]
# Env: MU, MU_S (default MU), DT (default 5e-4), SLOPES, RUN_DIR, TIMEOUT; DEME_CONTACT_E etc.
# pass through to the collision system. Output: $RUN_DIR/output/brick_sliding_<name>_s<slope>_*.
set -u
NAME=$1
shift
HERE=$(cd "$(dirname "$0")/../.." && pwd)
BIN="$HERE/bazel-bin/lib_bin/ptest/brick_sliding"
RUN_DIR=${RUN_DIR:-$HOME/work/fea/ref-runs/brick-compare-exp}
MU=${MU:-0.2}
MU_S=${MU_S:-$MU}
DT=${DT:-5e-4}
STEPS=$(python3 -c "print(round(0.3 / $DT))")
SLOPES=${SLOPES:-"01 02 03p3 03p4 03p1 03 03p2 04"}
TIMEOUT=${TIMEOUT:-$(python3 -c "print(int(150 * 5e-4 / $DT))")}
mkdir -p "$RUN_DIR"
ln -sfn "$HERE/data" "$RUN_DIR/data"
cd "$RUN_DIR" || exit 1
echo "variant $NAME: mu_s $MU_S mu_k $MU dt $DT steps $STEPS args [$*] E=${DEME_CONTACT_E:-default}" >> runs.txt
for s in $SLOPES; do
  for try in 1 2 3; do
    timeout "$TIMEOUT" "$BIN" --slope="$s" --constraint_type=0 --export_interval=0 --rigid_equiv_csv \
      --mu_s="$MU_S" --mu_k="$MU" --dt="$DT" --steps="$STEPS" --out_suffix="$NAME" "$@" \
      > "log_${NAME}_s$s.log" 2>&1
    rc=$?
    echo "  $NAME slope $s try $try exit $rc" >> runs.txt
    [ $rc -eq 0 ] && break
  done
done
