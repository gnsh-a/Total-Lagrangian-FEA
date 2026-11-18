#!/usr/bin/env bash
# IPC brick-on-slope variants (all 8 slopes, mu_k = 0.2).
#   test-scripts/ipc_proto/run_brick_ipc_variants.sh [log_dir]
# Batch 1, sudden gravity: --eps-v 1e-4 (_eps1e-4), --mu-s 0.25 (_sk), both (_sk_eps1e-4).
# Batch 2, gravity ramped over 50 ms from a start at dhat: _ramp, _sk_ramp, _sk_eps1e-4_ramp.
# Batch 3 (below), tilt test: full normal gravity from t = 0, along-slope part ramped over 100 ms.
set -u
LOG=${1:-/tmp}
mkdir -p "$LOG"
run() {
  uv run --no-project --python 3.10 --with numpy --with scipy --with matplotlib \
    python test-scripts/ipc_proto/brick_slope.py --mus 0.2 --workers 6 --tangent-ramp 0.1 "$@"
}
run --tag _tilt > "$LOG/ipc_tilt.log" 2>&1 &
run --mu-s 0.25 --tag _sk_tilt > "$LOG/ipc_sk_tilt.log" 2>&1 &
run --mu-s 0.25 --eps-v 1e-4 --tag _sk_eps1e-4_tilt > "$LOG/ipc_sk_eps1e-4_tilt.log" 2>&1 &
wait
