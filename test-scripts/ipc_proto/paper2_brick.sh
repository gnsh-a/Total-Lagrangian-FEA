#!/usr/bin/env bash
# TL-FEA Part II brick-on-slope test (tl_fea_num 07_unit_tests.tex, Sec. brick_slope), stock DEME vs IPC:
# slopes 01-04, mu_s 0.25 / mu_k 0.2, dt 5e-4, E 1e7, CoR 0, k_n 1e8, sudden release, 0.3 s.
# These are the stock demo defaults, so the runs are reused when present:
#   DEME: ref_brick_sliding.sh (no MU) -> $D/brick_sliding_s{key}_ctype0_E1e7
#   IPC:  brick_slope.py --mus 0.2 --mu-s 0.25 --tag _sk -> out/brick_slope/ipc_sk_s{key}_mu0.2.json
# Scores go to out/paper2/.
set -u
D=${DEME_OUT:-$HOME/work/fea/ref-runs/2fffca8/output}
S=01,02,03,04
O=test-scripts/ipc_proto/out/paper2
mkdir -p $O
missing=""
for k in ${S//,/ }; do [ -f "$D/brick_sliding_s${k}_ctype0_E1e7/rigid_equiv.csv" ] || missing="$missing $k"; done
[ -n "$missing" ] && SLOPES="$missing" bash test-scripts/ipc_proto/ref_brick_sliding.sh
missing=""
for k in ${S//,/ }; do [ -f "test-scripts/ipc_proto/out/brick_slope/ipc_sk_s${k}_mu0.2.json" ] || missing="$missing,$k"; done
py() { uv run --no-project --python 3.10 --with numpy --with scipy --with matplotlib python "$@"; }
[ -n "$missing" ] && py test-scripts/ipc_proto/brick_slope.py --mus 0.2 --mu-s 0.25 --tag _sk --slopes "${missing#,}"
py test-scripts/ipc_proto/analyze_brick.py --slopes $S --per-slope --json $O/results.json \
  --ipc "ipc=_sk:0.2:0.25" --deme "deme=$D/brick_sliding_s{key}_ctype0_E1e7:0.2:0.25:0" | tee $O/results.txt
