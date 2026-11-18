#!/usr/bin/env bash
# Videos of the four simple brick cases, IPC prototype vs stock DEME (DEME runs from
# ref_brick_sliding.sh with EXPORT=5, IPC runs from brick_slope.py, sudden release).
set -u
D=${DEME_OUT:-$HOME/work/fea/ref-runs/2fffca8/output}
O=test-scripts/ipc_proto/out/brick_slope
r() {
  uv run --no-project --python 3.10 --with numpy --with scipy --with matplotlib \
    python test-scripts/ipc_proto/animate_case.py "$@"
}
r --title "Case 1: frictionless slide" --tag _mu0 --mu 0.0 \
  --row 01=$D/brick_sliding_mu0_s01_ctype0_E1e7 --row 04=$D/brick_sliding_mu0_s04_ctype0_E1e7 \
  --out $O/case1_frictionless.mp4 &
r --title "Case 2: kinetic slide" --mu 0.2 \
  --row 03=$D/brick_sliding_mu0.2_s03_ctype0_E1e7 --row 04=$D/brick_sliding_mu0.2_s04_ctype0_E1e7 \
  --out $O/case2_kinetic.mp4 &
r --title "Case 3: clear stick" --mu 0.4 \
  --row 01=$D/brick_sliding_mu0.4_s01_ctype0_E1e7 --row 04=$D/brick_sliding_mu0.4_s04_ctype0_E1e7 \
  --out $O/case3_stick.mp4 &
r --title "Case 4: static / kinetic" --tag _sk --mu 0.2 --mu-s 0.25 \
  --row 01=$D/brick_sliding_s01_ctype0_E1e7 --row 04=$D/brick_sliding_s04_ctype0_E1e7 \
  --out $O/case4_static_kinetic.mp4 &
wait
