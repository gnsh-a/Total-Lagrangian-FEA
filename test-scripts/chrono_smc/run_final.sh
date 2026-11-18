#!/usr/bin/env bash
# Final Chrono SMC brick runs, the cases that IPC and DEME also have (single mu, sudden gravity, 0.3 s):
#   mu 0.2 x slopes 01 02 03 03p1 03p2 03p3 03p4 04,  mu 0.4 x 01 04,  mu 0 x 01 04.
# Multicore SMC, MultiStep (contact history), Hertz, box brick, dt 5e-4.
set -eu
cd "$(dirname "$0")"
mkdir -p out_final
run() { ./build-mc/brick_slope_mc --slope "$1" --mu "$2" --dt 5e-4 --tdispl multistep --out "out_final/s$1_mu$2.csv"; }
{
  for s in 01 02 03 03p1 03p2 03p3 03p4 04; do run "$s" 0.2; done
  for s in 01 04; do run "$s" 0.4; done
  for s in 01 04; do run "$s" 0; done
} | tee out_final/summary.txt
