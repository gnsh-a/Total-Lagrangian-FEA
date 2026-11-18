#!/usr/bin/env bash
# Multicore SMC, box brick, MultiStep (Hertz, 0.3 s), 12 runs:
#   stick check at dt 5e-4: slopes 01-04 x mu 0.25
#   dt sweep: slopes 01, 04 x mu 0.2 x dt 1e-4, 2e-4, 5e-4, 1e-3
set -eu
cd "$(dirname "$0")"
mkdir -p out_mc_dt
{
  for s in 01 02 03 04; do
    ./build-mc/brick_slope_mc --slope "$s" --mu 0.25 --dt 5e-4 --out "out_mc_dt/s${s}_mu0.25_dt5e-4.csv"
  done
  for s in 01 04; do
    for dt in 1e-4 2e-4 5e-4 1e-3; do
      ./build-mc/brick_slope_mc --slope "$s" --mu 0.2 --dt "$dt" --out "out_mc_dt/s${s}_mu0.2_dt${dt}.csv"
    done
  done
} | tee out_mc_dt/summary.txt
