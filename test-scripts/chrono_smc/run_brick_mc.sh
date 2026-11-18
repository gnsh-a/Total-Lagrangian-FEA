#!/usr/bin/env bash
# The 14 Multicore SMC brick runs (Hertz, 0.3 s): MultiStep for slopes 01-04 x (mu 0.2 dt 1e-4,
# mu 0.2 dt 5e-4, mu 0.25 dt 1e-4), plus OneStep for slopes 01 and 04 at mu 0.2 dt 1e-4.
set -eu
cd "$(dirname "$0")"
mkdir -p out_mc
{
  for cfg in "0.2 1e-4" "0.2 5e-4" "0.25 1e-4"; do
    read -r mu dt <<< "$cfg"
    for s in 01 02 03 04; do
      ./build-mc/brick_slope_mc --slope "$s" --mu "$mu" --dt "$dt" --tdispl multistep \
        --out "out_mc/s${s}_mu${mu}_dt${dt}_multistep.csv"
    done
  done
  for s in 01 04; do
    ./build-mc/brick_slope_mc --slope "$s" --mu 0.2 --dt 1e-4 --tdispl onestep --out "out_mc/s${s}_mu0.2_dt1e-4_onestep.csv"
  done
} | tee out_mc/summary.txt
