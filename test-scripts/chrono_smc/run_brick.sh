#!/usr/bin/env bash
# The 12 SMC brick runs: slopes 01-04 x (mu 0.2 dt 1e-4, mu 0.2 dt 5e-4, mu 0.25 dt 1e-4), Hertz, 0.3 s.
set -eu
cd "$(dirname "$0")"
mkdir -p out
for cfg in "0.2 1e-4" "0.2 5e-4" "0.25 1e-4"; do
  read -r mu dt <<< "$cfg"
  for s in 01 02 03 04; do
    ./build/brick_slope_smc --slope "$s" --mu "$mu" --dt "$dt" --out "out/s${s}_mu${mu}_dt${dt}.csv"
  done
done | tee out/summary.txt
