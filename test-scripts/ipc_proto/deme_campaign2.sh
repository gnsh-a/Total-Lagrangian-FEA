#!/usr/bin/env bash
# DEME batch 2: tilt-test loading (full normal gravity from t = 0, along-slope part ramped over
# 100 ms), plus stiffness / damping / dt diagnostics. Each variant runs all 8 slopes.
# (Batch 1's gravity ramp made DEME miss the slowly approaching brick: 0 contacts, NaN, GPU fault.)
set -u
V=test-scripts/ipc_proto/deme_variant.sh
$V tilt --tangent_ramp=0.1                                    # mu_s = mu_k = 0.2
MU_S=0.25 $V tilt_sk --tangent_ramp=0.1                       # mu_s 0.25, mu_k 0.2 (demo values)
MU=0 DEME_CONTACT_E=1e9 $V mu0_E1e9                           # frictionless, 10x stiffer contact
DEME_CONTACT_E=1e9 $V tilt_E1e9 --tangent_ramp=0.1            # 10x stiffer contact
$V tilt_nodamp --tangent_ramp=0.1 --eta_damp=0                # no Kelvin-Voigt damping (IPC has none)
DT=1e-4 $V tilt_dt1e-4 --tangent_ramp=0.1                     # 5x smaller step
