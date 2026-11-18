#!/usr/bin/env bash
# DEME diagnosis batch 1 (mu_s = mu_k = 0.2 unless noted), each on all 8 slopes.
set -u
V=test-scripts/ipc_proto/deme_variant.sh
$V v0                                              # new binary, demo defaults (sanity vs main)
MU=0 $V mu0                                        # frictionless: does the brick still slow/stop?
$V ramp --gravity_ramp=0.05                        # settle: ramp gravity over 50 ms
$V ramp_nodamp --gravity_ramp=0.05 --eta_damp=0    # + no Kelvin-Voigt damping (IPC has none)
$V ramp_nodamp_mass --gravity_ramp=0.05 --eta_damp=0 --mass_from_fe   # + DEME told the FE mass
DEME_CONTACT_E=1e9 $V E1e9 --gravity_ramp=0.05 --eta_damp=0 --mass_from_fe  # + 10x stiffer contact
