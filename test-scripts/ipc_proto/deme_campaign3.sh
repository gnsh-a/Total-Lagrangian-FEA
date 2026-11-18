#!/usr/bin/env bash
# DEME batch 3: does DEME converge with a stiffer contact / smaller step? (E = 1e9 is unstable at
# dt = 5e-4: NaN + GPU fault.) dt = 1e-4 runs use the informative slopes only.
set -u
V=test-scripts/ipc_proto/deme_variant.sh
S="01 03p3 03p1 03 04"
SLOPES=$S MU=0 DT=1e-4 DEME_CONTACT_E=1e9 $V mu0_E1e9_dt1e-4                       # frictionless, stiff
SLOPES=$S DT=1e-4 $V tilt_dt1e-4 --tangent_ramp=0.1                                # smaller step
SLOPES=$S DT=1e-4 DEME_CONTACT_E=1e9 $V tilt_E1e9_dt1e-4 --tangent_ramp=0.1        # stiff + small step
$V tilt_nodamp --tangent_ramp=0.1 --eta_damp=0                                     # no Kelvin-Voigt damping
