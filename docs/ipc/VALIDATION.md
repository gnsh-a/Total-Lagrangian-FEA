# Validation record

Environment: Bazel 9.3.1, CUDA 12.8, cuDSS 0.8, RTX 3070 Ti; optimized build.
Results are for the CPU contact implementation in this worktree. Known failures
and exclusions are explicit in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Automated coverage

`./scripts/validate_ipc.sh` builds/runs these test targets:

| Target | Coverage |
|---|---|
| `utest_t10_mass_positive` | 3 tests: positive exact quadrature, SPD consistent mass, curved geometry, moving targets, inverted reference rejection |
| `utest_ipc_contact` | 9 tests: raw barrier/friction derivatives, momentum balance, PSD matrix, swept CCD and thickness, filters, weighted mapping, checkpoints, distinct friction coefficients, self/static contact, repeatable parallel contact, initial containment |
| `utest_ipc_solver` | 14 tests: GPU SVK/MR derivatives, interior path inversion, BE/BDF2 translation, state/input/history rollback, adaptive callbacks, coupled momentum/friction, per-body parameters, damping rejection, prescribed/general constraints, changing contact graph and surface topology |
| Legacy targets | `utest_sparse_mass`, `utest_feat10_mesh`, `utest_feat10_von_mises`, `utest_feat10_cudss` |

No test interprets successful nonlinear convergence as a continuum accuracy
certificate. The raw Hessian finite-difference checks are separate from the PSD
search matrices. The aligned free-box friction failure is recorded separately;
it is not the two-body passing baseline.

## Runnable scenarios

The validation script runs 10 steps each for:

- Brick on a deformable, bottom-fixed base: SVK, BE, slope 10 degrees, mu .3.
- Sphere on a deformable, bottom-fixed base: 626 T10 elements, 1198 nodes, SVK,
  BE, mu .3.
- Two free deformable boxes: SVK, BE, mu 0.
- Two free deformable boxes: MR, BDF2, mu 0.

All use barrier stiffness 1000 and dt .001. The log tree is
`output/logs/ipc_validation/`; each case has its command/configuration, run log
and complete step CSV. A separate 20-step frictional brick run starting just
outside the activation band (`gap=.0032`, stiffness 1e5) covered contact onset and
matrix graph changes without retries.

## Parameter sensitivity

`python3 scripts/ipc_convergence.py --duration=.01` ran equal-duration, zero-
friction brick cases. All completed without retries and with positive sampled
Jacobians and active separation. The output is
`output/logs/ipc_convergence/summary.{json,csv}`.

| Case | Vertical displacement at .01 s (mm) | Final vertical contact force (N) |
|---|---:|---:|
| Baseline: dt .001, box resolution 1, dhat .003, stiffness 1000 | 1.108 | 2.433 |
| Half time step | 1.279 | 2.615 |
| Box resolution 2 | 0.510 | 4.975 |
| Half dhat | -0.114 | 2.743 |
| Double stiffness | 1.229 | 2.262 |

These differences are substantial. This is a sensitivity report, not a claim of
convergence. In particular, the initial .001 gap lies inside the activation band.
The script can also run sphere refinement with `--case=sphere`; the full sphere
refinement sweep has not been run for this delivery.

## Native Hertz comparison

`./scripts/run_ipc_hertz.sh --levels=1,2 --steps=4` generates straight-sided T10
quarter hemispheres (R=.15 m, E=1e7 Pa, nu=.3), uses symmetry constraints and a
prescribed top-face approach, and solves against a fixed triangulated plane with
the native IPC solver. There is no Python FE/contact solve in this harness.

It uses dt=1 and density=1e-8 to approximate quasi-static equilibrium with a
positive mass regularizer, rather than introducing material damping. The measured
inertial residual at the final step was about 2.4e-18 N. Four prescribed increments
reach 1 mm indentation, with dhat=2e-5 and stiffness=1000. Every increment completed
without retries; final minimum sampled det(F) was above .937, constraint errors
below 1e-9 m and residuals below 3e-6 N.

| Mesh | Nodes / elements | Total force | Hertz force | Force error | Effective radius error |
|---|---|---:|---:|---:|---:|
| Level 1 | 668 / 312 | 170.88 N | 179.45 N | -4.78% | +4.99% |
| Level 2 | 1012 / 493 | 174.37 N | 179.45 N | -2.83% | +0.27% |

Force is four times the quarter-model resultant. Effective radius is recovered
from the second radial moment of vertical contact force; the Hertz comparator
uses that measured total force. The finite hemisphere, SVK law, proxy geometry
and barrier regularization all contribute error. Two mesh levels are useful
verification evidence, not a full asymptotic convergence or pressure study.
Files are under `output/logs/ipc_hertz/`, including reproducible meshes, commands,
CSV histories, summaries and final VTU files.

## Performance observation

An isolated frictionless low-resolution sphere run before the final diagnostics
changes took 3.53 s for its first step and 1.26 s for its second. First-step phase
costs were approximately 2.37 s elastic evaluation/projection, .085 s contact,
.736 s feasibility/CCD, .193 s linear solve and .038 s constraint assembly.
This identifies likely optimization targets; it is not a stable benchmark or a
comparison against the old contact system. Parameter sweeps and Hertz runs are
accuracy/feasibility evidence, not comparative timing measurements.
