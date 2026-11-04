# Known issues and follow-up work

The delivery target is a working CPU IPC integration with explicit limits, not a
fully tuned or continuum-certified contact solver. Failed solves return failure
and roll back; none of the cases below is silently accepted as converged.

## 1. Aligned flat faces with friction can stall or cycle

**Open, numerical robustness.** Two identical, aligned free boxes with
`mu=0.3`, `eps_velocity=0.001`, `dhat=0.003`, barrier stiffness 1000 and `dt=0.001`
can exhaust Newton or friction outer iterations. Increasing iteration counts
alone did not resolve the observed fixed-point cycle. Geometry relaxation and
inexact inner solves were explored but did not reliably resolve it, so those
experiments are not included in the baseline implementation.

Reproduce without a long adaptive retry sequence:

```sh
IPC_LOG_DIR=output/logs/ipc_known_flat_friction scripts/run_ipc.sh \
  --case=two-body --steps=1 --stiffness=1000 --mu=.3 \
  --max_retries=0 --verbose
```

The corresponding frictionless run passes. Friction derivative tests, coupled
nonaligned T10 tests, and the brick/sphere friction demos pass. Thus friction is
implemented, but robustness for arbitrary contact configurations is not claimed.
The validation script explicitly uses `mu=0` for the aligned free-box scenario.
It does not hide this configuration as a skipped passing friction test.

Follow-up: inspect stencil/feature changes in the lag update, assess a better
Newton matrix for the signed improved-max terms, and evaluate controlled friction
regularization/continuation. Preserve exact energy/gradient checks and report any
change to the time discretization or friction law. Do not fix this by accepting
unconverged steps.

## 2. Coarse meshes and barrier settings materially affect answers

**Open, accuracy/calibration.** The short brick sensitivity sweep passes numerical
feasibility but changes displacement significantly under mesh/barrier refinement.
It is not mesh-converged. At 0.01 s, changing `dhat` from .003 to .0015 changes the
upper-body vertical displacement by about 1.22 mm. The initial gap is already
inside the activation band, so initial barrier force strongly influences this
short transient. Choose an initially unloaded gap outside the band for impact
studies, or equilibrate the preload deliberately.

The Hertz benchmark shows improved force and radius agreement with refinement,
but is only a two-level check of a finite hemisphere and linear collision proxy.
There is no certified pressure distribution, continuum error bound or production
calibration. Barrier stiffness is fixed per configuration; no automatic stiffness
selection or parameter continuation is implemented.

## 3. Collision and material validity have discretization limits

**Documented approximation.** CCD protects the four-triangle-per-face T10 proxy.
It does not protect the exact curved quadratic surface. Material inversion is
checked along search trajectories at 125 quadrature points, not everywhere in
the curved element. Refine and validate application meshes accordingly.
Containment is checked between closed, consistently oriented body boundaries;
open user-supplied surfaces cannot establish solid containment. Collision filters
intentionally disable checks for excluded pairs. Within-body checks require
`self_collision=true`.

## 4. CPU reference performance is not optimized

**Deferred tuning.** Local 30x30 elastic PSD projection, material path checks,
host assembly/transfers and changing sparse patterns can dominate. The initial
sphere run spent most time in elastic evaluation/projection and material/CCD
feasibility, rather than contact evaluation. The 125-point elastic integration
is deliberately conservative. No scalability or GPU-contact speedup is claimed.
Timing comparisons require isolated runs on the same hardware; some parameter
studies were run concurrently and are not performance benchmarks.

## 5. Dependency patch must be maintained

**Fixed locally, upstream status unverified.** Pinned Toolkit 1.6.1 needs the
edge-order/directional-distance-type canonicalization patch in `third_party/ipc`.
Without it, nearly parallel contact produced different energy and gradient
values at identical positions. A regression and the formerly failing flat
brick run pass with the patch. Verify an equivalent upstream fix before removing
it or upgrading the dependency.

## 6. Explicit scope limits

- Material damping is rejected when nonzero, per the chosen initial scope.
- GPU contact migration and mixed ANCF/T10 coupling are deferred.
- Contacting T10 bodies must share one FE object/solve; separate sequential solves
  do not provide coupled deformable–deformable contact.
- Material family is uniform within the aggregate; parameters and density can
  vary by body. Constitutive models beyond SVK/MR are unsupported.
- Loads are frozen dead loads during each attempt. Follower-load energy and
  tangent terms are not implemented.
- Fixed obstacle vertices are supported. Prescribed moving contact geometry must
  be represented as constrained T10 bodies; there is no moving-obstacle channel.
- Topology, material or constraint-layout mutation requires a new solver.
- General nonlinear constraints use an ALM Gauss–Newton search matrix, not the
  full indefinite constraint Hessian.
- Extremely small adaptive steps can encounter floating-point residual limits;
  failure is reported instead of weakening convergence criteria.
- Legacy mass integration is intentionally corrected; old trajectories can change
  even though legacy elasticity, damping and solver choices are retained.
