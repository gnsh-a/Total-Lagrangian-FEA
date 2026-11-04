# CPU IPC contact for T10 solids

This worktree adds a coupled, position-based implicit solver for deformable T10
solids using IPC Toolkit 1.6.1 contact on the CPU. Constitutive evaluation remains
on CUDA, and sparse factorization uses cuDSS. It supports SVK and Mooney–Rivlin,
deformable–deformable contact, optional self-contact, fixed obstacles, friction,
existing fixed/general constraints, backward Euler and variable-step BDF2.
**Material damping must be zero. GPU contact migration is deferred.**

## Build and run

Requirements: Bazel 9.3.1 (pinned in `.bazelversion`), C++17 compiler, CMake,
CUDA and cuDSS as configured by the existing repository. Tested with CUDA 12.8,
cuDSS 0.8 and an RTX 3070 Ti. Use optimized builds for simulations; fastbuild
appends `-O0` to the CUDA compilation and is substantially slower.

```sh
export CUDA_PATH=/usr/local/cuda-12.8
scripts/run_ipc.sh --case=brick --steps=20 --stiffness=1000
scripts/run_ipc.sh --case=sphere --steps=10 --mu=0 --stiffness=1000
scripts/run_ipc.sh --case=two-body --material=mr --integrator=bdf2 --mu=0
scripts/validate_ipc.sh
python3 scripts/ipc_convergence.py --duration=.02
scripts/run_ipc_hertz.sh --levels=1,2 --steps=4
```

`IPC_LOG_DIR` selects a run's log directory; use a separate directory per run.
`run_ipc.sh` saves build/run logs and defaults results to `<log-dir>/results`.
Pass `--out=...` to override the results directory. `--help` lists all options.
The sphere meshes use `--sphere_resolution=low|med|high`; boxes use
`--resolution=N`. `--export_interval=N` writes VTU files. `--verbose` includes
Newton energy, residual, step length, constraint error and friction convergence.

The Toolkit dependency is isolated behind IPC targets. Source archives and their
transitive dependencies are pinned by SHA-256 in `third_party/ipc/sources.json`.
Bazel downloads them; CMake builds offline. Legacy targets do not link IPC.
The lockfile was regenerated with the pinned Bazel version. No DEME submodule is
needed for the new IPC targets. The Hertz runner additionally uses `uv` to install
pinned Gmsh and NumPy versions for mesh generation; all solves use the C++ solver.
The old Python prototype is retained in `test-scripts/ipc_proto` for comparison.

## Integrating an application

1. Aggregate every mutually contacting T10 body into **one** `GPU_FEAT10_Data`,
   with canonical T10 connectivity and disjoint node IDs between separate bodies.
   Initialize it, call `Setup`, install zero-damping material(s), and set loads
   and constraints before constructing the solver. Per-body density and material
   parameters use the existing material-range API. The material family
   (SVK or MR) is uniform across this FE object.
2. Extract each body's boundary using `ExtractSurfaceTriMesh`. Build
   `contact::Surface::FromMeshes(reference_positions, surfaces)`. The extractor
   checks node IDs, repeated nodes, nonmanifold faces, shared midside conformity,
   and overlapping tetrahedra sharing a face.
3. Construct `contact::IpcContact(surface, config)` and transfer ownership to
   `IpcNewtonSolver(data, std::move(contact), params)`. Pass `nullptr` for an
   implicit FE-only solve. The FE data must outlive the solver.
4. Set initial velocity in node-major XYZ order. Optionally install a pure
   `SetStepInputs(time)` callback returning the dead-load force vector and optional
   fixed-node position targets or general scalar constraint targets. The callback
   is evaluated again at each attempted endpoint, including retries.
5. Call `Advance(output_interval)` and inspect `accepted`, or use `Solve()` to
   throw when an output interval cannot finish. `TryStep(h)` makes one attempt
   without adaptive subdivision. `Velocity()`, `time()` and `last_result()` give
   committed state and diagnostics. Positions and legacy stress output are
   synchronized on the existing FE object.

See `lib_bin/ipc/contact_demo.cc` for a complete example. Rebuild the solver after
changing topology, mass, material parameters or constraint layout. Mutating the
FE object behind a live solver is unsupported. The callback must not mutate
external application state: the solver can roll back its own inputs, not external
side effects. General constraints consume scalar targets in finalized row order;
fixed-node constraints consume an Nx3 target matrix.

## Numerical formulation

Let `x` be node-major positions, `M` the consistent mass, and `v=(x-y)/a`.
Each frozen ALM/friction inner solve minimizes

```text
E(x) = (x-xhat)^T M (x-xhat)/(2 a²) + Psi(x) - f^T(x-y)
       + B(x) + a D((x-y)/a) + lambda^T c(x) + rho ||c(x)||²/2.
```

`B` uses IPC's physical barrier, reference-area weighting and improved maximum
approximation. `stiffness` already includes the barrier multiplier; do not multiply
it a second time. `dhat` is the activation distance beyond `dmin`; both are lengths.
`D` is lagged, velocity-based smoothed Coulomb friction; `eps_velocity` is a speed.
`mu_static >= mu_kinetic >= 0` is required. Equal coefficients use ordinary
regularized Coulomb friction; distinct coefficients use Toolkit's smooth transition.
There is no restitution or Kelvin–Voigt parameter in this solver.

For backward Euler, `a=h`, `y=x_n`, `xhat=y+h*v_n`. For BDF2, with
`r=h/h_previous`, `a0=(1+2r)/(1+r)`, `a1=-(1+r)`, `a2=r²/(1+r)`:

```text
a = h/a0
 y = (-a1*x_n-a2*x_(n-1))/a0
 vhat = (-a1*v_n-a2*v_(n-1))/a0
 xhat = y+a*vhat
```

Startup and ratios outside `[0.5,2]` restart backward Euler. BDF2 friction uses
this same physical velocity, including the factor `a` in its incremental potential.

Energy and gradients are unmodified derivatives. The Newton matrix combines
mass, locally PSD-projected elastic/contact Hessians and `rho*Jc^T*Jc` for
constraints. It omits nonlinear constraint curvature (a Gauss–Newton ALM matrix),
while the constraint energy and gradient remain exact. Raw elastic/contact
Hessians are separately available and finite-difference tested. The matrix is an
SPD search metric, not a claim that the total physical Hessian is PSD.

Newton uses an energy Armijo search. Loads, multipliers, penalty and lagged
friction remain frozen during that search. Normal collisions are rebuilt at
**every** trial position. A small energy-relative floating-point allowance avoids
rejecting steps whose predicted decrease is below rounding error; residual
convergence is still required. ALM multipliers/penalty and friction geometry are
updated outside the search. Acceptance checks the force from fully refreshed
friction geometry. This fixed-point iteration can cycle on aligned flat faces;
that limitation is recorded in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Geometry, feasibility and state

A T10 boundary face is represented by four linear triangles using all six face
nodes. The scalar displacement map is
`surface_x = surface_rest + W*(full_x-full_rest)`. Gradient and local Hessian blocks
are mapped with `W^T` and `W^T H W`. Dynamic rows must preserve translations.
Fixed obstacle rows are zero; their vertices live entirely in `surface_rest`.
The backend interface exposes local blocks without leaking Toolkit types into
CUDA translation units. Moving obstacle geometry is not yet an input channel;
use constrained T10 bodies for prescribed contact motion.

Body IDs, optional excluded body pairs, self-collision choice and static/static
filtering apply consistently to barrier, friction and CCD. Initial intersecting
surfaces, violations of `dmin`, and containment between closed oriented body
boundaries are rejected. Arbitrary open or inconsistently oriented user-supplied
surfaces cannot provide solid-containment validation. Initial meshes must
represent valid nonoverlapping solids.

Every Newton direction is capped by Tight Inclusion CCD and a material path
check. The latter finds the first determinant-floor crossing of each cubic
`det(F(alpha))` at all 125 quadrature points, including interior crossings whose
endpoints are positive. Before commitment, the straight physical trajectory
between previous and next states is checked separately. These guarantees apply
to the **triangulated collision proxy and sampled material Jacobians**. They are
not a proof that the curved quadratic boundary or every point in a curved T10
volume is valid. Refine the surface/volume to control that approximation.

The dynamic cuDSS wrapper owns descriptors and buffers together. It unions all FE,
constraint, active normal and lagged friction entries. Changed CSR row/column
arrays invalidate symbolic data before buffer replacement; unchanged patterns
reuse analysis. Numerical factorization is redone each iteration and the solve
residual is checked.

Failed attempts restore positions, velocity, both BDF histories, time, loads,
fixed/general targets, multipliers, penalty and friction state. Adaptive stepping
halves failed steps and grows accepted ones up to the configured step. Exhausting
retries or the minimum step rolls back the **entire requested output interval**.
Failures are explicit; the application never receives an unconverged state as a
successful step.

## FE changes shared with legacy solvers

The old five-point tetrahedral rule has negative weight and is insufficient for
the quadratic mass integrand. Consistent mass now uses an independent positive
125-point Gauss–Duffy rule. It integrates `N_i*N_j*det(J_reference)` exactly for
polynomial quadratic isoparametric T10 geometry. Positive reference Jacobians and
density are checked. Immutable reference coordinates are separate from moving
constraint targets and current positions.

The new IPC elastic path also uses 125 positive points, with GPU energy/gradient/
tangent evaluation and CPU 30x30 local Hessian projection. Legacy elastic
quadrature, solver algorithms and damping behavior remain as before; legacy mass
values do change because the mass integration is corrected. Nonpolynomial elastic
energy on curved elements is numerical quadrature, not an exact integration claim.

## Toolkit patch

`third_party/ipc/edge_edge_canonicalization.patch` applies to Toolkit commit
`19bcab7e1f5a0d2addfcd28513422ff248ea504e`. The collision builder compares edge pairs
without ordering, but its distance types are directional. Reversed mollified
stencils could consequently merge different distances, producing thread-dependent
energy/gradient values. The patch canonicalizes edge IDs **and transposes the
distance type** before hashing/merging. The flat-contact repeatability regression
and brick runs cover this failure. Keep the patch until an upstream version with
an equivalent fix is verified; this is a local fix, not an upstream release claim.

## Validation and performance

See [VALIDATION.md](VALIDATION.md) for measured results and limitations.
`steps.csv` includes residuals, constraint error, sampled minimum Jacobian,
minimum active contact distance, line-search cap, contacts, sparse nonzeros,
symbolic analyses, retries, energy, momentum, kinetic energy and phase timings.
An infinite active distance means no stencil is inside the activation band; it
is not a measurement of the global nearest distance. Contact forces are nodal
forces, not nodal pressures. Consistent T10 nodal mass weights can be negative;
they are used only for integration, not as a diagonal lumped mass matrix.

The CPU implementation is a correctness reference. Expected performance work is
local Hessian projection, host sparse assembly/transfers, material feasibility,
contact broad/narrow phase and factorization reuse. The mapped block interface
and explicit matrix-pattern lifecycle leave a path to GPU contact without changing
the integration objective. Mixed ANCF/T10 coupling, GPU contact, material damping,
adaptive barrier-stiffness selection and curved-boundary collision certification
remain outside this version.
