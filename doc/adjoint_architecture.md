# Adjoint architecture

This document describes the adjoint implementation on the
`adjoint_hack26_time_dependant` development branch. It is both an extension
guide and the change record for the 2026 architecture cleanup. The cleanup is
intended to preserve the discrete operators and solver paths: it does not
silently change a scientific formula when reorganizing the software.

## Scope and support envelope

The implementation covers instantaneous Stokes objectives, reverse-time
temperature and composition advection, dynamic topography, conventional and
self-gravitating geoid objectives, trajectory replay and restart, and reduced
initial-temperature optimization with ROL. The two dedicated nonlinear solver
schemes remain the public entry points.

The time-dependent implementation is a fixed-mesh, same-MPI-layout workflow.
Trajectory and adjoint-restart files are not repartitioning checkpoints. The
current artificial-viscosity transpose implements the documented v1 active-set
linearization; changing an active set is outside that derivative. Unsupported
material-model derivatives and replay configurations fail explicitly during
setup.

## Top-level ownership

`Adjoint::Manager<dim>` is the simulator-facing facade. It parses the three
adjoint parameter groups, initializes the components, dispatches the selected
workflow, and exposes objective values, kernels, and initial-field gradients.
The production components below own the corresponding algorithms:

- `Objective::Manager<dim>` constructs objective plugins and dispatches them by
  capability.
- `InstantaneousStokesWorkflow<dim>` evaluates instantaneous objectives,
  assembles their Stokes right-hand sides, solves the adjoint systems, and
  accumulates physical-property kernels.
- `TimeDependentWorkflow<dim>` is the sole reverse-time production entry and
  coordinates terminal/running seeds, replay, the fixed three-level
  recurrence, checkpoint commits, and cleanup through the facade's scoped
  assembly primitives. The reverse sweep is implemented by this class rather
  than forwarded back to `Adjoint::Manager`.
- `StokesAdjointSolver<dim>` performs a Stokes adjoint solve from an assembled
  full-system right-hand side.
- `TrajectoryManager<dim>` owns the primary trajectory, optional dense replay
  reference, matching reverse-checkpoint lifetime, parsed storage/replay
  configuration, transient replay windows, sparse-anchor history, and replay
  counters. It also records accepted forward states and validates the durable
  trajectory identity stored in reverse checkpoints.
- `TrajectoryStore` owns the accepted-step journal, payload segments,
  manifest, sparse-anchor payloads, and cleanup transactions, while
  `RestartCheckpointManager` owns reverse-recurrence checkpoints. A published
  `latest` pointer is the transaction commit record.
- `InitialTemperatureControl<dim>` owns the reduced control layout, prolongation
  and restriction operators, and the finite-element L2 metric used by ROL.
- `InitialTemperatureOptimization<dim>` owns the optimizer parameters,
  distributed bounds, Lin--More/L-BFGS lifecycle, and independent final-state
  qualification.
- `Diagnostics::Manager<dim>` is the runtime entry point for storage summaries,
  replay status, solver comparisons, and MPI memory checkpoints.

`ScopedSimulatorState<dim>` is used whenever an adjoint operation temporarily
installs a solution, history, right-hand side, constraint set, time level,
pressure normalization, or rebuild flag. Destruction restores the complete
state on normal and exceptional exits. A caller may explicitly `restore()`
early or `release()` only when it intentionally transfers responsibility for
the installed state.

### Source layout

The public headers under `include/aspect/adjoint` and their implementations
under `source/adjoint` mirror the same responsibility-based directory layout:

- `core`: the simulator-facing facade, common state and value types, scoped
  simulator-state restoration, kernel accumulation, failure injection, and
  the instantaneous workflow;
- `objective`: objective capability interfaces, registration, and concrete
  temperature, composition, dynamic-topography, conventional-geoid, and
  self-gravitating-geoid objectives;
- `operators`: reusable discrete forward/transpose operators for Stokes,
  advection, artificial viscosity, material models, and boundary topography;
- `time_dependent`: reverse-time orchestration, trajectory storage and replay,
  and restart transactions;
- `optimization`: the initial-temperature control space and ROL adapters and
  lifecycle; and
- `diagnostics`: the centralized runtime diagnostics interface and the
  deliberately separated high-cost validation implementations.

New code belongs in the directory that owns its abstraction rather than in the
top-level `adjoint` directory. This keeps include paths descriptive and avoids
turning the facade into a dependency hub.

## Objective capability model

All objective types live under `Adjoint::Objective` and implement the small
capabilities they actually need:

- `Interface<dim>` provides evaluation and postprocessor dependencies.
- `InstantaneousInterface<dim>` provides an instantaneous Stokes seed, direct
  material kernels, and the physical-Stokes requirement.
- `TerminalInterface<dim>` provides terminal temperature or composition seeds.
- `RunningStokesInterface<dim>` provides a running objective contribution at
  an accepted time level.

This split prevents a terminal objective from inheriting meaningless default
no-op Stokes methods. Capability dispatch uses `dynamic_cast` at the plugin
boundary; production operator code receives a capability reference and does
not infer behavior from an objective name. The conventional geoid objective is
named `geoid`; `geoid density` was removed rather than retained as an alias.

To add an objective, derive from `Interface<dim>` and the required capability
interfaces, declare and parse its parameters in its own subsection, register
the plugin, and add a forward/transpose or Taylor test for every derivative it
provides. Evaluation should read a `ForwardStateView<dim>`, whose constructor
requires a complete set of non-owning state and history references. A view must
never outlive the trajectory or replay cache from which it was constructed.

## Instantaneous data flow

For an accepted forward state, the instantaneous workflow performs these
operations in order:

1. execute required forward postprocessors;
2. evaluate each objective and retain its named scalar contribution;
3. ask each `InstantaneousInterface` for its full-system Stokes seed;
4. solve the discrete adjoint Stokes system with homogeneous adjoint
   constraints and the same pressure-gauge convention as the forward system;
5. contract the adjoint state with direct and material-model derivatives; and
6. accumulate named fields in `KernelRepository<dim>`.

The transpose order is the reverse of the discrete forward dependency order.
Locally owned contributions are compressed before a global scalar reduction;
ghost values are read only from explicitly updated vectors.

## Reverse-time data flow

The trajectory journal stores every accepted state metadata record `q_n` and
the outgoing accepted step that advances it to `q_(n+1)`. This distinction is
necessary because the simulator chooses the outgoing step only after `q_n` has
been solved. A dense schedule is retained even in sparse mode.

The reverse recurrence keeps exactly three levels for temperature, Stokes
right-hand sides, and every composition field. The entries mean pending
current, previous, and previous-previous levels for `next_reverse_step`; the
fixed-size type makes the BDF history invariant explicit. At each reverse
level the workflow:

1. reconstructs `q_n`, `q_(n-1)`, and, for BDF2, `q_(n-2)`;
2. installs their physical times and accepted step sizes;
3. applies terminal and running objective seeds;
4. applies the transposes of advection stabilization, material properties, and
   Stokes-field dependence in strict reverse order;
5. advances the three-level adjoint history; and
6. writes a checkpoint before any restart-safe incremental cleanup.

Minimal mode archives temperature and composition and deterministically
recovers velocity and pressure. Stokes-cached mode archives those fields as
well. Sparse replay stores fixed-spacing anchor packages and rebuilds one
bounded interval at a time. A restart serializes semantic recurrence state and
archive identity, never a transient replay window.

### Restart transaction

Each rank first writes its payload into a generation-specific temporary
directory. Collective success is required before rank zero publishes shared
metadata, renames the generation directory, and atomically renames
`latest.tmp` to `latest`. Restore follows only `latest`; orphaned temporary
data therefore cannot become a checkpoint. Typed diagnostic injection points
can stop each transaction stage in tests without placing string comparisons in
the production path.

## Dynamic topography and geoid operators

`DynamicTopographyBoundaryOperator<dim>` represents either the surface or CMB
through the strong `DynamicTopographyBoundary` enumeration. It shares the
forward reduced-stress projection, topography/state transpose, pressure-mean
correction, and direct density/viscosity contractions. There is no implicit
surface default.

`SphericalHarmonicCoefficients` owns a valid inclusive degree interval and the
matching cosine/sine storage. Access through `(l,m)` checks both the degree
range and triangular layout; parallel vectors of unrelated lengths are not a
valid state.

The conventional and self-gravitating geoid retain separate user-facing
postprocessors and objectives, while sharing harmonic projection/evaluation,
boundary reduced stress, density moments, and coefficient output conventions.
Their mathematical chain is:

```
state and material fields
  -> density moments and surface/CMB reduced stress
  -> boundary topography coefficients
  -> conventional coefficient scaling or self-gravity spectral solve
  -> spherical-harmonic evaluation/output
```

The transpose walks this chain from coefficient seeds back through harmonic
projection, topography, density moments, and finally the state/material
variables. The self-gravity 4-by-4 operator contains no `Simulator` access: its
matrix solve, transpose solve, residual, and matrix-density contraction are
independently testable. Simulator-dependent collection and output live outside
that operator.

## Initial-temperature optimization lifecycle

`InitialTemperatureOptimization<dim>` owns the canonical state, fixed physical
temperature baseline, reduced control space, ROL configuration, and algorithm
lifecycle: reduced bounds, Lin--More/L-BFGS construction, callback accounting,
and the optional independent final-gradient and Taylor qualification. The
facade supplies stable value/gradient evaluation entry points but does not own
or configure optimizer state.

The canonical pre-forward state contains the solution and two history vectors,
linearization point, right-hand side, constraints, time-step state, pressure
normalization and scaling, solver counters, and matrix rebuild flags. A trial
control installs

```
T_initial = T_fixed + P m,
```

where `P` maps the owned reduced control to the physical temperature DoFs. A
value call restores the canonical state, installs the control, runs the forward
model, and creates the matching trajectory generation. A gradient call at the
same control consumes that live trajectory and maps the physical adjoint seed
back with `P^T` and the reduced L2 Riesz map. ROL Trial/Revert/Accept events are
generation-aware; a cache hit is decided collectively so all MPI ranks execute
the same PDE path.

Detailed milestone evidence is retained in
[`adjoint_o3_reset_audit.md`](adjoint_o3_reset_audit.md),
[`adjoint_o6_optimization.md`](adjoint_o6_optimization.md),
[`adjoint_o7_synthetic_twin.md`](adjoint_o7_synthetic_twin.md), and
[`adjoint_o8_mpi_optimization.md`](adjoint_o8_mpi_optimization.md). The scaled
mantle-convection replay contract is documented in
[`adjoint_td_gadopt_benchmark.md`](adjoint_td_gadopt_benchmark.md).

## Parameter migration

No compatibility aliases are installed. Existing input files must use the new
layout:

| Old entry directly below `Adjoint` | New location |
| --- | --- |
| trajectory storage mode/directory/segment/anchor options | `Adjoint / Time dependent` |
| replay, restart interval, stop-after-checkpoint, and cleanup options | `Adjoint / Time dependent` |
| time-dependent advection preconditioner | `Adjoint / Time dependent` |
| optimize-initial-temperature, bounds, tolerances, trust radius, L-BFGS history | `Adjoint / Optimization` |
| memory, replay, finite-difference, transpose, solver comparison, and fault-injection switches | `Adjoint / Diagnostics` |
| objective functional `geoid density` | objective functional `geoid` |
| objective subsection `Geoid density` | objective subsection `Geoid` |
| `Also output the spherical harmonic coefficients of ...` geoid aliases | `Output ... coefficients` |

`benchmarks/adjoint_time_dependent_temperature/normalize_parameter_layout.py`
performs the mechanical migration for the maintained cases and has an
idempotent `--check` mode.

## Diagnostics and tests

Production operators expose composable forward and transpose actions. Cheap
runtime reporting is routed through `Diagnostics::Manager`; perturbing or
high-cost finite-difference, Taylor, transpose, and cold-rebuild checks are
test concerns. `Diagnostics::AdvectionOperatorChecks` and
`Diagnostics::InitialTemperatureOptimizationChecks` are the explicit
test-driver adapters; the corresponding `run_*` methods are not part of the
production facade or operator API. Fault injection uses typed lifecycle,
checkpoint, append, and cleanup points and is disabled by default.

Focused tests cover spherical-harmonic layout/evaluation, the self-gravity
residual and transpose, initial-control projection and L2 metrics, distributed
ROL vectors, optimization parameter contracts, and ROL lifecycle behavior.
Integration cases retain BDF1/BDF2, composition, active artificial viscosity,
boundary-condition, dense/minimal/sparse replay, restart/failure, geoid, and
optimization coverage. Taylor tests require a second-order remainder with a
minimum observed order of 1.9 unless a documented case-specific tolerance is
stricter.

## 2026 cleanup change record

- introduced the objective capability interfaces and removed default no-op
  objective methods;
- introduced the instantaneous workflow, reusable Stokes adjoint solver,
  centralized diagnostics manager, and exception-safe simulator-state guard;
- moved ROL parameters, bound construction, Lin--More/L-BFGS lifecycle, and
  final accepted-state qualification into
  `InitialTemperatureOptimization<dim>`;
- made forward-state views complete on construction and removed unused
  dimension templates from result, adjoint state, trajectory, and restart
  storage types;
- replaced variable-length reverse histories with an explicit three-level
  representation;
- replaced string comparisons inside storage transactions with typed failure
  injection points;
- removed perturbing advection and entropy-viscosity checks from the public
  production-operator interfaces and compiled their finite-difference,
  constraint, and transpose implementations in dedicated diagnostics source
  files;
- made the harmonic coefficient layout an invariant-owning type and unified
  surface/CMB selection in the dynamic-topography boundary operator;
- stored conventional and self-gravitating geoid observations in the same
  invariant harmonic type and routed all coefficient tables through one
  degree/order writer;
- moved restart implementation details out of the public header;
- moved trajectory serialization, MPI-IO, manifest, and cleanup details behind
  a 180-line public interface (from an approximately 1,000-line header-only
  implementation);
- mirrored the adjoint public and implementation trees into `core`,
  `objective`, `operators`, `time_dependent`, `optimization`, and
  `diagnostics` responsibility directories;
- moved the complete reverse sweep into `TimeDependentWorkflow`, and moved
  trajectory configuration, forward recording, sparse-anchor packaging,
  replay-window ownership, and restart-identity validation into
  `TrajectoryManager`;
- split optimization, high-cost diagnostics, trajectory/replay support, and
  reverse-operator implementations out of `source/adjoint/core/manager.cc`;
- documented the implementation bodies of the mathematical operators and
  workflows, including weak forms, coefficient seeds, active-set derivatives,
  forward/transpose chains, BDF history indices, MPI ownership, replay and
  checkpoint transactions, and reduced L2/Riesz-map formulas;
- isolated unit-test helpers in file-specific namespaces so that the same
  tests compile in both unity and non-unity configurations;
- reorganized parameters and migrated maintained parameter files;
- renamed the conventional objective to `geoid`; and
- added this architecture, extension, migration, and validation record.

Scientific changes suspected during refactoring must be recorded as separate
issues and are not folded into the cleanup. Verification results are appended
to this document only after the corresponding build or test completes.

## Verification record

- Parameter-layout normalizer: idempotent (`0` files require migration).
- Parameter parsing: all 160 maintained adjoint `.prm` files validate after
  building the local material-model test plugin. Eight initially reported the
  missing plugin, then passed after it was built.
- Full Debug and Release unity builds pass after the final source-directory
  split. In non-unity mode, every adjoint translation unit compiles
  independently in both Debug and Release configurations. The repository-wide
  Debug non-unity rebuild was stopped after the adjoint objects completed; no
  adjoint compiler error was present.
- The focused harmonic, conventional/self-gravitating geoid, trajectory,
  restart, initial-temperature control/optimization, and ROL suite passes 629
  assertions in 18 test cases after the directory split and detailed source
  formulation-comment pass.
- Three-transition time-dependent replay smoke: objective
  `0.004837536386553313`; all recovered and replayed state, velocity, and
  gauge-normalized pressure relative differences are zero; reverse sweep and
  checkpoint commit complete successfully.
- Instantaneous 3D smoke cases: conventional combined surface/CMB geoid and
  SG2 mixed self-gravitating geoid both complete their forward Stokes solve,
  physical adjoint solve, dynamic-topography postprocessing, and kernel
  output on the 28,690-DoF fixture.
- The unfiltered repository-wide unit-test invocation was terminated by the
  host with exit status 137 before Catch emitted a test result. This is an
  environment/resource limitation, not recorded as a passing run.
- Parameter documentation was regenerated, including the new `Adjoint`
  parameter page.
- C++ whitespace check: `git diff --check` passes.
