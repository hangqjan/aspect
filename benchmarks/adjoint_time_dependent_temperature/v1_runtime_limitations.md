# Time-dependent temperature adjoint v1 runtime limitations

This note tracks which physical and numerical features are supported by the
current v1 implementation, which ones are explicitly rejected at runtime, and
which ones still need stronger runtime guards.

The goal of v1 is intentionally narrow: run a full forward transient, seed a
present-day temperature-field misfit, and compute gradients with respect to the
initial temperature degrees of freedom and, on the restricted path described
below, initial chemical-composition degrees of freedom for simple fixed-mesh
models.

## Explicitly rejected today

These checks are currently implemented in
`source/adjoint/core/manager.cc::validate_time_dependent_temperature_setup()`.

| Feature | Current status | Why it is unsafe for v1 | Current handling |
| --- | --- | --- | --- |
| Nonlinear solver schemes other than `single Advection, single Stokes, adjoint temperature` | Rejected | The reverse sweep assumes exactly one accepted temperature solve and one accepted Stokes solve per timestep, followed by a terminal reverse sweep. Iterated Picard/Newton schemes introduce multiple linearization states and additional residual derivatives. | `AssertThrow` |
| Non-chemical, particle, VOF, entropy, or discontinuous compositional fields | Rejected | These fields use update rules, transfers, or face fluxes that are not represented by the current transpose operators. | `AssertThrow` |
| Operator splitting | Rejected | Splitting reactions modify solution/history outside the main advection solve. Those updates require their own transpose operators. | `AssertThrow` |
| SUPG stabilization | Rejected | SUPG contributes matrix terms, history terms, velocity-dependent residual terms, and stabilization-parameter derivatives. The current v1 only implements the Galerkin velocity transpose path. | `AssertThrow`; use entropy viscosity |
| Mesh deformation | Rejected | The reverse sweep would need to restore mesh geometry, mesh velocity, quadrature geometry, and differentiate mesh-motion effects. | `AssertThrow` |
| Discontinuous temperature FE | Rejected | DG temperature needs transpose face fluxes, boundary fluxes, and limiter handling. The current implementation only assembles continuous cell terms. | `AssertThrow` |
| Box-geometry inflow-only fixed composition | Rejected when paired with an explicit normal prescribed-velocity component | The active Dirichlet set changes with flow direction, and its advection-boundary transpose is outside v1. Snapshot replay preserves the resulting constraint graph but does not supply the missing flux term. | The Box normal-velocity component-mask guard |
| Box-geometry prescribed normal velocity component | Rejected | A nonzero normal component makes the physical domain permeable and activates advection transpose boundary terms absent from v1. | `AssertThrow` from the prescribed-velocity component mask |

## Boundary conditions v1: impermeable-domain scope

Boundary Conditions v1 assumes a closed / impermeable domain: all physical
boundaries must satisfy `u.n = 0`. On prescribed-velocity boundaries, the
prescribed field must therefore be tangential. The adjoint snapshot stores the
forward `AffineConstraints` graph and homogenizes only its inhomogeneities;
this preserves hanging-node, periodic, and curved free-slip relations without
injecting forward Dirichlet values into the adjoint.

Supported:

- free slip;
- no slip / zero velocity;
- prescribed tangential velocity, including time-dependent tangential plate velocity;
- fixed temperature, fixed state-independent heat flux, and insulating temperature;
- fixed composition and zero-flux composition.

Unsupported:

- prescribed normal inflow/outflow and open/permeable boundaries;
- inflow-only composition boundary conditions;
- advection transpose boundary terms for `u.n != 0`.

For `Geometry model = box`,
`validate_time_dependent_temperature_setup()` rejects prescribed velocity masks
that include the known Cartesian normal component of the corresponding
boundary; this also rejects the inflow-only composition configuration whenever
it is paired with such an explicit flow. On curved geometries a component mask
cannot distinguish a tangential vector field represented in all Cartesian
components from a normal-flow field, so v1 deliberately does not use fragile
pointwise sampling: users must configure the prescribed field to satisfy
`u.n = 0`.

`adjoint_bc_inflow_composition.prm` is retained only as an expected-failure
diagnostic. Before this scope guard was introduced, it replayed the active
composition constraint graph with homogeneous adjoint constrained DoFs, yet its
Taylor test failed because `u.n != 0` activates missing advective transpose
boundary terms. It is not evidence of an `AffineConstraints` homogenization
failure and is excluded from the passing benchmark set.

## Implemented but still v1-limited

These parts exist and are exercised by the smoke benchmark and Taylor test, but
they are not yet general-purpose.

| Feature | Current status | Remaining limitation |
| --- | --- | --- |
| Entropy viscosity / artificial viscosity | Supported only on a restricted active-set path | The algebraic matrix transpose includes frozen artificial diffusion. In addition, the current branch assembles `bar_nu_art` for active `max(k,nu_art)` cells and propagates it through local `alpha=1`/`alpha=2` estimator branches. The tested `alpha=2` path includes the global entropy-variation denominator and extrapolated-range `average_field` transpose, and the `gamma*h*|eps(u)|` velocity-measure transpose. It still does not support smoothing, SUPG, mixed per-quadrature `max(k,nu_art)` branches, or the strict `alpha=1` entropy denominator dependencies on global field variation/global maximum velocity. |
| Transposed temperature solve `A_n^T z_n = bar_T_n` | Implemented and preconditioned | GMRES uses a freshly rebuilt `ILU(A^T)` for each restored snapshot. Diagnostic comparison modes retain unpreconditioned and experimental `ILU(A)` paths; see `adjoint_v1_milestones.md`. |
| Transposed composition solve `A_C,n^T z_C,n = bar_C,n` | Implemented and preconditioned for continuous FEM chemical fields | GMRES uses a freshly rebuilt `ILU(A^T)` and the composition solver tolerance. Composition history and velocity transpose use the composition equation coefficient `1`, not the temperature `rho c_p` coefficient. |
| BE/BDF2 history transpose | Basic Galerkin history terms implemented | Covers the standard `rho cp` mass/history contribution. More complex temperature-equation terms are not included. |
| Velocity transpose load | Galerkin velocity transpose plus restricted artificial-viscosity estimator velocity transpose implemented | Includes the entropy-viscosity `gamma*h*|eps(u)|` strain-rate contribution for the supported active-set path. Does not include SUPG velocity derivatives, melt/Darcy velocity derivatives, or explicit global-max-velocity derivative terms. |
| Stokes-to-temperature coupling `-S_T^T psi` | Implemented for buoyancy density and pointwise viscosity | The Stokes layer seeds material-output adjoints for density and viscosity and delegates constitutive derivatives to `MaterialModelAdjoint`. The validated viscosity path is `eta(T,C)` only; it does not include pressure or strain-rate viscosity, conductivity, heating, phase changes, or other material outputs. |
| Stokes-to-composition coupling `-S_C^T psi` | Implemented for buoyancy density and pointwise viscosity | The Stokes layer seeds material-output adjoints for density and viscosity and scatters resulting composition input adjoints back to the Stokes material state/history. It does not include compressible terms, melt terms, pressure viscosity, strain-rate viscosity, or non-density/non-viscosity Stokes material derivatives. |
| Temperature-equation composition coefficient transpose | Implemented for `rho c_p` dependence | The direct temperature equation and the artificial-viscosity estimator emit `bar(rho*cp)`, split it into `bar_rho` and `bar_cp`, and route those seeds through `MaterialModelAdjoint`. This validates `rho(T,C)` and `cp(T)` in the supported material-prefactor paths, but it is not a latent-heat or general material-derivative implementation. |
| Full transient finite-difference/Taylor validation | Implemented for the minimal benchmark | Validates one simple 2D direction in the v1 benchmark. It is not a replacement for operator-level tests or broader benchmark coverage. |
| Advection-dominated artificial-viscosity validation | Implemented for small temperature benchmarks | `adjoint_time_dependent_temperature_advection_dominated.prm` activates the artificial branch and passes the current one-direction Taylor test. It also runs isolated diagnostics for `R_T -> nu_art`, the alpha=2 local residual/entropy numerator, the scalar entropy denominator, the frozen-average `global_entropy_variation(T)` transpose, and the complete estimator transpose. The full estimator diagnostic is sensitive to finite-difference step size near active-set boundaries and is run with epsilon `1e-11`. |
| Composition-control validation | Partial | No-artificial-viscosity coupled composition checks pass for one-step and three-step runs. Active-AV direct terminal-composition objective checks now pass for three steps, validating the `C_final -> C0` composition advection/history/AV path. The present-day-temperature objective with `C0` control now also passes the active-AV step-2 and step-3 checks after making the temperature EV residual reconstruction forward-consistent. A custom/incompressible/real-density benchmark now stress-tests `C -> rho*cp -> temperature EV -> T_final`. |
| Terminal composition-field objective | Implemented for one continuous FEM chemical field | The no-AV and active-AV variants seed `bar_C_N` directly and validate the pure composition advection/history reverse path. They do not validate `C -> Stokes -> T` coupling. |
| `C -> Stokes RHS` transpose diagnostic | Implemented | The diagnostic finite-differences the assembled ASPECT Stokes RHS with respect to a composition perturbation and compares it with the local transpose action. It currently runs for the first compositional field and uses a deterministic algebraic perturbation/test vector. |
| Adjoint treatment of constrained advection DoFs | Implemented for the v1 continuous FE cases | The reverse solve uses a copy of the forward constraint graph with only inhomogeneities set to zero, then calls `distribute()` on that homogeneous graph. A runtime structural check verifies identical line indices and affine entries, including hanging-node, periodic, and free-slip relations; it does not merely zero constrained vector entries. |
| Transposed-advection constrained-DoF regression | Implemented as a benchmark diagnostic | The diagnostic reruns the actual transposed advection solve and asserts that constrained adjoint DoFs are homogeneous. The current Dirichlet-temperature benchmarks report 18 constrained temperature DoFs with `max |z| = 0`. |
| MaterialModelAdjoint constitutive VJP | Implemented for the v1 material inputs and seeded outputs | The generic central finite-difference VJP supports seeded density, specific heat, viscosity, and thermal conductivity outputs with respect to temperature, pressure, and composition inputs. The validated v1 physical use is `rho(T,C)`, `cp(T)`, and `eta(T,C)`. Nonzero pressure input adjoints are still rejected at the FE scatter boundary instead of silently dropped. |
| Temperature-equation `rho cp` material-prefactor transpose | Implemented for `rho(T,C)` and `cp(T)` | The direct temperature equation and the active entropy-viscosity `rho cp` paths now seed `bar_rho` and `bar_cp`, call `MaterialModelAdjoint`, and scatter the resulting temperature/composition input adjoints through the same linearization/history state used by the forward material evaluation. |
| Stokes viscosity coefficient transpose | Implemented for `eta(T,C)` | The Stokes transpose seeds `bar_eta = -2 eps(u):eps(psi) JxW`, routes it through `MaterialModelAdjoint`, and scatters to the Stokes material state/history used by the forward Stokes matrix assembly. A direct viscosity perturbation diagnostic bypasses `MaterialModelAdjoint` and verifies the physical coefficient derivative. Pressure- and strain-rate-dependent viscosity remain explicitly unsupported. |

## Missing runtime guards

These cases should be rejected explicitly before v1 is used more broadly. Some
of them may already fail indirectly, but they should not be allowed to produce a
plausible-looking incomplete gradient.

| Feature | How it can corrupt the adjoint | Suggested v1 handling |
| --- | --- | --- |
| AMR or mesh refinement after timesteps | The DoF space, constraints, and transfer operators change in time. Reverse mode needs the transpose of the mesh-transfer and constraint updates. | Reject adaptive refinement and require `Time steps between mesh refinement = 0`. |
| Material averaging | The forward operator uses averaged material properties, while the current adjoint uses pointwise material derivatives. The averaging operator itself is not transposed. | Reject non-none material averaging for active material derivatives, including `eta(T,C)`. |
| Compressible, ALA, TALA, or projected-density formulations | These add pressure, density, adiabatic, and mass-conservation terms that are not included in the reverse operators. | Only allow Boussinesq/incompressible v1 cases. |
| Compressible material models | A material model may be compressible even if other settings look simple. The current Stokes and temperature transpose paths assume incompressible behavior. | Require `material_model.is_compressible() == false`. |
| Melt transport | Adds fluid velocity, fluid pressure, porosity, melting reaction, and extra coupling terms. | Reject `include_melt_transport`. |
| Heating models | Shear heating, adiabatic heating, latent heat, and radiogenic heating may depend on temperature, pressure, velocity, or composition. Their derivatives are not included. | Only allow no heating or explicitly documented frozen/no-derivative heating. |
| Latent heat and phase transitions | Phase functions add temperature/pressure-dependent LHS/RHS terms to the heat equation. | Reject latent-heat and phase-transition configurations. |
| Temperature-dependent viscosity outside `eta(T,C)` | Pressure, strain-rate, averaging, or nonlinear-Stokes variants need derivatives not represented in the v1 Stokes material path. | Allow only the validated `eta(T,C)` path with no material averaging; reject `eta(p)` and `eta(strain_rate)`. |
| Composition-dependent viscosity outside `eta(T,C)` | More general composition viscosity settings may interact with averaging, phase logic, or unsupported material outputs. | Allow only the validated pointwise `eta(T,C)` material-output path. |
| Temperature-dependent conductivity | The temperature operator depends on temperature through `k(T)`, requiring additional transpose terms. | Reject variable conductivity models. |
| Composition-dependent conductivity or specific heat | The temperature equation depends on composition outside the current validated `rho(T,C)` and `cp(T)` paths. | Reject variable conductivity and composition-dependent `cp` until dedicated tests exist. |
| Variable density or `cp` in the temperature equation outside the validated form | Additional formulations may use density/specific heat in terms not covered by the direct and entropy-viscosity material-prefactor transposes. | Allow the validated `rho(T,C)` and `cp(T)` material-prefactor paths; keep unsupported pressure and broader formulation cases guarded. |
| Prescribed temperature fields or time-dependent constraints | The control space for initial temperature DoFs can diverge from the constrained algebraic space. Boundary/control semantics become ambiguous. | Document clearly and add tests before allowing. |
| MPI multi-rank Taylor validation | The solver can write per-rank gradient files, but the current Taylor script reads only rank 0. | Extend the script to merge all `initial_temperature_gradient_rank_*.txt` files before using MPI tests. |

## Artificial-viscosity boundary of the current implementation

The current artificial-viscosity adjoint is an active-set implementation for a
small subset of ASPECT's entropy-viscosity algorithm.

Implemented:

- `max(k, nu_art)` active artificial-diffusion branch seed:
  `bar_nu_art += -dt * integral grad(z) . grad(T)`.
- Reuse of the exact cellwise artificial viscosity and physical diffusion
  bounds from the advection assembly cache.
- Startup / fallback branch in which the estimator returns the maximum
  viscosity.
- `alpha = 1` and `alpha = 2` entropy-viscosity local branches when the limiter
  selects the entropy viscosity.
- Cellwise active quadrature point for `max_velocity`.
- Local entropy-viscosity `gamma*h*|eps(u)|` strain-rate contribution through
  the symmetric-gradient transpose of `eps(u)`.
- Cellwise active quadrature point for the weighted residual maximum.
- Residual transpose to historical temperature value, gradient, and laplacian
  at `T^{n-1}` and, for BDF2 steps, `T^{n-2}`.
- Residual transpose to historical velocity values.
- An advection-dominated benchmark with nonzero artificial-viscosity seed norm.
- Continuous FEM chemical-composition advection on the same no-SUPG,
  no-smoothing active-set path.

Still outside the boundary:

- `SUPG`.
- `Use artificial viscosity smoothing = true`.
- Mixed quadrature branches inside one cell where `nu_art > k` at some
  quadrature points but not all.
- Derivatives of physical conductivity, heating, latent heat, material
  averaging, and material-model additional outputs. The validated material
  prefactor paths include `rho(T,C)` and `cp(T)`.
- Explicit transpose of the `alpha = 1` entropy branch's global field variation
  and global maximum velocity denominator. The alpha=2 global entropy variation
  and extrapolated `average_field` path is implemented for the tested active
  set.
- Boundary disabling derivative for conduction-dominated Dirichlet temperature
  boundary cells. The current behavior treats that boolean active set as fixed.
- Compressible formulations remain rejected. `eps(u)` is the symmetric velocity
  gradient, not zero in compressible flow; only its trace equals `div(u)`.
- Melt/Darcy velocity and fluid velocity branches.
- Viscosity dependencies beyond pointwise `eta(T,C)`, latent heat, and general
  heating/material derivative paths.

## Validation status

The current branch has:

- A smoke benchmark at
  `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature.prm`.
- A finite-difference/Taylor script at
  `benchmarks/adjoint_time_dependent_temperature/run_taylor_test.py`.
- A successful one-direction Taylor check for the minimal 2D v1 benchmark, with
  finite-difference and adjoint directional derivatives agreeing to near machine
  precision.
- A successful multi-step Taylor check with `--end-step 3`, which exercises the
  BDF2 history transpose and velocity extrapolation transpose branches. In the
  checked run, the finest relative directional-derivative error was about
  `1.7e-11`.
- A successful advection-dominated `gamma = 1` Taylor check, which exercises
  the `gamma*h*|eps(u)|` artificial-viscosity estimator transpose. In the
  checked run, the finest relative directional-derivative error was about
  `4.4e-10`. Its standalone estimator diagnostic agrees to about `3.3e-5`
  relative error with diagnostic epsilon `1e-11`; the velocity/strain-rate
  component agrees to machine precision.
- A successful advection-dominated `alpha = 1` Taylor check, which exercises
  the currently active maximum-viscosity/startup-style estimator path in the
  benchmark. In the checked run, the finest relative directional-derivative
  error was about `1.9e-13`, and the standalone estimator diagnostic agrees to
  about `1.7e-9` with diagnostic epsilon `1e-11`.
- A broader Taylor sweep over all four current benchmarks, using end steps
  `1`, `3`, and `5` and perturbation amplitudes
  `1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4`. All 12 combinations passed and the
  Taylor remainders showed second-order convergence.
- A composition-control smoke benchmark at
  `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature_composition.prm`
  that writes `initial_composition_gradient_llsvp_rank_00000.txt`.
- A one-step no-artificial-viscosity composition check using
  `adjoint_time_dependent_temperature_composition_no_av.prm`. The tested
  directional derivative is near zero, so this is only a sanity check for the
  plumbing and output path.
- A stronger coupled no-artificial-viscosity composition benchmark at
  `adjoint_time_dependent_temperature_composition_coupled_no_av.prm`. This is
  intentionally kept as a validation target for the chain
  `C0 -> buoyancy density -> Stokes velocity -> final temperature objective`.
  This now passes. The one-step check gives finite-difference slope about
  `1.288e-7` and adjoint prediction about `1.276e-7`; the three-step check
  gives finite-difference slope about `7.999e-6` and adjoint prediction about
  `8.001e-6`, with approximately second-order Taylor remainder.
- The same coupled benchmark now runs a Phase-1
  `C -> assembled Stokes RHS` transpose diagnostic. It passes with
  `FD = -5.888458924416263`, `adjoint = -5.888458923258333`, and relative
  error about `2e-10`. This rules out the local density/buoyancy RHS transpose,
  sign, and composition component mapping as the primary source of the
  previously observed full-chain failure.
- The same diagnostic also checks `C -> Stokes solve` and
  the one-step `temperature-advection velocity transpose`. In the three-step
  coupled run, the Stokes-solve transpose relative error is about `2.7e-7`.
  The velocity diagnostic is skipped for multi-step runs because the current
  check probes the local `u_star` layer and does not compare the BDF2 history
  split. In the one-step run, the temperature-advection velocity transpose
  relative error is about `7.2e-8`.
- A terminal composition-field objective benchmark at
  `adjoint_time_dependent_composition_objective_no_av.prm`. This isolates the
  direct `C_final` objective and the reverse composition advection/history
  chain. The three-step Taylor check passes with finite-difference and adjoint
  directional derivatives agreeing to about `2e-13` relative error and
  second-order Taylor remainder.
- An active-artificial-viscosity terminal composition-field objective benchmark
  at `adjoint_time_dependent_composition_objective_av.prm`. This isolates
  `C_final -> C0` while keeping the composition entropy-viscosity path active.
  The three-step Taylor check passes with
  `FD = 1.2696868499639802e-02`,
  `adjoint = 1.2696868499425038e-02`, and relative slope error about
  `1.7e-11`.
- A one-step active-artificial-viscosity composition check using
  `adjoint_time_dependent_temperature_composition.prm`. After the latest
  alpha=2 estimator fixes, this one-step check passes tightly:
  finite-difference slope `-1.6688751075522035e-08`, adjoint slope
  `-1.6688783444144825e-08`, relative slope error about `1.9e-6`.
- A three-step active-artificial-viscosity composition check using the same
  benchmark now passes after fixing the temperature entropy-viscosity residual
  reconstruction. The step-2 check gives FD `-3.3197468211898506e-08` and
  adjoint `-3.3197553666812087e-08`, relative slope error about `2.6e-6`.
  The step-3 check gives FD `-4.9512464440903958e-08` and adjoint
  `-4.9512566069648305e-08`, relative slope error about `2.1e-6`.
- A custom/incompressible/real-density active-AV composition-control benchmark
  at `adjoint_time_dependent_temperature_composition_real_density_av.prm`.
  This stress-tests `C -> rho*cp -> temperature EV -> T_final` while keeping
  Stokes mass conservation incompressible. The step-2 check gives FD
  `-3.3402091856316041e-08` and adjoint `-3.3402148721503999e-08`, relative
  slope error about `1.7e-6`. The step-3 check gives FD
  `-4.9801399983062655e-08` and adjoint `-4.9801465935108224e-08`, relative
  slope error about `1.3e-6`.
- A standalone benchmark material VJP diagnostic at
  `material_model_adjoint_variable_cp_vjp.prm` validates mixed `rho(T,C)` and
  `cp(T)` seeds. The companion `material_model_adjoint_eta_vjp.prm` validates
  viscosity-only and mixed density/viscosity seeds for smooth `eta(T,C)`.
- Direct temperature material-prefactor Taylor benchmarks
  `adjoint_temperature_rho_t_direct.prm`, `adjoint_temperature_rho_t_ev.prm`,
  `adjoint_temperature_cp_t_direct.prm`, and `adjoint_temperature_cp_t_ev.prm`
  validate `rho(T)` and `cp(T)` in both direct and active entropy-viscosity
  temperature paths.
- Eta-only coupled benchmarks
  `adjoint_temperature_eta_c_stokes_no_av.prm`,
  `adjoint_temperature_eta_t_stokes_no_av.prm`, and
  `adjoint_temperature_eta_c_stokes_ev.prm` validate
  `C -> eta(C) -> Stokes -> T`, `T -> eta(T) -> Stokes -> T`, and the
  eta-dependent active-EV velocity coupling. The direct viscosity coefficient
  diagnostic is tight, and the old RHS/solve-only Stokes diagnostics are
  skipped for matrix-coefficient viscosity because they are not valid
  definitions of the `eta` sensitivity.
- Permanent `tests/` regression coverage for the decisive adjoint cases is
  deferred; the full validation cases remain in the benchmark directory for
  now.

Still missing:

- Matrix/operator dot-product tests for `A^T`, BE/BDF2 history transpose,
  velocity extrapolation transpose, and velocity transpose load.
- An explicit Stokes transpose solver and Stokes dot-product verification.
- Broader integration tests: advection-only one-step, coupled one-step, and
  BDF2 multi-step coverage beyond the validated tiny two-rank direct-Stokes
  restart/lifecycle fixture. Same-rank/same-partition MPI restart, collective
  checkpoint publication, safe-incremental cleanup, and transaction failure
  recovery are now validated; different rank counts and repartitioning are not.

## Practical rule for v1

### Sparse-replay runtime contract

Sparse replay is qualified only for fixed mesh, same-rank-count, same DoF
ownership/layout restarts. It retains the full accepted-step schedule and
replays an interval from its retained lower anchor; it does not invoke timestep
selection during replay. AMR, repartitioning, different-rank-count restarts,
Revolve scheduling, and unqualified plugin state are outside this contract.

Use `Trajectory storage mode = stokes cached` with sparse-reference validation:
the validation compares state, velocity, and gauge-normalized pressure and
therefore requires all three archived components. The minimal mode remains a
storage mode, not a full-state replay reference.

The fixed-mesh GMG qualification showed a bounded iterative reconstruction
variation (maximum relative velocity difference `2.91e-11`); its replay
tolerance is consequently `1e-10`, while direct-Stokes qualified cases remain
bitwise-identical in the checked gradients. This is a numerical-solver
tolerance, not an authorization for different topology or rank layout.

For now, the time-dependent temperature adjoint should be treated as valid only
for small, fixed-mesh, continuous-temperature, optional continuous FEM
chemical-composition, no-melt, Boussinesq/simple-material models using
`single Advection, single Stokes, adjoint temperature`.

Everything outside that envelope should either be explicitly rejected at runtime
or implemented with a matching transpose operator and a dot-product/Taylor test.
