# Artificial viscosity in the forward advection solve and adjoint consistency notes

This note summarizes how ASPECT currently computes and uses artificial
viscosity for the temperature and continuous FEM chemical-composition advection
equations, and what a strictly consistent time-dependent adjoint would need to
implement.

It is written as planning material for extending the current v1 adjoint beyond
the frozen-stabilization approximation.

## Current v1 status

The current time-dependent temperature adjoint supports the default
`Stabilization method = entropy viscosity` path in two layers. The same
advection-field machinery is now used for temperature and for continuous FEM
chemical-composition fields, but the composition path is still much less
validated.

Layer 1 is the frozen-coefficient transpose:

- The reverse sweep restores each forward snapshot.
- It reassembles the forward advection matrix for that snapshot.
- It applies the algebraic transpose of the assembled advection block,
  including the artificial-viscosity diffusion entries that are present in that
  matrix.

Layer 2 is the restricted active-set estimator transpose:

- The advection assembly caches the cellwise artificial viscosity and physical
  diffusion bounds that were actually used.
- After solving `A_n^T z_n = bar_T_n`, the adjoint assembles the artificial
  coefficient seed
  `bar_nu_art += -dt * integral grad(z) . grad(field)` on cells where
  `max(k, nu_art)` selects artificial diffusion.
- For `alpha = 1` and `alpha = 2`, this seed is propagated through the
  supported local branches of the entropy-viscosity estimator:
  maximum-viscosity startup/fallback, entropy-viscosity limiter, cellwise
  velocity maximum including the optional `gamma*h*|eps(u)|` strain-rate
  measure, cellwise weighted residual maximum, residual dependence on
  historical temperature values/gradients/laplacians, and residual dependence
  on historical velocity values.
- For the tested `alpha = 2` active set, the transpose now also includes the
  global entropy-variation denominator and the extrapolated field range used to
  define `average_field`.

Therefore the implemented adjoint is no longer only a frozen-stabilization
approximation for the tested path, but it is still not the strict discrete
adjoint of every branch in ASPECT's full nonlinear stabilization algorithm.

SUPG is different: the current v1 rejects SUPG at runtime because the SUPG
residual and stabilization parameter introduce additional transpose terms that
are not implemented.

## Current implementation boundary

Implemented in code:

- `source/adjoint/operators/artificial_viscosity_adjoint.cc`
  - `assemble_coefficient_seed()`
  - `run_coefficient_seed_diagnostic()`
  - `add_estimator_transpose()`
- `source/adjoint/operators/advection_adjoint_calculator.cc`
  - calls the coefficient seed immediately after the transposed advection solve
  - calls the estimator transpose before the Galerkin history and velocity
    transpose contributions
  - provides a constrained-DoF regression diagnostic for the transposed
    advection solve
  - provides an isolated `R_q -> nu_art` coefficient-seed diagnostic
  - provides a standalone entropy-viscosity estimator transpose diagnostic
    from cellwise `bar_nu_art` to old/old-old field and velocity states
- `source/simulator/assembly.cc`
  - caches the artificial viscosity and physical diffusion bounds from the
    actual advection assembly
- `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature_advection_dominated.prm`
  - activates the artificial diffusion branch and verifies that the seed is
    nonzero
  - runs the constrained-DoF diagnostic and the isolated
    artificial-viscosity coefficient-seed diagnostic
- `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature_advection_dominated_alpha1.prm`
  - activates the `alpha = 1` entropy-viscosity branch in an
    advection-dominated setting
- `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature_composition.prm`
  - exercises the continuous FEM chemical-composition path and writes an
    initial composition gradient
- `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_composition_objective_av.prm`
  - observes `C_final` directly and validates the active artificial-viscosity
    composition advection/history path without the `C -> Stokes -> T` chain

Hard runtime or active restrictions:

- `Stabilization method = SUPG` is rejected.
- `Use artificial viscosity smoothing = true` is rejected.
- `alpha = 1` and `alpha = 2` are supported on the documented local active-set
  paths.
- `gamma != 0` is supported for the local velocity measure through the
  symmetric-gradient transpose of `eps(u)`.
- Continuous temperature FE only.
- Continuous FEM chemical-composition fields only; no entropy fields,
  particles, VOF, or discontinuous composition.
- No melt, mesh deformation, or operator splitting.
- The current active-set seed supports only cells where all quadrature points
  select the same branch of `max(k, nu_art)`.

Strict-discrete-adjoint phase boundary:

- The layer `R_q -> nu_art` has now been isolated and checked by treating
  `nu_art` as an independent cellwise coefficient. On the advection-dominated
  temperature benchmark, the diagnostic tests 8 artificial-branch cells and
  gives `FD = -1.699675458344842e-4`,
  `adjoint = -1.699675458344841e-4`, relative error about `1.6e-16`.
- The same coefficient-seed diagnostic passes at machine precision for the
  `alpha = 1` benchmark and the `gamma != 0` benchmark. This validates the
  seed generated from the transposed advection solution, but not the
  downstream entropy-viscosity estimator derivative.
- The transposed advection solve now has a regression diagnostic for
  inhomogeneous forward Dirichlet boundaries. On the active-AV benchmark it
  finds 18 constrained temperature DoFs and `max |z| = 0`.

Still not implemented:

- Explicit transpose of the `alpha = 1` entropy denominator dependencies on
  global field variation and global maximum velocity.
- Neighbor-max smoothing transpose.
- Boundary disabling derivative for conduction-dominated Dirichlet cells; the
  boolean active set is treated as fixed.
- Physical/material derivative paths for `k(T,p)`, `rho`, `cp`, heating,
  latent heat, material averaging, melt/Darcy terms, and additional material
  outputs.
- SUPG matrix/RHS/tau transpose.
- Strict validation of composition-control gradients under active artificial
  viscosity; the current composition+AV Taylor check exposes a missing
  derivative by showing first-order remainder.

Note on `eps(u)`:

- In ASPECT's stabilization formula, `eps(u)` is the symmetric velocity
  gradient, `0.5*(grad u + grad u^T)`.
- It is not zero in compressible flow. In incompressible flow only
  `trace(eps(u)) = div(u)` is constrained to vanish; the shear/deviatoric
  strain-rate components can still be nonzero.
- Compressible formulations remain outside the current time-dependent
  temperature adjoint boundary for other reasons: mass conservation, density,
  pressure, and temperature-equation terms need additional transpose operators.

Validation currently recorded:

- The original diffusion-dominated smoke benchmark passes with zero
  artificial-viscosity seed.
- The advection-dominated smoke benchmark passes with nonzero seed norm
  around `1.2119e-4`.
- The original multi-step Taylor test passes with finest relative slope error
  around `1.7e-11`.
- The advection-dominated multi-step Taylor test passes with finest relative
  slope error around `1.4e-9`.
- The advection-dominated `gamma = 1` multi-step Taylor test passes with finest
  relative slope error around `3.0e-9`.
- The advection-dominated `alpha = 1` multi-step Taylor test passes with finest
  relative slope error around `2.5e-9`.
- A broader Taylor sweep over the four current benchmarks and end steps
  `1`, `3`, and `5` passes for perturbation amplitudes
  `1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4`. The Taylor remainders converge at
  observed order two across the sweep.
- A no-artificial-viscosity one-step composition-control smoke check passes on
  the absolute-slope criterion. The directional derivative is near zero, so it
  should be read as a plumbing sanity check rather than strong gradient
  validation.
- A terminal composition-field objective benchmark
  `adjoint_time_dependent_composition_objective_no_av.prm` passes a three-step
  Taylor test with second-order remainder. This isolates the direct
  `C_final -> C0` reverse composition advection/history path and does not use
  artificial viscosity.
- A stronger no-artificial-viscosity coupled composition-control check is kept
  in
  `benchmarks/adjoint_time_dependent_temperature/adjoint_time_dependent_temperature_composition_coupled_no_av.prm`.
  This case now passes after fixing the adjoint treatment of constrained
  advection DoFs: the transposed solve distributes a homogeneous copy of the
  forward constraint graph, preserving affine entries while removing only the
  forward inhomogeneities.
- A transposed-advection constrained-DoF regression diagnostic is enabled in
  the coupled no-AV and advection-dominated benchmarks. The no-AV coupled run
  reports 18 constrained temperature DoFs and `max |z| = 0`.
- The same coupled benchmark now includes a Phase-1
  `C -> assembled Stokes RHS` transpose diagnostic. That diagnostic passes to
  about `2e-10` relative error. The Stokes-solve transpose diagnostic passes to
  about `2.7e-7`. The temperature-advection velocity transpose diagnostic is
  intentionally run only for one-step checks; the current diagnostic probes the
  local `u_star` layer and does not compare the BDF2 history split.
- The advection-dominated active-AV benchmark now includes a Phase-1
  artificial-viscosity coefficient-seed diagnostic. Treating `nu_art` as an
  independent cellwise coefficient, the diagnostic passes with relative error
  about `1.6e-16`. A Taylor run of the same benchmark still passes with
  second-order remainder and finest relative slope error about `6.7e-13`.
- The same active-AV benchmark now includes a standalone estimator-transpose
  diagnostic. On the one-step startup/fallback branch, the field contribution
  is zero and the velocity maximum transpose agrees with finite differences to
  about `1.2e-6` at epsilon `1e-9`; the `gamma != 0` benchmark agrees to about
  `4.6e-6`.
- The benchmark also includes a frozen-denominator local entropy-residual
  diagnostic for the alpha=2 numerator
  `Rmax = max_q |r_q| |field_history - average_field|`. On an end-step-3 run,
  this local numerator diagnostic passes with
  `FD = -2.283692171110683`, `adjoint = -2.283691483465801`, and relative error
  about `3e-7` at epsilon `1e-10`. This validates the local argmax, absolute
  value, time-history residual, advection residual, and entropy-deviation
  multiplier for the tested smooth active set.
- The alpha=2 scalar denominator derivative
  `nu_entropy = c_R h^2 Rmax / global_entropy_variation` is now isolated by a
  diagnostic that treats `global_entropy_variation` as an independent scalar.
  On the same end-step-3 run it reports
  `global_entropy_variation = 1.475058404997293e-3`,
  `bar_global_entropy_variation = 1.371431106076628e-5`,
  `FD = 5.088009403555274e-6`,
  `adjoint = 5.08800940354429e-6`, and relative error about `2.2e-12`.
- The alpha=2 `global_entropy_variation(field_history)` transpose is now
  isolated by a frozen-average diagnostic. On the asymmetric end-step-3 run, it
  reports one active max location, no active min location for the selected
  branch, and agrees with finite differences at about `1.7e-9` relative error.
- The alpha=2 local numerator dependence on the scalar `average_field` is also
  checked inside the local entropy-residual diagnostic. On the end-step-3 run,
  the scalar average-field FD/adjoint values agree to about `4e-7` relative
  error.
- With the estimator diagnostic epsilon reduced to `1e-11` to avoid crossing
  nonsmooth active-set boundaries, the complete alpha=2 standalone estimator
  diagnostic now agrees to about `2.9e-5` relative error on the end-step-3
  advection-dominated benchmark. The remaining difference is consistent with
  finite-difference sensitivity around global min/max and active-branch
  boundaries; the lower-level diagnostics agree much more tightly.
- The temperature artificial-viscosity residual reconstruction now follows the
  forward temperature residual form
  `r = (rho*cp) * (dTdt + u.grad(T)) - k*Delta(T)` for the current no-heating,
  no-latent-heat v1 envelope. The reverse path therefore sends
  `bar_transport = (rho*cp) * bar_r` to the field and velocity terms, rather
  than using the bare `bar_r`.
- The temperature maximum-viscosity branch and entropy-residual branch now
  produce an optional `bar(rho*cp)` seed. When the temperature equation uses
  real density, this seed is propagated to composition fields through the same
  finite-difference `d(rho*cp)/dC` mechanism used by the temperature-equation
  material-coefficient transpose. In `reference density profile` mode, this
  material path is intentionally inactive because the forward temperature
  equation overwrites density by the reference profile.
- The alpha=1 advection-dominated benchmark currently validates the
  maximum-viscosity/startup-style path: with estimator epsilon `1e-11`, the
  full estimator diagnostic agrees to about `1.7e-9` relative error and the
  field contribution is zero. The stricter alpha=1 entropy branch that depends
  on global field variation and global maximum velocity still needs its own
  active-set benchmark and transpose.
- The `gamma != 0` alpha=2 benchmark validates the strain-rate contribution in
  the local velocity measure. With estimator epsilon `1e-11`, the velocity
  component agrees to machine precision and the full estimator diagnostic agrees
  to about `3.3e-5` relative error.

Forward definition of `global_entropy_variation` for `alpha = 2`:

- `average_field` is passed in from
  `get_extrapolated_advection_field_range(advection_field)` as
  `0.5 * (global_min_extrapolated_field + global_max_extrapolated_field)`.
- `get_entropy_variation(average_field, advection_field)` samples the history
  field on the advection-field quadrature:
  `field_history_q = 0.5 * (old_field_q + old_old_field_q)`.
- It forms
  `entropy_q = (field_history_q - average_field)^2`.
- It computes local/global `min_entropy`, `max_entropy`, the integrated entropy
  `integral entropy_q dV`, and the domain measure.
- It returns
  `max(max_entropy - average_entropy, average_entropy - min_entropy)`, where
  `average_entropy = integral entropy_q dV / integral 1 dV`.
- This means the strict transpose needs active-set derivatives for the selected
  outer max branch and for the global min/max entropy locations, plus the
  smooth integral-average entropy transpose and the `average_field` dependency.
  These terms are now implemented for the tested alpha=2 active-set path.
- After the alpha=2 temperature-estimator fixes, the active-artificial-viscosity
  one-step composition-control check now passes tightly:
  `FD = -1.6688751075522035e-08`,
  `adjoint = -1.6688783444144825e-08`, relative slope error about `1.9e-6`.
- Before the temperature-EV material-prefactor fix, the
  active-artificial-viscosity three-step composition-control check with a
  present-day temperature objective failed as a strict gradient test. An
  epsilon sweep from `1e-3` to `6.25e-5` showed first-order Taylor remainder and
  a persistent relative slope error about `1.31e-2`.
- The diagnostic split showed that a step-2 run already failed, while the
  composition standalone estimator diagnostic passed. This localized the bug to
  the coupled temperature-EV path, not to the direct composition estimator
  transpose or reverse-time bookkeeping.
- The direct terminal-composition objective benchmark with active composition
  artificial viscosity passes a three-step Taylor test:
  `J(0)=1.7014527392795081e-03`,
  adjoint directional derivative `1.2696868499425038e-02`,
  finite-difference derivative `1.2696868499639802e-02`, relative slope error
  about `1.7e-11`. This validates the `C_final -> C0` multi-step active-AV
  composition advection/history chain for the tested active set.
- In the failing present-day temperature objective benchmark, the composition
  estimator diagnostic at step 2 agrees tightly, but the temperature estimator
  diagnostic originally did not. The issue was the temperature EV adjoint
  reconstructing the residual as `dTdt + u.grad(T) - k*Delta(T)` instead of the
  forward-consistent prefactored form. After this fix, the present-day
  temperature objective with `C0` control passes the active-AV step-2 and
  step-3 Taylor checks with relative slope errors about `2.6e-6` and `2.1e-6`,
  respectively.
- `adjoint_time_dependent_temperature_composition_real_density_av.prm` is the
  stronger material-prefactor stress test. It uses
  `Formulation = custom`, `Mass conservation = incompressible`, and
  `Temperature equation = real density`, so the temperature equation samples
  material-model density while the Stokes mass equation remains in the v1
  incompressible envelope. The step-2 and step-3 `C0 -> T_final` Taylor checks
  pass with relative slope errors about `1.7e-6` and `1.3e-6`, respectively,
  validating the tested `C -> rho*cp -> temperature EV -> T_final` path.

## Relevant source locations

- `source/simulator/assembly.cc`
  - `Simulator<dim>::assemble_advection_system(...)`
  - Calls `get_artificial_viscosity(viscosity_per_cell, advection_field)` before
    assembling the advection system.
  - Stores the per-cell value into `scratch.artificial_viscosity`.

- `source/simulator/assemblers/advection.cc`
  - `Assemblers::AdvectionSystem<dim>::execute(...)`
  - Uses `scratch.artificial_viscosity` in the temperature matrix.
  - For entropy viscosity, the effective diffusion is
    `max(thermal_conductivity, artificial_viscosity)`.
  - For SUPG, `scratch.artificial_viscosity` is used as the SUPG parameter
    `tau`, but v1 rejects this path.

- `source/simulator/entropy_viscosity.cc`
  - `Simulator<dim>::get_artificial_viscosity(...)`
  - Computes cellwise artificial viscosity values.
  - `Simulator<dim>::compute_viscosity(...)`
  - Implements the entropy-viscosity formula and limiter.

- `source/simulator/parameters.cc`
  - Declares `Stabilization method = entropy viscosity|SUPG`.
  - Declares `Use artificial viscosity smoothing`, `alpha`, `cR`, `beta`, and
    `gamma`.

## Forward flow: where artificial viscosity enters

For each temperature advection assembly:

1. `assemble_advection_system(AdvectionField::temperature())` zeros the
   temperature block matrix and RHS.

2. It allocates a vector over active cells:

   ```text
   viscosity_per_cell[cell] = artificial viscosity for this advection field
   ```

3. It calls:

   ```text
   get_artificial_viscosity(viscosity_per_cell, temperature_field)
   ```

4. During cell assembly, the cell value is copied into:

   ```text
   scratch.artificial_viscosity
   ```

5. In `AdvectionSystem::execute`, for the temperature equation and
   `Stabilization method = entropy viscosity`, the effective diffusion is:

   ```text
   diffusion_constant = max(k, nu_art)
   ```

   where:

   ```text
   k      = material_model_outputs.thermal_conductivities[q]
   nu_art = scratch.artificial_viscosity
   ```

6. The diffusion matrix contribution contains:

   ```text
   dt * diffusion_constant * grad(phi_i) . grad(phi_j) * JxW
   ```

7. Thus the forward temperature solve contains artificial diffusion whenever
   `nu_art > k`.

## Forward flow: inputs to the entropy-viscosity estimator

`get_artificial_viscosity()` computes the cellwise coefficient from many pieces
of forward state. For temperature, important dependencies include:

- `old_solution`
- `old_old_solution`
- `current_linearization_point`
- old and old-old temperature values
- old and old-old temperature gradients
- old and old-old temperature laplacians
- old and old-old velocity values
- old and old-old strain rates
- old and old-old pressure values
- old and old-old pressure gradients
- material model outputs:
  - density
  - specific heat
  - thermal conductivity
  - equation-of-state properties
- heating model outputs:
  - heating source terms
  - latent heat LHS terms
- global field range
- global entropy variation
- global maximum velocity
- geometry length scale and cell diameter
- boundary condition classification
- optional boundary normal-flow checks
- optional artificial-viscosity smoothing over neighboring cells

This means `nu_art` is not a simple local function of current temperature. It is
a time-history, state-history, global-reduction, material-model, heating-model,
and limiter-dependent quantity.

## Entropy-viscosity formula structure

For continuous elements, the coefficient is approximately:

```text
nu_art = min(nu_max, nu_entropy)
```

The maximum viscosity branch is:

```text
nu_max = beta * max_advection_prefactor * max_velocity * h
```

where:

```text
max_velocity = max_q(|u_old/old_old| + gamma * h * |strain_rate_old/old_old|)
```

The entropy branch depends on the stabilization alpha:

For `alpha = 2`:

```text
nu_entropy = cR * h^2 * max_residual / global_entropy_variation
```

For `alpha = 1`:

```text
nu_entropy = cR * h * domain_diameter * max_velocity * max_residual
             / (global_u_infty * global_field_variation)
```

The residual used by the temperature advection assembler is based on:

```text
dT/dt + u . grad(T)
diffusion term k * laplacian(T)
heating source gamma
latent heat LHS contribution
rho * cp
```

For `alpha = 2`, the residual is additionally multiplied by:

```text
abs(T_history - average_field)
```

Startup behavior:

- If `timestep_number <= 1`, or entropy/field variation is near zero, the code
  returns `nu_max`.
- This means the first two iterations/steps are treated specially and can have a
  different derivative structure than later BDF2 steps.

Boundary behavior:

- For temperature, artificial viscosity can be disabled on conduction-dominated
  Dirichlet boundary cells.
- The code checks fixed temperature boundaries and whether boundary flow is
  zero/tangential or has normal flow.

Smoothing behavior:

- If `Use artificial viscosity smoothing = true`, the final `nu_art` on a cell
  may be replaced by the maximum over the cell and its neighbors.

## What the current adjoint already includes

The current v1 reverse temperature solve does:

```text
A_n^T z_n = bar_T_n
```

where `A_n` is the reassembled forward temperature matrix. Since `A_n` includes:

```text
dt * max(k, nu_art) * grad(phi_i) . grad(phi_j)
```

the algebraic transpose of that matrix is included.

This is enough for frozen-coefficient consistency:

```text
delta A = A(delta T, delta u, delta material outputs) with nu_art fixed
```

It is not enough for full consistency:

```text
delta A also contains dA/dnu_art * dnu_art/dstate
```

## Missing adjoint terms for strict consistency

For the strict discrete adjoint, if:

```text
R_T(T_n, T_{n-1}, T_{n-2}, u*, nu_art(...)) = 0
```

and:

```text
nu_art = F(T_{n-1}, T_{n-2}, u_{n-1}, u_{n-2}, p, material, heating, global reductions, limiters)
```

then the reverse sweep must include:

```text
bar_state += -(partial R_T / partial nu_art)^T z_n
             * partial nu_art / partial state
```

The immediate matrix sensitivity is local:

```text
partial R_i / partial nu_art
  contains dt * indicator(nu_art > k) * grad(phi_i) . grad(T_n)
```

or, in matrix form:

```text
delta A_ij = dt * indicator(nu_art > k) * delta_nu_art
             * grad(phi_i) . grad(phi_j) * JxW
```

The adjoint seed for the artificial-viscosity coefficient on a cell would be:

```text
bar_nu_art[cell] += sum_q dt * indicator(nu_art > k)
                    * (grad(z_n) . grad(T_n)) * JxW
```

Then `bar_nu_art` must be propagated through the estimator `F`.

## Difficult pieces in differentiating the estimator

### 1. `max(k, nu_art)`

The forward matrix uses:

```text
diffusion_constant = max(k, nu_art)
```

Derivative:

- If `nu_art > k`, derivative with respect to `nu_art` is 1.
- If `nu_art < k`, derivative with respect to `nu_art` is 0.
- If `nu_art == k`, the derivative is nonsmooth.

Planning options:

- Use an active-set derivative and ignore exact ties.
- Freeze the branch selected in the forward run.
- Smooth the max function if a differentiable formulation is required.

### 2. `min(nu_max, nu_entropy)`

The entropy estimator uses:

```text
nu_art = min(nu_max, nu_entropy)
```

Derivative:

- Propagate only through the active smaller branch.
- Exact ties are nonsmooth.

Planning options:

- Active-set derivative.
- Freeze selected branch.
- Smooth min approximation.

### 3. Quadrature max operations

The estimator takes maxima over quadrature points:

```text
max_residual
max_velocity
max_advection_prefactor
max_conductivity
```

Derivative:

- Propagate to the argmax quadrature point only.
- Multiple equal maxima are nonsmooth.

Planning options:

- Store argmax choices in the forward snapshot.
- Recompute argmax during reverse and accept active-set behavior.
- Use a smooth max if differentiability is required.

### 4. Absolute values

The residual uses absolute values, and `alpha = 2` multiplies by:

```text
abs(field - average_field)
```

Derivative:

- Sign-dependent.
- Nonsmooth at zero.

Planning options:

- Active-set sign derivative.
- Smooth absolute value.
- Freeze residual weights.

### 5. Global reductions

The estimator depends on global field range, entropy variation, and global max
velocity.

Derivative:

- Global min/max/range operations are nonsmooth and couple all ranks.
- A perturbation at one DoF can change the global active min/max location.

Planning options:

- Freeze global scalars.
- Store active locations and propagate through them.
- Use smooth global reductions for differentiable mode.

### 6. Boundary disabling logic

Temperature artificial viscosity can be set to zero on conduction-dominated
Dirichlet boundary cells.

Derivative:

- This is branch logic based on boundary type and flow normal/tangential tests.
- If a perturbation changes the branch, the derivative is discontinuous.

Planning options:

- Treat boundary disabling as frozen branch logic.
- Reject cases with nonzero artificial viscosity near controlled boundaries.
- Store branch decisions in snapshots.

### 7. Artificial-viscosity smoothing

With `Use artificial viscosity smoothing = true`, the final value is:

```text
nu_art[cell] = max(nu_art[cell], nu_art[neighbor cells])
```

Derivative:

- Propagate adjoint only to the cell/neighbor that supplied the maximum.
- Nonsmooth at ties.
- Cross-cell and cross-rank communication may be needed.

Planning options:

- Reject smoothing in v1.
- Store neighbor argmax choices.
- Implement transpose of the smoothing max operation.

### 8. Material and heating dependencies

The residual and scaling factors depend on material and heating outputs:

- density
- specific heat
- thermal conductivity
- heating source
- latent heat LHS
- possibly melt outputs and other additional outputs

Derivative:

- Requires material-model derivative infrastructure.
- Requires heating-model derivative infrastructure.

Planning options:

- For v1, restrict to constant `rho`, `cp`, `k`, no heating, no latent heat.
- Later, add derivative outputs to material/heating models.

## Snapshot requirements for strict artificial-viscosity adjoint

The current v1 snapshots store solution/history and linearization points. A
strict artificial-viscosity adjoint may also need to store or reproducibly
recompute:

- `nu_art[cell]` for each timestep and advection field.
- active branch of `max(k, nu_art)`.
- active branch of `min(nu_max, nu_entropy)`.
- argmax quadrature point for residual/velocity/conductivity/prefactors.
- signs of absolute-value terms.
- global range/entropy/max-velocity values and active locations.
- boundary disabling decisions.
- smoothing neighbor argmax choices.
- material/heating outputs used by the estimator, unless they are exactly
  reproducible from restored state.

Without storing these, reverse recomputation may choose a different branch after
roundoff-level differences, causing an inconsistent adjoint.

## Suggested implementation strategies

### Strategy A: frozen-stabilization v1

Treat `nu_art` as fixed data during reverse mode.

Required implementation:

- Keep current `A_n^T` reassembly and transpose solve.
- Document that artificial-viscosity coefficients are frozen.
- Add a runtime note/warning when entropy viscosity is active.
- Optionally output max/mean artificial viscosity per timestep for diagnostics.
- Reject `Use artificial viscosity smoothing = true` unless documented as frozen.

Pros:

- Simple and already close to current behavior.
- Stable for small benchmarks.
- Avoids differentiating nonsmooth max/min/abs/global reductions.

Cons:

- Not the strict discrete adjoint of the full stabilized forward code.
- Taylor tests may fail when perturbations significantly change `nu_art`.

### Strategy B: zero-artificial-viscosity v1

Allow entropy-viscosity mode only if the computed artificial viscosity is zero
or inactive everywhere:

```text
nu_art <= k
```

Required implementation:

- During snapshot recording or reassembly, compute max active artificial
  viscosity.
- Assert or warn if `nu_art > k` anywhere.

Pros:

- Gives a clean consistency envelope.
- Avoids hidden frozen-stabilization assumptions.

Cons:

- Too restrictive for advection-dominated problems.

### Strategy C: active-set strict adjoint

Differentiate the currently active max/min/abs branches.

Required implementation:

- Assemble `bar_nu_art[cell]` from the diffusion matrix sensitivity.
- Implement reverse pass through:
  - `max(k, nu_art)`
  - `min(nu_max, nu_entropy)`
  - quadrature maxima
  - absolute values
  - global reductions
  - boundary disabling
  - optional smoothing
  - residual formula
- Store active branches or recompute them exactly.

Pros:

- Closest to the actual discrete forward algorithm away from branch changes.

Cons:

- Large implementation.
- Nonsmooth behavior remains near branch switches.
- Requires careful MPI/global reduction adjoints.

### Strategy D: smooth differentiable stabilization

Replace hard max/min/abs with smooth approximations in a special adjoint-ready
mode.

Required implementation:

- Define smooth versions of max, min, abs, and possibly global reductions.
- Use the same smooth estimator in forward and adjoint.
- Differentiate the smooth estimator.

Pros:

- Mathematically cleaner for gradient-based optimization.

Cons:

- Changes the forward discretization.
- Requires design decisions about smoothing parameters.
- Not identical to current ASPECT entropy viscosity.

## Recommended near-term plan

For the current branch, the most practical next steps are:

1. Explicitly declare the current behavior as frozen-stabilization adjoint.

2. Add runtime diagnostics:

   ```text
   max artificial viscosity
   max(max(k, nu_art) - k)
   number of cells where nu_art > k
   ```

3. Add a parameter such as:

   ```text
   subsection Adjoint
     set Stabilization derivative mode = frozen coefficients | require inactive artificial viscosity
   end
   ```

4. In `require inactive artificial viscosity` mode, assert if artificial
   viscosity is active in any accepted timestep.

5. Extend the Taylor script to include a case where artificial viscosity is
   deliberately active. This will tell whether frozen mode remains adequate for
   the intended benchmark regime.

6. Defer strict active-set differentiation until after the core adjoint is
   covered by operator-level dot-product tests.

## Tests to add

### Frozen-coefficient matrix test

Goal:

```text
a^T A(nu_art_fixed) b == (A(nu_art_fixed)^T a)^T b
```

This verifies the algebraic transpose of the matrix including artificial
diffusion entries.

### Inactive artificial viscosity Taylor test

Set up a benchmark where:

```text
nu_art <= k
```

Expected:

- Full Taylor test passes.
- No artificial-viscosity derivative is needed because the active diffusion is
  physical conductivity.

### Active artificial viscosity frozen-mode Taylor test

Set up a benchmark where:

```text
nu_art > k
```

Expected:

- If perturbations do not significantly change `nu_art`, Taylor may still pass
  approximately.
- If perturbations change `nu_art`, Taylor should expose the missing derivative.

This test is useful to quantify how much the frozen approximation matters.

### Smoothing rejection test

If v1 rejects smoothing:

```text
set Use artificial viscosity smoothing = true
```

should produce a clear error.

### Branch-switch sensitivity test

Construct a case near:

```text
nu_art == k
nu_max == nu_entropy
residual == 0
```

Expected:

- Strict derivative is nonsmooth.
- Test should document behavior rather than expect clean convergence.

## Practical guidance for using current v1

Use current v1 only when one of the following is acceptable:

1. Artificial viscosity is inactive or negligible.
2. Artificial viscosity is intentionally treated as frozen stabilization.
3. The Taylor test is run for the exact benchmark and perturbation family of
   interest and passes.
4. The run stays inside the documented `alpha = 1` or `alpha = 2`,
   no-smoothing, no-SUPG active-set boundary, and the field is temperature or a
   continuous FEM chemical-composition field.

It is reasonable to describe the current branch as a restricted active-set
adjoint for the tested entropy-viscosity path. Do not describe it as a complete
strict discrete adjoint for all ASPECT stabilization options until the remaining
global reductions, smoothing, SUPG, material/heating derivative paths, and
branch-switch tests are implemented.
