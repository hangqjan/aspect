# O6 native ROL initial-temperature optimization

## Scope and production entry point

O6 adds an opt-in native ROL solve to the time-dependent temperature adjoint
workflow. The user parameter is:

```
subsection Adjoint
  set Solve mode = kernel | optimize
end
```

The default is `kernel`, which preserves the ordinary adjoint-kernel workflow.
`optimize` dispatches from `Simulator` through `Adjoint::Manager` to
`run_initial_temperature_optimization()`.

The v1 control is a reduced **increment**. With the captured canonical
initial-temperature field `T_fixed`, the physical initial condition is

```
T = T_fixed + P m .
```

Consequently `m = 0` exactly reconstructs the captured physical initial
temperature. Lower and upper optimization bounds apply to the coefficients of
`m`, not directly to physical temperature values.

## Optimizer and callback contract

The production method is `ROL::TypeB::LinMoreAlgorithm<double>` with an
explicit `ROL::lBFGS<double>` secant model. The vector metric is the reduced
temperature L2 metric. Exact Hessian, inverse-Hessian, tangent-linear, and
second-order-adjoint actions are intentionally unsupported; production runs
require zero `hessVec` and `invHessVec` callbacks.

`ROLReducedObjective` is generation-aware. A value evaluation creates the
live forward trajectory; a gradient at the same control reuses that trajectory
and performs only the reverse sweep. Trial, Revert, and Accept are handled by
the O5 accepted-generation cache protocol.

The scalar optimization contracts are shared by production parsing and the
focused `adjoint_optimization_parameters` unit test:

- L-BFGS history size is at least one.
- Gradient tolerance, step tolerance, and initial trust-region radius are
  strictly positive.
- The lower increment bound does not exceed the upper increment bound.

## O6 qualification record

The standard production three-iteration fixture is deterministic:

```
J = 0.0040227
||g|| = 0.000289202
forward = 4, reverse = 4
L-BFGS pairs = 3 (configured cap 3)
Initial/Trial/Revert/Accept = 1/3/0/3
preconditioner/hessVec/invHessVec = 12/0/0
```

The same point passed a fresh-objective audit with zero relative reduced-L2
gradient difference and zero maximum coefficient difference. A deterministic,
feasible reduced-L2 direction gave Taylor remainders at step lengths
`1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5` with observed order 2 over each
successive pair.

The active-bound fixture reached projected gradient norm zero after one
accepted step, without manually clamping the physical temperature. The
history-cap fixture with cap two retained exactly two pairs after three
accepted updates. A valid-path stress fixture converged after four accepted
updates with five forward and five reverse solves; RSS remained near 150 MiB
and did not grow with accepted iterates.

No natural Lin-More `Trial -> Revert` was observed in the bounded O6 probes:
the standard, tight-bound, and larger-radius fixtures accepted all trial
steps on this small near-quadratic benchmark. This is classified as **not
naturally exercised in the O6 fixture**, not a failure. O5 separately
validates Trial/Revert/Accept semantics, accepted-generation restoration, and
five-cycle memory stability.

Both omitted and explicit `Solve mode = kernel` runs retain the reference
terminal objective `0.004837537772946448` and do not enter optimizer code.

## Debug invalid-configuration limitation

The production scalar helper rejects every invalid O6 combination locally,
without MPI, Simulator construction, ROL construction, or a forward solve.
The normal ASPECT Debug top-level entry point catches exceptions after MPI
initialization, prints diagnostics, and calls `MPI_Abort(MPI_COMM_WORLD, 1)`.

An unrelated pre-existing `Dimension = 9` failure occurs in `main` before O6
code and exhibits the same resident-memory growth after its diagnostic. Thus
the non-terminating Debug/MPI abort behavior is framework-level rather than
an O6 validation defect. Expected-failure O6 validation is therefore covered
by the focused local helper; full-ASPECT invalid runs are deliberately not
used as a clean-exit test on this host.

## Unsupported v1 features

O6 does not support AMR during optimization, repartitioning, particles, mesh
deformation, free surface, uncontrolled mutable time-step plugins,
different-rank restart, optimizer restart, Revolve, exact Hessians,
tangent-linear models, or second-order adjoints.
