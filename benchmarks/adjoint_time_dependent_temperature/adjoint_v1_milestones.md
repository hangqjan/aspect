# Time-dependent discrete adjoint v1 milestones

## Validated milestone stack

- Core time-dependent discrete adjoint: validated.
- Entropy/artificial-viscosity adjoint: validated.
- MaterialModelAdjoint v1: validated.
- Boundary Conditions v1: validated.
- Adjoint preconditioning v1: validated.

## Adjoint preconditioning v1

### Design

Stokes uses ASPECT's existing forward-style block-preconditioner path. Under
the supported frozen/Picard v1 formulation, the Stokes operator is symmetric,
so the reduced forward and adjoint Stokes operators have the same structure.

Advection uses `ILU(A^T)` by default. The factorization is rebuilt for every
restored forward snapshot; no preconditioner is cached across reverse time
steps. The retained comparison modes are:

- `none`;
- `forward ilu experimental` (`ILU(A)` applied while solving `A^T`);
- `transpose ilu` (`ILU(A^T)`, the default).

The implementation reuses ASPECT's forward diagonal-strengthening retry policy.
Composition adjoint GMRES uses the composition solver tolerance.

### Close-out validation

The following same-snapshot, same-RHS diagnostic compares converged
unpreconditioned and `ILU(A^T)` transpose solves:

| Solve | Relative solution difference | None: residual / iterations | `ILU(A^T)`: residual / iterations |
| --- | ---: | ---: | ---: |
| Temperature | `2.10e-12` | `1.09e-24` / 29 | `2.18e-25` / 11 |
| Composition | `9.91e-13` | `3.17e-15` / 36 | `7.65e-16` / 11 |

The representative curved-free-slip reverse Stokes solve used the existing
block-GMG path and took `16+0` iterations. The curved-free-slip end-to-end
Taylor order was `2.00000001`.

One modest multistep reverse sample reported:

| Timer section | Cumulative time |
| --- | ---: |
| Build Stokes preconditioner (4 builds) | `0.000455 s` |
| Solve Stokes system (10 solves) | `0.0228 s` |
| Build adjoint temperature preconditioner (2 builds) | `0.000114 s` |
| Solve adjoint temperature system (2 solves) | `0.0000966 s` |
| Build adjoint composition preconditioner (3 builds) | `0.000212 s` |
| Solve adjoint composition system (3 solves) | `0.000178 s` |

The existing Stokes timing category combines forward and reverse solves. No
identity/unpreconditioned Stokes switch is available without restructuring the
existing ASPECT Stokes solver, so this validation intentionally does not add
one.

### Separate curved free-slip diagnostic issue

The optional curved-free-slip velocity-transpose diagnostic still reports a
large mismatch:

| Preconditioner | Relative error |
| --- | ---: |
| `none` | `0.7237447798478293` |
| `ILU(A^T)` | `0.7237447798475474` |

The difference is about `3e-13`; therefore the mismatch is pre-existing and
preconditioner-independent. It is a separate diagnostic issue, not a blocker
for preconditioning v1: the end-to-end curved-free-slip Taylor test remains
second order. Do not change the diagnostic as part of preconditioning work.

## Next major target: scalable trajectory replay and checkpointing

The next major development target is **minimal replay state plus
checkpoint/recomputation architecture** for scalable long-duration global
time-dependent adjoint simulations. Its objective is to store the minimum state
required for deterministic forward replay instead of storing a complete state
snapshot at every accepted timestep.

Planned scope only; no checkpointing implementation is included in v1:

- forward dependency audit;
- `MinimalReplayState` definition;
- sparse disk checkpoints;
- deterministic replay;
- sliding/ring-buffer history;
- checkpoint plus recomputation;
- later Revolve/binomial checkpoint scheduling;
- replay-fidelity diagnostics.
