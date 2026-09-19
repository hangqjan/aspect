# G5 non-self-gravitating geoid baseline

This is the acceptance record for the non-self-gravitating `geoid`
objective. It covers the shared mantle-density harmonic operator, the surface
and CMB dynamic-topography branches, and one summed physical Stokes adjoint.
It does not cover SG0 or any self-gravitating operator.

## Serial resource baseline

`/usr/bin/time -l build-release/aspect-release
benchmarks/adjoint_geoid_density/g5_cmb_m0.prm` completed with one forward and
one adjoint AMG solve (158 and 151 iterations), zero swaps, and a maximum RSS
of 315,719,680 bytes (about 301 MiB). A process sample independently measured
304,352 KiB RSS (about 297 MiB) and 436,332,480 bytes VSZ. The previous memory
report is therefore classified as:

```text
Transient/unreproduced host or UI memory observation
```

All G5 integration runs below were launched serially.

## Branch and solve-count matrix

| Case | Fixture | Result |
| --- | --- | --- |
| D | `adjoint_geoid_density.prm` | one forward GMG solve; no physical adjoint |
| T / D+T | `g5_surface_m0.prm` | 768 cells, forward + one AMG adjoint |
| B / D+B | `g5_cmb_m0.prm` | 768 cells, forward + one AMG adjoint |
| T+B / D+T+B | `g5_surface_cmb_m0.prm` | 768 cells, forward + one AMG adjoint |

The objective has no switch that removes `A_rho` from the forward observation
operator. “T”, “B”, and “T+B” mean that the density *control direction* is
held fixed, while the mandated mantle-density forward term remains present.

The surface and CMB `l=2,m=1` cosine and sine fixtures all completed on the
same 768-cell mesh, each with one forward and one physical adjoint solve.
They retain coverage of coefficient ordering, support-point projection, and
the inner-boundary sign/radial convention.

## Fixed-mesh full FD closure

The existing direct cellwise material wrapper selected the largest final
combined kernels: density cell 672 and raw-viscosity cell 644. Each entry is a
centered difference of the full D+T+B objective and the corresponding sum of
the direct mantle-density, surface-CBF, CMB-CBF, and implicit-Stokes kernels.

| Density step | Density relative error | Viscosity step | Viscosity relative error |
| ---: | ---: | ---: | ---: |
| 0.1 | 2.84e-7 | 1e17 | 2.28e-7 |
| 1 | 2.82e-7 | 1e18 | 1.17e-6 |
| 10 | 2.62e-7 | 1e19 | 7.67e-5 |

The small and middle viscosity steps are the clear FD convergence region; the
large raw-viscosity step has the expected truncation error. The final kernel
output keeps the terms distinct:

```text
mantle-density geoid harmonic / density
dynamic topography surface objective / density, viscosity
dynamic topography CMB objective / density, viscosity
incompressible Stokes volume / density, viscosity
```

Current largest-kernel non-axisymmetric closures are surface `l=2,m=1`
cosine: density cell 682 `4.84e-7`, raw-viscosity cell 654 `4.09e-7`; and
CMB `l=2,m=1` sine: density cell 565 `5.06e-9`, raw-viscosity cell 711
`4.48e-7`. The CMB harmonic transpose is below `1e-15`; CMB m=0/cosine/sine
coefficient-to-FE dualities agree to printed precision; fixed-seed top+CMB
RHS superposition is about `1.0e-16` relative.

## AMR and MPI

The fixed mesh gives objective `7.374797682e-3`. A uniformly finer 6,144-cell
mesh gives `8.633180791e-3`, with the same complete direct/implicit
decomposition. A protected one-step adaptive mesh (minimum level 1 and no
coarsening) realizes 796 cells / 30,227 DoFs and completes the forward plus
one-adjoint pair before and after the initial adaptation.

For the adaptive mesh, a raw-viscosity FD retains the 796-cell realization and
closes at `1.97e-7`. A raw-density perturbation changes the AMR decision: the
baseline has 796 final cells while the perturbed runs remain at 768 cells.
That difference is a derivative through mesh selection and is explicitly out
of scope. G5 differentiates the observation operator on a realized mesh only;
it does not differentiate AMR decisions. The unprotected default AMR setup
also allows coarsening and is not an appropriate convergence comparison.

MPI ranks 2 and 4 are **NOT AVAILABLE** on this host: PRTE fails before
ASPECT starts because the host forbids binding its TCP listener. The 1-rank
comparison is the serial baseline above.

## State isolation and limits

The manager clears the forward pressure-normalization adjustment only while
solving the physical adjoint and restores it, together with the forward state,
RHS, linearization point, and constraints, on normal and exception paths.
The normal-path and forced-exception restoration checks are retained from G3/G4.
A dedicated two-objective sequential regression was not added.

No exact density-contrast cancellation is assumed: CBF uses local support
density, configured above/below density, local gravity, and a support-point
denominator, whereas geoid scaling uses the legacy boundary reference/mean
density, reference gravity, and radial factor.

G5 excludes self-gravity, coupled surface/CMB feedback, internal-interface
deformation, moving geometry and gravity derivatives, point-observation
synthesis, self-gravity matrix derivatives, and self-gravitating
boundary-density derivatives.
