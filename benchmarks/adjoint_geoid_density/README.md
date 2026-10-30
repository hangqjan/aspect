# Geoid adjoint baselines

This directory contains the G5 non-self-gravitating `geoid` baseline
and the production self-gravitating geoid baseline. The G5 objective uses the
shared degree-major cosine/sine layout and evaluates

```text
N = A_rho rho + B_t h_t + B_b h_b
J = 1/2 (N - N_obs)^T W (N - N_obs)
```

## G5 non-self-gravitating baseline

`adjoint_geoid_density.prm` is the fast density-only case (D). The
`g5_topography_base.prm` model is a compact fixed-mesh 3-D spherical-shell
case. Its leaf files provide the physical-adjoint cases:

| File | Enabled branches | Matrix cases covered |
| --- | --- | --- |
| `g5_surface_m0.prm` | surface | T and D+T |
| `g5_cmb_m0.prm` | CMB | B and D+B |
| `g5_surface_cmb_m0.prm` | surface and CMB | T+B and D+T+B |
| `g5_surface_m1_cos.prm`, `g5_surface_m1_sin.prm` | surface | permanent l=2,m=1 ordering checks |
| `g5_cmb_m1_cos.prm`, `g5_cmb_m1_sin.prm` | CMB | permanent l=2,m=1 ordering and inner-boundary checks |

The `_m0` files isolate l=2,m=0 with weights `1,0,0`; the cosine and sine
files isolate l=2,m=1 by using a unit observation and weight only in that
slot. The mantle-density forward operator is always part of this objective.
For cases labelled “no/fixed density”, hold the density control fixed in the
directional derivative; there is intentionally no switch that changes the
forward observation operator.

## Architecture and invariants

```text
GeoidObjective
  -> GeoidHarmonics: A_rho, B_t, B_b forward and transpose operators
  -> DynamicTopographyCBFTranspose: top and bottom CBF transposes
  -> one summed physical Stokes-adjoint RHS
```

The mantle-density transpose contributes the kernel label
`mantle-density geoid harmonic`. The CBF helper emits separate surface and
CMB density/viscosity labels. The manager uses no physical adjoint solve for
D, and exactly one for every objective with either topography branch; it sums
the top and bottom RHS contributions before that solve.

Pressure normalization is stateful in the forward solve. Before the physical
adjoint solve, the manager sets its stored forward pressure-normalization
adjustment to zero, then restores the saved forward solution, RHS,
linearization point, constraints, and adjustment on both normal and exception
paths. This prevents a forward pressure gauge from leaking into an adjoint
initial guess or a later objective.

## Operator conventions and contrast diagnostic

The CBF transpose uses the local support-point mantle density, the configured
density above (surface) or below (CMB), local gravity magnitude, and the
selected boundary's outward normal. Its pressure correction is constructed
with the same selected-boundary area and mean-pressure convention as the
forward dynamic-topography calculation.

The geoid harmonic scalings use the legacy boundary mean/reference density,
the configured density above/below, the surface/reference gravity, and the
appropriate radial factor. Consequently the CBF denominator and geoid
contrast are not the same discretized quantity. No exact density-contrast
cancellation is assumed or forced.

## Canonical checks

Run a fixed-mesh smoke case with, for example:

```sh
build-release/aspect-release benchmarks/adjoint_geoid_density/g5_surface_cmb_m0.prm
```

Expected solve counts are one forward / zero physical adjoint for D, and one
forward / one physical adjoint for every file with surface and/or CMB enabled.
Use the existing direct-control finite-difference wrapper from the 3-D
dynamic-topography benchmark for centered raw-density and raw-viscosity
directions. Sweep at least three epsilons and only report relative errors in
a resolved, non-near-zero region.

The permanent fast acceptance set is the three harmonic transpose identities,
top and CMB coefficient-to-FE seed dualities, and fixed-seed top+CMB RHS and
density/viscosity-kernel superposition. The permanent medium set is the
three branch configurations above plus `adjoint_geoid_density.prm`. The four
`m=1` files prevent regressions in cosine/sine ordering, support-point
projection, radial factors, and CMB signs.

Recorded G0–G4 fixed-mesh references are: CMB harmonic transpose relative
error below `1e-15`; CMB FE dualities for m=0, cosine, and sine agree to the
printed precision; and fixed-seed top+CMB RHS superposition has relative
error about `1.0e-16`. Robust CMB m=1-sine FD closure at cell 145 was
`2.13e-5` for density and `9.28e-6` for raw viscosity. These values are
baselines, not tolerances for bitwise comparison.

## Scope limits

G5 excludes self-gravitation, coupled surface/CMB geoid feedback,
internal-interface deformation, moving-geometry derivatives, and
self-gravity-matrix or self-gravitating boundary-density derivatives. It
differentiates the observation operator on the realized mesh only, not AMR
decisions or mesh selection. It also does not add point-observation
synthesis. MPI comparisons may differ in their final digits because of
reduction order; compare coefficients and directional derivatives
numerically, not bitwise.

## Self-gravitating production baseline

The self-gravitating objective is selected independently with:

```text
subsection Adjoint
  set List of objectives = self-gravitating geoid
  subsection Self-gravitating geoid
    ...
  end
end
```

It solves the SG0 coupled degree-wise system rather than modifying the G5
`geoid` path. Its public controls are the minimum/maximum degrees
(`l_min >= 2`), observed cosine/sine coefficient lists for surface/CMB
topography and geoid, and shared nonnegative degree/order weights.  The
fixed external `Surface density above` and `CMB density below` entries must
match the required `Postprocess/Dynamic topography` density-above/below
settings; a mismatch fails at parse/use time rather than silently changing a
CBF or matrix convention.

The final raw-density decomposition is:

```text
self-gravitating surface density moment
self-gravitating CMB density moment
dynamic topography surface objective
dynamic topography CMB objective
incompressible Stokes volume
self-gravity matrix density term
```

For raw viscosity, only the two CBF terms and `incompressible Stokes volume`
are present; the coupled 4x4 matrix has no viscosity derivative.

| Purpose | Canonical asset |
| --- | --- |
| SG forward, axisymmetric/non-axisymmetric | `sg0_coupled_m0.prm`, `sg0_coupled_m1.prm` |
| Production SG density/viscosity FD and one-adjoint invariant | `sg2_mixed.prm` plus the documented cellwise wrapper directions |
| Non-axisymmetric production SG density FD | `sg2_nonaxis_surface_geoid_m1.prm` |
| Four coupled basis-seed architecture regressions | `sg2_basis_*.prm` |
| Realized-mesh SG smoke | `sg4_amr_smoke.prm` |
| Operator ladder (SG0–SG3) | `build-release/aspect-release --test 'Self-gravitating*'` |
| Non-self-gravitating baseline | `g5_surface_cmb_m0.prm` |

The production reverse chain is `A_l^T z=q_x`, followed by the existing
top/CMB CBF and moment transposes, one summed physical Stokes adjoint RHS,
and the separate matrix-density contraction `-z^T(delta A_l)x`.  It does not
differentiate gravity, radii, geometry, moving interfaces, external-density
parameters, AMR decisions, log viscosity, or time dependence.
