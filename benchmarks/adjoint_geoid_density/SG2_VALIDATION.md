# SG2 fixed-matrix physical transpose propagation

## Scope

SG2 connects the SG1 coupled spectral seeds to the existing physical adjoint
operators while deliberately treating the degree-wise coupled matrix as
fixed.  It does **not** include `-z^T (delta A_l) x`, boundary-density/
contrast derivatives in `A_l`, gravity/radius derivatives, moving interfaces,
log-viscosity, AMR-decision derivatives, or time dependence.

`SelfGravitatingGeoidObjective` is a coefficient-space quadratic test and
production-ready objective:

```text
J = 1/2 sum W (x-x_obs)^2,
x = (surface topography, CMB topography, surface geoid, CMB geoid).
```

It first applies the SG1 `A_l^-T` solve.  The four resulting physical-channel
seeds are then handled as follows:

```text
-z_s       -> surface reduced-stress harmonic transpose -> top CBF transpose
+z_b       -> CMB reduced-stress harmonic transpose     -> bottom CBF transpose
Gamma z_Nt -> surface volume-density-moment transpose
Gamma z_Nb -> CMB volume-density-moment transpose
```

The two CBF calls add to one `rhs`, so the objective manager executes one
physical Stokes adjoint solve, never one per boundary or spectral field.

## Exact physical interfaces

`assemble_boundary_reduced_stress_transpose()` is the reusable reverse of the
SG0 harmonic projection.  For a traction coefficient seed `q_tau`, it uses
the same real-harmonic convention and Gauss face quadrature as SG0 and forms

```text
q_tau dOmega = q_tau/r^2 dA,
q_h = q_tau (rho-rho_exterior) |g| / r^2.
```

Thus `q_h` is precisely the temperature-support-point FE seed expected by
the validated `DynamicTopographyCBFTranspose`, and its independently
accumulated pressure seed is `integral q_tau/r^2 dA`.  This preserves the
reduced-stress sign and mean-pressure transpose at both the surface and CMB;
no second CBF implementation was added.

`compute_self_gravitating_density_moment_transpose()` uses the exact SG0
volume quadrature, global degree-major real harmonic layout, and cell-average
kernel convention.  The two radial kernels are separate in code:

```text
I_t: r^l / R^(l+1)
I_b: Rc^l / r^(l+1)
```

The final kernel output keeps the requested decomposition:

```text
self-gravitating surface density moment / density
self-gravitating CMB density moment     / density
dynamic topography surface objective    / density, viscosity
dynamic topography CMB objective        / density, viscosity
incompressible Stokes volume            / density, viscosity
```

There is no explicit density-moment viscosity term and no log-viscosity
conversion.

## Serial validation

All runs used the established fixed 768-cell / 28,690-DoF spherical-shell
model and were launched serially.

`sg2_mixed.prm`, which has nonzero coupled seeds in all four solved fields,
completed with one 158-iteration forward Stokes solve and one 147-iteration
physical adjoint solve.  The native output contains all five contributions
listed above.

The `sg2_basis_*.prm` fixtures use SG0's native l=2 coefficients as their
observations except for one coefficient shifted by one.  Therefore each is an
exact unit `q_x` basis seed and has `J=0.5` (surface-topography output shows
`0.5000000000000009`, the remaining four show `0.5`):

| Fixture | Seed | Forward / adjoint iterations |
|---|---|---|
| `sg2_basis_surface_topography.prm` | `(1,0,0,0)` | 158 / 147 |
| `sg2_basis_cmb_topography.prm` | `(0,1,0,0)` | 158 / 151 |
| `sg2_basis_surface_geoid.prm` | `(0,0,1,0)` | 158 / 146 |
| `sg2_basis_cmb_geoid.prm` | `(0,0,0,1)` | 158 / 149 |
| `sg2_nonaxis_surface_geoid_m1.prm` | surface-geoid `l=2,m=1` cosine | 158 / 100 |

The `m=1` case is deliberately non-axisymmetric and exercises both the
harmonic-to-FE boundary assembly and the moment kernels.

For a frozen-matrix interior cell (active cell 672), the existing
benchmark-only additive material wrapper was used only through stdin-fed
temporary parameters.  Both top/bottom boundary densities remained printed
as 3300 kg/m3, so no boundary scalar entering `A_l` was perturbed.  Centered
differences of the complete SG2 objective gave:

| Control | Step | FD | summed direct + implicit kernel | Relative difference |
|---|---:|---:|---:|---:|
| raw density | 1 kg/m3 | -128.35704594 | -128.374 | 1.3e-4 |
| raw viscosity | 1e18 Pa s | 3.41085965e-22 | 3.41999e-22 | 2.7e-3 |

The visibly asymmetric one-sided density changes are normal at this step;
the centered derivative is the comparison.  This check is intentionally on
an interior control and therefore excludes SG3's matrix-material term.

## Regression

* SG0 axisymmetric: 768 cells, one 158-iteration solve, coupled residual
  `2.62e-16`.
* SG0 non-axisymmetric: 768 cells, one 252-iteration solve, residual
  `3.55e-16`.
* SG1 unit tests: all 199 assertions in 3 self-gravitating test cases pass.
* G5 combined non-SG regression: unchanged 768-cell, 158/146
  forward/adjoint pattern.
* `git diff --check`: pass.

MPI remains unavailable on this host because PRTE cannot bind its local TCP
listener before ASPECT starts.  No manager logic changed, and the G5
pressure-normalized regression therefore remains the pressure-gauge isolation
check.  AMR was not added: SG2 differentiates the fixed realized mesh only.

## Final status

```text
Top traction spectral -> FE transpose: PASS
Bottom traction spectral -> FE transpose: PASS
Top traction seed duality: PASS
Bottom traction seed duality: PASS
I_t density transpose: PASS
I_b density transpose: PASS
I_t transpose dot product: PASS
I_b transpose dot product: PASS
I_t + I_b kernel superposition: PASS
Top + bottom physical RHS superposition: PASS
One physical adjoint solve: PASS
Surface-topography basis seed propagation: PASS
CMB-topography basis seed propagation: PASS
Surface-geoid basis seed propagation: PASS
CMB-geoid basis seed propagation: PASS
Fixed-A_l density FD: PASS
Fixed-A_l raw-viscosity FD: PASS
Pressure-gauge isolation preserved: PASS
Non-axisymmetric SG2 case: PASS
SG0 forward preserved: PASS
SG1 spectral transpose preserved: PASS
G5 non-SG regression preserved: PASS
Matrix derivative -z^T(delta A)x: NOT IMPLEMENTED
SG2 status: PASS
Ready for SG3: YES
```
