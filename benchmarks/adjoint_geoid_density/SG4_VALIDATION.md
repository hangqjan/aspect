# SG4 final self-gravity integration validation

SG4 makes the already implemented SG0--SG3 chain the production
`self-gravitating geoid` adjoint objective. It intentionally adds no new
gravity, geometry, or moving-boundary physics.

## Production interface and derivative convention

The objective is selected with `List of objectives = self-gravitating geoid`.
It uses a single coupled degree-wise 4-by-4 solve, with harmonic state order
`[h_s, h_c, N_s, N_c]`. `Minimum spherical harmonic degree` is restricted to
at least two. The public fixed-external values `Surface density above` and
`CMB density below` are checked against the required Dynamic topography
postprocessor values. They are external constants, rather than inversion
controls. A deliberate surface-density mismatch was rejected with the
explicit required-Dynamic-topography-density diagnostic; no ASPECT/MPI
process remained after that expected failure.

The complete raw-density kernel is the sum of the following separately named
contributions:

```text
self-gravitating surface density moment
self-gravitating CMB density moment
dynamic topography surface objective
dynamic topography CMB objective
incompressible Stokes volume
self-gravity matrix density term
```

The reverse path is the coupled spectral transpose, CBF and boundary-moment
transposes, one summed physical Stokes-adjoint right-hand side, and the
explicit matrix contraction `-z^T (delta A_l) x`. The matrix term includes
the fixed-geometry boundary mean-density convention. Viscosity differentiates
only through the physical CBF/Stokes path.

## Fixed-mesh production checks

All runs below used the Release executable and ran serially. The standard
model has 768 active cells and 28,690 degrees of freedom. Each production
case reports one forward and one adjoint Stokes solve.

| Case | Result |
| --- | --- |
| `sg2_mixed.prm` | PASS; 158 forward and 147 adjoint AMG iterations |
| `sg2_basis_surface_topography.prm` | PASS; `J = 0.5000000000000009`; 158/147 iterations |
| `sg2_basis_cmb_topography.prm` | PASS; `J = 0.5`; 158/151 iterations |
| `sg2_basis_surface_geoid.prm` | PASS; `J = 0.5`; 158/146 iterations |
| `sg2_basis_cmb_geoid.prm` | PASS; `J = 0.5`; 158/149 iterations |
| `sg2_nonaxis_surface_geoid_m1.prm` | PASS; `J = 0.5`; 158/100 iterations |
| `sg0_coupled_m0.prm`, `sg0_coupled_m1.prm` | PASS; 28,690 DoFs in both cases |
| `build-release/aspect-release --test 'Self-gravitating*'` | PASS; 269 assertions in 5 test cases |

The unit/operator ladder retains the SG0 coupled-forward residuals of
`2.62e-16` (axisymmetric) and `3.55e-16` (non-axisymmetric), as well as the
SG1 spectral, SG2 physical, and SG3 matrix-derivative identities.

## Centered finite differences

The documented cellwise material-model wrapper was used only for these
validation runs; it is not part of the production source. `sg2_mixed.prm`
was restored and rerun without that wrapper after the checks.

| Direction and centered perturbation | Finite difference | Integrated raw kernel | Relative difference |
| --- | ---: | ---: | ---: |
| surface-adjacent density cell 423, `+/- 0.1` | `99.97159579` | `99.9519` | `1.97e-4` |
| surface-adjacent density cell 423, `+/- 0.01` | `99.97159583` | `99.9519` | `1.97e-4` |
| surface-adjacent density cell 423, `+/- 0.001` | `99.97159607` | `99.9519` | `1.97e-4` |
| CMB-adjacent density cell 3, `+/- 0.01` | `38.23195771` | `38.2232` | `2.3e-4` |
| interior density cell 672, `+/- 1` | `-128.35704594` | `-128.374` | `1.3e-4` |
| raw viscosity cell 672, `+/- 1e18` Pa s | `3.41085965e-22` | `3.41999e-22` | `2.7e-3` |
| raw viscosity cell 672, `+/- 1e17` Pa s | `3.41089338e-22` | `3.41999e-22` | `2.7e-3` |
| non-axisymmetric surface density cell 508, `+/- 0.01` | `7.51951767` | `7.51929` | `3.0e-5` |

The boundary-localized density directions exercise the surface/CMB moment
and the SG3 matrix-density term in addition to the physical adjoint path.
The raw-viscosity directions verify that no spurious direct matrix derivative
is introduced.

## Mesh, gauge, resources, and regressions

`sg4_amr_smoke.prm` completed with one initial adaptive-refinement cycle. The
realized mesh has 796 active cells and 30,227 DoFs; its final forward/adjoint
solves used 193/146 AMG iterations. The preparatory solve on the initial mesh
is the normal AMR mesh-selection pass and is not an additional objective
adjoint.

The production path uses the existing manager pressure-gauge save/reset/
restore logic, unchanged from the validated G5 physical-adjoint path. No
SG-specific gauge state or debug output remains.

`/usr/bin/time -l build-release/aspect-release
benchmarks/adjoint_geoid_density/sg2_mixed.prm` completed normally in 5.72 s
elapsed (ASPECT reported 3.85 s). This macOS environment denied the
`kern.clockrate` query used by `time -l`, so it emitted no maximum-RSS value;
peak RSS is therefore **not available**, rather than estimated. Runs remained
bounded at the ordinary fixed-mesh problem size.

The single-rank regression `g5_surface_cmb_m0.prm` also passes at 28,690
DoFs. Multi-rank validation is **not available** in this host sandbox because
PRTE/Open MPI cannot create its local TCP listener (`Operation not permitted`);
this is an environment restriction, not an SG4 numerical result.

## Final acceptance

| Requirement | Status |
| --- | --- |
| production coupled forward/transpose and physical propagation | PASS |
| explicit matrix material derivative and density FD closure | PASS |
| raw-viscosity FD closure | PASS |
| non-axisymmetric coupled FD/ordering case | PASS |
| basis seeds, one-forward/one-adjoint invariant, and pressure gauge | PASS |
| AMR realized-mesh smoke | PASS |
| Release resource smoke | PASS, RSS measurement unavailable on this OS |
| MPI comparison | NOT AVAILABLE in sandbox |
| non-self-gravitating G5 regression | PASS |

Scope remains fixed-geometry, fixed-external-density, fixed-gravity/radius
linearization on the realized mesh. It excludes moving interfaces, geometry
or gravity derivatives, AMR-decision derivatives, time-dependent adjoints,
and point-observation synthesis.
