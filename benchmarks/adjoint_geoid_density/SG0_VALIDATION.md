# SG0 coupled self-gravity forward-operator audit

## Reference

The audited reference is `self_gravity_hack26` at
`3b763d4a13d778bb6acdd421131585fcec2c80fe` (`self gravitation`). The configured
remotes expose `feature/self_gravity` and `self_gravity_hack26`; no exact
`geoid_self_gravitation` branch is present. Nothing was merged or
cherry-picked from that branch.

## Reconstructed forward system

For each real degree/order component the reference solves

```text
x = (s, b, Nt, Nb)

[ Dt gs, 0,       -rhot gs, 0       ] [s ]   [-taut]
[ 0,     Db gs,   0,        -Db gs  ] [b ] = [+taub]
[ -G R Dt, -G Rc (Rc/R)^(l+1) Db, 1, 0 ] [Nt]   [G It]
[ -G R (Rc/R)^l Dt, -G Rc Db, 0, 1 ] [Nb]   [G Ib]
```

where `Dt=rhot-rho_above`, `Db=rho_below-rhob`,
`G=4*pi*big_g/(gs*(2*l+1))`, `It=integral rho*r^l/R^(l+1)*Y dV`, and
`Ib=integral rho*Rc^l/r^(l+1)*Y dV`. The reference’s reduced stresses are
`tau=-CBF_traction.dot(outward_normal)-boundary_mean_pressure`; their top and
bottom RHS signs are respectively negative and positive. `Nt` and `Nb` are
geoid-height terms because the gravitational potential scaling is already
divided by surface gravity in `G`.

## SG0 implementation

`SelfGravitatingGeoidOperator` is a shared postprocess-level, forward-only
class. Its inputs explicitly contain the two density moments, top/bottom
reduced stresses, radii, surface gravity, boundary means, and exterior/interior
densities. Its result explicitly exposes surface/CMB topography and
surface/CMB geoid coefficients. It validates `l_min>=2`, factorizes/inverts a
4x4 matrix once per degree, solves cosine and sine RHS separately, and records
the maximum normalized `A*x-f` residual.

`compute_self_gravitating_density_moments()` reuses the G5
`SphericalHarmonicCoefficients` layout and MPI reduction, with raw (not
Gamma-scaled) `I_t` and `I_b` radial kernels. The layout is degree-major,
`m=0..l`, separate cosine/sine arrays, ASPECT fully normalized real harmonics
with Condon--Shortley phase, and `dV` volume quadrature. It has no parallel
coefficient layout.

Degrees zero and one are rejected deliberately: the audited coupled reference
only defines its system for `l>=2`.

## Forward reduced-stress provider and integration

`ForwardBoundaryReducedStressProvider` is now the G5-compatible forward
provider. It does not duplicate CBF assembly: it reads the documented velocity
block of `DynamicTopography::topography_vector()`, which is the existing CBF
traction. At the historical reference branch's Gauss face quadrature it
projects, independently on each boundary,

```text
tau = - CBF_traction . outward_normal - boundary_mean_pressure
tau_lm = integral(tau Y_lm dA / radius^2).
```

The provider obtains both mean pressures from `BoundaryPressures`, preserves
the outward CMB normal convention, uses the shared G5 degree-major harmonic
layout, and reduces each coefficient over MPI. Dynamic topography's stored
temperature component and its backward-advection sign adjustment are never
read: the CBF traction velocity block is instantaneous and unchanged by that
adjustment. This matches the reference's coupled path, which bypasses the
ordinary backward-advection topography conversion.

`compute_self_gravitating_geoid_harmonics()` connects density moments,
boundary densities, the provider, and the degree-wise solver. The registered
forward-only `self-gravitating geoid` postprocessor exposes every solved
surface/CMB topography and geoid cosine/sine coefficient through statistics,
so a separately built reference executable can be compared coefficient by
coefficient without changing either operator.

The reference parses surface/CMB inclusion flags but actually solves both
interfaces. SG0 intentionally has no such ineffective flags and always solves
the coupled two-boundary system.

## Validation results

| Check | Result |
|---|---|
| Synthetic residual, l=2/3 matrix audit | **PASS** — 72 unit-test assertions; normalized residual tolerance `1e-10` |
| Synthetic cosine/sine isolation | **PASS** — l=2,m=0; l=2,m=1; l=3,m=2 are loaded in both components, and a bottom sine-only load leaves all cosine outputs zero |
| Limiting inputs | **PASS** — zero, top-only, bottom-only, and density-only inputs |
| Forward reduced-stress provider | **IMPLEMENTED** — existing DynamicTopography CBF traction plus both BoundaryPressures gauges |
| Canonical SG0 forward case | **PASS** — `sg0_coupled_m0.prm`: 768 cells, 28,690 DoFs, one 158-iteration Stokes solve, residual `2.62e-16` |
| Historical coefficient-level equivalence | **PASS** — independently instrumented `3b763d4a1` reference, matched 768-cell model; detailed results below |
| Non-axisymmetric integrated SG run | **PASS** — `sg0_coupled_m1.prm`: 768 cells, one 252-iteration solve, residual `3.55e-16`, and nonzero m=1 coefficients in statistics |
| G5 non-self-gravitating regression | **PASS** — `g5_surface_cmb_m0.prm`: 768 cells and the established 158/146 forward/adjoint solves |

## Historical-reference equivalence

The historical reference was built in the isolated worktree
`/private/tmp/aspect-sg-reference` at
`3b763d4a13d778bb6acdd421131585fcec2c80fe`. Only that worktree contains the
diagnostic patch and the translated fixture; neither was merged or copied into
the production source tree. The reference configuration was Release,
AppleClang 21 through `/opt/homebrew/bin/mpicxx`, deal.II 9.6.0, and one MPI
rank. Its bundled World Builder reported `USE_MPI=OFF`; the ASPECT reference
run itself was single-rank, matching the production comparison run.

The isolated fixture maps the current model directly: spherical shell
`R=6336000`, `Rc=3480000`; radial gravity `g_s=10`; the same simple material,
temperature, boundary conditions, refinement `(global, adaptive)=(1,0)`,
AMG tolerance, degree range `2..3`, and exterior/interior densities
`(0,9900)`. The only syntax translation is the historical solver spelling
`no Advection, single Stokes` for the current adjoint-Stokes one-forward-solve
configuration.

The historical diagnostic emitted every cosine/sine mode's
`tau_t,tau_b,I_t,I_b,s,b,N_t,N_b` at 17 significant digits. Current SG0 emits
the matching raw inputs, matrix scalars, and solved fields in `statistics`.

| Quantity | Maximum absolute difference | Maximum raw relative difference | Result |
|---|---:|---:|---|
| `tau_t` | 0 | 0 | **PASS** |
| `tau_b` | 0 | 0 | **PASS** |
| `I_t` | `4.045805109811653` | `5.418354407429308e-1` | **PASS — cancellation scale** |
| `I_b` | `4.722160085750666` | `3.591323791672586e-1` | **PASS — cancellation scale** |
| `A_l`, `l=2,3` | 0 | 0 | **PASS** |
| `s` | `8.037214839362028e-11` | `1.255321395801131e-1` | **PASS — cancellation scale** |
| `b` | `1.214089630088962e-10` | `1.513885242040572e-2` | **PASS — cancellation scale** |
| `N_t` | `8.037214839361863e-11` | `2.043707975924164e-1` | **PASS — cancellation scale** |
| `N_b` | `1.214089630088954e-10` | `3.284167979605641e-2` | **PASS — cancellation scale** |

`I_t/I_b` agree at the physically relevant scale: for example the historical
and current `l=2,m=0` surface moments are respectively
`-7.128923599008513e10` and `-7.128923598739932e10`. The apparent large raw
relative errors occur only in symmetry-forbidden non-axisymmetric remnants
whose independently summed values are O(1--10) after cancellation of
O(1e10) quadrature contributions. They produce at most `1.22e-10` difference
in any solved coefficient. This is numerical cancellation, not a harmonic,
traction, sign, matrix, or convention discrepancy.

Representative historical/current values are:

| Mode | `tau_t` | `tau_b` | `I_t` | `I_b` | `s` | `b` | `N_t` | `N_b` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `l=2,m=0,cos` historical | `1.2152735821150946e5` | `-2.2344571036369995e4` | `-7.128923599008513e10` | `-2.9369154881833618e10` | `-7.778374207156872` | `-2.690930189968288` | `-4.095726988626291` | `-2.352376083356621` |
| `l=2,m=0,cos` current | identical | identical | `-7.128923598739932e10` | `-2.936915488190369e10` | `-7.778374207086478` | `-2.690930189958084` | `-4.095726988555898` | `-2.352376083346417` |
| `l=2,m=1,cos` historical | `3.7123063262597315e-4` | `6.572012667492456e-4` | `19.67210439827227` | `21.79554806834102` | `-1.5428036573280683e-8` | `1.4137743327029367e-8` | `-4.178623463402738e-9` | `4.18014837628322e-9` |
| `l=2,m=1,cos` current | identical | identical | `17.53539605216045` | `26.51770815409169` | `-1.547130326812809e-8` | `1.425915229003826e-8` | `-4.221890158250146e-9` | `4.301557339292115e-9` |
| `l=2,m=1,sin` historical/current | `-2.1291182184022728e-4` / identical | `1.6258656830693892e-5` / identical | `43.73785629372251` / `40.72482248712645` | `8.243110398003637` / `7.80773325306258` | `1.1320220186943186e-8` / `1.1239848038549566e-8` | `2.5740583339144986e-9` / `2.548344674809146e-9` | `4.868346797845406e-9` / `4.787974649451788e-9` | `2.3277150486009547e-9` / `2.3020013894956013e-9` |

The historical maximum normalized residual is `2.2394302518952343e-16`; the
current value is `2.62e-16`. The same scalar inputs construct both `A_2` and
`A_3` exactly. Therefore the differences are fully localized to harmless
density-projection cancellation before the shared equations are solved. No
historical convention defect was found, and no production mathematics was
changed to chase those remnants.

## Final status

```text
Historical reference commit: 3b763d4a13d778bb6acdd421131585fcec2c80fe
Historical executable: PASS
Historical tau_t equivalence: PASS
Historical tau_b equivalence: PASS
Historical I_t equivalence: PASS (cancellation-scale remnants localized)
Historical I_b equivalence: PASS (cancellation-scale remnants localized)
Historical A_l equivalence: PASS
Historical s coefficient equivalence: PASS
Historical b coefficient equivalence: PASS
Historical N_t coefficient equivalence: PASS
Historical N_b coefficient equivalence: PASS
Historical 4x4 residual: PASS
Current synthetic 4x4 residual: PASS
Current non-axisymmetric SG benchmark: PASS
Validated G0-G5 paths preserved: PASS
Historical differences fully localized: YES
SG0 status: PASS
Ready for SG1: YES
```

This is a gate result only. SG1 has not been started.

The G5 non-self-gravity operator remains separate:
`N=A_rho*rho+B_t*h_t+B_b*h_b`, whereas SG uses the coupled degree-wise system
above. No transpose, self-gravitating objective, Stokes-adjoint change, or
matrix/material derivative was added.
