# SG3 coupled-matrix raw-density derivative

## Matrix dependency audit

The audited SG0 construction for each degree is

```text
A(0,0) = Dt gs                 A(0,2) = -rho_t gs
A(1,1) = Db gs                 A(1,3) = -Db gs
A(2,0) = -Gamma R Dt           A(2,1) = -Gamma Rc (Rc/R)^(l+1) Db
A(3,0) = -Gamma R (Rc/R)^l Dt  A(3,1) = -Gamma Rc Db
```

where `Dt=rho_t-rho_above`, `Db=rho_below-rho_b`, and
`Gamma=4*pi*G/(gs*(2l+1))`.

| Matrix quantity | SG3 treatment |
|---|---|
| `rho_t`, `rho_b` boundary mantle means | differentiated raw-density controls |
| `Dt`, `Db` | differentiated through their production definitions |
| `rho_above`, `rho_below` | fixed external parameters |
| `gs`, `R`, `Rc`, `Gamma` | fixed gravity/geometry quantities |

In particular, the bottom contrast derivative is negative:
`d Db/d rho_b = -1`.  The explicit `-rho_t gs` entry and the top contrast
entries are both included, so top-density dependence is not double counted or
silently dropped.

## Implementation

`SelfGravitatingGeoidOperator` now exposes the two exact per-degree
`dA/d rho_bar` matrices and accumulates, across every degree/order and real
cosine/sine component,

```text
dJ/d rho_t = -sum z^T (dA/d rho_t) x
dJ/d rho_b = -sum z^T (dA/d rho_b) x.
```

No inverse or factorization derivative is used.

`BoundaryDensities` defines the forward means as area-weighted face-density
integrals.  Its transpose follows that exact discretization: `top`/`bottom`
boundary IDs; `introspection().face_quadratures.temperature`; face `JxW`;
division by the globally MPI-reduced boundary area; and cell-average kernel
storage.  Geometry and area variation are held fixed.  The resulting new,
separate raw-density contribution is named:

```text
self-gravity matrix density term
```

It is added once, independently of the two density moments, CBF direct
terms, and implicit Stokes kernel.  There is no matrix-viscosity contribution.

## Operator validation

`unit_tests/self_gravitating_geoid_operator.cc` now checks, for both top and
bottom means at degrees 2 and 3:

* every entry of the production `dA` against an independent construction;
* centered finite differences of `q^T x(A)` with RHS fixed, using both
  cosine and sine channels;
* linear superposition of independently seeded matrix contractions.

The Release self-gravity test filter reports **269 assertions in 5 cases**.

## Physical fixed-mesh finite differences

All integration runs were serial on the validated 768-cell / 28,690-DoF
model.  The benchmark-only additive material wrapper was supplied only via
temporary stdin parameters; no generated or production source was changed.

| Case | FD | complete density kernel | Relative difference |
|---|---:|---:|---:|
| Interior cell 672, density step 1 | -128.35704594 | -128.374 | 1.3e-4 |
| CMB-adjacent cell 3, density step 0.01 | 38.23145771 | 38.2232 | 2.2e-4 |
| Surface-adjacent cell 423, density step 0.01 | 99.97159583 | 99.9519 | 2.0e-4 |
| Non-axisymmetric `l=2,m=1` surface-geoid, cell 508, density step 0.01 | 7.51951767 | 7.51929 | 3.0e-5 |
| Interior cell 672, raw viscosity step 1e18 Pa s | 3.41085965e-22 | 3.41999e-22 | 2.7e-3 |

The matrix kernel is exactly zero at interior cell 672.  At the selected CMB
and surface cells its integrated values are respectively `-9.67288785e-6`
and `7.61150812e-6`, confirming its boundary localization and opposite
mean-density chains.  Matrix contractions also pass independent operator
superposition tests before this physical chaining is applied.

Raw viscosity is unchanged from SG2 because `A_l` has no viscosity input.
Gravity, radii, geometry, external densities, AMR decisions, and log
viscosity remain out of scope.

## Regression

* SG0 m=0: 768 cells, one 158-iteration solve, residual `2.62e-16`.
* SG0 m=1: 768 cells, one 252-iteration solve, residual `3.55e-16`.
* SG1/SG3 operator tests: 269 assertions in 5 cases pass.
* SG2 mixed: one 158-iteration forward plus one 147-iteration adjoint solve.
* G5 combined non-SG: unchanged 158/146 forward/adjoint pattern.
* `git diff --check`: pass.

MPI remains unavailable because PRTE cannot bind its local TCP listener before
ASPECT starts.  No manager code changed; established pressure-gauge isolation
is therefore preserved.  AMR is intentionally differentiated only on a
realized mesh and was not part of this fixed-mesh acceptance suite.

## Final status

```text
A_l density dependencies audited: YES

Independent dA entry tests: PASS

-z^T(dA)x operator-level FD: PASS

Top boundary mean-density transpose: PASS

CMB boundary mean-density transpose: PASS

Interior perturbation matrix term: ZERO

Surface-adjacent density matrix derivative: PASS

CMB-adjacent density matrix derivative: PASS

Top + CMB matrix-term superposition: PASS

Full physical density FD: PASS

Non-axisymmetric full density FD: PASS

Raw-viscosity FD preserved: PASS

Matrix viscosity derivative: ZERO

Gravity/radius derivatives: NOT IMPLEMENTED

SG0 forward preserved: PASS

SG1 spectral transpose preserved: PASS

SG2 fixed-A_l validation preserved: PASS

G5 non-SG regression preserved: PASS

MPI: NOT AVAILABLE

SG3 status: PASS

Ready for SG4: YES
```
