# SG1 coupled spectral-transpose validation

## Scope

SG1 adds only the fixed-matrix spectral reverse solve to
`SelfGravitatingGeoidOperator`:

```text
A_l^T z = q_x,
q_x = (q_s, q_b, q_Nt, q_Nb).
```

`apply_transpose()` takes arbitrary four-field harmonic seeds and returns the
four named components of `z`, plus named physical RHS-channel seeds:

```text
dJ/d tau_t = -z_s
dJ/d tau_b = +z_b
dJ/d I_t   = Gamma_l z_Nt
dJ/d I_b   = Gamma_l z_Nb.
```

The shared `matrix_for_degree()` is the only construction path for the SG0
matrix. Forward solves factorize `A_l` once per degree; SG1 factorizes `A_l^T`
once per degree. Neither path factorizes separately by order or real channel.

## Deliberate boundary

SG1 does not apply a CBF-traction transpose, a density-moment transpose, a
physical Stokes adjoint, a material kernel, parameter derivatives, or
`-z^T (delta A_l) x`. It holds `A_l` fixed and ends at the four spectral
physical-channel seeds above.

## Tests

`unit_tests/self_gravitating_geoid_operator.cc` uses deterministic nonzero
four-component `q_x` and physical perturbations for every retained real mode:

| Degree | Orders | Channels | Identity checked |
|---|---|---|---|
| 2 | 0, 1, 2 | cosine, sine | `q_x^T delta_x = seed_tau_t delta_tau_t + seed_tau_b delta_tau_b + seed_It delta_It + seed_Ib delta_Ib` |
| 3 | 0, 1, 2, 3 | cosine, sine | same |

For all 14 degree/order/channel cases, the tests independently reconstruct
`A_l`, verify every component of `A_l^T z-q_x` to a scale-aware `1e-10`
tolerance, verify the bilinear identity to the same tolerance, and directly
check each of the four RHS-seed signs/scalings. The SG test filter reports:

```text
All tests passed (199 assertions in 3 test cases)
```

## Regression

The serial Release G5 `g5_surface_cmb_m0.prm` regression remains unchanged:
768 active cells, 28,690 DoFs, and the established 158/146 forward/adjoint
Stokes solves. `git diff --check` passes.

## Status

```text
Generic A_l^T spectral solve: PASS
Arbitrary four-field q_x: PASS
RHS tau/I seed signs and Gamma scaling: PASS
Direct transpose residual: PASS
Bilinear transpose identity: PASS
CBF/density/physical-adjoint propagation: NOT IMPLEMENTED (out of SG1 scope)
delta A material/parameter term: NOT IMPLEMENTED (out of SG1 scope)
G0-G5 regression: PASS
SG1 status: PASS
```

No subsequent SG stage was started.
