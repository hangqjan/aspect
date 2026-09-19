# TD-GADOPT benchmark scaling and fidelity contract

## Scope

This document defines the corrected ASPECT reference configuration used before
the paused O9 sparse-replay qualification. The authoritative scaled coefficient
input is `benchmarks/adjoint_gadopt_mantle_convection/reference_forward.prm`;
the primary benchmark is its serial 32 by 32 refinement-5 overlay
`reference_forward_32.prm`. The original 16 by 16 input remains a smoke case.
Neither is a claim of discretization identity with the G-ADOPT tutorial.

## Scaling

The ASPECT case sets

```
rho = cp = H = alpha = g = eta = 1,  k = 1e-6.
kappa = k/(rho cp) = 1e-6.
tau = H^2/kappa = 1e6.
U = kappa/H = 1e-6.
P = eta*kappa/H^2 = 1e-6.
```

Consequently, the dimensional-to-nondimensional conversions are

```
t* = t/tau = 1e-6 t,
u* = u/U = 1e6 u,
p* = p/P = 1e6 p.
```

The configured ASPECT step is exactly `dt = 4`, so `dt* = 4e-6`. Eighty
accepted steps end at `t = 320` and `t* = 3.2e-4`.

With the usual thermal Rayleigh number,

```
Ra = rho alpha g H^3/(eta kappa) = 1e6.
```

With `x=Hx*`, `u=Uu*`, `t=tau t*`, and `p=Pp*`, division of the Stokes balance
by `eta*U/H^2` gives the nondimensional form (up to ASPECT's vertical-gravity
sign convention)

```
-div(2 eps(u*)) + grad(p*) = Ra T e_y,
Ra = rho alpha g DeltaT H^3/(eta kappa) = 1e6.
```

The temperature equation becomes

```
dT/dt* + u* . grad(T) - Laplacian(T) = 0.
```

Thus the selected time-rescaled ASPECT coefficients, rather than the rejected
`kappa=1, g=1e6` representation, produce the requested G-ADOPT-equivalent
thermal buoyancy coefficient.

## Fidelity table

| Component | G-ADOPT tutorial | ASPECT TD-GADOPT smoke benchmark |
| --- | --- | --- |
| Domain / mesh | 150 by 150 extruded quadrilaterals | primary: 32 by 32 box, global refinement 5; smoke: 16 by 16 |
| Velocity / pressure | CG2 / CG1 | Q2 / associated Q1 |
| Temperature | discontinuous Q2 (DQ2) | continuous Q1 |
| Initial-temperature control | CG1 | current reduced Q1-like control |
| Time integration | implicit midpoint | validated ASPECT BDF-family path |
| Timestep | `dt*=4e-6` | `dt=4`, hence `dt*=4e-6` |
| Trajectory | tutorial-specific implementation | dense, anchor spacing 1, retained |

The LBB (also called inf-sup) condition concerns the velocity-pressure pair.
It does not require the temperature field to use a particular degree; the Q1
temperature choice is an explicit, documented fidelity difference.

## Boundary and safety contract

Temperature is fixed to one at the bottom and zero at the top; side temperature
conditions remain natural. All walls are impermeable/free-slip through the
tangential-velocity boundary indicators. Runtime validation is serial, runs
only one build/test process group at a time, and terminates a small-fixture
ASPECT process if RSS grows monotonically beyond 2 GiB.

## Current stability qualification

The authoritative 16 by 16 smoke input is preserved unchanged. Its corrected
80-step Q1 run selected `dt = 4` at every transition but reached a small upper
overshoot at the terminal record, `max(T80) = 1.0064722257530734`. No
stabilization was added. The refinement-5 overlay `reference_forward_32.prm`
repeats the same physical and time configuration on 32 by 32 cells; its 80-step
serial run selected `dt = 4` throughout and retained nodal `0 <= T <= 1` over
all archived records. It is the stability-qualified source for subsequent truth
observations.

中文说明：本文档固定 O9 暂停期间的 TD-GADOPT 参考尺度。物理参数给出
`kappa=1e-6`、时间尺度 `tau=1e6`、速度尺度 `U=1e-6`，因此
`t*=1e-6t`、`u*=1e6u`；ASPECT 的 `dt=4` 正好对应无量纲 `dt*=4e-6`。
表格明确记录 G-ADOPT 与 ASPECT 的网格、温度离散和时间积分差异。LBB/
inf-sup 条件只约束速度-压力配对，并不要求温度必须为某一阶数。
