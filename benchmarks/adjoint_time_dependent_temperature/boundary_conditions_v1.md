# Boundary Conditions v1: impermeable physical boundaries

The time-dependent adjoint v1 is defined for a closed physical domain:

```
u . n = 0 on every physical boundary.
```

The forward snapshot retains the accepted forward `AffineConstraints` object.
For the reverse solve the same graph is copied and every inhomogeneity is set
to zero. Consequently, affine entries and coefficients, hanging-node
relations, periodic relations, and curved free-slip no-normal relations are
preserved; only forward Dirichlet values are removed from the adjoint.

## Supported

- Stokes free slip;
- zero velocity / no slip;
- prescribed tangential velocity and time-dependent tangential plate velocity;
- fixed temperature;
- fixed state-independent heat flux and insulating temperature;
- fixed composition and zero-flux composition.

## Unsupported

- prescribed normal inflow/outflow;
- open or otherwise permeable boundaries;
- inflow-only composition boundary conditions;
- any advection boundary condition requiring `u . n != 0`, including its
  missing advection-transpose boundary terms.

The runtime v1 guard is in
`Adjoint::Manager::validate_time_dependent_temperature_setup()`. For `Geometry
model = box`, it uses the prescribed-velocity component mask and the known face
normal to reject an explicit normal component. That configuration-level check
also rejects the inflow-only composition diagnostic. Curved geometry plugins
can represent a tangential field with every Cartesian component, so a mask-only
guard cannot reliably classify them. Those configurations remain supported when
the user ensures that the supplied field is tangential.

`adjoint_bc_inflow_composition.prm` is an expected-failure diagnostic, not a
passing regression. Its old snapshot diagnostic proved that the active
composition constraint graph was replayed and the adjoint constrained DoFs were
homogeneous. Its failed Taylor test instead exposes the deliberately omitted
normal-flow advection transpose boundary terms.
