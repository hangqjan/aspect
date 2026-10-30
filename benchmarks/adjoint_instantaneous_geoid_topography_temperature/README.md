# Instantaneous self-gravitating geoid and dynamic-topography inversion

This controlled 3-D spherical-shell twin uses the steady
`no Advection, adjoint Stokes` workflow. The reference temperature contains a
localized thermal anomaly; every inverse starts from the uniform 1600 K
background and reconstructs the initial (therefore instantaneous) temperature
field. Viscosity is fixed deliberately, so the experiment isolates the
temperature-to-density-to-buoyancy route.

The reference forward writes two native observation products:

* `results/reference/dynamic_topography/dynamic_topography_top.00000`;
* the coupled self-gravitating surface/CMB geoid coefficients in
  `results/reference/statistics`.

After the reference completes, extract its native coupled coefficients and
run the three inverses:

```bash
build-release/aspect-release benchmarks/adjoint_instantaneous_geoid_topography_temperature/reference.prm
python3 benchmarks/adjoint_instantaneous_geoid_topography_temperature/analysis/extract_self_gravitating_observation.py \
  --statistics benchmarks/adjoint_instantaneous_geoid_topography_temperature/results/reference/statistics \
  --output benchmarks/adjoint_instantaneous_geoid_topography_temperature/generated_self_gravitating_geoid_observation.prm
build-release/aspect-release benchmarks/adjoint_instantaneous_geoid_topography_temperature/inverse_dynamic_topography.prm
build-release/aspect-release benchmarks/adjoint_instantaneous_geoid_topography_temperature/inverse_geoid.prm
build-release/aspect-release benchmarks/adjoint_instantaneous_geoid_topography_temperature/inverse_combined.prm
```

`inverse_dynamic_topography.prm` consumes only the ASPECT-native surface
dynamic-topography output. `inverse_geoid.prm` uses only the coupled surface
and CMB geoid coefficients (zero topography field weights), and
`inverse_combined.prm` sums both functionals in the same reduced objective.

The two observables have different SI units and scales. The geoid-only input
uses a larger field weight to make its very small temperature gradient visible
to the trust-region solver; the combined input uses a smaller geoid weight to
balance the initial reduced-gradient norms of the two terms. Regenerate the
accepted-control comparison figure with:

```bash
conda run -n hang python benchmarks/adjoint_instantaneous_geoid_topography_temperature/analysis/plot_temperature_reconstructions.py \
  --root benchmarks/adjoint_instantaneous_geoid_topography_temperature/results \
  --output benchmarks/adjoint_instantaneous_geoid_topography_temperature/results/temperature_reconstruction_comparison.png
```
