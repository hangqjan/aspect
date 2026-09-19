# Cartesian Kernel Grid Scratch

Untracked scratch tooling for plotting a cellwise adjoint/model kernel next to a finite-difference kernel on a 2D Cartesian grid.

Quick smoke plot from an existing 4x4 FD-check run. For FD-check files, the left panel defaults to `full_control_gradient_derivative`, which includes direct objective terms and is the column that should match `primary_finite_difference_derivative`.

```bash
python3 scratch/adjoint_kernel_grid/plot_cartesian_kernel_grid.py \
  --fd-checks-file build-release/output-adjoint_dynamic_topography_volume_term_smoke/adjoint_finite_difference_checks_rank_00000.txt \
  --control viscosity \
  --cells-per-side 4 \
  --output scratch/adjoint_kernel_grid/viscosity_4x4.png \
  --error-output scratch/adjoint_kernel_grid/viscosity_4x4_relative_error.png
```

The main figure now annotates:

- paired cell count `n`
- Pearson correlation `r`
- L2 relative error
- median and max pointwise relative error
- max absolute error

For the first 16x16 Cartesian experiment, use:

```bash
cd build-release
./aspect-release ../scratch/adjoint_kernel_grid/adjoint_dynamic_topography_2d_box_gaussian_16x16.prm
```

That scratch prm uses a smooth Gaussian initial temperature anomaly:

```text
T(x,y) = 1600 + 100 exp(-((x-900000)^2 + (y-560000)^2) / (2 * 90000^2))
```

and increases the Cartesian mesh to `Initial global refinement = 4`, i.e. 16x16 cells.

Then plot the result:

```bash
python3 scratch/adjoint_kernel_grid/run_all_cell_fd_validation.py \
  --aspect build-release/aspect-release \
  --plugin build-release/adjoint_dynamic_topography_2d_spherical_plugin/libadjoint_dynamic_topography_2d_spherical.release.so \
  --controls viscosity \
  --cells-per-side 16
```

This runs the baseline, then every active cell with negative and positive perturbations, writes a CSV, the original 1:1 validation figure, and one pair of cell heatmaps per control. Use `--controls density,viscosity` to run both controls. For a quick smoke test, add `--cells 0`.

Default outputs:

```text
scratch/adjoint_kernel_grid/all_cell_fd_validation.csv
scratch/adjoint_kernel_grid/all_cell_fd_validation_1to1.png
scratch/adjoint_kernel_grid/all_cell_fd_validation_viscosity_heatmaps.png
scratch/adjoint_kernel_grid/all_cell_fd_validation_density_heatmaps.png
```

Use `--resume` to reuse completed case outputs inside `build-release/fd_validation_all_cells_2d_box_gaussian_16x16`.
