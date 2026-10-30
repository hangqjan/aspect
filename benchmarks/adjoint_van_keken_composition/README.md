# van Keken terminal-composition inverse benchmark

This is a twin experiment based on the smooth van Keken thermochemical
cookbook case. `reference_forward.prm` uses the classic 128-by-128 initial
mesh and final time, and writes its final `chemical` field as ASPECT
structured ASCII data. `inverse_initial_composition.prm` reads that exact file
as `Adjoint / Composition field / Observation file` and optimizes the initial
`chemical` field using the native ROL reduced-space workflow.

Build the benchmark-local exporter against the current ASPECT build:

```bash
cmake -S benchmarks/adjoint_van_keken_composition/plugin \
      -B build-release/van-keken-terminal-composition-plugin \
      -DAspect_DIR=$PWD/build-release
cmake --build build-release/van-keken-terminal-composition-plugin -j4
```

Run the reference on one MPI rank. The exporter deliberately overwrites its
file after each output event, so after a successful run it contains the final
accepted composition field:

```bash
build-release/aspect-release benchmarks/adjoint_van_keken_composition/reference_forward.prm
```

Then run the inverse (serially or with MPI). The structured observation file
is read independently by ASPECT's normal `AsciiDataInitial` implementation;
the exporter itself is not loaded by the inverse input.

```bash
mpirun -np 2 build-release/aspect-release \
  benchmarks/adjoint_van_keken_composition/inverse_initial_composition.prm
```

`Output accepted controls = true` in the inverse input writes the zero control
and every accepted initial-composition control to
`output-adjoint-van-keken-inverse/diagnostics/accepted_controls/`. Make the
G-ADOPT-style recovered-field and error panels with:

```bash
python3 benchmarks/adjoint_van_keken_composition/analysis/plot_initial_composition_iterations.py \
  --output-dir output-adjoint-van-keken-inverse
```

This creates `initial_composition_iterations.png` and
`initial_composition_error.png` under `diagnostics/plots`. The reference panel
uses the exact analytic initial composition from `reference_forward.prm`; it
does not depend on a separate visualization output at time zero.

For a resolution study, `reference_forward_ref5.prm` and
`reference_forward_ref6.prm` retain the same smooth field, physics, and final
time while selecting initial global refinement 5 and 6. Their matching inverse
inputs are `inverse_initial_composition_ref5.prm` and
`inverse_initial_composition_ref6.prm`, respectively. Each resolution has an
independent reference and inverse output directory.

The forward exporter is intentionally benchmark-local and serial because it
evaluates a deterministic uniform grid. It is not a general VTU/PVTU reader or
writer. Its `# POINTS: Nx Ny` format is already the documented ASPECT
full-field observation format, which is why the inverse needs no custom file
reader.
