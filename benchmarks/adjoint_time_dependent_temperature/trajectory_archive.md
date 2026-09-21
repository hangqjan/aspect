# Adjoint trajectory archive

The time-dependent adjoint stores its forward trajectory in
`<Output directory>/adjoint_trajectory/`. This is a fixed-mesh replay archive,
not an ASPECT restart checkpoint. MPI restart assumes the forward and reverse
sweeps use the same rank count, mesh, DoF numbering, and DoF ownership
partition. It deliberately does not support repartitioning or a different rank
count.

The physical backend is a shared MPI-IO archive: a root-published `manifest`
and finite `trajectory_000000.bin`, `trajectory_000001.bin`, … segments. A
logical accepted step remains the public unit of `store`, `load`, and cleanup.
Each rank serializes only its own payload; an `MPI_Exscan` byte prefix selects
its range, and `MPI_File_write_at_all` writes that range directly to the shared
segment. No trajectory values are gathered to rank zero. The manifest records
schema, storage mode, timestep metadata, segment/range, rank count, rank-local
ranges and sizes, and ownership fingerprints. It is the only replay authority:
bytes from a failed append are unlisted and never valid records.

## Dependency audit

| Reverse dependency | Source | Archive treatment |
|---|---|---|
| Current temperature | `solution`, `current_linearization_point` | Stored once in record `n` |
| Current compositions | `solution`, `current_linearization_point` | Stored once in record `n` |
| Current velocity/pressure | Stokes and advection assembly | Recovered from `T,C` in `minimal`; stored once in `stokes cached` |
| `q_(n-1)`, `q_(n-2)` fields and velocities | BDF2, extrapolated velocity, entropy viscosity | Loaded by record index into a three-state reverse cache |
| `time`, `dt`, prior `dt` | BDF coefficients and time-dependent BCs | Global metadata table; prior `dt` is `metadata[n-1]` |
| Current constraints | Stokes/advection assembly | Rebuilt from fixed mesh, time, and boundary managers; never archived |
| Stokes/advection linearization points | Picard assembly | Rebuilt from indexed states under the v1 `rho(T,C)`, `cp(T)`, `eta(T,C)` restrictions |
| Material outputs and entropy-viscosity data | Assembly/EV replay | Recomputed |
| Matrices, preconditioners, solver state | Linear solves | Rebuilt; never archived |

The supported configuration rejects scheduled or periodic mesh refinement.
Initial refinement before the forward trajectory is allowed because all stored
records share the resulting fixed mesh.

## Modes

```text
subsection Adjoint
  set Trajectory storage mode = minimal
end
```

`minimal` stores `T,C` only. `stokes cached` also stores `u,p`:

```text
subsection Adjoint
  set Trajectory storage mode = stokes cached
  set Run trajectory Stokes recovery diagnostic = true
end
```

The optional diagnostic samples the first, middle, and final records, recovers
Stokes from archived `T,C`, and compares velocity and gauge-normalized pressure
to the cached forward result. The tolerance is configured by
`Trajectory Stokes recovery tolerance`.

## Shared-file layout

`Trajectory records per segment` controls the finite shared-file grouping and
defaults to `4`:

```text
subsection Adjoint
  set Trajectory records per segment = 4
end
```

The value is part of the manifest contract and must be unchanged when a
reverse restart reopens the archive. It changes file count, not logical replay:
the reverse algorithm continues to address records only by `store(step)`,
`load(step)`, `has_record(step)`, and `reclaim(...)`. Each record contains a
contiguous sequence of rank-local byte ranges. The manifest names the segment,
record offset and size, and every rank's offset, size, ownership fingerprint,
and field-byte breakdown.

## Retention and cleanup policy (v1)

The v1 policy is **explicit retention, followed by explicit cleanup**. The
archive is created at the first accepted forward state in
`<Output directory>/adjoint_trajectory/`; it must remain intact through the
whole reverse sweep and any optional recovery diagnostic.

Archive lifetime is selected with `Trajectory cleanup policy`. `safe
incremental` deletes only records made obsolete by a fully published reverse
checkpoint. At `next_reverse_step = n`, records `[n,last]` are obsolete while
the next operation still needs `n-1,n-2,n-3`. Logical unavailability is first
published in the manifest. Rank zero then removes a shared segment only when
every record in it is unavailable; segments are never compacted. An interrupted
cleanup can therefore leave an extra reclaimable segment, but cannot publish a
record that names deleted data. `on success` retains every record
through the reverse sweep and deletes the archive collectively after the final
gradient is written; `keep` retains it after success. No policy deletes archive
data on an exception or manager destruction. The archive is not a cross-layout
replay artifact: a later process may use it only with the same MPI partitioning
and DoF numbering.

Use a fresh output directory for every ASPECT launch. Before reusing an output
directory, first verify that the preceding run has exited and that its
gradients, screen output, and any failure evidence have been retained as
needed. Then remove **only** that run's exact
`<Output directory>/adjoint_trajectory/` subdirectory. Do not remove the
parent output directory or use a broad wildcard. This prevents a later run
from confusing stale shared segments or a stale manifest with its own archive.

For failed or interrupted runs, retain the archive until diagnosis is complete;
then apply the same exact-subdirectory cleanup if desired. Safe-incremental
recovery repeats its committed deletion frontier before resuming the reverse
sweep, so a crash during deletion is safe to restart. A partially written
`adjoint_restart/checkpoint_*.tmp` directory is never a valid restart target;
only the published `adjoint_restart/latest` generation is selected.

## Measurements

The simulator reports per-mode field-byte totals, mean bytes per record,
metadata/cache object memory, and `VmPeak`/`VmRSS` at archive completion and
after the reverse sweep. MPI safe-incremental cleanup additionally emits one
`ASPECT_ADJOINT_MPI_CLEANUP` line per rank with the reverse frontier, retained
record count, rank-local deleted records/files, and reclaimed bytes. The fixed
timer section names are:

```text
Adjoint trajectory write
Adjoint trajectory read
Adjoint trajectory Stokes recovery
Adjoint trajectory state reconstruction
Adjoint trajectory reverse sweep
```

Enable ASPECT's `timing statistics` and `memory statistics` postprocessors in
the benchmark input to retain the global timer and standard object-memory
statistics alongside these archive-specific reports. Compare the two modes on
the same forward trajectory; no sparse checkpointing, recomputation windows,
Revolve, AMR replay, lossy compression, or cross-job replay is included here.

The shared-MPI backend's serial/MPI, lifecycle, append-failure, and 1001-record
file-count results are retained in
[`shared_mpi_trajectory_validation.md`](shared_mpi_trajectory_validation.md).
