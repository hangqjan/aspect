# Adjoint restart and trajectory lifecycle v1

This checkpoint is deliberately separate from ASPECT's ordinary forward
restart checkpoint. The forward checkpoint resumes `Y_n -> Y_(n+1)`; this one
resumes an already-started reverse recurrence.

## Reverse-state audit

| State | Checkpointed | Reconstructible | Reason |
|---|---:|---:|---|
| Current, previous, previous-previous temperature adjoints | yes | no | BDF and stabilization transpose can seed two earlier levels. |
| The same three levels for every composition adjoint | yes | no | Temperature material-prefactor and composition transpose modify them. |
| Current, previous, previous-previous Stokes-adjoint RHS | yes | no | Velocity/stabilization transpose accumulates RHS before the Stokes transpose. |
| Three-state forward cache | no | yes | Rebuilt from retained trajectory records. |
| Constraints, matrices, preconditioners, material outputs, entropy-viscosity intermediates | no | yes | Reconstructed from fixed mesh, recorded state and time. |
| Terminal objective values | yes | no | Preserves reported values without reevaluating terminal objectives. |
| Initial gradients | no, until completion | yes | They are the final pending reverse state. |
| `KernelRepository` | no | not accumulated | The current transient sweep does not add kernels during reverse. |
| Plugin-hidden reverse scalar state | no known state | n/a | The audited calculators have no persistent reverse scalar state. |

`next_reverse_step` is the next seed index to process. History level zero is
that index; levels one and two are its previous and previous-previous levels.
After processing step `r`, the checkpoint therefore stores
`next_reverse_step = r - 1`. One operation can only modify these three pending
levels.

## MPI restart contract

The v1 MPI implementation is validated for a new `mpirun` with the **same
MPI rank count, mesh, DoF numbering, and ownership partition** as the run that
wrote the archive. Every rank writes and restores only its own owned reverse
state. Restore checks the stored process count and the owned global DoF indices
in every rank-local vector payload.

It does not support a different rank count, repartitioning, a changed mesh, or
forward replay across a different execution layout. The trajectory itself uses
shared MPI-I/O segments; the reverse checkpoint remains rank-local payloads
plus root-published metadata. MatrixFree data, GMG hierarchies, matrices,
transfers, constraints, and preconditioners remain runtime-only state and are
never checkpointed.

## Transaction and cleanup

For checkpoint generation `N`, rank zero creates `checkpoint_N.tmp`, all ranks
write and close their `rank_XXXXX.bin` payload, and a collective success check
must pass before rank zero writes shared metadata. Rank zero then renames the
directory to `checkpoint_N`, and only after that collective result succeeds
does it publish `latest`. Every stage is followed by a collective decision, so
a rank-local failure leads every rank down the same controlled termination
path. A `.tmp` directory is never restored.

Before `latest` is published, restart selects the prior generation even if a
newly named checkpoint directory exists. After `latest` is published, the new
generation is authoritative. This distinction is intentional: the durable
restart publication point is `latest`, not the earlier directory rename.

For a checkpoint with `next_reverse_step = n`, the next reverse operation
reconstructs records `n-1`, `n-2`, and `n-3` (where present). Safe-incremental
cleanup therefore retains `0..n-1` and may reclaim `[n,last]`. The checkpoint
stores this cleanup frontier. On resume, cleanup is repeated to the committed
frontier before any remaining record is read, so removal interrupted part-way
through is idempotent and cannot make the committed reverse state unrecoverable.

Under MPI, logical unavailability is first published in the shared trajectory
manifest. Rank zero then removes a `trajectory_XXXXXX.bin` segment only when
every record it contains is obsolete; there is no compaction and no rank-local
trajectory payload file. A missing fully obsolete segment is a safe no-op,
which makes repeated cleanup and recovery from a partial cleanup safe.

`Trajectory cleanup policy = safe incremental` is the default. `on success`
retains all temporary data until final gradients are written. `keep` retains
archive, manifest, and restart generations. No policy depends on a destructor,
so an exception retains recoverable data.

The direct-solver lifecycle controls first exercised serial safe-incremental
commit, cleanup, interruption, and recovery at reverse step 10. The MPI M2--M4
fixtures extend this to two ranks. They use a tiny 268-DoF case, a persistent
guarded `mpirun -n 2` session, and a 1 GiB RSS kill guard (with `vmmap`/`sample`
capture from 500 MiB). The observed peak guarded total RSS was about 288 MiB.

| Milestone | Validation | Result |
|---|---|---|
| M2 | Mid-sweep stop at `next_reverse_step = 10`, then a fresh 2-rank resume | Gradient is textually identical to the uninterrupted two-rank baseline. |
| M3 | A rank-1 injected failure after tmp creation, payload write, collective payload success, metadata write, directory commit, and `latest` publication | Pre-publication failures resume generation 3 at step 15; the post-`latest` failure resumes generation 4 at step 10. No run deadlocked or restored a `.tmp` directory. |
| M4 | Safe-incremental stop/resume and failures before cleanup, after rank-local deletion, before cleanup reduction, before manifest publication, and after manifest publication | Every fresh 2-rank resume completed with a textually identical gradient. |

For the fixture, the baseline gradient L2 norm is
`0.013480703099710863`, the terminal objective is
`0.004837524602356089`, and every validated MPI resume had relative L2 error
and maximum absolute difference equal to zero. `on success` retains all data
during reverse and removes it collectively only after final gradients are
written. `keep` retains its 21-record archive and restart data after success.
`safe incremental` performs collective final archive removal after successful
gradient output. No policy depends on a destructor, so abnormal termination
leaves restartable data.

Fresh trajectory creation resets an existing directory only when it contains a
trajectory manifest (or legacy metadata signature); an unknown directory is
rejected rather than recursively deleted.

## Scope and limitations

## Sparse-anchor restart qualification

For sparse spacing `K`, a reverse checkpoint at `next_reverse_step = r`
requires the lower anchor of the interval containing record `r-1`. The
2026-09-08 direct-Stokes qualification verified K=5 at `r=13` (anchor 10) and
`r=10` (anchor 5), in serial and with two ranks. Each restart rebuilt the
five-transition disposable interval and reproduced the uninterrupted gradient
byte-for-byte. Safe-incremental cleanup retained the required anchor before
removing obsolete shared segments; injected failures after rank-local payload
deletion and before/after manifest publication recovered idempotently from the
authoritative checkpoint.

The active entropy-viscosity K=5 case also restarted at `r=5`, retained anchor
0, replayed five transitions, and reproduced the uninterrupted gradient
byte-for-byte. This does not broaden the fixed-mesh/same-layout contract.

MPI restart and lifecycle safety are validated only under the contract above.
Different rank counts, repartitioning, sparse checkpointing, Revolve, AMR
replay, and new adjoint physics are separate future milestones. The current
work uses finite shared-MPI trajectory segments, but does not reduce
forward-end peak archive size or change validated adjoint PDE algebra.

## Matrix-free cold-rebuild audit (R0)

The missing cold-resume dependency was not a matrix, MatrixFree object, MG
hierarchy, or preconditioner. `Simulator::pressure_scaling` is initialized to a
signaling NaN in the constructor and is intentionally not checkpointed. A normal
forward Stokes solve calls `compute_pressure_scaling_factor()` immediately before
assembly. The adjoint-resume branch skips that solve, so the first reconstructed
GMG pressure mass operator saw NaN scaling and produced a non-finite diagonal.

`Manager::add_stokes_field_transpose()` now recomputes that scalar from the
restored primal snapshot immediately before its primal-topology Stokes assembly.
This is the same runtime preparation as the normal solve, and serializes no
solver state.

`adjoint_restart_r0_cold_rebuild.prm` is the A/B test. It invalidates MatrixFree
and GMG DoF/preconditioner state immediately before the first reverse Stokes
solve, then lets the ordinary reconstruction path rebuild it. It completed the
first rebuild with positive pressure-mass diagonals on all three levels, ruling
out a generally invalid clean GMG rebuild (case A). The resumed R0 run initially
reported identical DoF and constraint counts but `nonfinite=4` on level 0; after
the pressure-scaling reconstruction it reports zero nonpositive/nonfinite entries
on every level and completes the reverse sweep.

With `Solver parameters/Matrix Free/Output cold rebuild diagnostics = true`,
the build emits `ASPECT_MF_GMG_LEVEL` and `ASPECT_MF_MASS_DIAGONAL` lines with
level DoF/owned/relevant counts, actual level constraint counts, viscosity table
shape, pressure scaling, component, nonpositive/nonfinite diagonal counts,
minimum positive entry, and the first offending pressure DoF.

The direct-solver control fixtures (`adjoint_restart_r0_direct_{baseline,stop,
resume}.prm`) exercised a step-10 mid-sweep interruption and recovery. The
resumed and uninterrupted initial-temperature gradient files were byte-identical
(relative L2 error 0). The analogous GMG resumed and uninterrupted R0 gradients
were also byte-identical after the pressure-scaling reconstruction.

## Runtime-state audit categories

Sparse replay/restart work classifies state deliberately: persistent physical
state (trajectory records), persistent adjoint state (the reverse checkpoint),
reconstructible runtime state (current constraints, linearization context, and
`pressure_scaling`), and derived solver state (matrices, MatrixFree data, MG
hierarchy, transfer, and preconditioners). `pressure_scaling` is in the third
category: it must be recomputed from the restored physical state, but must not
be serialized. The fourth category remains entirely runtime-reconstructed.
