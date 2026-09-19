# Shared MPI trajectory backend: validation record

This record covers the fixed-mesh, same-rank-count, same-DoF-ownership
trajectory contract. It was run on 2026-09-07 with the Debug MPI executable
`build-debug/aspect-debug`, two MPI ranks where stated, and the repository's
1 GiB RSS guard. It is a correctness and local-filesystem characterization,
not a distributed-filesystem performance claim.

## Backend contract

Each rank serializes only its local payload. `MPI_Exscan` computes the local
`uint64_t` prefix; `LargeCount::File_write_at_all_c` writes directly to that
range in one `trajectory_XXXXXX.bin` segment. Payload synchronization and close
complete before rank zero atomically replaces `manifest`. The manifest is the
only authority for replay. It holds the record step/time/dt, mode, schema,
segment and record range, MPI rank count, every rank's range and byte count,
field-byte metadata, and ownership fingerprint. No trajectory values are
gathered to rank zero.

The tested layout is intentionally not repartition-capable. It requires the
same rank count, fixed mesh, global DoF numbering, and owned-index partition.

## S0--S6 correctness and lifecycle

| Stage | Fixture and observation | Result |
|---|---|---|
| S0 | Serial shared-file smoke/reverse | PASS |
| S1 | Fresh two-rank M2 run: 21 records, six data segments, 77,448 data bytes and a 3,260-byte manifest | PASS |
| S2 | Gradient L2 `0.013480703099710863`; existing validated M2 gradient files byte-identical; supplied scalar baseline differs only by `~1.5e-16` relative rounding | PASS |
| S3 | Three-state replay loaded records 8, 7, 6 across segments 2 and 1 | PASS |
| S4 | M2 stop at the interior reverse checkpoint, then a new `mpirun -n 2` resume | PASS; cold restart and final gradient were byte-identical to fresh M2 |
| S5 | M4 safe-incremental stop/resume | PASS; at stop the manifest retained records 0--9 in three segments; only fully obsolete segments were deleted |
| S6 | Controlled append failures at local-serialization, collective-write, durable-before-manifest, and post-manifest points | PASS; no deadlock and no RSS-guard termination |

For S6, injection at record 6 yielded these durable frontiers:

| Injection point | Manifest records / last | Segment 1 bytes | Interpretation |
|---|---:|---:|---|
| After local serialization | 6 / 5 | 7,376 | No record-6 shared write occurred. |
| After collective write, before manifest | 6 / 5 | 11,064 | Record-6 bytes are trailing and unlisted. |
| After payload durability, before manifest | 6 / 5 | 11,064 | Record-6 bytes are trailing and unlisted. |
| After manifest publication | 7 / 6 | 11,064 | Record 6 is committed and visible. |

`open_existing_archive()` reads only manifest entries and `load()` uses the
manifest-published local offset and size. The successful M2 restart exercises
that reopen path; trailing bytes from the two pre-manifest append cases are not
named by any logical record and are consequently unreachable.

## S7 file-count characterization

All rows use 2 ranks, 1000 requested steps (1001 records including time zero),
`stokes cached`, `Trajectory cleanup policy = keep`, and the same 268-DoF
fixture. Archive bytes include the manifest. Timings are one local Debug run.

| Records/segment | Shared data files | Metadata files | Archive bytes | Write | Reverse / read |
|---:|---:|---:|---:|---:|---:|
| 1 | 1001 | 1 | 3,841,711 | 3.12 s | 5.81 s / 0.760 s |
| 4 | 251 | 1 | 3,843,880 | 2.40 s | 4.99 s / 0.628 s |
| 16 | 63 | 1 | 3,844,286 | 2.22 s | 4.81 s / 0.606 s |
| 64 | 16 | 1 | 3,844,641 | 2.25 s | 5.07 s / 0.642 s |
| 100 | 11 | 1 | 3,844,481 | 3.89 s | 11.5 s / 1.78 s |

The raw data payload was 3,691,688 bytes in every row. Data-file count is
`ceil(records / records_per_segment)` and does not depend on rank count. Thus
5000 steps with 100 records per segment needs about 50 shared data files (plus
the manifest), even at 1024 ranks.

## Reproduction inputs

- `adjoint_shared_trajectory_serial.prm`
- `adjoint_mpi_m2_baseline.prm`, `adjoint_mpi_m2_stop.prm`, and
  `adjoint_mpi_m2_resume.prm`
- `adjoint_mpi_m4_stop.prm` and `adjoint_mpi_m4_resume.prm`
- `adjoint_mpi_m2_shared_append_*.prm`
- `adjoint_mpi_m2_shared_scaling_{1,4,16,64,100}.prm`

Run MPI fixtures through `run_adjoint_mpi_m2_guard.zsh`; it samples process RSS,
captures diagnostics at 500 MiB, and terminates at 1 GiB. `git diff --check`
and `cmake --build build-debug --target aspect.exe.debug -j 2` passed after
the validation set.
