# Adjoint trajectory archive v1: close-out validation

This report records single-rank Debug measurements made on 2026-09-05 with
`build-debug/aspect-debug`. The archive is a fixed-mesh, lossless trajectory
format. All comparisons use the same physical input, timestep cap, and mesh
for the two storage modes. Timings are development-machine measurements, not
production performance claims.

The reproducible drivers are
[`run_taylor_test.py`](run_taylor_test.py) and
[`run_trajectory_scaling.py`](run_trajectory_scaling.py). The latter retains
per-run inputs and complete ASPECT screen output, and writes `summary.csv` and
`summary.json` to the user-selected results directory.

## Correctness

| Case | Minimal result | Stokes-cached result | Cross-mode result | Status |
|---|---|---|---|---|
| Constant-viscosity BDF2 Taylor (3 steps) | Taylor order `2.000`; finest slope error `1.99e-11` | Taylor order `2.000`; finest slope error `1.99e-11` | Gradient relative L2 `1.00e-12` | PASS |
| Iterative-Stokes recovery (21 records, linear tolerance `1e-12`) | 21 recovery solves | Cached comparison at records 0/10/20: velocity `< 8e-10`, gauge-normalized pressure `< 3e-14` | B0 gradient relative L2 `1.96e-15` | PASS |
| eta(T) direct Stokes recovery | n/a | Recovery error exactly zero; direct viscosity derivative relative error `3.3e-16` | n/a | PASS |
| eta(C) direct Stokes recovery | n/a | Recovery error exactly zero; viscosity-coefficient transpose relative error `4.15e-9` | n/a | PASS |
| Active entropy viscosity, 3 steps | Taylor order `1.99999967`; 4 recovery solves | Taylor order `1.99999967`; 0 recovery solves | Gradient relative L2 `1.18e-12` | PASS |
| Curved free slip, 3 steps | Taylor order `2.000`; 4 recovery solves | Taylor order `2.000`; 0 recovery solves | Gradient relative L2 `1.95e-12` | PASS |
| Time-dependent tangential velocity, 3 steps | Finest Taylor order `1.9963`; archived metadata `t={0,.01,.02}`, `dt={0,.01,.01}` | Direct-Stokes recovery is exactly zero at records 0/1/2 | Gradient relative L2 `5.94e-12` | PASS (see GMG note) |
| Composition objective, 4 records | 4 recovery solves | 0 recovery solves | Composition-gradient relative L2 `1.05e-12` | PASS |

The active-EV local entropy residual, global entropy denominator, and global
entropy variation diagnostics produced the same values in both modes (relative
errors `5.05e-7`, `1.98e-11`, and `1.70e-9`, respectively). This confirms that
the current/previous/previous-previous reconstructed cache path was exercised
without changing the validated EV algebra.

The known optional curved velocity-transpose diagnostic was not used as an
archive pass/fail criterion. The end-to-end Taylor test is second order in both
modes, and no archive-mode-specific discrepancy was observed.

For the time-dependent tangential-velocity input, a separate GMG recovery
diagnostic with `Linear solver tolerance = 1e-12` reported velocity relative
error `8.78e-5` at record zero (pressure error `1.41e-11`) and therefore did
not meet its independently requested `1e-8` velocity threshold. Repeating the
identical archived T,C/time/constraint reconstruction with ASPECT's direct
Stokes solver gives exactly zero velocity and pressure error at all three
records. The end-to-end minimal/cached Taylor and gradient comparisons above
remain mode-equivalent. This identifies an iterative inhomogeneous-constraint
solver reproducibility sensitivity, not missing archive state; it is retained
as a benchmark caveat rather than changing adjoint or archive physics.

## Temporal scaling (2D, 268 DoFs, one MPI rank)

Each row has `N+1` archived records because ASPECT writes the accepted state at
time zero and after every requested step. Byte counts include metadata.

| Requested steps / records | Mode | Archive bytes | Bytes/record | Write / read (s) | Recovery (s) / solves | Reverse (s) | Gradient difference |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 / 21 | minimal | 14,335 | 682 | .00338 / .000437 | .0788 / 21 | .176 | `1.96e-15` |
| 20 / 21 | stokes cached | 46,087 | 2,194 | .00905 / .00112 | 0 / 0 | .124 | `1.96e-15` |
| 100 / 101 | minimal | 69,031 | 683 | .0203 / .00250 | .387 / 101 | .926 | `4.44e-15` |
| 100 / 101 | stokes cached | 221,743 | 2,195 | .0306 / .00512 | 0 / 0 | .545 | `4.44e-15` |
| 250 / 251 | minimal | 171,615 | 683 | .0585 / .00571 | .962 / 251 | 2.33 | `9.01e-15` |
| 250 / 251 | stokes cached | 551,127 | 2,195 | .0831 / .0126 | 0 / 0 | 1.37 | `9.01e-15` |
| 500 / 501 | minimal | 342,583 | 683 | .165 / .0122 | 1.94 / 501 | 4.70 | `8.85e-15` |
| 500 / 501 | stokes cached | 1,100,095 | 2,195 | .210 / .0259 | 0 / 0 | 2.75 | `8.85e-15` |
| 1,000 / 1,001 | minimal | 684,531 | 683 | .454 / .0231 | 3.87 / 1,001 | 9.40 | `2.35e-14` |
| 1,000 / 1,001 | stokes cached | 2,198,043 | 2,195 | .548 / .0543 | 0 / 0 | 5.48 | `2.35e-14` |

The measured bytes per record are stable (`682–683` minimal and
`2,194–2,195` cached), verifying linear archive growth. At 1,001 records the
minimal storage saving is `68.86%`; its reverse-time penalty is `71.5%`, of
which Stokes recovery is `41.2%` of the minimal reverse sweep. Read/write I/O
was below `6.6%` of whole-run time on this local filesystem.

## Spatial scaling (3D, fixed 20 requested steps, one MPI rank)

The paired 3D S3 level has 20,381 DoFs: 14,739 velocity, 729 pressure, and
4,913 temperature. The field decomposition is reported directly by ASPECT.

| DoFs / records | Mode | Archive bytes / record | Total archive | Write / read (s) | Recovery / step (s) | Reverse / step (s) | Current / cache memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20,381 / 21 | minimal | 39,338 | 826,111 | .00816 / .00320 | .272 | .771 | 245,340 / 736,020 B |
| 20,381 / 21 | stokes cached | 163,098 | 3,425,071 | .0174 / .00889 | 0 | .533 | 245,340 / 736,020 B |

The S3 gradient relative L2 difference is `2.97e-15`. Minimal storage saves
`75.88%`; its reverse penalty is `44.6%`, and Stokes recovery is `35.3%` of
its reverse time. The platform's deal.II memory-stat API returned `VmPeak =
VmRSS = 0 kB`, so those process-level numbers are explicitly unavailable here;
the object-level fields above remain valid.

An exploratory S4 minimal run reached 148,661 DoFs (107,811 velocity, 4,913
pressure, 35,937 temperature). Its 21-record archive was 6,038,143 B
(287,530 B/record), reverse time was 128 s, recovery time was 44 s, and the
three-record reverse cache was 5,354,100 B. It demonstrates a substantially
larger practical Debug load, but it is not included in a paired result because
the cached counterpart was deliberately not run after the 196 s minimal run.

## Combined stress benchmark

The S3 mesh with 100 requested steps (101 records) combines spatial and
temporal load without a Taylor sweep.

| DoFs / records | Mode | Archive bytes | Bytes/record | Write / read (s) | Recovery (s) / solves | Reverse (s) | Gradient relative L2 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20,381 / 101 | minimal | 3,973,287 | 39,339 | .0455 / .0165 | 27.8 / 101 | 83.9 | `8.34e-15` |
| 20,381 / 101 | stokes cached | 16,473,047 | 163,099 | .0876 / .0447 | 0 / 0 | 55.7 | `8.34e-15` |

For this combined case, minimal storage saves `75.88%`; the reverse-time
penalty is `50.6%`; Stokes recovery is `33.1%` of minimal reverse time; and
archive I/O is below `0.15%` of whole-run time.

## Linear storage projections

These are linear extrapolations from measured bytes per record, not measured
global-model runs. They exclude filesystem metadata and assume the same DoF
layout, rank count, and lossless format.

| Measured level | Mode | 1,000 records | 5,000 records | 10,000 records |
|---|---|---:|---:|---:|
| 2D, 268 DoFs | minimal | 683 kB | 3.42 MB | 6.83 MB |
| 2D, 268 DoFs | stokes cached | 2.20 MB | 10.98 MB | 21.95 MB |
| 3D S3, 20,381 DoFs | minimal | 39.34 MB | 196.69 MB | 393.38 MB |
| 3D S3, 20,381 DoFs | stokes cached | 163.10 MB | 815.49 MB | 1.63 GB |

## Close-out status

## Sparse replay Phase C/D qualification (2026-09-08)

The authoritative executable for this qualification was the non-unity MPI
Debug build, `build-no-unity-mpi/aspect-debug`. It rebuilt
`source/adjoint/core/manager.cc`; the trajectory-store and restart-checkpoint
headers were consequently compiled through that translation unit. The
accepted-step journal, anchor replay, and checkpoint recovery all use the
recorded timestep schedule and the ordinary simulator advance/start/solve path.

The 21-record, 268-DoF direct-Stokes control completed serial and two-rank
checks. K=5 restarts at `next_reverse_step=13` and `10` selected lower anchors
`10` and `5`, respectively; they replayed five transitions after restart and
produced byte-identical rank-local gradient files. Safe incremental cleanup
retained only the required lower anchor and reclaimed obsolete shared segments.
The serial and two-rank rank-local payload-deletion recovery fixture also
completed with byte-identical gradients. The serial logical global comparison
against the two-rank K=5 gradient was L2 `8.011591853955987e-13`, relative L2
`5.943007419354939e-11`, and max absolute `2.901637263796886e-13` over 82
global DoFs.

| Qualification | Result | Replay/reference result |
|---|---|---|
| Active entropy/artificial viscosity (`alpha=1`), direct Stokes | PASS | K=1/K=5 gradients byte-identical; state, velocity, and gauge-pressure differences 0 |
| Time-dependent tangential prescribed velocity, direct Stokes | PASS | K=1/K=5 gradients byte-identical; state, velocity, and gauge-pressure differences 0 |
| GMG, fixed mesh | PASS at configured `1e-10` replay tolerance | max state `1.87e-16`, velocity `2.91e-11`, pressure `3.05e-16`; K=1/K=5 gradient relative L2 `8.45e-16` |
| Active-EV K=5 restart at reverse step 5 | PASS | lower anchor 0, five restart-induced transitions, byte-identical gradient |

The physical reference archive must use `Trajectory storage mode = stokes
cached` whenever state/velocity/pressure replay validation is requested. A
minimal reference does not contain velocity or pressure and is not a valid
reference for those components.

### Anchor-spacing measurement

All rows below are fresh serial Debug runs on the same 20-step control case.
Archive bytes include the dense schedule and anchor payloads. Wall time is a
local development-machine measurement only; maximum resident set size was
about 151 MB in every run. Every sparse gradient was byte-identical to K=1.

| K | Archive bytes | Reduction vs K=1 | Replay transitions | Peak window states | Wall time (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 79,414 | 0% | 0 | 0 | 1.27 |
| 5 | 49,649 | 37.5% | 20 | 8 | 1.98 |
| 10 | 27,335 | 65.6% | 20 | 13 | 1.30 |
| 20 | 16,178 | 79.6% | 20 | 21 | 1.33 |
| 50 | 16,178 | 79.6% | 20 | 21 | 1.98 |

The subsequent [progressive stress report](trajectory_archive_v1_stress.md)
supersedes this document's earlier 3-D scaling limit: it includes paired
148,661-DoF measurements, a 1.13M-DoF qualification, four 3-D temporal levels,
and the practical debug-runtime limit. The v1 lifecycle policy is now explicit:
retain `<Output directory>/adjoint_trajectory/` through reverse completion and
diagnosis, then have the owning workflow delete that exact subdirectory before
the output directory is reused. With that operational policy, the restricted
fixed-mesh, same-process v1 archive/restore scope is fully closed out.
