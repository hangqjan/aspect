# Adjoint Trajectory Archive v1: Progressive Stress Results

## Scope

This is a measurement-only campaign. It uses the existing 3-D fixed-box,
single-rank, production-GMG benchmark with the same simple material model,
temperature objective, zero compositional fields, tolerances (`1e-12` Stokes
linear tolerance), and archive implementation in both modes. Measurements use
the ASPECT debug executable, so absolute timings are a local development limit
rather than release-build throughput. No archive, solver, or adjoint-physics
change was made for these runs.

`/usr/bin/time -l` supplies process-level memory because the ASPECT/deal.II
`VmPeak` and `VmRSS` fields are `0` on this macOS host. The host has 16 GiB of
physical memory; the safety threshold was therefore 11.2--12.0 GiB. All cases
use one MPI rank, so the process maximum is both the maximum and mean across
ranks. Archive sizes below are decimal MB; MiB/GiB are stated explicitly.

The raw per-case JSON and CSV contain all runner fields, including the archive
object, current-record, recovered-state, and three-state-cache byte counts:

- `/private/tmp/adjoint-trajectory-stress-r2-t50-20260905/summary.json`
- `/private/tmp/adjoint-trajectory-stress-r3-t50-20260905/summary.json`
- `/private/tmp/adjoint-trajectory-stress-r4-t50-20260905/summary.json`
- `/private/tmp/adjoint-trajectory-stress-r5-q5-20260905/summary.json`
- `/private/tmp/adjoint-trajectory-stress-r3-t250-20260905/summary.json`

For every comparison,
`E_g = ||g_min - g_cached||_2 / max(||g_cached||_2, 1e-50)`.
The denominator is much larger than `1e-50` in every reported case.

## Spatial scaling

The geometric fixed-mesh hierarchy produces 3,041, 20,381, 148,661, and
1,134,437 total DoFs rather than each nominal target in the plan. Block order
is velocity / pressure / temperature / composition; composition is zero in
every case. `S5-Q5` is the mandatory 5-step safety qualification, while the
first three levels completed the normal 50-step paired measurement.

| Level | Total DoFs (u / p / T / C) | Steps / records | Mode | MB/record | Archive MB | Recovery s/record | Reverse s/record | OS peak RSS GiB | `E_g` |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| S2 | 3,041 (2,187 / 125 / 729 / 0) | 50 / 51 | minimal | 0.00587 | 0.299 | 0.0355 | 0.1045 | 0.153 | 6.55e-15 |
| S2 | same | 50 / 51 | stokes cached | 0.02438 | 1.243 | 0 | 0.0718 | 0.153 | 6.55e-15 |
| S3 | 20,381 (14,739 / 729 / 4,913 / 0) | 50 / 51 | minimal | 0.03934 | 2.006 | 0.2784 | 0.8275 | 0.198 | 6.87e-15 |
| S3 | same | 50 / 51 | stokes cached | 0.16310 | 8.318 | 0 | 0.5588 | 0.198 | 6.87e-15 |
| S4 | 148,661 (107,811 / 4,913 / 35,937 / 0) | 50 / 51 | minimal | 0.28753 | 14.664 | 2.1373 | 6.5098 | 0.524 | 5.28e-15 |
| S4 | same | 50 / 51 | stokes cached | 1.18934 | 60.656 | 0 | 4.7255 | 0.495 | 5.28e-15 |
| S5-Q5 | 1,134,437 (823,875 / 35,937 / 274,625 / 0) | 5 / 6 | minimal | 2.19703 | 13.182 | 18.6667 | 43.3333 | 1.728 | 1.68e-15 |
| S5-Q5 | same | 5 / 6 | stokes cached | 9.07554 | 54.453 | 0 | 33.0000 | 1.737 | 1.68e-15 |

Field-wise archive breakdown for the same cases (MB except metadata, which is
bytes):

| Level / mode | Temperature | Composition | Velocity | Pressure | Metadata |
|---|---:|---:|---:|---:|---:|
| S2 minimal / cached | 0.298 / 0.298 | 0 / 0 | 0 / 0.893 | 0 / 0.051 | 1,381 |
| S3 minimal / cached | 2.005 / 2.005 | 0 / 0 | 0 / 6.014 | 0 / 0.298 | 1,381 |
| S4 minimal / cached | 14.663 / 14.663 | 0 / 0 | 0 / 43.987 | 0 / 2.005 | 1,381 |
| S5-Q5 minimal / cached | 13.182 / 13.182 | 0 / 0 | 0 / 39.546 | 0 / 1.725 | 136 |

The reverse cache is exactly three reconstructed full states: 0.112, 0.736,
5.354, and 40.842 MB at S2--S5. The corresponding current-record/recovered
Stokes-state sizes in minimal mode are 0.037/0.037, 0.245/0.245,
1.785/1.785, and 13.614/13.614 MB. Metadata objects are 4,295 bytes at the
50-step levels and 711 bytes for S5-Q5 (cached mode differs by 8 bytes).
Cached mode has no recovered Stokes-state allocation. These bounded values
show no cache growth with reverse progress.

At S4, minimal mode saved 75.82% of archive storage, with a 37.76% reverse
time penalty; recovery was 32.83% of minimal reverse time. At S5-Q5 the same
figures were 75.79%, 31.31%, and 43.08%. I/O was at most 0.15% of total time
in these two minimal cases and 0.12%/0.08% in cached mode, respectively.

## 3-D temporal scaling at fixed 20,381 DoFs

The 20-, 50-, 100-, and 250-step measurements use exactly the S3 mesh. The
100-step point is also the combined spatial-temporal case. Process RSS was
not captured for the earlier pre-wrapper 20- and 100-step runs; all ASPECT
internal Vm values were unavailable on this host, not zero-memory readings.

| Steps / records | Mode | Archive MB | MB/record | Write / read s | Recovery s | Reverse s | Recovery / reverse | OS peak RSS GiB | `E_g` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 / 21 | minimal | 0.826 | 0.03934 | 0.0082 / 0.0032 | 5.72 | 16.2 | 35.31% | — | 2.97e-15 |
| 20 / 21 | stokes cached | 3.425 | 0.16310 | 0.0174 / 0.0089 | 0 | 11.2 | — | — | 2.97e-15 |
| 50 / 51 | minimal | 2.006 | 0.03934 | 0.0217 / 0.0084 | 14.2 | 42.2 | 33.65% | 0.198 | 6.87e-15 |
| 50 / 51 | stokes cached | 8.318 | 0.16310 | 0.0561 / 0.0235 | 0 | 28.5 | — | 0.198 | 6.87e-15 |
| 100 / 101 | minimal | 3.973 | 0.03934 | 0.0455 / 0.0165 | 27.8 | 83.9 | 33.13% | — | 8.34e-15 |
| 100 / 101 | stokes cached | 16.473 | 0.16310 | 0.0876 / 0.0447 | 0 | 55.7 | — | — | 8.34e-15 |
| 250 / 251 | minimal | 9.874 | 0.03934 | 0.1250 / 0.1590 | 67.5 | 208.0 | 32.45% | 0.297 | 2.32e-14 |
| 250 / 251 | stokes cached | 40.938 | 0.16310 | 0.2240 / 0.1170 | 0 | 140.0 | — | 0.315 | 2.32e-14 |

For 250 steps, the minimal/cached storage saving is 75.88%; the minimal
reverse-time penalty is 48.57%; and I/O is 0.095% / 0.150% of total elapsed
time for minimal / cached. Minimal mode performed exactly one recovery solve
per record (21, 51, 101, and 251); cached mode performed none. The three-state
cache stayed 0.736 MB for every temporal length, while archive metadata grew
linearly from 559 bytes (21 records) to 6,959 bytes (251 records).

## Combined stress case

The completed combined case is **20,381 DoFs x 100 accepted steps** (101
records). It is intentionally below the plan's aspirational 500k x 250 case:
the next fixed refinement is 1.13M DoFs, whose short paired qualification
already costs 13.9 minutes. The completed combined result is nevertheless a
meaningful simultaneous spatial-and-temporal load and exercises both archive
modes through a full forward/reverse cycle.

| Case | Mode | Archive MB | Reverse s | Recovery fraction | Peak memory | Gradient difference |
|---|---|---:|---:|---:|---:|---:|
| S3 x 100 | minimal | 3.973 | 83.9 | 33.13% | not captured | 8.34e-15 |
| S3 x 100 | stokes cached | 16.473 | 55.7 | 0% | not captured | 8.34e-15 |

## Recovery diagnostic and solver health

The cached S4 50-step diagnostic sampled records 0, 25, and 50 and then
completed the ordinary reverse sweep successfully. With GMG and the configured
`1e-12` linear tolerance, its velocity errors were `3.445e-10`, `3.498e-10`,
and `3.580e-10`; gauge-normalized pressure errors were `1.044e-13` at all
three records. Each sampled recovery converged in `13+0` GMG iterations and
passed the `1e-8` diagnostic threshold. Standard ASPECT output does not expose
a scalar final residual here; convergence is controlled by the configured
linear tolerance. The diagnostic's `/usr/bin/time -l` report showed 0 swaps
and 570,753,024 bytes peak RSS.

This does not supersede the documented caveat for time-dependent inhomogeneous
tangential velocity: its small-case GMG replay discrepancy was approximately
`8.78e-5`, while direct recovery was exact. The stress benchmark has static
tangential velocity and supplies no contradictory evidence of missing archive
state.

## Scaling interpretation and practical limit

`S_record` tracks total DoFs very closely. Minimal bytes/record grow from
5,867 to 39,339 to 287,531 to 2,197,030 bytes as DoFs grow from 3,041 to
20,381 to 148,661 to 1,134,437. Cached bytes/record grow from 24,379 to
163,099 to 1,189,339 to 9,075,542. The near-matching refinement ratios support
`S_record = O(N_DoF)` for both modes. At fixed S3, archive sizes and write/
read/recovery times grow approximately linearly with records; bytes/record
remain stable to the reported precision. There is no evidence of hidden archive
growth, metadata accumulation beyond linear per-record metadata, cache leakage,
or nonlinear I/O degradation.

The largest completed spatial configuration is **S5-Q5: 1,134,437 DoFs x 5
steps**, with a maximum measured RSS of 1,865,285,632 bytes (1.74 GiB) and zero
swaps. It is memory-safe at only about 15% of the 11.2--12.0 GiB stop range,
but it is the practical local-development time limit for this debug campaign:
the paired Q5 run took 469 s + 364 s. Applying the observed S4 Q5-to-50
multiplier projects an S5 50-step pair at roughly 69 min minimal plus 57 min
cached, or about 2.1 hours. The largest normal 50-step spatial measurement is
therefore **S4: 148,661 DoFs x 50 steps**. This is a runtime stop, not an
archive correctness or memory failure.

## Storage estimates from the largest measured DoF level

These are estimates, not measurements. They assume the measured S5-Q5
bytes/record remain unchanged and include `steps + 1` stored records.

| Accepted timesteps | Records | Minimal archive | Stokes-cached archive |
|---:|---:|---:|---:|
| 1,000 | 1,001 | 2.199 GB | 9.085 GB |
| 5,000 | 5,001 | 10.987 GB | 45.387 GB |
| 10,000 | 10,001 | 21.972 GB | 90.764 GB |

A purely linear, cautious extrapolation from S5 to 2,000,000 DoFs gives about
3.873 MB/record minimal and 16.000 MB/record cached. It is only a storage
estimate; it is not a predicted solver time or memory limit for an untested
global model.

## Archive retention and cleanup

V1 uses an explicit retention policy. The archive remains in
`<Output directory>/adjoint_trajectory/` after a successful reverse sweep and
also after a failed or interrupted run. This is intentional: it preserves
postmortem evidence, while providing no restart or cross-job-replay guarantee.
The workflow that owns the output directory must retain the files until the
gradient and logs have been accepted, then delete **only that exact archive
subdirectory** before the output directory is reused. Every new run should use
a fresh output directory. The detailed policy and stale-record rationale are
in [trajectory_archive.md](trajectory_archive.md#retention-and-cleanup-policy-v1).

## Result

The milestone's success conditions are met: four increasing spatial levels,
four increasing 3-D temporal levels, one completed combined case, paired-mode
comparisons, archive/timing/memory/gradient data, and a practical local limit
are all recorded. The explicit retention/cleanup policy closes the final v1
operational gap. The restricted fixed-mesh, same-process archive/restore v1 is
therefore fully closed out. The data support proceeding to the next planned
milestone: sparse trajectory checkpointing with deterministic forward-window
replay.
