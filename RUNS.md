# Run record

Every training run executed in this project, with the evidence available for
each. **All four completed runs belong to the v1 pilot, which is VOID as a test
of the hypothesis** — see the banner at the top of
[`docs/preregistration.md`](docs/preregistration.md). **No v2 arm has been run.**

## How these timings were obtained, and what they are worth

**The timings below are filesystem modification times of run artifacts, read on
2026-08-02 and reconstructed after the fact. They were not recorded at run
time.** Nothing in the v1 run pipeline wrote a wall-clock timestamp:
`training_log.jsonl` records per-epoch durations only, and `provenance.json`
records configuration and hardware but no clock reading and no commit hash.

This makes the table self-reported. An mtime is machine-local, mutable, and
carries no attestation. A reader who does not trust it has no independent way to
check it from this repository, and should not treat it as though they do.

Two conventions used throughout:

- **Start** is the mtime of `provenance.json` and `config.yaml`, which the
  trainer writes together at startup, a few seconds after process launch.
- **End** is the mtime of `checkpoint_final.pt`, written on completion.

**The commit column is inferred, not recorded.** It is the last commit whose
timestamp precedes the run's start, i.e. what `HEAD` must have been when the run
began. This is an inference from two independent records — the commit log and
the artifact mtimes — not something the run itself captured. It assumes the
working tree matched `HEAD` at launch, which **was not always true**; see the
note on `oneshot_prune_seed0` below.

Runs from this point forward record the commit hash and UTC start and end times
in `provenance.json` natively, so this reconstruction is not repeated.

## v1 pilot — VOID

1 seed x 100 epochs per arm. Run order was dense, rigl, static_sparse,
oneshot_prune.

| Arm | Start | End | Duration | `HEAD` at start (inferred) |
|---|---|---|---|---|
| `dense_seed0` | 2026-07-19 20:34:58 | 2026-07-19 23:32:25 | 2h 57m | `9ecbee7` |
| `rigl_seed0` | 2026-07-19 23:32:33 | 2026-07-20 02:39:39 | 3h 07m | `76564ab` |
| `static_sparse_seed0` | 2026-07-20 02:39:53 | 2026-07-20 05:54:08 | 3h 14m | `76564ab` |
| `oneshot_prune_seed0` | 2026-07-20 08:33:55 | 2026-07-20 11:51:08 | 3h 17m | `1226ceb` |

All times are local (UTC-05:00).

### Notes on individual runs

**`oneshot_prune_seed0` launched from a working tree that did not match
`HEAD`.** Its launch log is named `launch_20260720_083347`, placing process
start at 08:33:47; `provenance.json` was written 8 seconds later. The commit
that added the detached-launch script it was started with, `a322478`, is
timestamped 08:33:58 — 3 seconds *after* the run had already recorded its
provenance. The script therefore existed uncommitted at launch. The inferred
`HEAD` of `1226ceb` is correct as far as it goes, but the tree was dirty, and
that is the general reason the commit column is an inference rather than a
record.

**The gap between `static_sparse_seed0` and `oneshot_prune_seed0`** (05:54 to
08:33 on 2026-07-20) covers the CUDA device-ordering investigation committed in
`4e3ec2a` and `1226ceb`, and the detached-launch change in `a322478`.

**All four arms ran on the 8 GB RTX 3070 Ti, not the intended 16 GB card.**
This is established directly only for `oneshot_prune_seed0`, whose
`provenance.json` records `device_name`. The other three recorded only the
string `cuda:0`, which a run on either card would report; that they also ran on
the 3070 Ti is documented in the README and in `4e3ec2a`, not demonstrated by
their own artifacts.

**Training is not bitwise reproducible at fixed seed.** Determinism flags were
deliberately left off so that later arms stayed comparable with earlier ones.
The flags are recorded in `provenance.json` for `oneshot_prune_seed0` only.

## v2 — nothing has been run

The v2 gates are `STATUS: PROPOSED — NOT IN EFFECT`
([`docs/preregistration_v2.md`](docs/preregistration_v2.md)). No
`sparse_momentum` arm of either initialization has been executed. The most
recent artifact of any kind in `results/` is dated 2026-07-20 11:51, which
precedes the first v2 gate commit (`7a0c026`, 2026-07-21 07:39) by about 29
hours.

## Other artifacts in `results/`

`results/smoke/` and `results/launchcheck/` hold throughput and launch smoke
tests from 2026-07-19, used to settle dataloader worker counts before the pilot.
They are not pilot arms, were not run against any gate, and are excluded from
the table.
