# Protocol history

The record of what the v1 pilot revealed, why its gates are void, and why
restating them for a redesigned mechanism is not post-hoc threshold tuning.

This document exists so that the failure is legible to someone who did not run
the study. The commit history is the primary evidence; this is the narrative
that makes it checkable.

## Timeline

| | Commit | Timestamp |
|---|---|---|
| v1 Gate A and Gate B defined | `e2a9e20` | 2026-07-19 15:22:42 |
| v1 Gate B budget clause added | `a7ea388` | 2026-07-19 15:51:55 |
| Gate B under-strictness recorded, threshold left frozen | `76564ab` | 2026-07-19 20:40:43 |
| `dense_seed0` starts | — | 2026-07-19 20:34:58 |
| `rigl_seed0` starts | — | 2026-07-19 23:32:33 |
| Device-ordering and determinism corrections | `4e3ec2a` | 2026-07-20 07:27:38 |
| `oneshot_prune_seed0` finishes, pilot ends | — | 2026-07-20 11:51:08 |
| v1 pilot declared void | `f435b37` | 2026-07-21 07:28:15 |
| Global redistribution added | `e72c832` | 2026-07-21 07:35:14 |
| v2 gates proposed | `7a0c026` | 2026-07-21 07:39:33 |
| Gate B1 revised to the ERK-ray residual | `a3ab865` | 2026-07-21 16:24:34 |

Run times are reconstructed filesystem mtimes, not records. See
[`../RUNS.md`](../RUNS.md) for what that evidence is worth.

## Why the v1 pilot cannot test the hypothesis

**The defect is provable from the specification alone. No data is required.**

The v1 specification required that the number of connections pruned equal the
number regrown **per layer**, at every update. Equal prune and regrow counts
within a layer make that layer's live-connection count invariant. Every layer's
density is therefore constant across the entire run, fixed at its 0.30
initialization.

The layer-wise density trajectory is the quantity the study exists to measure.
Under this rule it is flat by construction, for every sparse arm, before any
batch is loaded.

This is directly observable in the published artifacts. In
`results/rigl_seed0/trajectory.csv`, each of the 11 sparsified layers holds
**exactly one** density value across all 100 epochs; the spread within every
layer is 0.0.

### Gate A's density condition was unsatisfiable

Gate A required at least one encoder stage's mean density to depart from 0.30 by
more than 0.05. No layer's density could depart from 0.30 by anything. The
measured maximum deviation on `rigl_seed0` was 0.0000, and no other value was
reachable.

A gate that cannot be satisfied by any possible run is not a test.

### Gate B passed, vacuously

Gate B compared the final allocation to the ERK prior. It evaluated True, but
the quantity it measured — the deep-versus-ERK budget gap — is the fixed
distance between the uniform-0.30 initialization and ERK. That distance is
present at step 0, is unchanged by training, and reflects no reallocation at
all. A pass meant only that uniform is not ERK, which is true by arithmetic.

### The mechanism was also wrong for the question

Standard RigL (Evci et al. 2020) holds per-layer sparsity fixed at
initialization and rewires only *within* layers. Even without the conservation
rule above, it could not move capacity *between* layers, which is the entire
question. The instrument did not match the measurement.

### Where the error originated

Both failures originate in the v1 specification, not in its implementation. The
masking code enforced per-layer conservation exactly as written and carried a
test asserting that it did. That test is precisely what should have been a
per-layer-density-*can*-change test instead. The redesign adds that test:
`tests/test_redistribute.py` asserts that per-layer density changes across an
update, which is the assertion whose absence let a 22-hour pilot measure a
constant.

## Gate B was found under-strict before the run it judges

The 3-point budget clause in Gate B exists so that a departure confined to
stages 0–2 — 6.1 percent of encoder parameters — cannot pass on the density view
alone. The pre-registration justified the clause by that parameter share.

The clause was then measured against the case it was written to exclude:

- At the ERK allocation, stages 0–2 hold **16.0 percent** of the live budget.
- ERK already pins stages 0 and 1 at density 1.000, so the only shallow movement
  available is driving stage 2 from its ERK density of 0.720 up to 1.0.
- Doing that, at constant overall density, takes the shallow block to about
  **20.4 percent** of the live budget.
- The largest possible shallow-only departure is therefore about **4.4 points**,
  which **clears the 3-point threshold.**

The clause constrains, but it does not exclude the case it was written for. A
threshold near 5 points would.

**This was recorded at `76564ab`, 2026-07-19 20:40:43 — about 2 hours 51 minutes
before the `rigl` arm started at 23:32:33, and roughly six hours before it
finished.** The finding predates the run it would have judged.

**The threshold was not amended.** It was left frozen at 3 points, the argument
was recorded, and the limitation was reported rather than fixed. The threshold
text is byte-identical in every version of `preregistration.md`; `git log -p
-S'more than 3 points' -- docs/preregistration.md` confirms it.

The measured bound is pinned by
`tests/test_analysis.py::test_shallow_only_departure_has_bounded_budget_effect`,
so it cannot drift silently.

## Non-monotonicity is sufficient but not necessary

Recorded at the same commit. The pre-registration named a mid-depth bulge —
stages 2–3 ending above their ERK allocation — as a discriminating departure,
and noted that ERK is monotone decreasing in depth.

Both statements are true, but ERK already pins stages 0 and 1 at density 1.000.
For a stage-density sequence to become non-monotone, some deeper stage must
exceed a shallower one, which means exceeding 1.0 where the shallow stages are
already saturated. A trajectory can therefore be a large, genuine, task-specific
departure from ERK at stages 2–3 and still be monotone decreasing overall.

`tests/test_analysis.py::test_mid_depth_bulge_passes_gate_b` constructs exactly
such a case: it clears both Gate B conditions while remaining monotone.

Treating a monotone result as "therefore ERK-like" would be a mistake. The gate
conditions themselves — per-stage departure plus the budget clause — do the
deciding. `monotone_in_depth` is reported alongside the gates as one diagnostic
among several.

The gate conditions were not changed. The prose that had called
non-monotonicity "the sharpest available departure" was corrected at `4e3ec2a`,
mid-pilot, and explicitly labelled a description-only correction. That diff
shows no gate condition was touched.

## Corrections made during the pilot

**Device ordering.** All four arms ran on an 8 GB card rather than the intended
16 GB card. `CUDA_DEVICE_ORDER` was unset, so CUDA used `FASTEST_FIRST`, which
ranks the two devices opposite to the order `nvidia-smi` reports; the configured
`cuda:0` therefore resolved to the smaller device. The runs fit and are valid,
but did not use the intended hardware, and their provenance recorded only the
string `cuda:0`, which a run on either device would report. Device selection is
now by name, with a startup assertion and the resolved name recorded in
provenance.

**Determinism.** An earlier claim that training is deterministic given the seed
was wrong. `use_deterministic_algorithms`, `cudnn.benchmark` and
`cudnn.deterministic` are all False and `CUBLAS_WORKSPACE_CONFIG` is unset, so
3D convolution backward accumulates non-deterministically. What *is*
deterministic is the data pipeline, which is keyed on `(seed, epoch, global
sample index)` and is independent of worker count and iteration order.
Determinism was deliberately not enabled mid-pilot, because doing so would have
made the remaining arms non-comparable with those already finished.

## What survives the void

Two observations survive, **recorded as observations only**. Neither is a
finding, and neither is evidence for the hypothesis, which the pilot could not
test.

**1. Validation loss.** `rigl_seed0` reached the lowest final validation loss of
the four arms: −0.8491, against dense −0.8292, `oneshot_prune` −0.8296 and
`static_sparse` −0.8126. This is n=1, it is loss rather than Dice, and RigL
rewired only within layers. It says nothing about capacity allocation. Values
are reproducible from `results/*/training_log.jsonl`.

**2. Regrowth informativeness, with an important qualification.** The probe
measures top-k overlap between regrowth scores computed on a
foreground-oversampled batch and on a background-dominated batch, at the same
masked positions. Low overlap means the regrowth criterion is reading the batch
rather than the task.

At the first three probes the overlap falls with depth, which is what the
original record reported:

| Step | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| 3125 | 0.745 | 0.628 | 0.529 | 0.399 | 0.369 | 0.305 |
| 9375 | 0.651 | 0.714 | 0.443 | 0.339 | 0.321 | 0.286 |
| 15625 | 0.886 | 0.801 | 0.556 | 0.375 | 0.287 | 0.248 |
| 21875 | 0.550 | 0.502 | 0.411 | 0.482 | 0.833 | **1.000** |

**At the final probe the trend inverts.** Stages 4 and 5 rise to 0.833 and
1.000, with three individual layers at exactly 1.000. Averaged over all four
probes the depth ordering is 0.708, 0.661, 0.485, 0.399, 0.452, 0.460 — not
monotone.

The original record described this observation as falling *monotonically* with
depth, from about 0.75 at stage 0 to about 0.28 at stage 5. That description
fits the first three probes and does not fit the fourth. It is corrected here
and in the VOID banner of `preregistration.md`.

The final probe at step 21875 falls after the prune rate has cosine-annealed to
zero at 75 percent of training, so the mask is frozen by then; an overlap of
exactly 1.000 across three layers is consistent with a degenerate top-k
selection rather than with a genuine rise in task-informativeness. **That
explanation is not established here.** The probe as instrumented does not
distinguish the two, and the raw values are given above rather than being
interpreted.

Because of the inversion, this observation does **not** retire the
"task-independent regrowth signal" candidate explanation of a null, which is
what the original record claimed for it. That candidate explanation remains
open, and the v2 protocol carries the regrowth-informativeness probe forward as
a reported diagnostic for that reason.

Source data: `results/rigl_seed0/regrowth_informativeness.csv`.

## The redesign

Restating gates for a redesigned mechanism is not post-hoc tuning. No answer to
the original question was obtainable under the v1 specification, for any data
whatsoever, so there was no result to tune toward. The old gates are void
because the apparatus could not test them, not because the numbers came out
wrong. They are preserved in place, marked VOID, rather than deleted.

The redesigned mechanism conserves density **globally** rather than per layer:
total live connections across the encoder are fixed, while each layer's density
is free to move. This makes the measured quantity capable of varying, which is
the single thing v1 lacked. Design and gates are in
[`preregistration_v2.md`](preregistration_v2.md), which is `STATUS: PROPOSED —
NOT IN EFFECT`.
