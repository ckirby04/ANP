# Pre-registration v2 (PROPOSED — PENDING APPROVAL, NOT IN EFFECT)

> **This document is a PROPOSAL.** No gate here is operative. Nothing runs until
> Clark approves these conditions and thresholds. Until then the redesigned
> mechanism (`sparse_momentum`, global conservation) has code and tests but no
> agreed success criteria. The voided v1 gates are in `preregistration.md`.

## What changed and why the restatement is legitimate

The v1 pilot is void: per-layer conservation made per-layer density invariant,
so the layer-wise density trajectory was flat by construction and Gate A's
density condition was unsatisfiable. The redesigned dynamic arm conserves
density **globally** (total live connections fixed at 0.30·N across the
encoder), so per-layer density is free to move and capacity can migrate between
layers. Restating gates for a mechanism that can actually move the measured
quantity is not post-hoc tuning: no answer was obtainable under v1 for any data.

Mechanism: SNFS-style sparse momentum (Dettmers & Zettlemoyer 2019). Prune the
lowest-magnitude fraction per layer (floor-protected); redistribute the freed
budget across layers by mean momentum magnitude; regrow within a layer at the
highest-momentum dead positions. Floor = 0.05 (config-exposed).

## Pilot arm set (revised 2026-07-21)

Four arms: `dense`, `static_sparse`, `sparse_momentum_uniform_init`,
`sparse_momentum_erk_init`. `oneshot_prune` is dropped from the pilot — it bears
on the Dice comparison, not the allocation question — and returns for the full
matrix.

The two `sparse_momentum` arms are identical except for where the budget sits at
step 0 (both start at global 0.30). Running both is the strongest test
available: if uniform-init and ERK-init converge to the same allocation, and it
is not ERK, the result is robust to initialization, which retires the standing
objection that the starting point determined the destination. This also
promotes the ERK-init run from a contingent pass-A/fail-B rescue (v1) to a core
arm.

## Fixed reference quantities (this architecture, density 0.30, floor 0.05)

- Global live budget: 4,205,261 of 14,017,536 encoder weights.
- Per-stage parameter share: s0 0.20%, s1 1.18%, s2 4.73%, s3 18.93%, s4 35.50%,
  s5 39.45%.
- Uniform-init per-stage budget share: 0.20 / 1.18 / 4.73 / 18.94 / 35.50 / 39.45 %.
- ERK per-stage budget share: 0.66 / 3.95 / 11.36 / 22.29 / 30.09 / 31.65 %.
- Shallow(0-2)/deep(3-5) budget: uniform 6.11 / 93.89 %; ERK 15.97 / 84.03 %.
- **D_ERK(uniform init) = 26.41 points** (per-stage budget-share L1 distance to ERK).

## Proposed Gate A: the instrument moved, and stabilized

Two conditions, both required.

**A1 (movement).** Over the final 30% of training, at least one encoder stage's
mean density departs from the 0.30 initialization by more than **0.05**.

- *Proof the quantity is free to vary (required by the redesign).* Under global
  conservation Σ n_live is fixed, but each stage's n_live is independently free
  in [ceil(0.05·n_stage), min(n_stage, budget − Σ_other floors)]. Every stage's
  interval contains densities both below 0.30 (down to the 0.05 floor) and above
  it (up to ≥0.38 even for the most budget-constrained stage). So a >0.05
  departure is attainable for at least one stage. Under the void v1 per-layer
  rule this quantity was identically 0 — that contrast is exactly what this
  condition now tests.
- *Max attainable.* Downward: 0.25 (0.30→floor) for every stage. Upward: +0.70
  for stages 0-3, +0.45 for stage 4, +0.38 for stage 5 (budget-capped). The
  0.05 bar is 5–14× below the ceiling and is 0 at initialization, so it is
  attainable and not trivially met.
- *Raw reported:* per-stage mean density over the final 30%, and the full
  per-stage density trajectory.

**A2 (stability).** Mean Kendall tau ≥ **0.8** between the per-stage density
ranking at each of the last 10 checkpoints and the ranking at the final
checkpoint. (Carried over from v1 unchanged.)

- *Max attainable:* tau ∈ [−1, 1]; the prune rate cosine-anneals to 0 at 75% of
  training, so the final 25% is frozen and a converged run gives tau = 1.0.
  Attainable; fails only if the allocation is still moving at the end.
- *Raw reported:* the 10 per-checkpoint tau values.

Gate A passes iff A1 and A2.

## Proposed Gate B: task-specific, and NOT the ERK prior

Under global redistribution ERK is now a real attractor — "drifts toward ERK"
is a live outcome, not the fixed offset it was in v1. The v1-style magnitude
test (D_ERK(final) large) does not work: it measures distance, not direction,
so a network that moves 15 points straight toward ERK and stops at 11 still
sits far from ERK and passes. That is exactly the Evci et al. replication Gate B
exists to exclude.

**The fix is geometric (Clark's formulation).** Let v = budget_share(final) −
budget_share(init) and u = budget_share(ERK) − budget_share(init), both in the
sum-zero tangent space. Decompose v along the ERK direction:

  - **ERK-ward component** = v · û
  - **Residual** = ‖v − (v·û)û‖  (movement ERK does not point at)

A pure drift toward ERK, of any magnitude, has residual **exactly 0** by
construction. A task-specific allocation ERK does not point at has a
substantial residual. Direction, not magnitude, is what discriminates.

Implemented and tested in `erk_ray_decomposition` (src/analysis/trajectory.py).
Computed on per-stage budget shares (6-dim), the granularity of the capacity
claim. Units: budget-share percentage points (Euclidean).

**B1 (moved off the ERK ray).** At the final checkpoint the residual exceeds
**5 points**.

> **Revised 2026-07-21, commit `a3ab865`, before any v2 arm was run.** B1 was
> first stated in `7a0c026` as a magnitude test: D_ERK(final), the L1 distance
> from ERK in per-stage budget share, exceeding **8 points**. That test measures
> distance, not direction. A run that moved 15 points straight toward ERK and
> stopped at 11 would still sit 11 from ERK and pass it, while having done
> nothing but partially reproduce the ERK prior — the exact outcome Gate B
> exists to exclude. The residual test replaces it and is the operative
> statement. Both formulations predate every v2 run; no v2 arm has been run
> under either. The superseded 8-point text is in the diff of `a3ab865`.

- *Calibration (why 5).* Every replication case scores residual exactly 0.00:
  staying at init, drifting fully to ERK, and drifting partway and stopping
  (e.g. the "moves 15 toward ERK, stops at 11" case → residual 0.00, ERK-ward
  +11.00). Task-specific allocations score: a mild task move ~3.9, a mid-stage
  bulge that keeps the shallow layers sparse (unlike ERK) ~8.4, a strong bulge
  ~10.2, an anti-ERK deep-enrichment ~54. The init→ERK axis length ‖u‖ is 12.37
  points, so a 5-point residual is ~40% of the entire ERK scale — a substantial
  off-axis excursion, clearly above frozen-tail settling jitter (the prune rate
  anneals to 0 by 75%, so the final 25% is frozen), and it cleanly clears a
  ~3.9 "mild" move without stamping it task-specific.
- *Max attainable.* Derived, not assumed: the maximum residual over the
  reachable budget polytope (each stage's share in [floor·paramshare/0.30,
  paramshare/0.30], summing to 1) is **64.39 points**, at the corner that
  masses the budget into stage 4. So the 5-point bar is ~13× below the ceiling
  — attainable with large margin, and not trivially met, since any positive
  residual already requires movement ERK does not predict.
- *Raw reported, always:* the residual, the ERK-ward component, the residual
  ratio (residual / ‖v‖), and the full D_ERK(t) trajectory. The ratio and the
  trajectory handle the ambiguous middle — a large ERK-ward component with a
  small residual reads as mostly replication — without a hard rule, as
  proposed.

**B2 (movement floor).** At least one stage's final budget share differs from
its uniform-init budget share by more than **3 points**.

- *Why keep it.* B1 already implies real movement (residual > 5 forces ‖v‖ > 5),
  so B2 is a secondary guard rather than the primary one, retained per Clark's
  instruction. It states the floor directly in per-stage budget space.
- *Max attainable.* Stage 5 can drop from 39.45% to floor (~6.6%) = 32.8 pts;
  the shallow block can rise 14.27 pts. The 3-pt bar is ~10× below the ceiling
  and 0 at init.
- *Raw reported:* per-stage |final − uniform-init| budget share.

Gate B passes iff B1 and B2.

## Diagnostics reported, not gated

- **Monotonicity in depth.** ERK is monotone decreasing; a mid-depth bulge is
  one non-ERK signature. As corrected in v1, it is sufficient but not necessary
  (ERK pins stages 0-1 dense, so a real departure can still be monotone).
  Reported, not gated.
- **Floor-binding.** Any stage sitting at the 0.05 floor at the end has its
  migration signal clipped there. Report which stages are floor-bound; if many
  are, the 0.30 budget / 0.05 floor pairing is too tight and must be revisited
  before the result is trusted.
- **Churn** and **regrowth-informativeness**, as in v1.

## Open judgment calls for Clark (flagged, not resolved)

1. **B1 threshold (5 pts residual).** The init→ERK axis is 12.37 points long and
   the maximum attainable residual over the reachable budget polytope is 64.39,
   so 5 points is roughly 40 percent of the ERK scale and about 13x below the
   ceiling. It clears a mild task-specific move (~3.9 in the calibration table)
   without stamping it task-specific, and any positive residual already requires
   movement ERK does not predict. A stricter or looser cut is defensible.
2. **A large ERK-ward component with a small residual is genuinely ambiguous.**
   The residual test cleanly excludes pure drift toward ERK, which scores
   exactly 0. It does not sharply separate the middle: a run can post a large
   ERK-ward component and a residual near the threshold, which reads as mostly
   replication with a small task-specific component. No threshold decides that
   case. This is why the residual ratio (residual / ‖v‖), the ERK-ward
   component, and the full D_ERK(t) trajectory are all required raw outputs
   rather than only the endpoint.
3. **Two movement thresholds** (A1 density 0.05; B2 budget 3 pts) are not
   redundant: shallow stages can move density a lot while moving little budget,
   and deep stages the reverse.
