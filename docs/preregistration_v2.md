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
is a live outcome, not the fixed offset it was in v1. Gate B must therefore
reject two nulls at once: the allocation collapsing to ERK (rediscovery), and
the allocation not really moving (the v1 vacuous-pass failure). Two conditions,
both required.

**B1 (not ERK).** At the final checkpoint the per-stage budget-share L1 distance
to ERK, D_ERK(final), exceeds **8 points**.

- *Max attainable / triviality.* D_ERK ∈ [0 (exact ERK), ~52 (budget massed
  opposite to ERK)]; init sits at 26.41. So >8 is attainable. **It is NOT
  sufficient alone**: staying at uniform init gives 26.41 > 8, which is exactly
  the v1 vacuous pass. B1 must be paired with B2.
- *Raw reported:* per-stage |final − ERK| budget share, and the full D_ERK(t)
  trajectory so the direction of drift is visible.

**B2 (not uniform either).** At least one stage's final budget share differs
from its uniform-init budget share by more than **3 points**.

- *Why.* B1 rejects ERK; B2 rejects "far from ERK only because it never moved."
  Together they require the allocation to settle at a third place — distinct
  from both the initialization and the prior — which is the task-specific
  finding. B2 guards the void loophole directly in budget space, as Gate A's
  A1 does in density space.
- *Max attainable.* Stage 5 can drop from 39.45% to floor (~6.6%) = 32.8 pts;
  the shallow block can rise 14.27 pts. The 3-pt bar is ~10× below the ceiling
  and 0 at init. Attainable, not trivial.
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

1. **B1 threshold (8 pts).** Init is 26.4 from ERK, exact ERK is 0. 8 pts
   demands the allocation land clearly closer to ERK than to init before it
   counts as "not ERK." A stricter or looser cut is defensible.
2. **Partial drift toward ERK is genuinely ambiguous.** If D_ERK(final) lands
   between 8 and ~20, the allocation moved but partway toward ERK. B1 passes,
   yet calling that "task-specific" vs "partial rediscovery" is a judgment the
   thresholds cannot make. This is why the full D_ERK(t) trajectory is a
   required raw output rather than only its endpoint.
3. **Two movement thresholds** (A1 density 0.05; B2 budget 3 pts) are not
   redundant: shallow stages can move density a lot while moving little budget,
   and deep stages the reverse.
