# Pre-registration

Committed before the pilot run. The point of this document is that a flat or
null result is a real result, rather than something reinterpreted after seeing
the plot.

Nothing in here may be edited after the pilot begins. Revisions go in a dated
amendment section at the bottom, with the reason.

## Pilot design

1 seed x 4 arms x 100 epochs, approximately 22 hours. The pilot tests whether
the instrument works, not whether the method wins. Dice is not a pilot
criterion.

Scaling to 3 seeds x 200 epochs (approximately 5.3 days) is conditional on the
gates below.

## Gate A: does anything move, stably

> The pilot passes if, over the final 30 percent of training, the mean layer
> density of at least one encoder stage differs from 0.30 by more than 0.05,
> and the ordering of stage densities is stable across the last 10 logged
> checkpoints.

Ordering stability is scored as Kendall tau between the stage-density ranking
at each of the last 10 checkpoints and the ranking at the final checkpoint.
Stable means mean tau >= 0.8.

Gate A failing means the RigL update interval and drop fraction are mistuned
for 3D conv and need retuning before any full matrix is worth running.

**Gate A alone is not sufficient**, for the reason in Gate B.

## Gate B: is the movement distinguishable from the ERK null

The Erdos-Renyi-Kernel prior (Evci et al. 2020) allocates density by
(sum of weight dims) / (product of weight dims). It is a task-independent
parameter-per-activation heuristic. Computed in closed form for this
architecture at overall density 0.30 by `src/sparsity/erk.py`:

| stage | ERK density | deviation from 0.30 | share of encoder params |
|---|---|---|---|
| 0 | 1.000 | +0.700 | 0.2% |
| 1 | 1.000 | +0.700 | 1.2% |
| 2 | 0.720 | +0.420 | 4.7% |
| 3 | 0.353 | +0.053 | 18.9% |
| 4 | 0.254 | -0.046 | 35.5% |
| 5 | 0.241 | -0.059 | 39.4% |

ERK is monotone decreasing in depth.

### Density is not the headline readout; budget share is

Encoder parameter counts are extremely uneven, so per-layer density overstates
small stages. Stages 0-2 together hold 6.1 percent of encoder parameters: they
can swing from 0.30 to 1.000 density, a 3.3x change that looks dramatic on a
density plot, while moving only 3.3 points of the live-parameter budget.

Share of the live-parameter budget is what "capacity allocation" means, and it
is the headline figure. Computed by `live_budget_share` in
`src/sparsity/erk.py`:

| stage | share of params | budget share, uniform | budget share, ERK | delta |
|---|---|---|---|---|
| 0 | 0.2% | 0.2% | 0.7% | +0.5 |
| 1 | 1.2% | 1.2% | 3.9% | +2.8 |
| 2 | 4.7% | 4.7% | 11.4% | +6.6 |
| 3 | 18.9% | 18.9% | 22.3% | +3.4 |
| 4 | 35.5% | 35.5% | 30.1% | -5.4 |
| 5 | 39.4% | 39.4% | 31.7% | -7.8 |

Aggregated: ERK moves **9.9 points of budget** from the deep stages (3-5) to
the shallow ones (0-2), 93.9 percent to 84.0 percent. That is the full size of
the null's effect on capacity allocation, and it is far less dramatic than the
density view suggests.

Both views are logged. Gate B is evaluated on **both**, and a result where the
two disagree is reported as such rather than represented by whichever is more
striking.

Note that **ERK alone moves stage 5 by 0.059, which already clears Gate A's
0.05 threshold.** A run that merely rediscovers ERK would pass Gate A and
look like a result. It is not one. Deep-stage drain is predicted by ERK for
any conv net regardless of task, so observing it says nothing about nnU-Net's
allocation being wrong for meningioma.

Gate B is therefore the scientific gate:

> The trajectory is task-informative only if the final stage-density
> allocation departs from the ERK allocation by more than 0.05 in at least one
> stage, in a direction ERK does not predict, **and** the deep-to-shallow
> budget split departs from ERK's 84.0 / 16.0 by more than 3 points.

The budget clause exists so that a departure confined to stages 0-2, which are
6.1 percent of parameters, cannot pass on the density view alone.

The sharpest available departure is **non-monotonicity**. ERK is monotone
decreasing in depth. A mid-depth bulge, meaning stages 2-3 ending above their
ERK allocation while stages 4-5 sit at or below theirs, cannot be produced by
ERK and would be a genuine task-specific finding.

## Directional predictions

Stated in advance so they can be wrong.

### Prediction 1 (Kirby)

RigL drains the deepest stages (features 320, 320) and enriches stages 2-3.

Reasoning: nnU-Net's channel schedule caps at 320 by a hardcoded rule rather
than anything data-derived, and meningioma is a large, high-contrast,
dural-based target that plausibly does not need much deep global context.

Status against the null: **discriminating.** Enrichment peaking at stages 2-3,
above their ERK allocation, is the mid-depth bulge Gate B asks for. ERK
predicts stage 2 at 0.720 and stage 3 at 0.353 as part of a monotone decrease,
not as a peak.

### Prediction 2 (Claude)

RigL drains stages 4-5 and enriches stages 0-1, following the ratio of
parameters to spatial positions.

Reasoning: stage 5 operates on a 4 x 5 x 7 grid while carrying 2.76M
parameters per conv, roughly 19,700 parameters per spatial position, against
0.4 for stage 1. Gradient magnitude at a masked position accumulates over
spatial positions, so high-resolution early layers generate large gradients
over few parameters.

Status against the null: **not discriminating, and recorded as such.** This
prediction was made before the ERK allocation was computed, and it turns out
to be approximately the ERK allocation itself. If it is confirmed, it confirms
Evci et al. 2020 and says nothing about this architecture or this task. It is
retained here as a record of what was predicted, not as a hypothesis worth
confirming.

### Secondary prediction (Claude)

Within each stage, the first conv, which expands channels and carries the
stride, retains density better than the second conv. Untested in the
literature at 3D and cheap to read off the same CSV.

## Outcomes and what each licenses

| Gate A | Gate B | Reading |
|---|---|---|
| fail | n/a | Instrument mistuned. Retune update interval and drop fraction for 3D before spending the full matrix. No scientific claim. |
| pass | fail | RigL reallocates, but along ERK lines. Paper is "RigL rediscovers ERK in 3D segmentation," a replication. The capacity-misallocation claim is not supported. **Triggers the ERK-init follow-up below.** |
| pass | pass, matching Prediction 1 | Confirmed directional hypothesis. nnU-Net's allocation is misconfigured for this target in a specific, predicted way. |
| pass | pass, other direction | nnU-Net's capacity allocation is non-obvious and does not follow the standard prior. Publishable, weaker, honest. |

## Contingent follow-up: ERK-initialized RigL

Pre-specified here so that if it is run, it is a planned contingency and not a
post-hoc rescue attempt.

**Trigger condition, and the only one:** Gate A passes and Gate B fails. It is
run in no other branch.

**Why only that branch.** All arms initialize at uniform density 0.30, which
means ERK sits away from the starting point and acts as an attractor. If Gate
B passes, task-specific departure from the null is already demonstrated and
ERK-init adds nothing. If Gate A fails, the instrument is mistuned and no
initialization question is meaningful yet. Only in the pass-A / fail-B cell is
the result ambiguous between two distinct explanations:

1. There is no task-specific signal; RigL converges to ERK because ERK is
   correct for this architecture.
2. There is a weaker task-specific signal that uniform initialization swamped,
   because drift from uniform toward ERK dominates the trajectory.

**Design.** One additional arm, `rigl_erk_init`, identical to `rigl` except
that the initial mask is drawn at the ERK per-layer densities rather than
uniform 0.30. Overall density remains 0.30, and the per-update density
conservation constraint is unchanged.

**Decision rule, fixed now.** Starting at ERK, explanation 1 predicts the
allocation stays at ERK; explanation 2 predicts it departs. The follow-up
supports a task-specific signal if the final allocation departs from its ERK
starting point by more than 0.05 density in at least one stage and more than 3
points of deep-to-shallow budget split, using the same thresholds as Gate B.
Staying put is a null result and is reported as one.

**Disambiguating a static allocation.** If the allocation stays at ERK, a
mistuned drop fraction and a genuinely converged mask look identical at the
tail. They differ in history: a mistuned drop fraction gives low churn from
step one because the mask was never asked to move, while a converged mask
gives high churn early that decays as the topology settles. Churn is
`(n_pruned + n_regrown) / n_live` per update, already derivable from the
trajectory CSV, so this is read off the **early-training segment** of the
existing pilot data at zero extra cost. Read it before concluding anything
from a flat tail.

**Cost.** 1 seed x 100 epochs, approximately 5.5 hours, before any decision to
extend to 3 seeds.

## Out of scope

The sub-5mm lesion bucket does not support small-lesion or micrometastasis
detection claims on this dataset. See `docs/dataset_report.md`. Stratified
Dice is still reported, labeled as agreement with annotation noise in that
bucket. Any metastasis claim requires BraTS-METS and is not made here.
