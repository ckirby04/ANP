# ANP: Artificial Neural Plasticity

Does a 3D segmentation network benefit from being allowed to reallocate its own
connectivity during training, rather than training a fixed architecture to
convergence?

The scientific question is not whether nnU-Net can be compressed. It is whether
**the pattern of where connections migrate** reveals that nnU-Net's
auto-configured capacity allocation is wrong for the task it was configured for.

**The primary readout is the layer-wise sparsity trajectory, not the Dice
score.** Dice establishes that the method did not break. The trajectory is the
result.

## Arms

Four arms with identical data splits, schedule, augmentation, and optimizer
settings. Only the connectivity treatment differs.

| Arm | Treatment |
|---|---|
| `dense` | Standard nnU-Net 3D, no sparsity. Control. |
| `static_sparse` | Random sparse mask at target density, fixed for all of training. |
| `oneshot_prune` | Train dense, prune to target density by weight magnitude at a set epoch, fine-tune. |
| `rigl` | RigL dynamic sparse training: periodic prune-by-magnitude, regrow-by-gradient-magnitude, at fixed density. |

`static_sparse` is the arm that determines whether any observed effect is due to
dynamic reallocation or merely to the network being overparameterized. It is not
optional.

Target density is 0.30, configurable. Sparsity is applied to encoder 3x3x3 conv
layers only. The stem conv, all 1x1x1 convs, seg heads, decoder, normalization,
and biases stay dense.

Masking is implemented as a **multiplicative mask on weights**, never as
`torch.sparse`. RigL's regrowth criterion needs the dense gradient at masked
positions, which sparse tensors would not provide.

## Data

BraTS-MEN 2023 (meningioma), 1000 cases, binarized to whole tumor. Single fold:
train on fold 0 (680), model selection on its validation split (170), headline
numbers on the held-out test set (150). No ensembling, no multi-fold.

See [`docs/dataset_report.md`](docs/dataset_report.md) for the full dataset
characterization.

## Pre-registration

The pilot's pass/fail criteria and both directional predictions are fixed in
[`docs/preregistration.md`](docs/preregistration.md), committed before the run.
A flat trajectory is a real result, not something to be squinted past.

Two gates. **Gate A** asks whether density moves at all and settles into a
stable stage ordering; failing it means the RigL hyperparameters are mistuned
for 3D. **Gate B** asks whether the movement is distinguishable from the
Erdos-Renyi-Kernel prior, which allocates density by a task-independent
parameter-per-activation rule and is computed in closed form by
`src/sparsity/erk.py`.

Gate B exists because ERK alone shifts the deepest stage by 0.059 and would
pass Gate A on its own. Deep-stage drain is what ERK predicts for any conv net
regardless of task, so observing it would not support the capacity
misallocation claim. The discriminating signature is **non-monotonicity**: ERK
is monotone decreasing in depth, so a mid-depth bulge at stages 2-3 above their
ERK allocation cannot be produced by the null.

## Known limitations

These are deviations from canonical practice, recorded up front rather than
discovered later.

**The sub-5mm lesion bucket is annotation artifact, not small lesions.** Over
250 cases, 92.5 percent of sub-5mm ground-truth components are attributable to
dural-tail fragmentation, rim partial-volume, or label speckle; 64 percent are a
single voxel. The stratified metric is reported honestly as agreement with
annotation noise in that bucket. It does not support claims about
micrometastasis or small-lesion detection, which would require a metastases
dataset. Analysis and evidence in `docs/dataset_report.md`.

**`oneshot_prune` prunes a model that has not fully converged.** The canonical
recipe prunes after convergence. At the epoch budget used here the dense arm has
not reached the plateau the prior 315-epoch reference run did, so the pruning
step lands on a still-improving model. This is a real deviation and is not
presented as the canonical method.

**Deep supervision is on.** nnU-Net's default deep-supervised loss changes
encoder gradient magnitudes, which is what RigL's regrowth criterion reads. It
is held identical across all four arms, but it is a confound on the trajectory
and is named as one.

## On efficiency

This method does not make training faster. Unstructured sparsity on this
hardware runs at dense speed or worse, because the masks are dense multiplies
and the gradients are dense. No efficiency claim is made anywhere in this work.

## Layout

```
configs/     one YAML per arm; seeds via CLI override
src/
  data/      BraTS-MEN loading, splits, preprocessing
  models/    nnU-Net 3D wrapper
  sparsity/  masking, RigL scheduler, pruning ops
  metrics/   Dice, lesion-wise stratification, connected components
  train.py
  evaluate.py
scripts/     PowerShell run drivers, diagnostics
tests/
results/     CSVs and checkpoints, gitignored
notebooks/   analysis of the trajectory CSVs
```

The model is instantiated from the existing `nnUNetPlans.json` rather than
reimplemented, because that plan file *is* nnU-Net's auto-configured capacity
allocation and therefore the object under study.

## Scientific output

`results/trajectory/*.csv`, one row per sparsified layer per logging step:

```
run_id, seed, arm, step, layer_name, stage, density, n_pruned, n_regrown,
n_weights, n_live, live_budget_share
```

Designed to be joined and plotted. This file is the result.

`density` and `live_budget_share` are both logged because they can tell
different stories. Encoder parameter counts are uneven enough that stages 0-2,
at 6.1 percent of parameters, can swing 0.30 to 1.000 density while moving only
3.3 points of budget. Density answers "how connected is this layer"; budget
share answers "where did the capacity go", which is the question the paper
asks. `n_weights` and `n_live` are logged so any aggregation, per stage or
per shallow/deep split, is derivable without re-running.
