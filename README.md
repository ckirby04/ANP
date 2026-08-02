# ANP: layer-wise capacity reallocation in 3D segmentation

nnU-Net configures its own architecture from a dataset fingerprint. This study
asks whether that auto-configured allocation of capacity across encoder depth is
wrong for the task it was configured for, and whether a network allowed to move
its own connections during training will move them somewhere systematically
different.

**The outcome measure is the layer-wise sparsity trajectory, not Dice.** Dice
establishes only that a method did not break. Where capacity migrates is the
result.

## The hypothesis, and how it fails

**Hypothesis.** Given a fixed connection budget and freedom to redistribute it
across layers during training, the network converges on an allocation that is
(a) different from where it started and (b) not explainable by a
task-independent prior.

The prior that has to be excluded is **ERK** (Erdős–Rényi–Kernel, Evci et al.
2020), which allocates density by a parameter-per-activation rule that knows
nothing about the task. ERK is computed in closed form for this architecture by
`src/sparsity/erk.py`.

The hypothesis is falsified if any of the following happens:

- **The allocation does not move.** Per-layer density stays at its
  initialization, or moves and never stabilizes. This is Gate A.
- **The allocation moves, but along the ERK direction.** A network that drifts
  toward ERK has rediscovered a known prior, not found anything about
  meningioma or about nnU-Net. This is Gate B, and it is the scientific gate.
- **The two initializations disagree.** Uniform-init and ERK-init arms that
  converge to different allocations mean the starting point determined the
  destination, and neither result says anything about the task.

A flat trajectory is a real result and is reported as one.

## Protocol history

This section exists because the study's central claim is that its decision
thresholds were fixed before the runs they judge. That claim is checkable
against the commit history, and the commands to check it are given below.

### v1: pre-registration and pilot

| | Commit | Timestamp |
|---|---|---|
| Gate A and Gate B defined | `e2a9e20` | 2026-07-19 15:22:42 |
| Gate B budget clause added | `a7ea388` | 2026-07-19 15:51:55 |
| First pilot arm starts | — | 2026-07-19 20:34:58 |
| First *sparse* arm (`rigl`) starts | — | 2026-07-19 23:32:33 |
| Last pilot arm finishes | — | 2026-07-20 11:51:08 |

Gate A required at least one encoder stage's mean density to depart from its
0.30 initialization by more than 0.05, with a stable stage ordering. Gate B
required the final allocation to depart from ERK by more than 0.05 density in
some stage *and* to move the deep/shallow budget split by more than 3 points.

The dynamic arm was RigL. Four arms ran at one seed for 100 epochs each: dense,
`rigl`, `static_sparse`, `oneshot_prune`.

### What the pilot revealed

**The v1 design could not test the hypothesis, for a reason provable from the
specification with no data at all.**

The v1 specification required equal prune and regrow counts *per layer*. That
makes each layer's density invariant across every update. The layer-wise density
trajectory — the quantity the whole study is built to measure — was therefore
flat at 0.30 by construction, for every sparse arm, before a single batch was
loaded.

Two consequences:

- **Gate A's density condition was unsatisfiable.** No layer's density could
  depart from 0.30 at all. Measured maximum deviation on `rigl_seed0` was
  0.0000, and could not have been anything else.
- **Gate B passed vacuously.** The deep-vs-ERK budget gap it measured is the
  fixed distance between uniform-0.30 and ERK. It is present at step 0, never
  changes, and reflects no reallocation whatsoever.

Compounding this, standard RigL holds per-layer sparsity fixed at
initialization and rewires only *within* layers. The mechanism could not move
capacity between layers even in principle. It was the wrong instrument for a
between-layer question.

Both errors originate in the v1 specification, not in its implementation. The
masking code enforced per-layer conservation exactly as written, and had a test
asserting it. That test is precisely what should have been a
per-layer-density-*can*-change test instead.

### Gate B was found under-strict before the run it judges, and left frozen

While writing tests for the gate arithmetic, the 3-point budget clause in Gate B
was measured against the case it was written to exclude. The clause exists so
that a departure confined to stages 0–2 — 6.1 percent of encoder parameters —
cannot pass on the density view alone. The measured bound is that driving all of
stages 0–2 to fully dense at constant overall density moves about **4.4 points**
of budget, which **clears the 3-point threshold.** The clause constrains, but
does not exclude the case it was written for. A threshold near 5 points would.

This was recorded at `76564ab`, **2026-07-19 20:40:43** — about 2 hours 51
minutes *before* the `rigl` arm started at 23:32:33, and roughly six hours
before it finished.

**The threshold was not amended.** It was identified as too loose before the run
it would judge, and left frozen at 3 points, with the argument recorded in the
repository and the limitation reported rather than fixed. The threshold text is
byte-identical in every version of `docs/preregistration.md`, which
`git log -p -S'more than 3 points'` will confirm.

A second argument recorded at the same commit: non-monotonicity in depth is a
*sufficient but not necessary* signature of a non-ERK allocation, because ERK
already pins stages 0 and 1 at density 1.000, so a genuine mid-depth departure
can still be monotone. The gate conditions were not changed; the prose claiming
non-monotonicity was "the sharpest available departure" was corrected at
`4e3ec2a` and explicitly labelled a description-only correction.

### The void, and the v2 redesign

| | Commit | Timestamp |
|---|---|---|
| v1 pilot declared void | `f435b37` | 2026-07-21 07:28:15 |
| Global redistribution added | `e72c832` | 2026-07-21 07:35:14 |
| v2 gates proposed | `7a0c026` | 2026-07-21 07:39:33 |
| Gate B1 revised to the ERK-ray residual | `a3ab865` | 2026-07-21 16:24:34 |

The v1 pilot is **void as a test of the hypothesis.** Every v1 gate, threshold,
prediction and outcome-table cell is void with it. The content was preserved in
place under a VOID banner rather than deleted, because the record of the error
is part of the result.

Restating gates for a redesigned mechanism is not post-hoc threshold tuning: no
answer to the original question was obtainable under the v1 specification, for
any data whatsoever, so there was no result to tune toward. The voiding commit
precedes the redesign commit, which is fixed by the parent chain and not merely
asserted by timestamps.

### Checking the timeline

Commit *order* is secured by the parent-hash chain — each commit cryptographically
commits to its parent, so the sequence cannot be altered without changing every
subsequent hash. Timestamps are separately checkable against GitHub's record of
the initial commit and of pushes.

```
git log --graph --oneline                     # void precedes redesign precedes gates
git log -p --follow docs/preregistration.md   # every edit to the v1 gates
git log -p -S'more than 3 points' -- docs/preregistration.md   # threshold never changed
git show 4e3ec2a -- docs/preregistration.md   # the one mid-pilot edit, gates untouched
git log --format='%h %ad %s' --date=iso -S'IN EFFECT' -- docs/preregistration_v2.md
```

`docs/preregistration.md` was edited twice after it was written. Both diffs are
above. One predates every run; the other, `4e3ec2a`, was made mid-pilot and
*weakened* a claim, retracting an overstatement about non-monotonicity. Neither
touched a gate condition. Note that the document's own rule was to place
revisions in a dated amendment section at the bottom; the correction was made
inline and the VOID banner was added at the top, so the document does not follow
its own amendment procedure.

## Current design

**Mechanism: SNFS-style sparse momentum** (Dettmers & Zettlemoyer 2019),
implemented in `src/sparsity/redistribute.py`. Prune the lowest-magnitude
fraction per layer, floor-protected; redistribute the freed budget across layers
in proportion to each layer's mean momentum magnitude; regrow within a layer at
the highest-momentum dead positions.

The change that matters: **conservation is global, not per-layer.** Total live
connections across the encoder are fixed at 0.30·N, while each layer's density
is free to move. Capacity can migrate between layers, which is the quantity v1
could not vary. A minimum density floor of 0.05 keeps every layer functional and
is config-exposed.

Sparsity applies to encoder 3×3×3 conv layers only — 11 layers, 14,017,536
weights, 45.5 percent of the 30.8M network. Stem conv, 1×1×1 convs, decoder, seg
heads, normalization and biases stay dense. Masking is a multiplicative mask on
weights, never `torch.sparse`, because the regrowth criterion needs the dense
gradient at masked positions.

### Planned arms

| Arm | Treatment |
|---|---|
| `dense` | Standard nnU-Net 3D, no sparsity. Control. |
| `static_sparse` | Random sparse mask at 0.30, fixed for all of training. |
| `sparse_momentum_uniform_init` | Global redistribution from a uniform-0.30 start. |
| `sparse_momentum_erk_init` | Global redistribution from an ERK-shaped start. |

`static_sparse` separates dynamic reallocation from the network merely being
overparameterized. It is not optional. The two initializations are the
robustness test: if both converge to the same non-ERK allocation, the result
does not depend on the starting point.

`oneshot_prune` is dropped from the pilot — it bears on the Dice comparison, not
the allocation question — and returns for the full matrix.

### v2 gates — PROPOSED, NOT IN EFFECT

The thresholds below are **not operative.** They become operative only through a
commit that changes the status line in
[`docs/preregistration_v2.md`](docs/preregistration_v2.md) to `IN EFFECT`; that
commit's hash and timestamp are the freeze record. No arm may be run against
gates that are not IN EFFECT.

- **A1** — at least one stage's mean density departs from 0.30 by more than
  **0.05** over the final 30% of training.
- **A2** — mean Kendall tau ≥ **0.8** across the last 10 checkpoints.
- **B1** — the **ERK-ray residual** exceeds **5 points**. Decompose the
  budget-share move into its component along the init→ERK direction and its
  orthogonal residual. A pure drift toward ERK, of any magnitude, has residual
  exactly 0. Direction, not distance, is what discriminates a task-specific
  allocation from a replication of ERK.
- **B2** — at least one stage's final budget share differs from its uniform-init
  budget share by more than **3 points**.

B1 replaced an earlier magnitude test (distance from ERK exceeding 8 points) at
`a3ab865`, before any v2 arm was run. A distance test passes a network that
moves most of the way toward ERK and stops short, which is the exact outcome
Gate B exists to exclude. Full derivation, calibration table and
maximum-attainable bounds are in `docs/preregistration_v2.md`.

## Status

**No v2 arm has been run.** The redesigned mechanism has code and tests but no
operative success criteria, and the gates above are PROPOSED.

**The four v1 arms ran on 2026-07-19/20 and are void.** Their aggregate
artifacts are published under `results/` — `provenance.json`, `trajectory.csv`
and `training_log.jsonl` per arm — so the v1 failure is inspectable rather than
merely described. `results/rigl_seed0/trajectory.csv` shows a density column
that is constant, which is the failure, visible directly.

Two observations survive the void, recorded as observations only — not as
findings, and not as evidence for the hypothesis the run could not test.
`rigl_seed0` reached the lowest final validation loss of the four arms
(−0.8491 against dense −0.8292), at n=1, on loss rather than Dice, with
within-layer rewiring only. The regrowth-informativeness probe falls with depth
at its first three measurements and **inverts at the fourth**, which the
original record did not capture; the correction and the per-probe values are in
[`docs/protocol_history.md`](docs/protocol_history.md).

Per-run timings and the commit each arm ran from are in
[`RUNS.md`](RUNS.md), with the strength of that evidence stated there.

Not built: held-out test-set evaluation. There is no `evaluate.py`. The 150-case
test cohort has no preprocessed data and would need a preprocessing pass before
any Dice number could be produced. No Dice number is reported anywhere in this
repository.

## Reproduction

Requires an nnU-Net-preprocessed BraTS-MEN 2023 dataset in `3d_fullres`
configuration. **The dataset is not distributed here and must be obtained from
its own source under its own terms.** No scans, labels, or per-subject derived
data are in this repository.

```
pip install nnunetv2 torch numpy scipy nibabel pyyaml pytest
```

Point the code at your copy of the data with **`ANP_DATA_ROOT`**, which should
be the directory containing `nnUNet/nnUNet_preprocessed/Dataset002_BraTS_MEN`:

```
$env:ANP_DATA_ROOT = 'D:\BraTS-MEN'      # PowerShell
export ANP_DATA_ROOT=/data/BraTS-MEN      # bash
```

Individual arms can override it by uncommenting `preprocessed_dir` and `raw_dir`
in the relevant `configs/*.yaml`. Then:

```
.\scripts\run_arm.ps1 -Arm dense -Seed 0
.\scripts\run_arm.ps1 -Arm sparse_momentum_uniform_init -Seed 0
python -m pytest tests/ -q
```

The test suite runs without the dataset. Tests that read the real preprocessed
cases skip with a message naming `ANP_DATA_ROOT` and what to set it to; the rest
pass. A clone with no data should show passes and skips, never failures.

Run scripts are PowerShell and were written for Windows. `src/train.py` is
platform-independent and can be invoked directly:

```
python src/train.py configs/dense.yaml --set training.seed=0
```

Analysis of the trajectory CSVs is in `notebooks/trajectory_analysis.ipynb`.

## Limitations

**v1 run timings are reconstructed, not recorded.** The v1 pipeline wrote no
wall-clock timestamp and no commit hash. The times in `RUNS.md` are filesystem
mtimes read after the fact, and the commit column there is inferred from the
log, not read from any artifact. One arm demonstrably launched from a working
tree that did not match its inferred commit. Runs from `44c957c` onward record
commit, tree-dirty state and UTC timestamps natively.

**The freeze mechanism is documentary, not enforced.** Nothing in the code
branches on gate status. `train.py` will run a v2 arm today regardless of what
the status line says. The check available to a reader is that the commit
flipping the status must predate the first run it judges, and both are visible
in the history.

**The v1 runs' configurations are not reproducible from this repository.** Each
run's `config.yaml` is gitignored because it embeds absolute dataset paths from
the machine it ran on. The published `provenance.json` carries a
`config_digest`, which establishes that two runs used the same configuration but
does not reconstruct it.

**Training is not bitwise reproducible.** `use_deterministic_algorithms` is off,
`cudnn.benchmark` and `cudnn.deterministic` are False, `CUBLAS_WORKSPACE_CONFIG`
is unset. 3D convolution backward accumulates non-deterministically. The data
pipeline *is* deterministic — patch sampling is keyed on `(seed, epoch, global
sample index)`, independent of worker count and iteration order — but the run as
a whole is reproducible in distribution, not exactly. Determinism was
deliberately not enabled mid-pilot, because it would have made later arms
non-comparable with earlier ones.

**All four v1 arms ran on an 8 GB card, not the intended 16 GB card.** CUDA's
default `FASTEST_FIRST` ordering disagrees with `nvidia-smi`, so the configured
`cuda:0` resolved to the smaller device. The runs fit and are valid, but did not
use the intended hardware. This is established directly only for
`oneshot_prune_seed0`, whose provenance records a device name; the other three
recorded only `cuda:0`, which a run on either card would report. Device
selection is now by name with a startup assertion.

**Deep supervision is on.** nnU-Net's deep-supervised loss changes encoder
gradient magnitudes, which is what the regrowth criterion reads. It is held
identical across arms, but it is a confound on the trajectory.

**The sub-5mm lesion bucket is annotation artifact, not small lesions.** Over
250 cases, 92.5 percent of sub-5mm ground-truth components are attributable to
dural-tail fragmentation, rim partial-volume or label speckle; 64 percent are a
single voxel. The stratified metric is reported as agreement with annotation
noise in that bucket. It supports no claim about micrometastasis or small-lesion
detection. See `docs/dataset_report.md`.

**Scope.** One seed, one fold, 100 epochs per arm in the pilot. No ensembling,
no multi-fold, no held-out test evaluation. This is a pilot designed to test
whether the instrument works, not whether the method wins.

**No efficiency claim is made.** Unstructured sparsity on this hardware runs at
dense speed or worse: the masks are dense multiplies and the gradients are
dense.

This is a research repository. It is not clinical software, has not been
validated for any clinical or diagnostic use, and nothing in it should be
treated as production-ready.

## Layout

```
configs/     one YAML per arm; seeds via CLI override
docs/        pre-registration v1 (void), v2 (proposed), dataset report
src/
  data/      dataset loading, deterministic splits, augmentation
  models/    nnU-Net 3D wrapper and device resolution
  sparsity/  masking, ERK, RigL, global redistribution, controllers
  metrics/   Dice, lesion-wise stratification, connected components
  analysis/  trajectory loading, gate evaluation, ERK-ray decomposition
  train.py
scripts/     PowerShell run drivers, diagnostics
tests/
notebooks/   trajectory analysis
results/     per-arm aggregate artifacts; weights and logs gitignored
RUNS.md      when each arm ran, and how strong that evidence is
```

The network is instantiated from the existing `nnUNetPlans.json` rather than
reimplemented, because that plan file *is* nnU-Net's auto-configured capacity
allocation, and therefore the object under study.

## Trajectory schema

`results/<run_id>/trajectory.csv`, one row per sparsified layer per logging
step:

```
run_id, seed, arm, step, epoch, layer_name, stage, density, n_pruned,
n_regrown, n_weights, n_live, live_budget_share
```

`density` and `live_budget_share` are both logged because they can tell
different stories. Encoder parameter counts are uneven enough that stages 0–2,
at 6.1 percent of parameters, can swing from 0.30 to 1.000 density while moving
only about 3.3 points of budget. Density answers "how connected is this layer";
budget share answers "where did the capacity go", which is the question being
asked. `n_weights` and `n_live` are logged so any aggregation is derivable
without re-running.

## Licensing and attribution

**This repository is MIT licensed.** See [`LICENSE`](LICENSE).

**No third-party source is vendored here.** nnU-Net and its ecosystem are
imported as dependencies, not copied into this repository, so they are governed
by their own terms and not by the license above. Versions tested, with license
as declared in each distribution's own metadata:

| Dependency | Version tested | License |
|---|---|---|
| nnU-Net (`nnunetv2`) | 2.6.4 | Apache 2.0 |
| `batchgenerators` | 0.25.1 | Apache 2.0 |
| `acvl_utils` | 0.2.5 | Apache 2.0 |
| PyTorch | 2.10.0+cu128 | BSD-3-Clause |
| `nibabel` | 5.3.3 | MIT |
| `dynamic_network_architectures` | 0.4.3 | not declared in package metadata; unknown |

Methods this work builds on, cited rather than reimplemented from scratch:
RigL (Evci et al. 2020) for the prune-and-regrow schedule and the ERK
allocation; sparse momentum (Dettmers & Zettlemoyer 2019) for global
redistribution.

**Data.** BraTS-MEN 2023 is governed by its own usage terms, which apply to the
dataset and not to the code in this repository. No part of it is redistributed
here in any form, including derived per-subject quantities. Obtaining the data,
and complying with those terms, is the responsibility of whoever runs this.
