# Notes for Clark

Newest entry at the top.

---

## Session 1 — overnight autonomous run

### Status

Steps 3, 4 and 5 are complete, committed, and tested. **88 tests pass.** The
four-arm pilot (1 seed x 100 epochs, order dense / rigl / static_sparse /
oneshot_prune) is running. Nothing has been pushed. Nothing outside `G:\ANP`
was written or modified.

Commits this session:

| commit | contents |
|---|---|
| `9578c96` | model wrapper, training loop, dense config, step 3 |
| `b325aa2` | Dice and lesion stratification, step 4 |
| `9ecbee7` | sparsity machinery, all four arm configs, run scripts, step 5 |

### The single most important thing to look at first

**The step-3 timing investigation, and specifically that my first two
measurements were both wrong in ways that looked fine.** Details below under
"Surprises". The short version: I reported 0.364 s/iter from a median, the
mean was 1.27 s/iter, and the actual cause was a configuration mistake of
mine, not anything inherent. Final measured rate is 0.360 s/iter with mean,
median and p90 all agreeing. If any single thing in this session deserves your
scepticism, it is that I nearly certified a data-bound loop as compute-bound.

### Step 3 timing number and gate branch

**Final: 0.360 s/iter mean, 0.360 p90.** Pure-GPU floor measured separately at
0.332 s/iter. Reference baseline was 0.71 s/iter.

**Branch taken: under 0.85 s/iter, compute-bound, proceeded to step 4.**

The route there was not direct, and the intermediate numbers are worth keeping:

| configuration | s/iter (mean) | note |
|---|---|---|
| 4 train / 4 val workers | 1.27 | over the 1.20 gate |
| 12 train / 12 val workers | 1.09 | barely improved |
| 14 train / 3 val workers | **0.360** | mean = median = p90 |

Pilot projection at this rate: about 105 s/epoch, roughly 2.9 h per arm and
about 12 h for the four-arm pilot.

### Decisions I made that were not pre-answered

Each of these is reversible; the reasoning is given so you can overrule it.

**1. Reproduce nnU-Net's augmentation by calling its own transform builders,
rather than reimplementing them or subclassing `nnUNetTrainer` wholesale.**
Identical augmentation across arms is satisfied by any consistent choice, but
the 0.71 s/iter and 0.90 pseudo-Dice reference points only transfer if the
pipeline matches the reference run. Writing a fresh augmentation stack
unattended was the highest-risk silent-bug surface in the build. Confined to
`src/data/augmentation.py`.

**2. `data.num_workers = 12` default, `data.val_num_workers = 3`.** See
"Surprises" item 2. The asymmetry is the fix for the throughput problem, and
it is also a correctness fix: at 16 train + 16 val workers the run crashes
outright.

**3. Validation is 25 batches of patch-size samples, every epoch.** Not
pre-specified. It costs about 7-16 s per epoch. I kept it every epoch rather
than every N epochs because it is cheap at this cost and the val-loss curve is
worth having per-epoch. Trivially changed via `Trainer.n_val_batches`.

**4. Face adjacency (6-neighbour) as the default connected-component
connectivity**, rather than 26-neighbour. 26-neighbour merges lesions that
touch only at a corner, which would undercount lesions in exactly the small
buckets under scrutiny. Configurable, and a test contrasts the two.

**5. Detection threshold `min_overlap = 0.1`** for a per-lesion detection.
Not pre-specified. It is config-exposed and tested at the boundary.

**6. Set git `user.email` / `user.name` repo-local only.** Already flagged and
you approved keeping it.

**RigL hyperparameters were left at the given defaults** — interval 100, initial
drop fraction 0.3, cosine decay to zero at 75%. I changed nothing, so there is
nothing to log there.

### Surprises

**1. My step-3 timing was wrong twice, and the second failure was mine.**

First I measured 0.364 s/iter on a 5-case subset. That is a lower bound, not a
measurement: with 5 cases the blosc2 blocks sit in the OS cache. On the full
680-case split it went to 1.48 s/iter implied.

Then I reported 0.368 s/iter median from the full split, which was also wrong,
because the loader stalls periodically and the median steps neatly over the
stalls. Mean was 1.27. **The reported statistic mattered more than the
measurement.** I have changed the training loop to log mean, median and p90
together, and to print the mean, precisely so this cannot recur silently.

The cause turned out to be my own configuration: the validation loader was
holding as many persistent worker processes as the training loader, so 24+
processes contended for memory bandwidth and blosc2 file handles and starved
the training stream. Dropping validation to 3 workers took it to 0.360 s/iter
with mean, median and p90 all in agreement. It was never an inherent property
of the data path.

**2. 16 train + 16 val workers crashes outright**, with
`RuntimeError: Error while getting the buffer` from blosc2 inside a validation
worker. This is a hard failure, not a slowdown, and I would have hit it
mid-pilot rather than during a smoke test if I had not been chasing the timing
number. Root cause is process count against the blosc2 files.

**3. nnU-Net's deep-supervision scales are NOT reversed, and I reversed them.**
`_get_deep_supervision_scales` returns highest-resolution-first, matching the
order the network emits its outputs. My `[::-1]` paired every output with a
wrong-resolution target. This crashed loudly, which is the benign version;
had the resolutions happened to be compatible it would have trained on
mismatched targets and produced plausible-looking garbage.

**4. The training dataloader must crop nnU-Net's *initial* patch size
`[224, 228, 189]`, not the final `[128, 160, 112]`.** `SpatialTransform` crops
down after rotating and scaling. My step-2 loader sampled at the final size,
which would have left rotation border artifacts absent from the reference run.
Consequence worth knowing: the volumes are only about 140x175x135, so **66
percent of every initial patch is zero padding**, and the resulting 154 MB
per-sample allocation is the reason the data path is memory-bandwidth bound
rather than CPU bound. Loader throughput does not improve past about 8 workers.

**5. The `-1` label must survive to the transforms.** This corrects the step-2
finding rather than contradicting it. `MaskImageTransform` reads `seg < 0` to
mask the image, and only afterwards does `RemoveLabelTansform(-1, 0)` strip it.
`use_mask_for_norm` is `[True, True, True, True]` here, so that transform is
active. Remapping `-1` to background in the dataset, as step 2 did
unconditionally, would have silently turned it into a no-op. The remap now
applies only in raw mode; the transform path passes `-1` through.

**6. A momentum-clearing bug that a test caught.** `reset_optimizer_state` was
indexing the momentum buffer with flat indices while the buffer holds the
weight's 5-D shape. Instead of zeroing the regrown positions it was zeroing
whole output channels, and it only raised an error once an index happened to
exceed dim 0. Fixed to flatten first.

**7. PowerShell 5.1 writes a BOM with `-Encoding utf8`**, which made the
generated arm configs unparseable. The config loader rejected the unknown
top-level key loudly instead of ignoring it, which is the behaviour I want, but
worth knowing if you generate configs from PowerShell.

**8. An early observation from the regrowth-informativeness probe, recorded and
NOT interpreted.** In the smoke run, top-k overlap between a
foreground-oversampled and a background-dominated batch decreases with depth:
0.69 at stage 0, 0.61 and 0.56 at stage 1, 0.39 and 0.37 at stage 2. This is a
40-iteration smoke run with an untrained network and means nothing on its own.
Flagging only because if the pattern persists in the pilot it bears on the
third-explanation-of-a-null question.

### What I did NOT do

- **Did not push.** Per instruction. Local commits only.
- **Did not run the ERK-init arm.** It is pre-registered as contingent on the
  pass-A / fail-B cell and that cell is not determined.
- **Did not interpret pilot results against the gates.** Numbers only, once the
  pilot finishes.
- **Did not change any pre-registered gate, threshold, prediction or
  outcome-table cell.** The only edit to `docs/preregistration.md` was the
  pre-answered third-interpretation note, added as prose in the contingency
  section, not as a gate change.
- **Did not write `evaluate.py` or the held-out test-set evaluation.** The 150
  test cases have no preprocessed `.b2nd` data and need preprocessing at
  evaluation time. That is real work and it is not on the critical path for
  the pilot, whose readout is the trajectory rather than Dice.
- **Did not re-engineer the oversized-initial-patch data path.** It is the
  reason the loop sits at 0.360 rather than nearer the 0.332 GPU floor, but
  changing it risks diverging from nnU-Net's augmentation semantics, which is
  not a thing to do unattended.
- **Did not tune RigL hyperparameters.** Defaults were numerically stable.

### Open questions for you

1. **Validation currently measures loss only, not Dice.** The pilot's readout
   is the trajectory, so this does not block it, but before the full matrix
   you will want per-epoch validation Dice for model selection. It needs
   sliding-window inference, which is a distinct piece of work.
2. **The 150-case held-out cohort needs a preprocessing pass** before any
   headline number can be produced.
