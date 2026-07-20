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

### Correction: two claims I made that were wrong

**1. I said training is deterministic given the seed. It is not.** Verified
rather than assumed, at your prompting:

```
use_deterministic_algorithms : False
cudnn.benchmark              : False
cudnn.deterministic          : False
CUBLAS_WORKSPACE_CONFIG      : <unset>
```

We set none of these anywhere. 3D conv backward on cuDNN uses non-deterministic
atomic accumulation, so runs are not bitwise reproducible at fixed seed.

What is deterministic is the **data pipeline**: patch sampling is keyed on
`(seed, epoch, global sample index)`, independent of worker count and iteration
order. That is what `src/data/dataset.py` documents and it is true. I
over-generalised it to the whole training run when I told you the
`static_sparse` slowdown could not affect the trajectory. That inference was
unsupported. Corrected in README limitations, and `provenance.json` now records
the determinism flags so every run carries its own evidence.

Per your instruction determinism was **not** enabled, since that would make the
remaining arms non-comparable with `dense_seed0` and `rigl_seed0`.

**2. Every pilot arm ran on the wrong GPU.** `CUDA_DEVICE_ORDER` was unset, so
CUDA used `FASTEST_FIRST`, which ranks the two cards differently from
`nvidia-smi`:

```
default (what every run used):        PCI_BUS_ID (intended):
  cuda:0 -> RTX 3070 Ti   8.0 GiB       cuda:0 -> RTX 5060 Ti  15.9 GiB
  cuda:1 -> RTX 5060 Ti  15.9 GiB       cuda:1 -> RTX 3070 Ti   8.0 GiB
```

So `device: cuda:0` in every config resolved to the **8 GB** card. The runs fit
at about 87 percent of its VRAM and are valid, but they did not use the
intended hardware, and `provenance.json` recorded only `"cuda:0"` — which is
not evidence of anything, since it is what a run on either card would report.

This also inverts the contention diagnosis. The unattributed 3070 Ti load was
*us*; the process on the 5060 Ti is a **PI-CAI evaluation job** (PID 7844,
`eval_test_set_picai.py`, started 07-19 08:48, 18,315 CPU-s), which is not part
of this project and which you have asked me to leave alone. The
`static_sparse` slowdown to 0.998 s/iter was CPU and memory-bandwidth
contention with it, not GPU contention.

Fixed: `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set in `train.py` before `torch` is
imported and in `run_arm.ps1`; the device is resolved **by name** with the
index only as a fallback; a startup assertion fails loudly with the device name
and VRAM if the wrong card or too little memory is selected;
`torch.cuda.set_device` is called so `autocast` and `GradScaler` act on the
selected card; and provenance records the resolved name, VRAM, device order and
full visible-device inventory. `tests/test_device.py` covers all of it,
including a source scan asserting no code path targets bare `cuda` or `cuda:0`.

**`oneshot_prune_seed0` was NOT restarted** and continues on the 3070 Ti, per
your constraint. It is therefore consistent with the other three pilot arms,
all of which also ran there.

### Sharing the machine with PI-CAI

PI-CAI is expected to hold the **5060 Ti** for about 10 hours from 2026-07-20
morning. Two things follow.

**The device fix points future runs straight at it.** Correcting
`CUDA_DEVICE_ORDER` means `cuda:0` now resolves to the 5060 Ti, which is
PI-CAI's card. Left alone, the next matrix launch would have created direct GPU
contention with it. `scripts/run_arm.ps1` and `run_all.ps1` now take
`-Gpu 5060ti|3070ti`, which sets `device`, `require_device_name` and
`require_min_vram_gb` together.

**Overriding the index alone does not work, and does not fail either.** The
name match is what decides, so `--set device=cuda:1` on its own resolves back
to whatever `require_device_name` names, emits a warning, and runs on the card
you were trying to leave. That is a worse failure mode than an abort, because
it looks like it worked. Use `-Gpu`. Pinned by
`test_index_override_alone_is_ineffective_and_warns`.

**Recommendation while PI-CAI runs:** keep ANP on the **3070 Ti**
(`-Gpu 3070ti`), and use `-Workers 8` or so rather than the default 12. The
loader is memory-bandwidth bound, not just CPU bound, so it competes with
PI-CAI for more than cores. This also happens to be the consistent choice:
**all four pilot arms ran on the 3070 Ti**, so keeping the full matrix there
makes wall-clock comparable across every run rather than splitting the matrix
across two cards.

`oneshot_prune_seed0` was not restarted and remains on the 3070 Ti at its
original 12 workers.

### Arguments that a pre-registered element may be wrong

Per your standing instruction I am logging these and **proceeding under the
existing rules unchanged**. Nothing in `docs/preregistration.md` was modified.
Both surfaced while writing tests for the gate arithmetic, which is exactly
when you would want them to.

**1. The "non-monotonicity" signature may be close to unattainable in the
density view.** The pre-registration names a mid-depth bulge, stages 2-3 above
their ERK allocation, as the discriminating departure, and says ERK is monotone
decreasing in depth. Both true. But ERK already pins stages 0 and 1 at density
1.000. For the stage-density sequence to become non-monotone, some deeper stage
must exceed a shallower one, which means exceeding 1.0 where the shallow stages
are already saturated. A trajectory can therefore be a large, genuine,
task-specific departure from ERK at stages 2-3 and still be monotone decreasing
overall.

My test `test_mid_depth_bulge_passes_gate_b` constructs exactly such a case: it
clears both Gate B conditions and is still monotone. So **monotonicity is a
sufficient but not necessary signature**, and treating a monotone result as
"therefore ERK-like" would be a mistake. Gate B's actual conditions, per-stage
departure plus the budget clause, do the real work and are unaffected. I would
suggest demoting monotonicity from "the sharpest available departure" to one
diagnostic among several, but that is your call and I have not touched it.
`monotone_in_depth` is reported alongside the gates so you can see both.

**2. The 3-point budget clause has less headroom than implied.** The
pre-registration justifies the budget clause by noting stages 0-2 are 6.1
percent of parameters, so a departure confined to them should not pass. I
measured the actual bound. At ERK, stages 0-2 hold 16.0 percent of the live
budget; driving all of them to fully dense at constant overall density takes
them to about 20.4 percent. So **the largest possible shallow-only departure is
roughly 4.4 points, which clears the 3-point threshold.** The clause does
constrain, but it does not fully exclude the case it was written for. A
threshold near 5 points would; 3 does not. Frozen, unchanged, recorded here.
`test_shallow_only_departure_has_bounded_budget_effect` pins the measured bound
so it cannot drift.

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
