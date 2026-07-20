# Notes for Clark

Newest entry at the top.

---

## Session 1 — overnight autonomous run

**Status:** in progress. This file is updated as work completes; if it still
says "in progress" at the top, the run was interrupted mid-step.

### The single most important thing to look at first

Not yet determined. Will be filled in before the session ends.

### Progress log

**Step 3 (dense arm end-to-end) — in progress.**

Built `src/models/network.py`. The architecture is instantiated from
`nnUNetPlans.json` via nnU-Net's own `get_network_from_plans` rather than
reimplemented, per the approved design.

Verified consistency that the whole ERK comparison depends on: the layer names
and weight shapes read off the real instantiated network match exactly what
`src/sparsity/erk.py` derives independently from the plans file. If those had
drifted, the ERK null would not have been joinable to the observed trajectory
and the mismatch would probably not have surfaced until figure-making.

Resulting selection: **11 sparsifiable layers, 14,017,536 parameters**, which
is 45.5 percent of the 30,788,586-parameter network. Stem excluded, decoder
excluded, 3x3x3 only, as pre-registered.

### Decisions made that were not pre-answered

**Reproducing nnU-Net's augmentation rather than writing my own.** The four
arms need identical augmentation, and that is satisfied by any consistent
choice. But the 178 s/epoch and 0.90 pseudo-Dice reference points only
transfer if the pipeline matches the reference run. Writing a fresh
augmentation stack unattended overnight is the highest-risk silent-bug surface
in this build, so I am calling nnU-Net's own `get_training_transforms` and
`get_validation_transforms` from inside this project's training loop, rather
than either reimplementing them or subclassing `nnUNetTrainer` wholesale.

This keeps the module boundaries in the spec while removing the risk of a
subtly different augmentation invalidating the baseline comparison. Reversible:
it is confined to the transform construction in the training module.

### Surprises

None yet this session beyond the step-2 `-1` finding already recorded in git.

### What I did not do

Nothing deliberately skipped yet.
