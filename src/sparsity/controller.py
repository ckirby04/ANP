"""Connectivity treatment per arm, and the trajectory CSV.

The trajectory CSV is the scientific output of this project. One row per
sparsified layer per logging step:

    run_id, seed, arm, step, epoch, layer_name, stage, density,
    n_pruned, n_regrown, n_weights, n_live, live_budget_share

`density` and `live_budget_share` are both logged because they tell different
stories: the shallow stages hold about 6 percent of encoder parameters, so they
can swing from 0.30 to 1.00 density while moving almost no capacity. Density
answers "how connected is this layer", budget share answers "where did the
capacity go". Raw counts are logged so any aggregation is derivable later
without re-running.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from models.network import sparsifiable_encoder_convs, stage_of

from .erk import erk_densities
from .masking import MaskedLayers
from .redistribute import sparse_momentum_update
from .rigl import cosine_drop_fraction, regrowth_scores, rigl_update, topk_overlap

TRAJECTORY_FIELDS = (
    "run_id", "seed", "arm", "step", "epoch", "layer_name", "stage",
    "density", "n_pruned", "n_regrown", "n_weights", "n_live",
    "live_budget_share",
)

PROBE_FIELDS = (
    "run_id", "seed", "arm", "step", "layer_name", "stage", "k",
    "topk_overlap_fg_vs_bg", "n_dead",
)


class SparsityController:
    """Base: owns masks, logs the trajectory. Subclasses define the treatment."""

    arm = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.masked: MaskedLayers | None = None
        self.run_dir: Path | None = None
        self._last_stats: dict[str, dict[str, int]] = {}
        self._epoch = 0
        self._probe_steps: set[int] = set()

    # --- lifecycle --------------------------------------------------------

    def attach(self, network, run_dir: Path, cfg) -> None:
        convs = sparsifiable_encoder_convs(
            network, include_stem=cfg.sparsity.include_stem)
        self.masked = MaskedLayers(convs)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._traj_path = self.run_dir / "trajectory.csv"
        self._probe_path = self.run_dir / "regrowth_informativeness.csv"
        for path, fields in ((self._traj_path, TRAJECTORY_FIELDS),
                             (self._probe_path, PROBE_FIELDS)):
            if not path.exists():
                with open(path, "w", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=list(fields)).writeheader()

        n = self.cfg.sparsity.n_informativeness_probes
        if n > 0:
            total = self.cfg.total_steps
            self._probe_steps = {
                int(round((i + 0.5) * total / n)) for i in range(n)}

        self.initialize_masks()

    def initialize_masks(self) -> None:
        """Default: fully dense. Subclasses override."""

    def target_densities(self) -> dict[str, float]:
        return {name: self.cfg.sparsity.density for name in self.masked.masks}

    # --- hooks called from the training loop ------------------------------

    def before_optimizer_step(self) -> None:
        """Mask gradients so momentum does not accumulate on dead weights."""
        if self.masked is not None:
            self.masked.apply_to_grads()

    def after_optimizer_step(self, step: int, trainer) -> None:
        if self.masked is None:
            return
        self.masked.apply()
        self._epoch = trainer.epoch
        every = self.cfg.logging.trajectory_every_n_steps
        if every and step % every == 0:
            self.log_trajectory(step)

    def epoch_summary(self) -> dict:
        if self.masked is None:
            return {}
        return {
            "overall_density": self.masked.overall_density(),
            "n_live": self.masked.total_live(),
        }

    # --- logging ----------------------------------------------------------

    def log_trajectory(self, step: int) -> None:
        total_live = self.masked.total_live()
        rows = []
        for name in self.masked.masks:
            n_w = self.masked.n_weights(name)
            n_l = self.masked.n_live(name)
            stats = self._last_stats.get(name, {})
            rows.append({
                "run_id": self.cfg.run_id,
                "seed": self.cfg.seed,
                "arm": self.cfg.arm,
                "step": step,
                "epoch": self._epoch,
                "layer_name": name,
                "stage": stage_of(name),
                "density": n_l / n_w,
                "n_pruned": stats.get("n_pruned", 0),
                "n_regrown": stats.get("n_regrown", 0),
                "n_weights": n_w,
                "n_live": n_l,
                "live_budget_share": (n_l / total_live) if total_live else 0.0,
            })
        with open(self._traj_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(TRAJECTORY_FIELDS)).writerows(rows)
        # Counts describe the interval just ended, so clear them.
        self._last_stats = {}

    # --- diagnostic -------------------------------------------------------

    def maybe_probe_informativeness(self, step: int, trainer) -> None:
        """Is the regrowth criterion reading the task or the batch?

        Computes regrowth scores on the same masked positions under a
        foreground-oversampled batch and a background-dominated batch, and logs
        the top-k selection overlap. Logged, not interpreted: low overlap means
        the criterion is batch-dependent, which is one candidate explanation
        for a null trajectory, distinct from a mistuned drop fraction.
        """
        if self.masked is None or step not in self._probe_steps:
            return
        try:
            fg = self._scores_under(trainer, oversample=1.0)
            bg = self._scores_under(trainer, oversample=0.0)
        except Exception as exc:  # diagnostics must never kill a run
            print(f"[{self.cfg.run_id}] informativeness probe failed at step "
                  f"{step}: {exc}", flush=True)
            return

        rows = []
        for name in self.masked.masks:
            n_dead = self.masked.n_weights(name) - self.masked.n_live(name)
            k = max(1, int(round(self.cfg.sparsity.initial_drop_fraction
                                 * self.masked.n_live(name))))
            k = min(k, n_dead) if n_dead else 0
            rows.append({
                "run_id": self.cfg.run_id,
                "seed": self.cfg.seed,
                "arm": self.cfg.arm,
                "step": step,
                "layer_name": name,
                "stage": stage_of(name),
                "k": k,
                "topk_overlap_fg_vs_bg": topk_overlap(fg[name], bg[name], k),
                "n_dead": n_dead,
            })
        with open(self._probe_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=list(PROBE_FIELDS)).writerows(rows)

    def _scores_under(self, trainer, oversample: float) -> dict[str, torch.Tensor]:
        """Dense-gradient regrowth scores from one batch at a given oversample rate."""
        from data.augmentation import build_train_transforms
        from data.dataset import BraTSMENPatches

        ds = BraTSMENPatches(
            preprocessed_data_dir=Path(trainer.cfg.data.preprocessed_dir)
            / f"nnUNetPlans_{trainer.cfg.data.configuration}",
            identifiers=trainer.train_ids,
            patch_size=trainer.initial_patch_size,
            length=trainer.batch_size,
            seed=trainer.cfg.seed + 9973,   # a stream the training run never uses
            oversample_foreground=oversample,
            transforms=build_train_transforms(
                trainer.plan.patch_size, trainer.ds_scales, trainer.use_mask),
        )
        samples = [ds[i] for i in range(trainer.batch_size)]
        data = torch.stack([s["data"] for s in samples]).to(trainer.device)
        target = [torch.stack([s["target"][i] for s in samples]).to(trainer.device)
                  for i in range(len(samples[0]["target"]))]

        was_training = trainer.network.training
        trainer.network.train()
        trainer.network.zero_grad(set_to_none=True)
        with torch.autocast(trainer.device.type, enabled=trainer.scaler is not None):
            loss = trainer.loss(trainer.network(data), target)
        loss.backward()
        scores = {name: regrowth_scores(self.masked, name).clone()
                  for name in self.masked.masks}
        trainer.network.zero_grad(set_to_none=True)
        if not was_training:
            trainer.network.eval()
        return scores

    # --- persistence ------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "arm": self.arm,
            "masks": self.masked.state_dict() if self.masked else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("masks") and self.masked is not None:
            self.masked.load_state_dict(state["masks"])


class StaticSparseController(SparsityController):
    """Random mask at target density, fixed for all of training.

    This is the arm that separates "dynamic reallocation helped" from "the
    network was simply overparameterized".
    """

    arm = "static_sparse"

    def initialize_masks(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        densities = self.target_densities()
        # Draw on CPU so the mask is identical regardless of device.
        for name, mask in self.masked.masks.items():
            n = mask.numel()
            k = int(round(densities[name] * n))
            flat = torch.zeros(n, dtype=torch.bool)
            if k:
                flat[torch.randperm(n, generator=gen)[:k]] = True
            self.masked.masks[name] = flat.view_as(mask).to(mask.device)
        self.masked.apply()


class OneShotPruneController(SparsityController):
    """Train dense, prune by magnitude at a set step, fine-tune.

    Note the pre-registered limitation: at the pilot epoch budget the dense
    phase has not converged, so this prunes a still-improving model. That is a
    deviation from the canonical prune-after-convergence recipe and is
    documented rather than presented as the standard method.
    """

    arm = "oneshot_prune"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prune_step = int(round(cfg.sparsity.prune_at_frac * cfg.total_steps))
        self.pruned = False

    def initialize_masks(self) -> None:
        pass  # dense until the prune step

    def after_optimizer_step(self, step: int, trainer) -> None:
        if not self.pruned and step >= self.prune_step:
            stats = self.masked.prune_by_magnitude(self.target_densities())
            self._last_stats = {
                name: {"n_pruned": n, "n_regrown": 0} for name, n in stats.items()}
            self.pruned = True
            print(f"[{self.cfg.run_id}] pruned to density "
                  f"{self.masked.overall_density():.4f} at step {step}", flush=True)
        super().after_optimizer_step(step, trainer)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["pruned"] = self.pruned
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.pruned = bool(state.get("pruned", False))


class RiglController(SparsityController):
    """Periodic prune-by-magnitude, regrow-by-gradient, at fixed density."""

    arm = "rigl"

    def initialize_masks(self) -> None:
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        densities = self.target_densities()
        for name, mask in self.masked.masks.items():
            n = mask.numel()
            k = int(round(densities[name] * n))
            flat = torch.zeros(n, dtype=torch.bool)
            if k:
                flat[torch.randperm(n, generator=gen)[:k]] = True
            self.masked.masks[name] = flat.view_as(mask).to(mask.device)
        self.masked.apply()

    def drop_fraction_at(self, step: int) -> float:
        s = self.cfg.sparsity
        return cosine_drop_fraction(step, s.initial_drop_fraction,
                                    self.cfg.total_steps, s.drop_decay_end_frac)

    def before_optimizer_step(self) -> None:
        # Deliberately does NOT mask gradients here. The dense gradient must
        # survive until after_optimizer_step, where the RigL update reads it.
        pass

    def after_optimizer_step(self, step: int, trainer) -> None:
        interval = self.cfg.sparsity.update_interval
        if step > 0 and step % interval == 0:
            frac = self.drop_fraction_at(step)
            if frac > 0:
                self._last_stats = rigl_update(self.masked, frac,
                                               optimizer=trainer.optimizer)
        self.masked.apply()
        self._epoch = trainer.epoch
        self.maybe_probe_informativeness(step, trainer)
        every = self.cfg.logging.trajectory_every_n_steps
        if every and step % every == 0:
            self.log_trajectory(step)


class SparseMomentumController(SparsityController):
    """Global sparse redistribution: density conserved globally, not per layer.

    The redesigned dynamic arm. Unlike RigL, per-layer density is free to move,
    so capacity can migrate between layers, which is the mechanism the
    scientific question requires. See src/sparsity/redistribute.py.
    """

    arm = "sparse_momentum"

    def initialize_masks(self) -> None:
        """Draw the initial mask per layer at uniform or ERK density.

        Both modes hit the same global 0.30 total; they differ only in where
        the budget sits at step 0. Running both and confirming they converge to
        the same non-ERK allocation removes the objection that the
        initialization determined the destination.
        """
        if self.cfg.sparsity.init_mode == "erk":
            shapes = {name: tuple(m.shape) for name, m in self.masked.masks.items()}
            densities = erk_densities(shapes, self.cfg.sparsity.density)
        else:
            densities = self.target_densities()

        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        for name, mask in self.masked.masks.items():
            n = mask.numel()
            k = int(round(densities[name] * n))
            k = max(0, min(n, k))
            flat = torch.zeros(n, dtype=torch.bool)
            if k:
                flat[torch.randperm(n, generator=gen)[:k]] = True
            self.masked.masks[name] = flat.view_as(mask).to(mask.device)
        self.masked.apply()

    def prune_rate_at(self, step: int) -> float:
        s = self.cfg.sparsity
        return cosine_drop_fraction(step, s.initial_drop_fraction,
                                    self.cfg.total_steps, s.drop_decay_end_frac)

    def before_optimizer_step(self) -> None:
        # Do NOT mask gradients: the momentum buffer must accumulate a dense
        # signal at dead positions for the redistribution and regrowth steps.
        pass

    def after_optimizer_step(self, step: int, trainer) -> None:
        interval = self.cfg.sparsity.update_interval
        if step > 0 and step % interval == 0:
            rate = self.prune_rate_at(step)
            if rate > 0:
                self._last_stats = sparse_momentum_update(
                    self.masked, trainer.optimizer, rate,
                    self.cfg.sparsity.min_density_floor)
        self.masked.apply()
        self._epoch = trainer.epoch
        self.maybe_probe_informativeness(step, trainer)
        every = self.cfg.logging.trajectory_every_n_steps
        if every and step % every == 0:
            self.log_trajectory(step)


_CONTROLLERS = {
    "static_sparse": StaticSparseController,
    "oneshot_prune": OneShotPruneController,
    "rigl": RiglController,
    "sparse_momentum": SparseMomentumController,
}


def make_controller(cfg):
    if cfg.arm not in _CONTROLLERS:
        raise ValueError(f"no controller for arm {cfg.arm!r}")
    return _CONTROLLERS[cfg.arm](cfg)


def erk_reference(masked: MaskedLayers, density: float) -> dict[str, float]:
    """The ERK null allocation for these layers, for joining to the trajectory."""
    shapes = {name: tuple(m.shape) for name, m in masked.masks.items()}
    return erk_densities(shapes, density)
