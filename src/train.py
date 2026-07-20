"""Training entry point for one arm at one seed.

Schedule, optimizer, loss and augmentation follow nnU-Net's own trainer so the
prior fold-0 run remains a valid reference point. The only thing that differs
between arms is the connectivity treatment, which is applied through a
`SparsityController` supplied by src/sparsity/.

Everything needed to resume lives in the checkpoint: model, optimizer, AMP
scaler, RNG state, step counter and any masks. A crash costs at most the work
since the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Set before torch is imported, so it is in place before the CUDA driver
# initializes. CUDA's default FASTEST_FIRST ordering does not agree with
# nvidia-smi's, and under it `cuda:0` on this machine is the 8 GB card.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, load_config  # noqa: E402
from data.augmentation import (  # noqa: E402
    build_train_transforms,
    build_val_transforms,
    configure_augmentation,
)
from data.dataset import BraTSMENPatches  # noqa: E402
from data.splits import load_split, verify_split_available  # noqa: E402
from models.network import (  # noqa: E402
    build_network,
    count_parameters,
    describe_cuda_devices,
    load_plan,
    resolve_device,
)

CHECKPOINT = "checkpoint_latest.pt"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(plan, deep_supervision: bool):
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    loss = DC_and_CE_loss(
        {"batch_dice": plan.batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": False},
        {}, weight_ce=1, weight_dice=1, ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss)

    if not deep_supervision:
        return loss

    # Exponentially decaying weights, coarsest output dropped entirely.
    scales = plan.deep_supervision_scales
    weights = np.array([1 / (2 ** i) for i in range(len(scales))])
    weights[-1] = 0
    weights = weights / weights.sum()
    return DeepSupervisionWrapper(loss, weights)


def make_dataloader(cfg: Config, plan, identifiers, transforms, patch_size,
                    batch_size: int, length: int, start_index: int = 0,
                    persistent: bool = True, workers: int | None = None,
                    prefetch: int | None = None):
    ds = BraTSMENPatches(
        preprocessed_data_dir=Path(cfg.data.preprocessed_dir) / f"nnUNetPlans_{cfg.data.configuration}",
        identifiers=identifiers,
        patch_size=patch_size,
        length=length,
        seed=cfg.seed,
        oversample_foreground=cfg.data.oversample_foreground,
        transforms=transforms,
        start_index=start_index,
    )
    workers = cfg.data.num_workers if workers is None else workers
    prefetch = cfg.data.prefetch_factor if prefetch is None else prefetch
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=bool(workers) and persistent,
        prefetch_factor=prefetch if workers else None,
        drop_last=True,
    )
    return ds, loader


def move_target(target, device):
    if isinstance(target, list):
        return [t.to(device, non_blocking=True) for t in target]
    return target.to(device, non_blocking=True)


class Trainer:
    def __init__(self, cfg: Config, controller=None):
        self.cfg = cfg
        self.controller = controller
        self.device = resolve_device(cfg.device,
                                     cfg.require_device_name,
                                     cfg.require_min_vram_gb)
        set_seeds(cfg.seed)

        pre = Path(cfg.data.preprocessed_dir)
        self.plan = load_plan(pre, cfg.data.configuration)
        self.batch_size = cfg.train.batch_size or self.plan.batch_size

        split = load_split(pre, cfg.data.fold)
        data_dir = pre / f"nnUNetPlans_{cfg.data.configuration}"
        verify_split_available(split, data_dir)
        train_ids = split.train
        val_ids = split.val
        if cfg.data.limit_cases:
            train_ids = train_ids[: cfg.data.limit_cases]
            val_ids = val_ids[: max(1, cfg.data.limit_cases // 2)]
        self.split = split
        self.train_ids, self.val_ids = train_ids, val_ids

        aug = configure_augmentation(self.plan.patch_size)
        self.initial_patch_size = aug["initial_patch_size"]
        self.n_val_batches = 25
        ds_scales = self.plan.deep_supervision_scales if cfg.train.deep_supervision else None

        self.use_mask = [True] * self.plan.n_input_channels
        self.ds_scales = ds_scales
        # The training stream is built lazily in fit(), after any resume has
        # set self.step, so the stream can start at the right sample offset.
        self.train_ds = self.train_loader = self._train_iter = None
        self.val_ds, self.val_loader = make_dataloader(
            cfg, self.plan, val_ids,
            build_val_transforms(ds_scales),
            self.plan.patch_size, self.batch_size,
            max(self.batch_size, self.n_val_batches * self.batch_size),
            workers=cfg.data.val_num_workers,
            prefetch=cfg.data.val_prefetch_factor)

        self.network = build_network(self.plan, cfg.train.deep_supervision).to(self.device)
        self.loss = build_loss(self.plan, cfg.train.deep_supervision)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), cfg.train.initial_lr,
            weight_decay=cfg.train.weight_decay, momentum=cfg.train.momentum,
            nesterov=cfg.train.nesterov)
        self.scaler = torch.amp.GradScaler("cuda") if (
            cfg.train.amp and self.device.type == "cuda") else None

        self.epoch = 0
        self.step = 0
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.controller is not None:
            self.controller.attach(self.network, self.run_dir, cfg)

    # --- schedule ---------------------------------------------------------

    def _set_lr(self):
        # nnU-Net's poly schedule, stepped per epoch.
        lr = self.cfg.train.initial_lr * (1 - self.epoch / self.cfg.train.epochs) ** 0.9
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    # --- checkpointing ----------------------------------------------------

    def save_checkpoint(self, name: str = CHECKPOINT):
        payload = {
            "epoch": self.epoch,
            "step": self.step,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "config": self.cfg.to_dict(),
            "split_digest": self.split.digest,
        }
        if self.controller is not None:
            payload["sparsity"] = self.controller.state_dict()
        tmp = self.run_dir / (name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.run_dir / name)

    def load_checkpoint(self, name: str = CHECKPOINT) -> bool:
        path = self.run_dir / name
        if not path.is_file():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scaler and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.epoch = payload["epoch"]
        self.step = payload["step"]
        rng = payload.get("rng") or {}
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        if self.controller is not None and payload.get("sparsity"):
            self.controller.load_state_dict(payload["sparsity"])
        if payload.get("split_digest") != self.split.digest:
            raise RuntimeError(
                "checkpoint was trained on a different data split "
                f"({payload.get('split_digest')} vs {self.split.digest})")
        return True

    # --- training ---------------------------------------------------------

    def train_step(self, batch) -> tuple[float, float]:
        data = batch["data"].to(self.device, non_blocking=True)
        target = move_target(batch["target"], self.device)

        self.optimizer.zero_grad(set_to_none=True)
        amp = self.scaler is not None
        with torch.autocast(self.device.type, enabled=amp):
            output = self.network(data)
            loss = self.loss(output, target)

        if amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.cfg.train.grad_clip)
            if self.controller is not None:
                self.controller.before_optimizer_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.cfg.train.grad_clip)
            if self.controller is not None:
                self.controller.before_optimizer_step()
            self.optimizer.step()

        if self.controller is not None:
            self.controller.after_optimizer_step(self.step, self)

        return float(loss.detach()), float(grad_norm)

    def _build_train_stream(self):
        """One continuous loader for the whole run, offset to the current step.

        Recreating this per epoch would respawn workers, which costs about 21 s
        each time on Windows.
        """
        remaining = max(0, self.cfg.total_steps - self.step)
        self.train_ds, self.train_loader = make_dataloader(
            self.cfg, self.plan, self.train_ids,
            build_train_transforms(self.plan.patch_size, self.ds_scales, self.use_mask),
            self.initial_patch_size, self.batch_size,
            length=remaining * self.batch_size,
            start_index=self.step * self.batch_size,
        )
        self._train_iter = iter(self.train_loader)

    def run_epoch(self) -> dict:
        self.network.train()
        lr = self._set_lr()

        losses, iter_times = [], []
        t_epoch = time.perf_counter()
        t_last = t_epoch
        for _ in range(self.cfg.train.iters_per_epoch):
            try:
                batch = next(self._train_iter)
            except StopIteration:
                break
            loss, _ = self.train_step(batch)
            losses.append(loss)
            self.step += 1

            now = time.perf_counter()
            iter_times.append(now - t_last)
            t_last = now

            if self.cfg.logging.checkpoint_every_n_steps and \
                    self.step % self.cfg.logging.checkpoint_every_n_steps == 0:
                self.save_checkpoint()

        t_train = time.perf_counter() - t_epoch
        t_val0 = time.perf_counter()
        val = self.validate()
        t_val = time.perf_counter() - t_val0

        # Mean, not median. The loader stalls periodically when it is the
        # bottleneck, which makes the median look healthy while the epoch takes
        # three times as long. Mean is the number that predicts wall clock.
        stats = {
            "epoch": self.epoch,
            "lr": lr,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_time_s": t_train,
            "val_time_s": t_val,
            "epoch_time_s": t_train + t_val,
            "s_per_iter_mean": float(np.mean(iter_times)) if iter_times else float("nan"),
            "s_per_iter_median": float(np.median(iter_times)) if iter_times else float("nan"),
            "s_per_iter_p90": float(np.percentile(iter_times, 90)) if iter_times else float("nan"),
            "n_iters": len(losses),
        }
        stats.update(val)
        return stats

    @torch.no_grad()
    def validate(self, max_batches: int | None = None) -> dict:
        self.network.eval()
        max_batches = self.n_val_batches if max_batches is None else max_batches
        losses = []
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            data = batch["data"].to(self.device, non_blocking=True)
            target = move_target(batch["target"], self.device)
            with torch.autocast(self.device.type, enabled=self.scaler is not None):
                losses.append(float(self.loss(self.network(data), target)))
        return {"val_loss": float(np.mean(losses)) if losses else float("nan")}

    def fit(self, resume: bool = True) -> list[dict]:
        if resume and self.load_checkpoint():
            # flush, because stdout is block-buffered when redirected to a log
            # file and this line is the confirmation that a resume worked.
            print(f"[{self.cfg.run_id}] resumed at epoch {self.epoch} "
                  f"step {self.step}", flush=True)

        self._build_train_stream()

        log_path = self.run_dir / "training_log.jsonl"
        history = []
        while self.epoch < self.cfg.train.epochs:
            stats = self.run_epoch()
            if self.controller is not None:
                stats.update(self.controller.epoch_summary())
            history.append(stats)

            with open(log_path, "a") as fh:
                fh.write(json.dumps(stats) + "\n")
            print(f"[{self.cfg.run_id}] epoch {self.epoch} "
                  f"loss {stats['train_loss']:.4f} val {stats['val_loss']:.4f} "
                  f"train {stats['train_time_s']:.1f}s val {stats['val_time_s']:.1f}s "
                  f"({stats['s_per_iter_mean']:.3f} s/iter mean, "
                  f"{stats['s_per_iter_p90']:.3f} p90)", flush=True)

            self.epoch += 1
            self.save_checkpoint()

        self.save_checkpoint("checkpoint_final.pt")
        return history


def build_controller(cfg: Config):
    """Construct the arm's connectivity treatment, if any."""
    if not cfg.uses_sparsity:
        return None
    from sparsity.controller import make_controller
    return make_controller(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train one arm at one seed.")
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", action="append", default=[],
                    metavar="section.key=value",
                    help="override a config field, repeatable")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    trainer = Trainer(cfg, controller=build_controller(cfg))

    cfg.save(trainer.run_dir / "config.yaml")
    params = count_parameters(trainer.network)
    provenance = {
        "run_id": cfg.run_id,
        "arm": cfg.arm,
        "seed": cfg.seed,
        "config_digest": cfg.digest(),
        "split_digest": trainer.split.digest,
        "n_train_cases": len(trainer.train_ids),
        "n_val_cases": len(trainer.val_ids),
        "patch_size": list(trainer.plan.patch_size),
        "initial_patch_size": list(trainer.initial_patch_size),
        "batch_size": trainer.batch_size,
        "parameters": params,
        "device": str(trainer.device),
        # The index alone is not evidence of which card ran the job: earlier
        # runs recorded "cuda:0" while executing on the 8 GB card.
        "device_name": (torch.cuda.get_device_name(trainer.device)
                        if trainer.device.type == "cuda" else "cpu"),
        "device_vram_gb": (round(torch.cuda.get_device_properties(
            trainer.device).total_memory / 1024 ** 3, 2)
            if trainer.device.type == "cuda" else None),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "<unset>"),
        "visible_cuda_devices": describe_cuda_devices(),
        "torch": torch.__version__,
        "determinism": {
            # Recorded because it is NOT enforced. See docs and README.
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        },
    }
    with open(trainer.run_dir / "provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
    print(json.dumps(provenance, indent=2), flush=True)

    trainer.fit(resume=not args.no_resume)


if __name__ == "__main__":
    main()
