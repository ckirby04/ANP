"""Run configuration.

A run must be reproducible from its config alone, so everything that affects
the result lives here and is serialized into the run directory. Seeds are
overridable from the CLI because the arm matrix varies only in seed and arm.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# `rigl` is retained so the voided pilot remains reproducible; its per-layer
# conservation cannot answer the between-layer question (see the VOID banner in
# docs/preregistration.md). `sparse_momentum` is the redesigned dynamic arm,
# which conserves density globally and lets capacity migrate between layers.
ARMS = ("dense", "static_sparse", "oneshot_prune", "rigl", "sparse_momentum")


@dataclass
class DataConfig:
    preprocessed_dir: str = r"G:\BraTS-MEN\nnUNet\nnUNet_preprocessed\Dataset002_BraTS_MEN"
    raw_dir: str = r"G:\BraTS-MEN\nnUNet\nnUNet_raw\Dataset002_BraTS_MEN"
    configuration: str = "3d_fullres"
    fold: int = 0
    oversample_foreground: float = 0.33
    # Matches nnU-Net's get_allowed_n_proc_DA default. Augmentation runs on an
    # oversized initial patch, so this is the throughput-limiting setting: at 4
    # workers the loop is data-bound at about 1.27 s/iter, at 12 it is not.
    num_workers: int = 12
    prefetch_factor: int = 3
    # Validation workers stay resident alongside the training workers, so this
    # is deliberately small. Validation samples the final patch size with no
    # spatial augmentation and is cheap. Setting it equal to num_workers puts
    # 24+ processes on the blosc2 files at once, which fails outright with
    # "Error while getting the buffer" and starves the training loader.
    val_num_workers: int = 3
    val_prefetch_factor: int = 2
    # Cap on cases used, for smoke tests. 0 means the whole split.
    limit_cases: int = 0


@dataclass
class TrainConfig:
    epochs: int = 100
    iters_per_epoch: int = 250
    batch_size: int = 0  # 0 means take the planned batch size
    initial_lr: float = 1e-2
    weight_decay: float = 3e-5
    momentum: float = 0.99
    nesterov: bool = True
    grad_clip: float = 12.0
    amp: bool = True
    deep_supervision: bool = True


@dataclass
class SparsityConfig:
    """Connectivity treatment. `arm` selects which of these fields apply."""

    density: float = 0.30
    include_stem: bool = False
    # RigL
    update_interval: int = 100
    initial_drop_fraction: float = 0.3
    # Cosine decay of the drop fraction reaches zero at this fraction of training.
    drop_decay_end_frac: float = 0.75
    # oneshot_prune: fraction of total training at which the prune happens.
    prune_at_frac: float = 0.5
    # sparse_momentum (global redistribution): per-layer density floor, so no
    # layer can be pruned to the point where it stops passing a signal. 0.05
    # leaves substantial migration room (ERK's lowest stage is 0.24, uniform is
    # 0.30) while keeping every layer functional. Config-exposed for sweeping;
    # whether any layer sits AT the floor is reported, since a binding floor
    # clips the migration signal.
    min_density_floor: float = 0.05
    # Regrowth-informativeness diagnostic, run at this many evenly spaced points.
    n_informativeness_probes: int = 4


@dataclass
class LoggingConfig:
    results_dir: str = "results"
    trajectory_every_n_steps: int = 250
    checkpoint_every_n_steps: int = 1000
    log_every_n_steps: int = 50


@dataclass
class Config:
    arm: str = "dense"
    seed: int = 0
    device: str = "cuda:0"
    # The device is selected by NAME, not by this index. CUDA's default
    # FASTEST_FIRST ordering does not match nvidia-smi's, and on this machine
    # the default `cuda:0` is the 8 GB card, not the 16 GB one. Set to "" to
    # select purely by index, which is not recommended.
    require_device_name: str = "RTX 5060 Ti"
    require_min_vram_gb: float = 15.0
    run_id: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARMS}")
        if not 0.0 < self.sparsity.density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {self.sparsity.density}")
        if not 0.0 < self.sparsity.initial_drop_fraction < 1.0:
            raise ValueError(
                f"initial_drop_fraction must be in (0, 1), got "
                f"{self.sparsity.initial_drop_fraction}")
        if self.sparsity.update_interval < 1:
            raise ValueError("update_interval must be >= 1")
        if not self.run_id:
            self.run_id = f"{self.arm}_seed{self.seed}"

    @property
    def uses_sparsity(self) -> bool:
        return self.arm != "dense"

    @property
    def total_steps(self) -> int:
        return self.train.epochs * self.train.iters_per_epoch

    @property
    def run_dir(self) -> Path:
        return Path(self.logging.results_dir) / self.run_id

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def digest(self) -> str:
        import hashlib
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


_SECTIONS = {
    "data": DataConfig,
    "train": TrainConfig,
    "sparsity": SparsityConfig,
    "logging": LoggingConfig,
}


def _coerce(current: Any, value: str) -> Any:
    """Parse a CLI override string against the type of the existing value."""
    if isinstance(current, bool):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def from_dict(raw: dict) -> Config:
    raw = copy.deepcopy(raw or {})
    sections = {}
    for name, cls in _SECTIONS.items():
        payload = raw.pop(name, {}) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown keys in {name}: {sorted(unknown)}")
        sections[name] = cls(**payload)

    known_top = {f.name for f in dataclasses.fields(Config)} - set(_SECTIONS)
    unknown = set(raw) - known_top
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return Config(**raw, **sections)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply `section.key=value` CLI overrides.

    Overrides are applied to the raw mapping before construction so that
    validation in __post_init__ sees the final values.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key=value")
        key, value = item.split("=", 1)
        parts = key.split(".")
        if len(parts) == 1:
            target, leaf = raw, parts[0]
        elif len(parts) == 2 and parts[0] in _SECTIONS:
            target = raw.setdefault(parts[0], {})
            leaf = parts[1]
        else:
            raise ValueError(f"override key {key!r} does not name a config field")

        # Coerce against the dataclass default so types survive the CLI.
        cls = _SECTIONS[parts[0]] if len(parts) == 2 else Config
        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        if leaf not in defaults:
            raise ValueError(f"override key {key!r} does not name a config field")
        current = target.get(leaf, defaults[leaf])
        target[leaf] = _coerce(current, value)

    cfg = from_dict(raw)
    # run_id is regenerated after overrides so a seed override renames the run.
    if not (raw.get("run_id") or "").strip():
        cfg.run_id = f"{cfg.arm}_seed{cfg.seed}"
        cfg.__post_init__()
    return cfg
