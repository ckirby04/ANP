"""nnU-Net 3D network, built from the dataset's own plans file.

The architecture is not reimplemented here. It is instantiated from
`nnUNetPlans.json` via nnU-Net's own builder, because that plan file *is* the
auto-configured capacity allocation this experiment studies. Hand-rolling an
equivalent-looking UNet would quietly change the object under test.

This module also owns the definition of which layers are sparsifiable, and the
layer naming used by the trajectory CSV and by `src/sparsity/erk.py`. Those
names must agree or the ERK null cannot be compared to the observed
trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


@dataclass(frozen=True)
class PlanSpec:
    """The subset of the plans file this project depends on."""

    patch_size: tuple[int, ...]
    batch_size: int
    strides: tuple[tuple[int, ...], ...]
    features_per_stage: tuple[int, ...]
    n_conv_per_stage: tuple[int, ...]
    kernel_sizes: tuple[tuple[int, ...], ...]
    n_input_channels: int
    n_classes: int
    batch_dice: bool
    arch_class_name: str
    arch_kwargs: dict
    arch_kwargs_req_import: tuple[str, ...]

    @property
    def deep_supervision_scales(self) -> list[list[float]]:
        """Resolution of each decoder output relative to the input patch.

        Matches nnUNetTrainer._get_deep_supervision_scales: cumulative product
        of the strides, inverted, dropping the coarsest entry. The list is NOT
        reversed; it runs highest resolution first, matching the order the
        network emits its deep-supervision outputs. Reversing it silently
        pairs each output with the wrong-resolution target.
        """
        return [list(i) for i in
                1 / np.cumprod(np.vstack(self.strides), axis=0)][:-1]


def load_plan(preprocessed_dir: str | Path,
              configuration: str = "3d_fullres") -> PlanSpec:
    """Read the plans and dataset json into a PlanSpec."""
    p = Path(preprocessed_dir)
    with open(p / "nnUNetPlans.json") as fh:
        plans = json.load(fh)
    with open(p / "dataset.json") as fh:
        dataset = json.load(fh)

    if configuration not in plans["configurations"]:
        raise KeyError(f"configuration {configuration!r} not in plans; "
                       f"have {sorted(plans['configurations'])}")
    cfg = plans["configurations"][configuration]
    arch = cfg["architecture"]
    kw = arch["arch_kwargs"]

    labels = dataset["labels"]
    if "ignore" in labels:
        raise NotImplementedError(
            "dataset declares an ignore label; the -1 handling in "
            "src/data/dataset.py assumes there is none")

    return PlanSpec(
        patch_size=tuple(cfg["patch_size"]),
        batch_size=int(cfg["batch_size"]),
        strides=tuple(tuple(s) for s in kw["strides"]),
        features_per_stage=tuple(kw["features_per_stage"]),
        n_conv_per_stage=tuple(kw["n_conv_per_stage"]),
        kernel_sizes=tuple(tuple(k) for k in kw["kernel_sizes"]),
        n_input_channels=len(dataset["channel_names"]),
        n_classes=len(labels),
        batch_dice=bool(cfg["batch_dice"]),
        arch_class_name=arch["network_class_name"],
        arch_kwargs=kw,
        arch_kwargs_req_import=tuple(arch["_kw_requires_import"]),
    )


def build_network(plan: PlanSpec, deep_supervision: bool = True) -> nn.Module:
    """Instantiate the planned architecture."""
    return get_network_from_plans(
        arch_class_name=plan.arch_class_name,
        arch_kwargs=plan.arch_kwargs,
        arch_kwargs_req_import=list(plan.arch_kwargs_req_import),
        input_channels=plan.n_input_channels,
        output_channels=plan.n_classes,
        allow_init=True,
        deep_supervision=deep_supervision,
    )


def sparsifiable_encoder_convs(
        network: nn.Module,
        include_stem: bool = False) -> dict[str, nn.Conv3d]:
    """Encoder convs eligible for sparsification, keyed by `stage{s}.conv{c}`.

    Selection, per the pre-registered spec:
      - encoder only; the decoder, seg heads, norms and biases stay dense
      - 3x3x3 kernels only; 1x1x1 convs are excluded
      - the stem (stage0.conv0, 4 input channels) stays dense by default

    The returned keys are the layer names written to the trajectory CSV and
    used by src/sparsity/erk.py, so the observed trajectory and the ERK null
    are directly joinable.
    """
    encoder = getattr(network, "encoder", None)
    if encoder is None or not hasattr(encoder, "stages"):
        raise AttributeError(
            f"{type(network).__name__} has no encoder.stages; the layer-naming "
            "assumption in this module does not hold for this architecture")

    out: dict[str, nn.Conv3d] = {}
    for s, stage in enumerate(encoder.stages):
        # StackedConvBlocks exposes its blocks as .convs; each block's conv is
        # at .conv. Fall back to a module scan if that layout ever changes.
        blocks = getattr(stage, "convs", None)
        if blocks is None:
            blocks = [m for m in stage.modules() if isinstance(m, nn.Conv3d)]
            blocks = [_Wrap(c) for c in blocks]

        for c, block in enumerate(blocks):
            conv = getattr(block, "conv", block)
            if not isinstance(conv, nn.Conv3d):
                continue
            if tuple(conv.kernel_size) != (3, 3, 3):
                continue
            if s == 0 and c == 0 and not include_stem:
                continue
            out[f"stage{s}.conv{c}"] = conv

    if not out:
        raise RuntimeError("no sparsifiable encoder convs found")
    return out


class _Wrap:
    """Adapter so a bare Conv3d looks like a conv block."""

    def __init__(self, conv):
        self.conv = conv


def layer_shapes(convs: dict[str, nn.Conv3d]) -> dict[str, tuple[int, ...]]:
    """Weight shapes keyed by layer name, for the ERK allocation."""
    return {name: tuple(conv.weight.shape) for name, conv in convs.items()}


def count_parameters(network: nn.Module) -> dict[str, int]:
    """Total and encoder-sparsifiable parameter counts, for run provenance."""
    convs = sparsifiable_encoder_convs(network)
    return {
        "total": sum(p.numel() for p in network.parameters()),
        "trainable": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "sparsifiable": sum(c.weight.numel() for c in convs.values()),
        "n_sparsifiable_layers": len(convs),
    }


def stage_of(layer_name: str) -> int:
    """Encoder stage index from a `stage{s}.conv{c}` layer name."""
    if not layer_name.startswith("stage"):
        raise ValueError(f"not a stage-qualified layer name: {layer_name!r}")
    return int(layer_name[len("stage"):].split(".", 1)[0])


def device_from_config(device: str) -> torch.device:
    """Resolve a configured device string, failing loudly if unavailable.

    Single device only. Falling back to CPU silently would turn a 22-hour
    pilot into an unfinishable one without any visible error.
    """
    d = torch.device(device)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA but torch.cuda.is_available() is False")
        index = 0 if d.index is None else d.index
        n = torch.cuda.device_count()
        if index >= n:
            raise RuntimeError(f"config requests {d} but only {n} CUDA device(s) visible")
    return d
