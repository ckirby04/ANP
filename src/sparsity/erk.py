"""Erdos-Renyi-Kernel (ERK) sparsity allocation.

ERK (Evci et al. 2020, "Rigging the Lottery") assigns each layer a density
proportional to (sum of weight dims) / (product of weight dims), scaled so the
overall parameter density hits the target. It is the standard non-uniform
sparse-training prior and encodes a parameter-per-activation heuristic: wide,
deep, low-resolution layers get sparser, narrow high-resolution layers get
denser.

This project uses ERK as the **null model** for the layer-wise density
trajectory, not as an initialization. RigL run at uniform density will drift
somewhere; the scientific question is whether it drifts toward ERK, which
would be a rediscovery of a known prior and says nothing about the task, or
away from ERK in a task-specific direction, which is the actual finding.

See docs/preregistration.md.
"""

from __future__ import annotations

import math


def erk_densities(shapes: dict[str, tuple[int, ...]], target_density: float,
                  max_iters: int = 100) -> dict[str, float]:
    """Per-layer ERK densities hitting `target_density` overall.

    Layers whose raw ERK score would push them past density 1.0 are pinned
    dense and removed from the scalable pool, and the remaining budget is
    redistributed. This is the standard iterative correction.

    Args:
        shapes: layer name -> weight shape, e.g. (out, in, kd, kh, kw).
        target_density: overall fraction of live weights across `shapes`.

    Returns:
        layer name -> density in (0, 1].
    """
    if not 0.0 < target_density <= 1.0:
        raise ValueError(f"target_density must be in (0, 1], got {target_density}")

    n_params = {k: math.prod(s) for k, s in shapes.items()}
    total = sum(n_params.values())
    budget = target_density * total

    raw = {k: sum(s) / math.prod(s) for k, s in shapes.items()}
    dense: set[str] = set()

    for _ in range(max_iters):
        pool = [k for k in shapes if k not in dense]
        # Budget left after the pinned-dense layers take their full share.
        remaining = budget - sum(n_params[k] for k in dense)
        denom = sum(raw[k] * n_params[k] for k in pool)
        if denom <= 0:
            raise ValueError("no scalable layers remain")
        eps = remaining / denom

        over = [k for k in pool if raw[k] * eps > 1.0]
        if not over:
            out = {k: 1.0 for k in dense}
            out.update({k: raw[k] * eps for k in pool})
            return {k: out[k] for k in shapes}
        dense.update(over)

    raise RuntimeError("ERK allocation did not converge")


def live_budget_share(shapes: dict[str, tuple[int, ...]],
                      densities: dict[str, float]) -> dict[str, float]:
    """Each layer's share of the total live-parameter budget.

    Per-layer density is a misleading primary readout when parameter counts are
    this uneven: the shallow stages hold roughly 6 percent of encoder
    parameters, so a stage can swing from 0.30 to 0.90 density while moving
    almost no budget. Share of live parameters is what "capacity allocation"
    actually means, and it is the headline figure.
    """
    live = {k: math.prod(shapes[k]) * densities[k] for k in shapes}
    total = sum(live.values())
    if total <= 0:
        raise ValueError("no live parameters")
    return {k: live[k] / total for k in shapes}


def encoder_conv_shapes(features: list[int], n_conv_per_stage: list[int],
                        kernel_sizes: list[list[int]], in_channels: int,
                        skip_stem: bool = True) -> dict[str, tuple[int, ...]]:
    """Weight shapes of the encoder convs, named to match the trajectory CSV.

    Names are `stage{s}.conv{c}`. The stem (stage0.conv0) is excluded by
    default because it stays dense in this project.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    prev = in_channels
    for s, (feat, n_conv) in enumerate(zip(features, n_conv_per_stage)):
        k = tuple(kernel_sizes[s])
        for c in range(n_conv):
            c_in = prev if c == 0 else feat
            if not (skip_stem and s == 0 and c == 0):
                shapes[f"stage{s}.conv{c}"] = (feat, c_in, *k)
        prev = feat
    return shapes
