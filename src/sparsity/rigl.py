"""RigL: prune by weight magnitude, regrow by dense-gradient magnitude.

Per update, per layer: drop the k smallest-magnitude live weights and grow the
k largest-magnitude gradients among currently-dead positions. k is the same
number on both sides, so layer density is held exactly constant by
construction rather than approximately. `assert_density_preserved` in the tests
checks this holds after a real update.

The drop fraction follows a cosine decay to zero, so late training stops
rewiring and the topology settles. Interval, initial fraction and the point at
which the decay reaches zero are all config-exposed: the published defaults are
tuned for 2D ResNets on ImageNet and there is no reason to expect them to
transfer to 3D conv.
"""

from __future__ import annotations

import math

import torch


def cosine_drop_fraction(step: int, initial: float, total_steps: int,
                         decay_end_frac: float) -> float:
    """Drop fraction at `step`, decaying as a cosine to zero.

    Reaches exactly zero at `decay_end_frac` of total training and stays there.
    """
    if total_steps <= 0:
        return 0.0
    end = decay_end_frac * total_steps
    if end <= 0 or step >= end:
        return 0.0
    return float(initial / 2 * (1 + math.cos(math.pi * step / end)))


@torch.no_grad()
def rigl_update(masked, drop_fraction: float,
                optimizer=None) -> dict[str, dict[str, int]]:
    """One prune-and-regrow step across all masked layers.

    Requires `weight.grad` to hold the DENSE gradient, i.e. the gradient of the
    unmasked weight. Call before masking gradients and before the optimizer
    step.

    Returns per-layer counts. n_pruned equals n_regrown for every layer.
    """
    if not 0.0 <= drop_fraction <= 1.0:
        raise ValueError(f"drop_fraction must be in [0, 1], got {drop_fraction}")

    stats: dict[str, dict[str, int]] = {}
    for name, conv in masked.layers.items():
        mask = masked.masks[name]
        n = mask.numel()
        live = mask.flatten()
        n_live = int(live.sum().item())

        k = int(math.floor(drop_fraction * n_live))
        # Cannot grow more than there are dead positions to grow into.
        k = min(k, n - n_live)
        if k <= 0:
            stats[name] = {"n_pruned": 0, "n_regrown": 0, "n_live": n_live}
            continue

        weight = conv.weight.detach().flatten()
        grad = conv.weight.grad
        if grad is None:
            raise RuntimeError(
                f"{name} has no gradient; rigl_update needs the dense gradient "
                "and must be called after backward()")
        grad = grad.detach().flatten()

        # Drop: smallest magnitude among live. Dead positions are pushed to
        # +inf so they cannot be selected for dropping.
        drop_score = torch.where(live, weight.abs(),
                                 torch.full_like(weight, float("inf")))
        drop_idx = torch.topk(drop_score, k, largest=False).indices

        # Grow: largest gradient magnitude among dead. Live positions are
        # pushed to -inf so they cannot be selected for growing. Positions
        # just dropped are still marked live here, so a weight cannot be
        # dropped and immediately regrown in the same update.
        grow_score = torch.where(live, torch.full_like(grad, float("-inf")),
                                 grad.abs())
        grow_idx = torch.topk(grow_score, k).indices

        new = live.clone()
        new[drop_idx] = False
        new[grow_idx] = True
        masked.masks[name] = new.view_as(mask)

        # A regrown connection starts at exactly zero, as in the paper, and
        # must not inherit momentum accumulated while it was dead.
        weight_view = conv.weight.view(-1)
        weight_view[grow_idx] = 0.0
        if optimizer is not None:
            masked.reset_optimizer_state(optimizer, name,
                                         grow_idx.view(-1))

        n_after = int(new.sum().item())
        if n_after != n_live:
            raise AssertionError(
                f"{name}: density not preserved, {n_live} live before and "
                f"{n_after} after (pruned {k}, regrown {k})")
        stats[name] = {"n_pruned": k, "n_regrown": k, "n_live": n_after}

    masked.apply()
    return stats


@torch.no_grad()
def regrowth_scores(masked, name: str) -> torch.Tensor:
    """Gradient magnitudes at currently-dead positions, for the diagnostic.

    Live positions are returned as -inf so a top-k over this tensor selects
    only candidates RigL could actually grow.
    """
    conv = masked.layers[name]
    if conv.weight.grad is None:
        raise RuntimeError(f"{name} has no gradient")
    live = masked.masks[name].flatten()
    grad = conv.weight.grad.detach().flatten().abs()
    return torch.where(live, torch.full_like(grad, float("-inf")), grad)


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    """Fraction of the top-k positions shared between two score vectors.

    1.0 means the two batches would regrow identical connections; near 0 means
    the regrowth criterion is dominated by whatever the batch happened to
    contain rather than by anything stable about the task.
    """
    if k <= 0:
        return float("nan")
    k = min(k, a.numel(), b.numel())
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / k
