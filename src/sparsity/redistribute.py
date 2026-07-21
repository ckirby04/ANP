"""Global sparse redistribution (SNFS-style sparse momentum).

The voided pilot held density constant PER LAYER, which made the layer-wise
density trajectory flat by construction. This module holds density constant
GLOBALLY instead: the total number of live connections across the sparsified
encoder is fixed, but each layer's density is free to move. That is the
mechanism the scientific question actually requires, since capacity can now
migrate between layers.

Mechanism, per update (Dettmers & Zettlemoyer 2019, "Sparse Networks from
Scratch"):

  1. Prune, per layer: remove the lowest-magnitude fraction `prune_rate` of a
     layer's live weights. Floor-protected, so no layer is pruned below the
     minimum density.
  2. Redistribute, globally: the freed budget is handed back to layers in
     proportion to each layer's mean momentum magnitude over its live weights,
     a direct gradient-derived estimate of which layers are using capacity
     efficiently.
  3. Regrow, within a layer: turn on the dead positions with the largest
     momentum magnitude.

Total live count is conserved exactly: sum of regrown equals sum of pruned.
Per-layer density changes by (regrown_i - pruned_i), which is generally nonzero.

Momentum is read from the SGD optimizer's momentum buffer, which is an
exponential average of the gradient. Because gradients are NOT masked at dead
positions (the mask multiplies weights, not gradients), the buffer accumulates
a meaningful signal at dead positions too, which is what the regrowth step
reads. This is the same dense-gradient property RigL relied on.
"""

from __future__ import annotations

import math

import torch


def _largest_remainder(raw: list[float], total: int) -> list[int]:
    """Round `raw` to integers summing exactly to `total` (largest remainder)."""
    floors = [int(math.floor(x)) for x in raw]
    deficit = total - sum(floors)
    if deficit <= 0:
        # Trim from the largest fractional parts if we overshot (rare).
        order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i])
        for i in order[:(-deficit)]:
            floors[i] -= 1
        return floors
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:deficit]:
        floors[i] += 1
    return floors


def _allocate(total_removed: int, contrib: list[float],
              capacity: list[int]) -> list[int]:
    """Split `total_removed` across layers by `contrib`, capped at `capacity`.

    Guarantees sum(result) == total_removed exactly, and result[i] <=
    capacity[i]. Excess from capped layers is re-split among layers with spare
    room, in proportion to their contribution, iterating until placed. Global
    capacity always covers total_removed, so this terminates.
    """
    n = len(contrib)
    alloc = [0] * n
    remaining = total_removed
    active = set(range(n))

    while remaining > 0 and active:
        csum = sum(contrib[i] for i in active)
        if csum <= 0:
            # No signal left; fill remaining capacity in a fixed order.
            for i in sorted(active):
                room = capacity[i] - alloc[i]
                take = min(room, remaining)
                alloc[i] += take
                remaining -= take
                if remaining == 0:
                    break
            break

        raw = [(contrib[i] / csum * remaining if i in active else 0.0)
               for i in range(n)]
        want = _largest_remainder(raw, remaining)

        placed = 0
        newly_full = []
        for i in list(active):
            room = capacity[i] - alloc[i]
            give = min(want[i], room)
            alloc[i] += give
            placed += give
            if alloc[i] >= capacity[i]:
                newly_full.append(i)
        remaining -= placed
        for i in newly_full:
            active.discard(i)
        if placed == 0:
            # Nothing landed this pass (all wanted layers full); force-place.
            for i in sorted(active):
                room = capacity[i] - alloc[i]
                take = min(room, remaining)
                alloc[i] += take
                remaining -= take
                if alloc[i] >= capacity[i]:
                    active.discard(i)
                if remaining == 0:
                    break
            break
    return alloc


@torch.no_grad()
def sparse_momentum_update(masked, optimizer, prune_rate: float,
                           min_density: float) -> dict[str, dict[str, int]]:
    """One global prune / redistribute / regrow step.

    Args:
        masked: MaskedLayers owning the encoder conv masks.
        optimizer: the SGD optimizer, for its momentum buffers.
        prune_rate: fraction of each layer's live weights to prune this step.
        min_density: per-layer density floor; no layer is pruned below it.

    Returns per-layer {n_pruned, n_regrown, n_live}. Sum of n_pruned equals
    sum of n_regrown across layers.
    """
    if not 0.0 <= prune_rate <= 1.0:
        raise ValueError(f"prune_rate must be in [0, 1], got {prune_rate}")
    if not 0.0 <= min_density < 1.0:
        raise ValueError(f"min_density must be in [0, 1), got {min_density}")

    names = list(masked.layers)
    mom, live_mask, weight = {}, {}, {}
    for name in names:
        conv = masked.layers[name]
        state = optimizer.state.get(conv.weight, {})
        buf = state.get("momentum_buffer")
        if buf is None:
            # No momentum yet (before the first optimizer step). Nothing to do.
            return {name: {"n_pruned": 0, "n_regrown": 0,
                           "n_live": masked.n_live(name)} for name in names}
        mom[name] = buf.detach().abs().flatten()
        live_mask[name] = masked.masks[name].flatten()
        weight[name] = conv.weight.detach().abs().flatten()

    # --- prune phase, floor-protected ---
    n_prune = {}
    for name in names:
        n = live_mask[name].numel()
        n_live = int(live_mask[name].sum().item())
        floor_count = int(math.ceil(min_density * n))
        want = int(math.floor(prune_rate * n_live))
        n_prune[name] = max(0, min(want, n_live - floor_count))
    total_removed = sum(n_prune.values())

    if total_removed == 0:
        return {name: {"n_pruned": 0, "n_regrown": 0,
                       "n_live": int(live_mask[name].sum().item())}
                for name in names}

    # --- redistribution signal: mean live-weight momentum per layer ---
    contrib = []
    for name in names:
        lm = live_mask[name]
        n_live = int(lm.sum().item())
        contrib.append(float(mom[name][lm].mean().item()) if n_live else 0.0)

    # Capacity = dead slots that will exist after pruning this layer.
    capacity = []
    for name in names:
        n = live_mask[name].numel()
        n_live = int(live_mask[name].sum().item())
        capacity.append(n - (n_live - n_prune[name]))
    regrow = dict(zip(names, _allocate(total_removed, contrib, capacity)))

    # --- apply prune, then regrow, per layer ---
    stats = {}
    for name in names:
        conv = masked.layers[name]
        lm = live_mask[name].clone()
        n = lm.numel()

        k_prune = n_prune[name]
        if k_prune > 0:
            drop_score = torch.where(lm, weight[name],
                                     torch.full_like(weight[name], float("inf")))
            drop_idx = torch.topk(drop_score, k_prune, largest=False).indices
            lm[drop_idx] = False

        k_grow = regrow[name]
        if k_grow > 0:
            grow_score = torch.where(lm, torch.full_like(mom[name], float("-inf")),
                                     mom[name])
            grow_idx = torch.topk(grow_score, k_grow).indices
            lm[grow_idx] = True
            wv = conv.weight.view(-1)
            wv[grow_idx] = 0.0
            masked.reset_optimizer_state(optimizer, name, grow_idx.view(-1))

        masked.masks[name] = lm.view_as(masked.masks[name])
        stats[name] = {"n_pruned": k_prune, "n_regrown": k_grow,
                       "n_live": int(lm.sum().item())}

    # Exact global conservation: total pruned == total regrown.
    tp = sum(s["n_pruned"] for s in stats.values())
    tr = sum(s["n_regrown"] for s in stats.values())
    if tp != tr:
        raise AssertionError(
            f"global conservation violated: pruned {tp} != regrown {tr}")

    masked.apply()
    return stats
