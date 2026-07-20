"""Multiplicative weight masks.

Masks are dense boolean tensors multiplied into the weights, never
`torch.sparse`. Two reasons, both load-bearing:

  - RigL regrows connections by the magnitude of the DENSE gradient at
    currently-masked positions. A sparse tensor would not produce a gradient
    there at all, so the regrowth criterion could not be computed.
  - Unstructured sparse kernels are slower than dense ones on this hardware.

This buys no speedup and is not intended to. It is an instrument for observing
where connectivity migrates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedLayers:
    """Owns the masks for a set of conv layers and keeps weights consistent.

    The mask is applied after every optimizer step, so masked weights are held
    at exactly zero. Gradients are left dense: `weight.grad` is the gradient of
    the unmasked weight, which is exactly what RigL's regrowth criterion needs.
    """

    def __init__(self, layers: dict[str, nn.Conv3d], device=None):
        if not layers:
            raise ValueError("no layers to mask")
        self.layers = dict(layers)
        self.masks: dict[str, torch.Tensor] = {
            name: torch.ones_like(conv.weight, dtype=torch.bool,
                                  device=device or conv.weight.device)
            for name, conv in self.layers.items()
        }

    # --- basic accounting -------------------------------------------------

    def n_weights(self, name: str) -> int:
        return self.masks[name].numel()

    def n_live(self, name: str) -> int:
        return int(self.masks[name].sum().item())

    def density(self, name: str) -> float:
        return self.n_live(name) / self.n_weights(name)

    def total_weights(self) -> int:
        return sum(m.numel() for m in self.masks.values())

    def total_live(self) -> int:
        return sum(int(m.sum().item()) for m in self.masks.values())

    def overall_density(self) -> float:
        return self.total_live() / self.total_weights()

    # --- application ------------------------------------------------------

    @torch.no_grad()
    def apply(self) -> None:
        """Zero every masked weight. Call after each optimizer step."""
        for name, conv in self.layers.items():
            conv.weight.mul_(self.masks[name])

    @torch.no_grad()
    def apply_to_grads(self) -> None:
        """Zero gradients at masked positions.

        Used between computing the dense gradient (which RigL needs) and taking
        the optimizer step, so that momentum does not accumulate on dead
        connections and silently drift them away from zero.
        """
        for name, conv in self.layers.items():
            if conv.weight.grad is not None:
                conv.weight.grad.mul_(self.masks[name])

    @torch.no_grad()
    def reset_optimizer_state(self, optimizer, name: str,
                              positions: torch.Tensor) -> None:
        """Clear momentum at newly regrown positions.

        A regrown connection inherits whatever momentum accumulated while it
        was dead. Left alone that gives it a large spurious first step.

        `positions` are FLAT indices into the weight. The momentum buffer has
        the weight's full shape, so it must be flattened before indexing;
        indexing it directly would silently zero whole output channels.
        """
        conv = self.layers[name]
        state = optimizer.state.get(conv.weight)
        if not state:
            return
        buf = state.get("momentum_buffer")
        if buf is not None:
            buf.view(-1)[positions.view(-1)] = 0.0

    # --- initialization ---------------------------------------------------

    @torch.no_grad()
    def randomize(self, densities: dict[str, float], generator=None) -> None:
        """Draw a random mask per layer at the given per-layer density.

        The live count per layer is exact, not binomial: `round(density * n)`
        positions are chosen without replacement. Sampling each weight
        independently would make the realised density fluctuate and would
        make "density held exactly constant" untestable.
        """
        for name, mask in self.masks.items():
            n = mask.numel()
            k = int(round(densities[name] * n))
            k = max(0, min(n, k))
            flat = torch.zeros(n, dtype=torch.bool, device=mask.device)
            if k:
                idx = torch.randperm(n, generator=generator, device=mask.device)[:k]
                flat[idx] = True
            self.masks[name] = flat.view_as(mask)
        self.apply()

    @torch.no_grad()
    def prune_by_magnitude(self, densities: dict[str, float]) -> dict[str, int]:
        """Keep the largest-magnitude weights per layer at the given density."""
        pruned = {}
        for name, conv in self.layers.items():
            mask = self.masks[name]
            n = mask.numel()
            k = max(0, min(n, int(round(densities[name] * n))))
            before = int(mask.sum().item())

            flat = conv.weight.detach().abs().flatten()
            # Only currently-live weights are eligible to survive.
            flat = torch.where(mask.flatten(), flat,
                               torch.full_like(flat, -1.0))
            new = torch.zeros(n, dtype=torch.bool, device=mask.device)
            if k:
                idx = torch.topk(flat, k).indices
                new[idx] = True
            self.masks[name] = new.view_as(mask)
            pruned[name] = before - int(new.sum().item())
        self.apply()
        return pruned

    # --- persistence ------------------------------------------------------

    def state_dict(self) -> dict:
        return {name: m.detach().cpu() for name, m in self.masks.items()}

    def load_state_dict(self, state: dict) -> None:
        missing = set(self.masks) - set(state)
        extra = set(state) - set(self.masks)
        if missing or extra:
            raise ValueError(
                f"mask state does not match the network: missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}")
        for name, m in state.items():
            target = self.masks[name]
            if tuple(m.shape) != tuple(target.shape):
                raise ValueError(
                    f"mask shape mismatch for {name}: {tuple(m.shape)} vs "
                    f"{tuple(target.shape)}")
            self.masks[name] = m.to(target.device).bool()
        self.apply()
