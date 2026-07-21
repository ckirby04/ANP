"""Tests for global sparse redistribution.

The single most important test here is the one that would have caught the
voided pilot: per-layer density CAN change across an update. The old per-layer
rule made that impossible, and no test asserted it should be possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsity.masking import MaskedLayers  # noqa: E402
from sparsity.redistribute import (  # noqa: E402
    _allocate,
    sparse_momentum_update,
)


def build(density=0.3, seed=0):
    """Three conv layers with a real SGD-momentum optimizer and populated buffers."""
    torch.manual_seed(seed)
    layers = {
        "stage1.conv0": nn.Conv3d(8, 16, 3, bias=False),
        "stage1.conv1": nn.Conv3d(16, 16, 3, bias=False),
        "stage2.conv0": nn.Conv3d(16, 32, 3, bias=False),
    }
    m = MaskedLayers(layers)
    m.randomize({k: density for k in layers},
                generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD([c.weight for c in layers.values()], lr=0.1, momentum=0.9)
    # Populate momentum buffers with a real step.
    for c in layers.values():
        c.weight.grad = torch.randn_like(c.weight)
    opt.step()
    return m, opt


def set_momentum(m, opt, per_layer_scale):
    """Overwrite momentum buffers so a layer's mean |momentum| is controllable."""
    for name, conv in m.layers.items():
        buf = opt.state[conv.weight]["momentum_buffer"]
        buf.copy_(torch.full_like(buf, per_layer_scale[name]))


# --- exact global conservation --------------------------------------------

def test_global_density_conserved_exactly():
    m, opt = build(0.3)
    before = m.total_live()
    for c in m.layers.values():
        c.weight.grad = torch.randn_like(c.weight)
    opt.step()
    stats = sparse_momentum_update(m, opt, prune_rate=0.3, min_density=0.05)
    assert m.total_live() == before
    tp = sum(s["n_pruned"] for s in stats.values())
    tr = sum(s["n_regrown"] for s in stats.values())
    assert tp == tr > 0


def test_global_conserved_over_many_updates():
    m, opt = build(0.3)
    before = m.total_live()
    for _ in range(20):
        for c in m.layers.values():
            c.weight.grad = torch.randn_like(c.weight)
        opt.step()
        sparse_momentum_update(m, opt, prune_rate=0.2, min_density=0.05)
        assert m.total_live() == before


@pytest.mark.parametrize("density", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_global_conserved_across_settings(density, rate):
    m, opt = build(density)
    before = m.total_live()
    for c in m.layers.values():
        c.weight.grad = torch.randn_like(c.weight)
    opt.step()
    sparse_momentum_update(m, opt, prune_rate=rate, min_density=0.05)
    assert m.total_live() == before


# --- THE test that would have caught the voided pilot ----------------------

def test_per_layer_density_can_change():
    """Per-layer density MUST be free to move under global conservation.

    This is the assertion whose absence let the per-layer-conservation pilot
    run for 22 hours measuring a quantity that was constant by construction.
    One layer is given dominant momentum so it gains budget while others lose.
    """
    m, opt = build(0.3)
    before = {name: m.n_live(name) for name in m.masks}
    # stage2.conv0 gets by far the largest momentum -> it should gain live weights.
    set_momentum(m, opt, {"stage1.conv0": 1.0, "stage1.conv1": 1.0,
                          "stage2.conv0": 50.0})
    sparse_momentum_update(m, opt, prune_rate=0.3, min_density=0.05)
    after = {name: m.n_live(name) for name in m.masks}

    assert after != before, "no per-layer density changed; global rule is inert"
    assert after["stage2.conv0"] > before["stage2.conv0"], \
        "the high-momentum layer did not gain capacity"
    assert any(after[n] < before[n] for n in before), \
        "no layer gave up capacity"
    # And the total is still exactly conserved.
    assert sum(after.values()) == sum(before.values())


def test_high_momentum_layer_gains_low_loses():
    m, opt = build(0.3)
    before = {name: m.n_live(name) for name in m.masks}
    set_momentum(m, opt, {"stage1.conv0": 0.01, "stage1.conv1": 1.0,
                          "stage2.conv0": 1.0})
    sparse_momentum_update(m, opt, prune_rate=0.3, min_density=0.05)
    after = {name: m.n_live(name) for name in m.masks}
    # The starved layer should not gain, and should typically lose.
    assert after["stage1.conv0"] <= before["stage1.conv0"]


# --- floor -----------------------------------------------------------------

def test_no_layer_falls_below_floor():
    m, opt = build(0.3)
    floor = 0.1
    # Give one layer near-zero momentum so redistribution keeps starving it.
    for _ in range(40):
        set_momentum(m, opt, {"stage1.conv0": 1e-6, "stage1.conv1": 1.0,
                              "stage2.conv0": 1.0})
        sparse_momentum_update(m, opt, prune_rate=0.5, min_density=floor)
        for name in m.masks:
            assert m.density(name) >= floor - 1e-9, \
                f"{name} fell to {m.density(name):.4f}, below floor {floor}"


def test_floor_binds_and_holds_the_layer_there():
    """A relentlessly starved layer converges to the floor, not below it."""
    m, opt = build(0.3)
    floor = 0.1
    for _ in range(60):
        set_momentum(m, opt, {"stage1.conv0": 1e-9, "stage1.conv1": 1.0,
                              "stage2.conv0": 1.0})
        sparse_momentum_update(m, opt, prune_rate=0.5, min_density=floor)
    assert m.density("stage1.conv0") == pytest.approx(floor, abs=0.02)


# --- sanity: the mover is not always the same layer ------------------------

def test_layer_that_changes_most_is_not_always_the_same():
    """Different momentum profiles must move different layers.

    A rule that always moved the same layer regardless of signal would be
    reallocating on something other than the momentum, which is the whole
    point of the mechanism.
    """
    movers = set()
    for step in range(6):
        m, opt = build(0.3, seed=step)
        # Rotate which layer has the dominant momentum.
        names = list(m.masks)
        scale = {n: 1.0 for n in names}
        scale[names[step % len(names)]] = 40.0
        before = {n: m.n_live(n) for n in names}
        set_momentum(m, opt, scale)
        sparse_momentum_update(m, opt, prune_rate=0.3, min_density=0.05)
        after = {n: m.n_live(n) for n in names}
        deltas = {n: abs(after[n] - before[n]) for n in names}
        movers.add(max(deltas, key=deltas.get))
    assert len(movers) > 1, f"the same layer always moved most: {movers}"


# --- allocation helper -----------------------------------------------------

def test_allocate_sums_to_total_and_respects_capacity():
    alloc = _allocate(100, contrib=[0.5, 0.3, 0.2], capacity=[40, 40, 40])
    assert sum(alloc) == 100
    assert all(a <= c for a, c in zip(alloc, [40, 40, 40]))


def test_allocate_spills_excess_from_capped_layer():
    # Layer 0 wants ~90 but can hold only 10; the rest must absorb the spill.
    alloc = _allocate(100, contrib=[0.9, 0.05, 0.05], capacity=[10, 200, 200])
    assert sum(alloc) == 100
    assert alloc[0] == 10


def test_allocate_zero_contrib_still_places_by_capacity():
    alloc = _allocate(50, contrib=[0.0, 0.0, 0.0], capacity=[100, 100, 100])
    assert sum(alloc) == 50


def test_update_is_noop_before_momentum_exists():
    torch.manual_seed(0)
    layers = {"stage1.conv0": nn.Conv3d(4, 4, 3, bias=False)}
    m = MaskedLayers(layers)
    m.randomize({"stage1.conv0": 0.3})
    opt = torch.optim.SGD([layers["stage1.conv0"].weight], lr=0.1, momentum=0.9)
    before = m.total_live()
    stats = sparse_momentum_update(m, opt, prune_rate=0.3, min_density=0.05)
    assert m.total_live() == before
    assert all(s["n_pruned"] == 0 for s in stats.values())
