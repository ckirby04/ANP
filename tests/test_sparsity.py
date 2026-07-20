"""Sparsity machinery tests.

The load-bearing assertion is that a RigL update preserves density EXACTLY,
per layer, not approximately. Everything else in the trajectory CSV is
uninterpretable if that drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sparsity.erk import encoder_conv_shapes, erk_densities, live_budget_share  # noqa: E402
from sparsity.masking import MaskedLayers  # noqa: E402
from sparsity.rigl import (  # noqa: E402
    cosine_drop_fraction,
    regrowth_scores,
    rigl_update,
    topk_overlap,
)


def tiny_layers(seed=0):
    torch.manual_seed(seed)
    return {
        "stage1.conv0": nn.Conv3d(8, 16, 3, bias=False),
        "stage1.conv1": nn.Conv3d(16, 16, 3, bias=False),
        "stage2.conv0": nn.Conv3d(16, 32, 3, bias=False),
    }


def masked_with_grads(density=0.3, seed=0):
    layers = tiny_layers(seed)
    m = MaskedLayers(layers)
    m.randomize({k: density for k in layers},
                generator=torch.Generator().manual_seed(seed))
    for conv in layers.values():
        conv.weight.grad = torch.randn_like(conv.weight)
    return m


# --- masks -----------------------------------------------------------------

def test_randomize_hits_exact_density_per_layer():
    m = masked_with_grads(0.3)
    for name in m.masks:
        n = m.n_weights(name)
        assert m.n_live(name) == int(round(0.3 * n))


def test_apply_zeros_masked_weights_exactly():
    m = masked_with_grads(0.3)
    for name, conv in m.layers.items():
        dead = ~m.masks[name]
        assert torch.all(conv.weight[dead] == 0.0)
        # Live weights must not have been touched.
        assert torch.any(conv.weight[m.masks[name]] != 0.0)


def test_apply_to_grads_zeros_dead_positions():
    m = masked_with_grads(0.3)
    m.apply_to_grads()
    for name, conv in m.layers.items():
        assert torch.all(conv.weight.grad[~m.masks[name]] == 0.0)


def test_prune_by_magnitude_keeps_the_largest():
    layers = {"stage1.conv0": nn.Conv3d(4, 4, 3, bias=False)}
    m = MaskedLayers(layers)
    w = layers["stage1.conv0"].weight
    with torch.no_grad():
        w.copy_(torch.arange(w.numel(), dtype=w.dtype).view_as(w))
    m.prune_by_magnitude({"stage1.conv0": 0.25})
    n = w.numel()
    k = int(round(0.25 * n))
    kept = m.masks["stage1.conv0"].flatten().nonzero().flatten()
    assert len(kept) == k
    # arange means the largest values are the last k indices.
    assert set(kept.tolist()) == set(range(n - k, n))


def test_state_dict_roundtrip():
    m = masked_with_grads(0.3)
    before = {k: v.clone() for k, v in m.masks.items()}
    state = m.state_dict()
    m.randomize({k: 0.9 for k in m.masks})
    m.load_state_dict(state)
    for k in before:
        assert torch.equal(m.masks[k], before[k])


def test_state_dict_rejects_mismatched_network():
    m = masked_with_grads(0.3)
    state = m.state_dict()
    state.pop("stage1.conv0")
    with pytest.raises(ValueError, match="does not match"):
        m.load_state_dict(state)


# --- RigL: the density conservation guarantee ------------------------------

def test_rigl_preserves_density_exactly_per_layer():
    m = masked_with_grads(0.3)
    before = {name: m.n_live(name) for name in m.masks}
    stats = rigl_update(m, drop_fraction=0.3)
    for name in m.masks:
        assert m.n_live(name) == before[name], f"{name} changed live count"
        assert stats[name]["n_pruned"] == stats[name]["n_regrown"]


def test_rigl_preserves_density_across_many_updates():
    m = masked_with_grads(0.3)
    before = {name: m.n_live(name) for name in m.masks}
    for _ in range(15):
        for conv in m.layers.values():
            conv.weight.grad = torch.randn_like(conv.weight)
        rigl_update(m, drop_fraction=0.2)
        for name in m.masks:
            assert m.n_live(name) == before[name]


@pytest.mark.parametrize("density", [0.05, 0.1, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5])
def test_rigl_preserves_density_across_settings(density, frac):
    m = masked_with_grads(density)
    before = {name: m.n_live(name) for name in m.masks}
    rigl_update(m, drop_fraction=frac)
    for name in m.masks:
        assert m.n_live(name) == before[name]


def test_rigl_actually_rewires():
    """Conservation is trivially satisfied by doing nothing; it must not."""
    m = masked_with_grads(0.3)
    before = {k: v.clone() for k, v in m.masks.items()}
    stats = rigl_update(m, drop_fraction=0.3)
    for name in m.masks:
        assert stats[name]["n_pruned"] > 0
        assert not torch.equal(m.masks[name], before[name])


def test_rigl_drops_smallest_weights_and_grows_largest_gradients():
    layers = {"stage1.conv0": nn.Conv3d(2, 2, 3, bias=False)}
    m = MaskedLayers(layers)
    conv = layers["stage1.conv0"]
    n = conv.weight.numel()

    live = torch.zeros(n, dtype=torch.bool)
    live[: n // 2] = True
    m.masks["stage1.conv0"] = live.view_as(conv.weight)

    with torch.no_grad():
        w = torch.zeros(n)
        w[: n // 2] = torch.arange(1, n // 2 + 1, dtype=torch.float)
        conv.weight.copy_(w.view_as(conv.weight))
    g = torch.zeros(n)
    g[n // 2:] = torch.arange(1, n - n // 2 + 1, dtype=torch.float)
    conv.weight.grad = g.view_as(conv.weight)

    k = int(np.floor(0.25 * (n // 2)))
    rigl_update(m, drop_fraction=0.25)
    new = m.masks["stage1.conv0"].flatten()

    # The k smallest live weights are indices 0..k-1.
    assert not new[:k].any()
    # The k largest dead gradients are the final k indices.
    assert new[n - k:].all()


def test_rigl_zeroes_regrown_weights():
    m = masked_with_grads(0.3)
    before = {k: v.clone() for k, v in m.masks.items()}
    rigl_update(m, drop_fraction=0.3)
    for name, conv in m.layers.items():
        regrown = m.masks[name] & ~before[name]
        assert regrown.any()
        assert torch.all(conv.weight[regrown] == 0.0)


def test_rigl_clears_momentum_at_regrown_positions():
    m = masked_with_grads(0.3)
    params = [c.weight for c in m.layers.values()]
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.9)
    opt.step()   # populate momentum buffers
    for conv in m.layers.values():
        buf = opt.state[conv.weight]["momentum_buffer"]
        buf.fill_(5.0)

    before = {k: v.clone() for k, v in m.masks.items()}
    rigl_update(m, drop_fraction=0.3, optimizer=opt)
    for name, conv in m.layers.items():
        regrown = (m.masks[name] & ~before[name]).flatten()
        buf = opt.state[conv.weight]["momentum_buffer"].flatten()
        assert torch.all(buf[regrown] == 0.0)


def test_rigl_requires_gradients():
    m = masked_with_grads(0.3)
    for conv in m.layers.values():
        conv.weight.grad = None
    with pytest.raises(RuntimeError, match="no gradient"):
        rigl_update(m, drop_fraction=0.3)


def test_rigl_zero_drop_fraction_is_a_noop():
    m = masked_with_grads(0.3)
    before = {k: v.clone() for k, v in m.masks.items()}
    stats = rigl_update(m, drop_fraction=0.0)
    for name in m.masks:
        assert torch.equal(m.masks[name], before[name])
        assert stats[name]["n_pruned"] == 0


def test_rigl_cannot_grow_more_than_dead_positions():
    """At high density there are few dead slots; k must clamp, not overflow."""
    m = masked_with_grads(0.95)
    before = {name: m.n_live(name) for name in m.masks}
    rigl_update(m, drop_fraction=0.5)
    for name in m.masks:
        assert m.n_live(name) == before[name]


# --- drop fraction schedule ------------------------------------------------

def test_cosine_decay_starts_at_initial_and_reaches_zero():
    total, end = 1000, 0.75
    assert cosine_drop_fraction(0, 0.3, total, end) == pytest.approx(0.3)
    assert cosine_drop_fraction(750, 0.3, total, end) == 0.0
    assert cosine_drop_fraction(999, 0.3, total, end) == 0.0


def test_cosine_decay_is_monotone():
    vals = [cosine_drop_fraction(s, 0.3, 1000, 0.75) for s in range(0, 800, 10)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_cosine_decay_halfway():
    # cos(pi/2) = 0 -> initial/2 at half of the decay window.
    assert cosine_drop_fraction(375, 0.3, 1000, 0.75) == pytest.approx(0.15)


# --- regrowth informativeness diagnostic -----------------------------------

def test_regrowth_scores_exclude_live_positions():
    m = masked_with_grads(0.3)
    for name in m.masks:
        s = regrowth_scores(m, name)
        live = m.masks[name].flatten()
        assert torch.all(torch.isinf(s[live]) & (s[live] < 0))
        assert torch.all(torch.isfinite(s[~live]))


def test_topk_overlap_bounds():
    a = torch.arange(100, dtype=torch.float)
    assert topk_overlap(a, a, 10) == 1.0
    assert topk_overlap(a, -a, 10) == 0.0


def test_topk_overlap_partial():
    a = torch.zeros(100)
    b = torch.zeros(100)
    a[:10] = torch.arange(10, 0, -1, dtype=torch.float)
    b[5:15] = torch.arange(10, 0, -1, dtype=torch.float)
    assert topk_overlap(a, b, 10) == pytest.approx(0.5)


# --- ERK -------------------------------------------------------------------

def test_erk_hits_overall_density():
    shapes = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                                 [[3, 3, 3]] * 6, 4)
    d = erk_densities(shapes, 0.30)
    n = {k: int(np.prod(v)) for k, v in shapes.items()}
    live = sum(n[k] * d[k] for k in shapes)
    assert live / sum(n.values()) == pytest.approx(0.30, abs=1e-9)


def test_erk_is_monotone_decreasing_in_depth():
    """The null's signature. Gate B looks for departures from this."""
    shapes = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                                 [[3, 3, 3]] * 6, 4)
    d = erk_densities(shapes, 0.30)
    by_stage = {}
    for name, dens in d.items():
        stage = int(name[5])
        n = int(np.prod(shapes[name]))
        acc = by_stage.setdefault(stage, [0, 0])
        acc[0] += n * dens
        acc[1] += n
    means = [by_stage[s][0] / by_stage[s][1] for s in sorted(by_stage)]
    assert all(b <= a + 1e-9 for a, b in zip(means, means[1:])), means


def test_erk_densities_never_exceed_one():
    shapes = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                                 [[3, 3, 3]] * 6, 4)
    for dens in erk_densities(shapes, 0.30).values():
        assert 0.0 < dens <= 1.0


def test_budget_share_sums_to_one():
    shapes = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                                 [[3, 3, 3]] * 6, 4)
    d = erk_densities(shapes, 0.30)
    assert sum(live_budget_share(shapes, d).values()) == pytest.approx(1.0)


def test_uniform_budget_share_equals_parameter_share():
    """Under a uniform mask, capacity share is exactly parameter share."""
    shapes = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                                 [[3, 3, 3]] * 6, 4)
    share = live_budget_share(shapes, {k: 0.30 for k in shapes})
    total = sum(int(np.prod(v)) for v in shapes.values())
    for k, v in shapes.items():
        assert share[k] == pytest.approx(int(np.prod(v)) / total)
