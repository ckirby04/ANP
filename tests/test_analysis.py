"""Tests for the gate evaluation.

The gates are frozen, so the thing to test is that the arithmetic implements
what docs/preregistration.md says, including the case the pre-registration
explicitly warns about: a pure ERK-rediscovery trajectory passes Gate A and
must fail Gate B.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.trajectory import (  # noqa: E402
    churn,
    deep_shallow_split,
    erk_ray_decomposition,
    erk_reference,
    evaluate_gates,
    kendall_tau,
    stage_budget_shares,
    stage_densities,
)
from sparsity.erk import encoder_conv_shapes, erk_densities  # noqa: E402

SHAPES = encoder_conv_shapes([32, 64, 128, 256, 320, 320], [2] * 6,
                             [[3, 3, 3]] * 6, 4)


def renormalize(densities, target=0.30):
    """Scale per-layer densities so the OVERALL density is exactly `target`.

    Every arm holds total density fixed, so a synthetic trajectory that does
    not is not comparable to a real one: budget shares would shift for the
    trivial reason that the budget itself changed size. Layers that would
    exceed 1.0 are pinned dense and the remainder is rescaled, as in ERK.
    """
    n = {k: prod(v) for k, v in SHAPES.items()}
    total = sum(n.values())
    budget = target * total
    pinned = set()
    for _ in range(100):
        pool = [k for k in densities if k not in pinned]
        remaining = budget - sum(n[k] for k in pinned)
        denom = sum(densities[k] * n[k] for k in pool)
        scale = remaining / denom
        over = [k for k in pool if densities[k] * scale > 1.0]
        if not over:
            out = {k: 1.0 for k in pinned}
            out.update({k: densities[k] * scale for k in pool})
            return out
        pinned.update(over)
    raise RuntimeError("renormalize did not converge")


def make_rows(densities_by_layer, n_steps=12, step_size=250,
              pruned=0, regrown=0):
    """Synthetic trajectory at fixed per-layer densities."""
    rows = []
    for i in range(n_steps):
        step = i * step_size
        live = {n: int(round(densities_by_layer[n] * int(prod(s))))
                for n, s in SHAPES.items()}
        total_live = sum(live.values())
        for name, shape in SHAPES.items():
            n_w = int(prod(shape))
            rows.append({
                "run_id": "t", "seed": 0, "arm": "rigl", "step": step,
                "epoch": i, "layer_name": name, "stage": int(name[5]),
                "density": live[name] / n_w,
                "n_pruned": pruned, "n_regrown": regrown,
                "n_weights": n_w, "n_live": live[name],
                "live_budget_share": live[name] / total_live,
            })
    return rows


def prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


# --- helpers ---------------------------------------------------------------

def test_kendall_tau_identical_and_reversed():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_stage_density_is_parameter_weighted():
    """A 27k-parameter layer must not count as much as a 2.7M one."""
    d = {n: 0.30 for n in SHAPES}
    d["stage5.conv0"] = 0.10
    rows = make_rows(d, n_steps=1)
    stages = stage_densities(rows, 0)
    # stage5 has two layers of equal size, so the mean is halfway.
    assert stages[5] == pytest.approx(0.20, abs=1e-3)


def test_budget_shares_sum_to_one():
    rows = make_rows({n: 0.30 for n in SHAPES}, n_steps=1)
    assert sum(stage_budget_shares(rows, 0).values()) == pytest.approx(1.0)


def test_uniform_split_matches_parameter_share():
    rows = make_rows({n: 0.30 for n in SHAPES}, n_steps=1)
    shallow, deep = deep_shallow_split(stage_budget_shares(rows, 0))
    assert shallow == pytest.approx(6.1, abs=0.2)
    assert deep == pytest.approx(93.9, abs=0.2)


def test_erk_reference_matches_preregistered_numbers():
    """The null recorded in docs/preregistration.md."""
    rows = make_rows({n: 0.30 for n in SHAPES}, n_steps=1)
    dens, share = erk_reference(rows, 0.30)
    assert dens[0] == pytest.approx(1.000, abs=1e-3)
    assert dens[2] == pytest.approx(0.720, abs=1e-3)
    assert dens[5] == pytest.approx(0.241, abs=1e-3)
    shallow, deep = deep_shallow_split(share)
    assert shallow == pytest.approx(16.0, abs=0.2)
    assert deep == pytest.approx(84.0, abs=0.2)


def test_churn_uses_live_denominator():
    rows = make_rows({n: 0.30 for n in SHAPES}, n_steps=1,
                     pruned=100, regrown=100)
    c = churn(rows, 0)
    n_live = [r["n_live"] for r in rows if r["layer_name"] == "stage5.conv0"][0]
    assert c["stage5.conv0"] == pytest.approx(200 / n_live)


# --- Gate A ----------------------------------------------------------------

def test_flat_uniform_trajectory_fails_gate_a():
    """Nothing moved: a real result, and it must register as one."""
    report = evaluate_gates(make_rows({n: 0.30 for n in SHAPES}))
    assert not report.gate_a_density_condition
    assert report.gate_a_ordering_condition   # stable, just not moving
    assert not report.gate_a


def test_large_stable_departure_passes_gate_a():
    d = {n: 0.30 for n in SHAPES}
    for n in SHAPES:
        if n.startswith("stage5"):
            d[n] = 0.10
    report = evaluate_gates(make_rows(d))
    assert report.gate_a_density_condition
    assert report.gate_a_ordering_condition
    assert report.gate_a


def test_unstable_ordering_fails_gate_a():
    """Movement without a stable stage ordering is not a pass."""
    rows = []
    for i in range(12):
        d = {n: 0.30 for n in SHAPES}
        # Alternate which stage is drained, so the ranking never settles.
        target = "stage5" if i % 2 == 0 else "stage1"
        for n in SHAPES:
            if n.startswith(target):
                d[n] = 0.05
        rows.extend(r for r in make_rows(d, n_steps=1) if True
                    for r in [dict(r, step=i * 250, epoch=i)])
    report = evaluate_gates(rows)
    assert report.gate_a_density_condition
    assert not report.gate_a_ordering_condition
    assert not report.gate_a


# --- Gate B: the ERK-rediscovery case --------------------------------------

def test_erk_trajectory_passes_gate_a_but_fails_gate_b():
    """The exact scenario the pre-registration warns about.

    A run that merely rediscovers ERK moves stage 5 by 0.059, clearing Gate
    A's 0.05 threshold, and would look like a result. Gate B must reject it.
    """
    report = evaluate_gates(make_rows(erk_densities(SHAPES, 0.30)))
    assert report.gate_a, "ERK alone should clear Gate A"
    assert not report.gate_b_density_condition
    assert not report.gate_b_budget_condition
    assert not report.gate_b, "ERK rediscovery must not pass Gate B"
    assert report.monotone_in_depth


def test_mid_depth_bulge_passes_gate_b():
    """The discriminating signature: stages 2-3 above their ERK allocation,
    paid for by stages 4-5, at constant overall density."""
    d = dict(erk_densities(SHAPES, 0.30))
    for n in SHAPES:
        if n.startswith(("stage2", "stage3")):
            d[n] = min(1.0, d[n] + 0.25)
        if n.startswith(("stage4", "stage5")):
            d[n] = max(0.01, d[n] - 0.12)
    report = evaluate_gates(make_rows(renormalize(d)))
    assert report.gate_a
    assert report.gate_b_density_condition
    assert report.gate_b_budget_condition
    assert report.gate_b


def test_shallow_only_departure_has_bounded_budget_effect():
    """Quantifies the headroom the budget clause actually provides.

    Stages 0-2 hold 6.1 percent of parameters and ERK already pins stages 0-1
    dense, so the largest departure available to the shallow stages alone is
    driving stage 2 from its ERK 0.720 up to 1.0. That is a real but small
    budget shift. This test records the size of it rather than asserting a
    convenient outcome: see docs/protocol_history.md, which argues the 3-point
    threshold has less headroom against a shallow-only departure than the
    pre-registration implies. The threshold is frozen and is not changed here.
    """
    d = dict(erk_densities(SHAPES, 0.30))
    for n in SHAPES:
        if n.startswith(("stage0", "stage1", "stage2")):
            d[n] = 1.0
    report = evaluate_gates(make_rows(renormalize(d)))
    # The whole shallow block cannot move more than about 5 points of budget.
    assert report.budget_dev_from_erk_pts < 5.0


def test_moderate_shallow_departure_fails_budget_condition():
    """A partial shallow-only move stays under the 3-point clause."""
    d = dict(erk_densities(SHAPES, 0.30))
    for n in SHAPES:
        if n.startswith("stage2"):
            d[n] = min(1.0, d[n] + 0.10)
    report = evaluate_gates(make_rows(renormalize(d)))
    assert not report.gate_b_budget_condition
    assert not report.gate_b


def test_monotonicity_detects_non_monotone_allocation():
    """Non-monotone in DENSITY requires a deeper stage above a shallower one.

    Note this is hard to achieve against ERK, which already pins stages 0-1 at
    1.0; a mid-depth bulge relative to ERK can still be monotone overall. The
    argument is recorded in docs/protocol_history.md. Here the check is exercised on
    a uniform baseline where it is attainable.
    """
    d = {n: 0.30 for n in SHAPES}
    for n in SHAPES:
        if n.startswith("stage3"):
            d[n] = 0.9
    assert not evaluate_gates(make_rows(renormalize(d))).monotone_in_depth


# --- v2 Gate B geometry: ERK-ray decomposition ----------------------------

# Per-stage budget shares (fractions summing to 1) for the real architecture.
_INIT = {0: 0.00197, 1: 0.01183, 2: 0.04734, 3: 0.18935, 4: 0.35503, 5: 0.39448}
_ERK = {0: 0.00657, 1: 0.03945, 2: 0.11364, 3: 0.22289, 4: 0.30092, 5: 0.31653}


def test_pure_erk_drift_has_zero_residual():
    """The replication case Gate B must exclude, at any magnitude."""
    for frac in (0.25, 0.5, 1.0, 0.89):
        final = {s: _INIT[s] + frac * (_ERK[s] - _INIT[s]) for s in _INIT}
        d = erk_ray_decomposition(final, _INIT, _ERK)
        assert d["residual"] == pytest.approx(0.0, abs=1e-6)
        assert d["erk_ward"] > 0 if frac > 0 else True


def test_staying_at_init_has_zero_everything():
    d = erk_ray_decomposition(dict(_INIT), _INIT, _ERK)
    assert d["residual"] == pytest.approx(0.0, abs=1e-9)
    assert d["move_norm"] == pytest.approx(0.0, abs=1e-9)


def test_task_specific_allocation_has_substantial_residual():
    """A move ERK does not point at registers a nonzero residual."""
    # Enrich stage 2 well beyond ERK, keep shallow-most sparse.
    final = {0: 0.002, 1: 0.012, 2: 0.14, 3: 0.30, 4: 0.30, 5: 0.246}
    tot = sum(final.values())
    final = {s: v / tot for s, v in final.items()}
    d = erk_ray_decomposition(final, _INIT, _ERK)
    assert d["residual"] > 5.0
    assert 0.0 < d["residual_ratio"] <= 1.0


def test_residual_is_orthogonal_to_erk_direction():
    final = {0: 0.01, 1: 0.02, 2: 0.10, 3: 0.25, 4: 0.30, 5: 0.32}
    tot = sum(final.values())
    final = {s: v / tot for s, v in final.items()}
    d = erk_ray_decomposition(final, _INIT, _ERK)
    # residual^2 + erk_ward^2 == move_norm^2 (Pythagoras in the decomposition)
    assert d["residual"] ** 2 + d["erk_ward"] ** 2 == pytest.approx(
        d["move_norm"] ** 2, rel=1e-6)


def test_report_formats_without_interpreting():
    from analysis.trajectory import format_report
    text = format_report(evaluate_gates(make_rows({n: 0.30 for n in SHAPES})))
    assert "Gate A" in text and "Gate B" in text
    # The report must not editorialise.
    for word in ("success", "failure", "confirms", "proves", "interesting"):
        assert word not in text.lower()
