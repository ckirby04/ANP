"""Analysis of the layer-wise density trajectory.

This module computes the quantities the pre-registered gates are defined over.
It deliberately reports numbers and boolean conditions, and does NOT decide
whether a result is interesting. Gate A and Gate B are stated in
docs/preregistration.md and are frozen; the functions here evaluate their
conditions as a factual matter.

Two views of every trajectory are carried throughout, because they can
disagree: per-layer `density`, and `live_budget_share`, the layer's share of
the total live-parameter budget. The shallow stages hold about 6 percent of
encoder parameters, so they can swing from 0.30 to 1.00 density while moving
almost no capacity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparsity.erk import erk_densities, live_budget_share

SHALLOW_STAGES = (0, 1, 2)
DEEP_STAGES = (3, 4, 5)


def load_trajectory(path: str | Path) -> list[dict]:
    """Read a trajectory CSV, coercing numeric columns."""
    ints = {"seed", "step", "epoch", "stage", "n_pruned", "n_regrown",
            "n_weights", "n_live"}
    floats = {"density", "live_budget_share"}
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out = dict(row)
            for k in ints:
                if k in out and out[k] != "":
                    out[k] = int(out[k])
            for k in floats:
                if k in out and out[k] != "":
                    out[k] = float(out[k])
            rows.append(out)
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    return rows


def load_many(results_dir: str | Path) -> list[dict]:
    """Every trajectory.csv under a results directory, concatenated."""
    rows = []
    for p in sorted(Path(results_dir).glob("*/trajectory.csv")):
        rows.extend(load_trajectory(p))
    return rows


def steps(rows: list[dict]) -> list[int]:
    return sorted({r["step"] for r in rows})


def stage_densities(rows: list[dict], step: int) -> dict[int, float]:
    """Parameter-weighted mean density per stage at one step.

    Weighted by parameter count, not a plain mean over layers: an unweighted
    mean would let a 27k-parameter layer count as much as a 2.7M one.
    """
    at = [r for r in rows if r["step"] == step]
    if not at:
        raise KeyError(f"no rows at step {step}")
    out = {}
    for stage in sorted({r["stage"] for r in at}):
        layers = [r for r in at if r["stage"] == stage]
        n = sum(r["n_weights"] for r in layers)
        out[stage] = sum(r["n_live"] for r in layers) / n
    return out


def stage_budget_shares(rows: list[dict], step: int) -> dict[int, float]:
    """Share of the live-parameter budget held by each stage at one step."""
    at = [r for r in rows if r["step"] == step]
    total = sum(r["n_live"] for r in at)
    if total == 0:
        return {r["stage"]: 0.0 for r in at}
    out = {}
    for stage in sorted({r["stage"] for r in at}):
        out[stage] = sum(r["n_live"] for r in at if r["stage"] == stage) / total
    return out


def deep_shallow_split(shares: dict[int, float]) -> tuple[float, float]:
    """(shallow, deep) fractions of the live budget, as percentages."""
    shallow = 100 * sum(v for k, v in shares.items() if k in SHALLOW_STAGES)
    deep = 100 * sum(v for k, v in shares.items() if k in DEEP_STAGES)
    return shallow, deep


def churn(rows: list[dict], step: int) -> dict[str, float]:
    """(pruned + regrown) / live per layer at one step.

    The disambiguator for a static allocation. A mistuned drop fraction gives
    low churn from step one; a converged mask gives high churn early that
    decays. Read the early-training segment, not only the tail.
    """
    at = [r for r in rows if r["step"] == step]
    return {r["layer_name"]: ((r["n_pruned"] + r["n_regrown"]) / r["n_live"]
                              if r["n_live"] else 0.0) for r in at}


def erk_reference(rows: list[dict], density: float) -> tuple[dict[int, float],
                                                             dict[int, float]]:
    """ERK stage densities and stage budget shares for this architecture.

    Derived from the layer shapes recorded in the trajectory itself, so the
    null is computed against exactly the layers that were sparsified.
    """
    first = steps(rows)[0]
    at = [r for r in rows if r["step"] == first]
    shapes = {r["layer_name"]: (r["n_weights"],) for r in at}
    stage_of = {r["layer_name"]: r["stage"] for r in at}

    # erk_densities needs true weight shapes to form its (sum dims / prod dims)
    # score, which a flat parameter count cannot provide. Recover the shape
    # from the layer name via the known nnU-Net encoder schedule.
    shapes = _shapes_from_names(sorted(shapes))
    dens = erk_densities(shapes, density)
    share = live_budget_share(shapes, dens)

    by_stage_d, by_stage_s = {}, {}
    for name, d in dens.items():
        s = stage_of[name]
        acc = by_stage_d.setdefault(s, [0.0, 0])
        acc[0] += np.prod(shapes[name]) * d
        acc[1] += np.prod(shapes[name])
        by_stage_s[s] = by_stage_s.get(s, 0.0) + share[name]
    return ({s: v[0] / v[1] for s, v in by_stage_d.items()}, by_stage_s)


def _shapes_from_names(names: list[str]) -> dict[str, tuple[int, ...]]:
    """Weight shapes for nnU-Net's planned encoder, keyed by layer name."""
    features = [32, 64, 128, 256, 320, 320]
    shapes = {}
    prev = 4
    for s, feat in enumerate(features):
        for c in (0, 1):
            name = f"stage{s}.conv{c}"
            c_in = prev if c == 0 else feat
            if name in names:
                shapes[name] = (feat, c_in, 3, 3, 3)
        prev = feat
    missing = set(names) - set(shapes)
    if missing:
        raise ValueError(f"unrecognised layer names: {sorted(missing)}")
    return shapes


def erk_ray_decomposition(final_shares: dict[int, float],
                          init_shares: dict[int, float],
                          erk_shares: dict[int, float]) -> dict[str, float]:
    """Decompose the budget-share move into ERK-ward and residual components.

    Implements the v2 Gate B geometry. Let v = share(final) - share(init) and
    u = share(ERK) - share(init), both in the sum-zero tangent space (shares
    sum to 1, so their differences sum to 0). Then:

      erk_ward = v . uhat            (movement along the ERK direction)
      residual = || v - erk_ward * uhat ||   (movement ERK does not point to)

    Units are budget-share percentage points (Euclidean). A pure drift toward
    ERK, of any magnitude, has residual 0 by construction; a task-specific
    allocation ERK does not point at has a substantial residual. All figures
    are returned; the pre-registration gates and reports on residual, and also
    reports erk_ward and the residual ratio raw so partial drift is visible.
    """
    stages = sorted(final_shares)
    v = np.array([100 * (final_shares[s] - init_shares[s]) for s in stages])
    u = np.array([100 * (erk_shares[s] - init_shares[s]) for s in stages])
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-12:
        raise ValueError("ERK coincides with init; ERK direction undefined")
    uhat = u / u_norm
    erk_ward = float(v @ uhat)
    residual_vec = v - erk_ward * uhat
    residual = float(np.linalg.norm(residual_vec))
    v_norm = float(np.linalg.norm(v))
    return {
        "residual": residual,
        "erk_ward": erk_ward,
        "residual_ratio": (residual / v_norm) if v_norm > 1e-12 else 0.0,
        "move_norm": v_norm,
        "erk_axis_norm": u_norm,
    }


@dataclass
class GateReport:
    """Factual evaluation of the pre-registered gate conditions.

    Reports whether each condition holds. It does not say what that means.
    """

    final_step: int
    tail_stage_densities: dict[int, float]
    tail_stage_shares: dict[int, float]
    erk_stage_densities: dict[int, float]
    erk_stage_shares: dict[int, float]
    max_abs_dev_from_uniform: float
    max_abs_dev_from_erk: float
    kendall_tau_mean: float
    shallow_pct: float
    deep_pct: float
    erk_shallow_pct: float
    erk_deep_pct: float
    budget_dev_from_erk_pts: float
    gate_a_density_condition: bool
    gate_a_ordering_condition: bool
    gate_a: bool
    gate_b_density_condition: bool
    gate_b_budget_condition: bool
    gate_b: bool
    monotone_in_depth: bool

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall tau between two rankings, over identical keys."""
    n = len(a)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def evaluate_gates(rows: list[dict],
                   target_density: float = 0.30,
                   tail_fraction: float = 0.30,
                   density_threshold: float = 0.05,
                   budget_threshold_pts: float = 3.0,
                   tau_threshold: float = 0.8,
                   n_ordering_checkpoints: int = 10) -> GateReport:
    """Evaluate the frozen Gate A and Gate B conditions.

    Thresholds default to the pre-registered values and are parameters only so
    the tests can exercise the logic. Do not change them for a real run.
    """
    all_steps = steps(rows)
    final = all_steps[-1]
    cutoff = final * (1 - tail_fraction)
    tail_steps = [s for s in all_steps if s >= cutoff] or [final]

    # Gate A, density condition: mean stage density over the tail, versus
    # uniform, in at least one stage.
    per_stage: dict[int, list[float]] = {}
    for s in tail_steps:
        for stage, d in stage_densities(rows, s).items():
            per_stage.setdefault(stage, []).append(d)
    tail_dens = {stage: float(np.mean(v)) for stage, v in per_stage.items()}
    max_dev_uniform = max(abs(d - target_density) for d in tail_dens.values())
    gate_a_density = max_dev_uniform > density_threshold

    # Gate A, ordering condition: Kendall tau between each of the last N
    # checkpoints' stage ranking and the final ranking.
    check_steps = all_steps[-n_ordering_checkpoints:]
    final_rank = stage_densities(rows, final)
    keys = sorted(final_rank)
    taus = [kendall_tau([stage_densities(rows, s)[k] for k in keys],
                        [final_rank[k] for k in keys]) for s in check_steps]
    tau_mean = float(np.mean(taus)) if taus else 1.0
    gate_a_ordering = tau_mean >= tau_threshold

    # Gate B: departure from ERK on both the density and the budget view.
    erk_dens, erk_share = erk_reference(rows, target_density)
    max_dev_erk = max(abs(tail_dens[s] - erk_dens[s]) for s in tail_dens)
    gate_b_density = max_dev_erk > density_threshold

    tail_shares = stage_budget_shares(rows, final)
    shallow, deep = deep_shallow_split(tail_shares)
    erk_shallow, erk_deep = deep_shallow_split(erk_share)
    budget_dev = abs(deep - erk_deep)
    gate_b_budget = budget_dev > budget_threshold_pts

    ordered = [tail_dens[s] for s in sorted(tail_dens)]
    monotone = all(b <= a + 1e-9 for a, b in zip(ordered, ordered[1:]))

    return GateReport(
        final_step=final,
        tail_stage_densities=tail_dens,
        tail_stage_shares=tail_shares,
        erk_stage_densities=erk_dens,
        erk_stage_shares=erk_share,
        max_abs_dev_from_uniform=max_dev_uniform,
        max_abs_dev_from_erk=max_dev_erk,
        kendall_tau_mean=tau_mean,
        shallow_pct=shallow,
        deep_pct=deep,
        erk_shallow_pct=erk_shallow,
        erk_deep_pct=erk_deep,
        budget_dev_from_erk_pts=budget_dev,
        gate_a_density_condition=gate_a_density,
        gate_a_ordering_condition=gate_a_ordering,
        gate_a=gate_a_density and gate_a_ordering,
        gate_b_density_condition=gate_b_density,
        gate_b_budget_condition=gate_b_budget,
        gate_b=gate_b_density and gate_b_budget,
        monotone_in_depth=monotone,
    )


def format_report(report: GateReport) -> str:
    """Human-readable gate conditions. States facts, draws no conclusions."""
    lines = [
        f"final step: {report.final_step}",
        "",
        "stage    density   ERK      dev      budget%   ERK budget%",
    ]
    for s in sorted(report.tail_stage_densities):
        d = report.tail_stage_densities[s]
        e = report.erk_stage_densities[s]
        lines.append(
            f"  {s}     {d:.4f}   {e:.4f}   {d-e:+.4f}   "
            f"{100*report.tail_stage_shares.get(s,0):6.2f}    "
            f"{100*report.erk_stage_shares.get(s,0):6.2f}")
    lines += [
        "",
        f"shallow/deep budget split : {report.shallow_pct:.1f} / {report.deep_pct:.1f}",
        f"ERK split                 : {report.erk_shallow_pct:.1f} / {report.erk_deep_pct:.1f}",
        f"budget departure from ERK : {report.budget_dev_from_erk_pts:.2f} points",
        f"monotone decreasing in depth: {report.monotone_in_depth}",
        "",
        f"Gate A density condition  (>0.05 from 0.30): {report.gate_a_density_condition} "
        f"(max {report.max_abs_dev_from_uniform:.4f})",
        f"Gate A ordering condition (tau >= 0.8)     : {report.gate_a_ordering_condition} "
        f"(mean tau {report.kendall_tau_mean:.4f})",
        f"Gate A                                     : {report.gate_a}",
        f"Gate B density condition  (>0.05 from ERK) : {report.gate_b_density_condition} "
        f"(max {report.max_abs_dev_from_erk:.4f})",
        f"Gate B budget condition   (>3 points)      : {report.gate_b_budget_condition}",
        f"Gate B                                     : {report.gate_b}",
        "",
        "These are the pre-registered conditions evaluated as facts.",
        # docs/preregistration.md is the source of truth for these conditions;
        # this function reports them and does not interpret them.
        "Interpretation follows the protocol in docs/preregistration.md.",
    ]
    return "\n".join(lines)
