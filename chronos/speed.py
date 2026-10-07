"""Speed governor: trade scene speed for compute.

Staleness hurts because the world moves while a plan ages. A facility can
slow the world down (conveyor speed, robot max velocity). Fewer items per
hour per robot, but each robot tolerates staler plans, so one GPU can serve
more robots. The quantity a warehouse buys is items/hour per GPU:

    throughput = sum_i  v_i * success_i(v_i)        (v = speed multiplier)

subject to the same deadline SLO and an absolute quality floor (success no
worse than the dedicated robot at nominal speed, minus delta).

A speed *family* is one fitted curve per speed level. Displacement scaling
(p_v(s) = p_1(s v)) does not hold even on the toy task because speed also
changes how long episodes run, so curves are fitted per level, not rescaled.
"""
from __future__ import annotations

import numpy as np

from .admission import SLO, make_fleet, reference
from .sim import ChronosPolicy, LatencyModel, Simulator


def evaluate_speeds(templates, family: dict, base_models: dict, n: int, lat=LatencyModel(),
                    slo=SLO(), horizon_s=10.0, ref=None):
    """For one fleet size, try each speed level of the speed-controllable
    tasks (`family`: task -> {v: model}). Returns rows {v, feasible, throughput,...}."""
    ref = reference(templates, base_models, lat, horizon_s) if ref is None else ref
    levels = sorted({v for fam in family.values() for v in fam})
    rows = []
    for v in levels:
        models = {**base_models, **{t: fam[v] for t, fam in family.items()}}
        sim = Simulator(make_fleet(templates, n), lat, ChronosPolicy(models=models, max_age=slo.s_max))
        m = sim.run(horizon_s)
        succ = sim.success(models)
        speed = np.array([v if r.task in family else 1.0 for r in sim.robots])
        loss = succ - ref[np.arange(n) % len(templates)]
        ok = (m["miss_rate"] <= slo.eps and succ.mean() >= ref.mean() - slo.delta
              and np.quantile(loss, slo.tail_q) >= -slo.delta_tail)
        rows.append({"v": v, "n": n, "feasible": bool(ok), "throughput": float((speed * succ).sum()),
                     "success": float(succ.mean()), "miss_rate": float(m["miss_rate"])})
    return rows


def governed(rows):
    """Best feasible speed level for this fleet size (None if none is feasible)."""
    ok = [r for r in rows if r["feasible"]]
    return max(ok, key=lambda r: r["throughput"]) if ok else None
