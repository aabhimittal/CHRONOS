"""Admission control: largest fleet N one GPU can serve under the SLOs

    P(deadline miss) <= eps,  E[staleness] <= s_max,
    mean success >= dedicated_success - delta.

The reference is the per-robot-dedicated deployment (N = 1, one GPU each).
Feasibility is not guaranteed monotone in N, so we scan upward and stop after
`patience` consecutive infeasible N instead of bisecting.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .sim import ChronosPolicy, LatencyModel, Robot, Simulator


@dataclass
class SLO:
    eps: float = 0.01        # max deadline-miss rate
    s_max: float = 1.0       # max mean staleness (s)
    delta: float = 0.02      # max absolute success loss vs dedicated


def make_fleet(templates: list[Robot], n: int, seed: int = 0, aligned: bool = False) -> list[Robot]:
    """Cycle through task templates. Random release phases and work-cycle
    offsets avoid lock-step bursts unless `aligned=True` (worst case)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = templates[i % len(templates)]
        cyc = sum(d for _, d in t.phases) if t.phases else 0.0
        out.append(replace(t, phase=0.0 if aligned else float(rng.uniform(0, t.period)),
                           cycle_offset=0.0 if aligned else float(rng.uniform(0, cyc))))
    return out


def evaluate(policy_factory, templates, n, lat, models, horizon_s, seed=0):
    sim = Simulator(make_fleet(templates, n, seed), lat, policy_factory())
    m = sim.run(horizon_s)
    m["success"] = float(sim.success(models).mean())
    m["n"] = n
    return m


def capacity(policy_factory, templates, models, lat=LatencyModel(), slo=SLO(),
             horizon_s=20.0, n_max=256, patience=3, seed=0, n_start=1):
    """Returns (max feasible N, reference success, list of per-N metrics).

    `n_start` skips the scan below a known-feasible size (e.g. a fraction of
    `oracle.fast_capacity`); the result is only valid if n_start is feasible."""
    ref = np.mean([evaluate(lambda: ChronosPolicy(), [tpl], 1, lat, models, horizon_s, seed)["success"]
                   for tpl in templates])
    best, fails, log = 0, 0, []
    for n in range(max(n_start, 1), n_max + 1):
        m = evaluate(policy_factory, templates, n, lat, models, horizon_s, seed)
        m["feasible"] = (m["miss_rate"] <= slo.eps and m["mean_staleness"] <= slo.s_max
                         and m["success"] >= ref - slo.delta)
        log.append(m)
        if m["feasible"]:
            best, fails = n, 0
        else:
            fails += 1
            if fails >= patience:
                break
    return best, float(ref), log
