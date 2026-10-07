"""Admission control: largest fleet N one GPU can serve under the SLOs

    P(deadline miss) <= eps,  E[staleness] <= s_max,
    mean success >= dedicated_success - delta,
    quantile_{tail_q}(success_i - dedicated_i) >= -delta_tail   (per robot).

The per-robot term matters for curve-aware scheduling, which deliberately
skips some robots: a fleet mean can look fine while a minority is starved.

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
    delta: float = 0.02      # max mean success loss vs dedicated
    tail_q: float = 0.05     # per-robot guarantee: this quantile of per-robot loss...
    delta_tail: float = 0.05 # ...must not exceed this (no starved minority)


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


def reference(templates, models, lat, horizon_s, seed=0) -> np.ndarray:
    """Dedicated-GPU success of each template (one robot alone on a GPU)."""
    from .sim import ChronosPolicy as _P
    return np.array([evaluate(_P, [t], 1, lat, models, horizon_s, seed)["success"] for t in templates])


def evaluate(policy_factory, templates, n, lat, models, horizon_s, seed=0, ref=None, sim_cls=Simulator):
    """Run one fleet. With `ref` (per-template dedicated success), also reports
    per-robot loss: `loss_tail` is the `tail_q` quantile of success_i - ref_i."""
    sim = sim_cls(make_fleet(templates, n, seed), lat, policy_factory())
    m = sim.run(horizon_s)
    succ = sim.success(models)
    m["success"] = float(succ.mean()) if n else 1.0
    m["success_min"] = float(succ.min()) if n else 1.0
    if ref is not None and n:
        loss = succ - np.asarray(ref)[np.arange(n) % len(templates)]
        m["loss_per_robot"] = loss
    m["n"] = n
    return m


def feasible(m, ref_mean, slo) -> bool:
    ok = (m["miss_rate"] <= slo.eps and m["mean_staleness"] <= slo.s_max
          and m["success"] >= ref_mean - slo.delta)
    if "loss_per_robot" in m:
        m["loss_tail"] = float(np.quantile(m.pop("loss_per_robot"), slo.tail_q))
        ok = ok and m["loss_tail"] >= -slo.delta_tail
    return bool(ok)


def capacity(policy_factory, templates, models, lat=LatencyModel(), slo=SLO(),
             horizon_s=20.0, n_max=256, patience=3, seed=0, n_start=1, sim_cls=Simulator):
    """Returns (max feasible N, reference success, list of per-N metrics).

    `n_start` skips the scan below a known-feasible size (e.g. a fraction of
    `oracle.fast_capacity`); the result is only valid if n_start is feasible."""
    refs = reference(templates, models, lat, horizon_s, seed)
    ref = float(refs.mean())
    best, fails, log = 0, 0, []
    for n in range(max(n_start, 1), n_max + 1):
        m = evaluate(policy_factory, templates, n, lat, models, horizon_s, seed, ref=refs, sim_cls=sim_cls)
        m["feasible"] = feasible(m, ref, slo)
        log.append(m)
        if m["feasible"]:
            best, fails = n, 0
        else:
            fails += 1
            if fails >= patience:
                break
    return best, float(ref), log
