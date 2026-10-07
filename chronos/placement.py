"""Multi-GPU fleet placement.

Given M robots across task types, how many GPUs does each placement need?
  * mixed:      round-robin. Every GPU sees the same task mix, so curve-aware
                batching can skip the insensitive robots on every GPU.
  * segregated: one pool per task type, each sized to that task's own
                single-GPU capacity (insensitive robots pack densely).
The answer depends entirely on the curves; neither strategy dominates.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .admission import SLO, capacity
from .sim import ChronosPolicy, LatencyModel


def gpus_needed(counts: dict, templates: dict, models, strategy="mixed", lat=LatencyModel(),
                slo=SLO(), horizon_s=10.0) -> dict:
    """counts: task -> number of robots; templates: task -> Robot template."""
    pf = lambda: ChronosPolicy(models=models, max_age=slo.s_max)
    if strategy == "segregated":
        per = {}
        for task, m in counts.items():
            cap, _, _ = capacity(pf, [templates[task]], models, lat, slo, horizon_s)
            per[task] = math.ceil(m / cap) if cap else math.inf
        return {"gpus": sum(per.values()), "per_task": per}
    total = sum(counts.values())
    pattern = proportional_pattern(counts)
    cap, _, _ = capacity(pf, [replace(templates[t]) for t in pattern], models, lat, slo, horizon_s)
    return {"gpus": math.ceil(total / cap) if cap else math.inf, "per_gpu": cap}


def proportional_pattern(counts: dict, length: int = 20) -> list:
    """Task sequence whose every prefix tracks the fleet's proportions."""
    total = sum(counts.values())
    seen = {t: 0 for t in counts}
    out = []
    for k in range(1, length + 1):
        t = max(counts, key=lambda t: counts[t] / total * k - seen[t])
        seen[t] += 1
        out.append(t)
    return out


def cheapest_fleet(counts: dict, templates: dict, models, gpu_types: dict, slo=SLO(),
                   horizon_s=10.0) -> dict:
    """Cheapest mix of GPU types for a fleet (mixed placement on every GPU).

    gpu_types: name -> (LatencyModel, cost per hour). Each type's per-GPU
    capacity is simulated on the fleet's task mix; the bulk goes on the type
    is searched exhaustively over single-type plans and two-type mixes (a few
    small cheap GPUs can beat one mostly idle big one for the remainder).
    """
    total = sum(counts.values())
    pattern = [replace(templates[t]) for t in proportional_pattern(counts)]
    pf = lambda: ChronosPolicy(models=models, max_age=slo.s_max)
    cap = {name: capacity(pf, pattern, models, lat, slo, horizon_s)[0] for name, (lat, _) in gpu_types.items()}
    usable = {n: c for n, c in cap.items() if c > 0}
    if not usable:
        return {"capacity": cap, "plan": None, "cost": math.inf}
    cost = {n: gpu_types[n][1] for n in usable}
    # exhaustive over "n_a GPUs of type a, the rest on the cheapest cover by b";
    # covers every single-type plan and every two-type mix (types are few)
    best, plan = math.inf, None
    for a in usable:
        for na in range(math.ceil(total / usable[a]) + 1):
            rem = total - na * usable[a]
            for b in usable:
                nb = max(0, math.ceil(rem / usable[b]))
                c = na * cost[a] + nb * cost[b]
                if c < best - 1e-12:
                    best, plan = c, {k: v for k, v in ((a, na), (b, nb)) if v}
                    if a == b:
                        plan = {a: na + nb}
    return {"capacity": cap, "plan": plan, "cost": best,
            "single_type_cost": {n: math.ceil(total / usable[n]) * cost[n] for n in usable}}
