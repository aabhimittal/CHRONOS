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
