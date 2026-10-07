"""Online admission control with the closed-form oracle.

Robots ask to join over time (Poisson arrivals, exponential lifetimes). The
controller answers each request in microseconds with oracle.admissible() for
the fleet's *current* task mix; rejected robots go elsewhere (another GPU).
The admitted schedule is then simulated to check the SLO actually held.
"""
from __future__ import annotations

import numpy as np

from .oracle import admissible
from .sim import Robot


def arrivals(rate, mean_life, horizon_s, tasks, seed=0):
    """[(t_arrive, t_leave, task)] for a Poisson stream of join requests."""
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while True:
        t += rng.exponential(1 / rate)
        if t >= horizon_s:
            return out
        out.append((t, t + rng.exponential(mean_life), tasks[rng.integers(len(tasks))]))


def admit(requests, models, lat, slo=None, policy="oracle", calibration=1.0):
    """Decide each request in arrival order. policy: "oracle" or "all"."""
    admitted, live = [], []
    for t, leave, task in requests:
        live = [r for r in live if r[1] > t]                     # robots that already left
        counts = {}
        for _, _, tk in live + [(t, leave, task)]:
            counts[tk] = counts.get(tk, 0) + 1
        if policy == "all" or admissible(counts, models, lat, slo, calibration=calibration):
            live.append((t, leave, task))
            admitted.append((t, leave, task))
    return admitted


def fleet(admitted, seed=0, hz=10.0):
    rng = np.random.default_rng(seed)
    return [Robot(task, hz=hz, phase=float(rng.uniform(0, 1 / hz)), start=t, stop=leave)
            for t, leave, task in admitted]
