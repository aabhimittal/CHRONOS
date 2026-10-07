"""Event-triggered refresh: refresh when the scene changes, not when the plan is old.

Model. Each robot's scene changes at Poisson times (rate `rate`, per s). A
plan is *valid* while no change has happened since its frame was captured.
Steps run with hazard h_valid on a valid plan and h_invalid on an invalid one.
Averaging over change times gives back an ordinary age curve,

    h(s) = h_valid + (h_invalid - h_valid) * (1 - exp(-rate * s)),

so age-based scheduling with `expected_curve()` is the fair baseline.

Detector. The robot runs a cheap change detector (frame differencing,
action-head uncertainty): it reports a real change after `delay` s with
probability `p_detect`, plus false alarms at `false_rate` per s. With
trigger="event" the scheduler refreshes flagged robots (and any older than
max_age); everything else keeps its plan. All parameters are synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hazard import DEFAULT_EDGES, HazardModel
from .sim import ChronosPolicy, Simulator


@dataclass
class SceneModel:
    rate: float = 0.3          # scene changes per second
    h_valid: float = 1e-4      # per-step hazard on a valid plan
    h_invalid: float = 2e-2    # per-step hazard on an invalidated plan
    delay: float = 0.05        # detector latency (s)
    p_detect: float = 0.9      # detector recall
    false_rate: float = 0.1    # false alarms per second
    horizon: float = 60.0      # exposure steps (as HazardModel.horizon)

    def expected_curve(self, task="scene") -> HazardModel:
        mid = np.minimum((DEFAULT_EDGES[:-1] + np.minimum(DEFAULT_EDGES[1:], 6.0)) / 2, 6.0)
        h = self.h_valid + (self.h_invalid - self.h_valid) * (1 - np.exp(-self.rate * mid))
        return HazardModel(DEFAULT_EDGES, h, self.horizon, task)


class EventSimulator(Simulator):
    def __init__(self, robots, latency, policy, scene: SceneModel, horizon_s: float, seed=0, **kw):
        super().__init__(robots, latency, policy, seed=seed, **kw)
        self.scene = scene
        rng = np.random.default_rng(seed + 1)
        n_ev = lambda r: rng.poisson(r * horizon_s * 1.2 + 1)
        self.events, self.flags = [], []
        for _ in robots:
            ev = np.sort(rng.uniform(0, horizon_s * 1.2, n_ev(scene.rate)))
            det = ev[rng.random(len(ev)) < scene.p_detect] + scene.delay
            fa = rng.uniform(0, horizon_s * 1.2, n_ev(scene.false_rate))
            self.events.append(ev)
            self.flags.append(np.sort(np.concatenate([det, fa])))
        self.valid = np.zeros(len(robots))
        self.invalid = np.zeros(len(robots))

    def flagged(self, i, t):
        """A detector flag raised after this robot's plan was captured."""
        f = self.flags[i]
        k = np.searchsorted(f, t, side="right")
        return k > 0 and f[k - 1] > self.plan_t0[i]

    def run_actions(self, t, jobs, k=None):
        t_end = super().run_actions(t, jobs, k)
        for _, i in jobs:
            r = self.robots[i]
            times = t_end + np.arange(r.chunk) / r.hz
            # plan invalid at time x if some change happened in (plan_t0, x]
            ev = self.events[i]
            last = np.searchsorted(ev, times, side="right")
            first_after = np.searchsorted(ev, self.plan_t0[i], side="right")
            bad = last > first_after
            self.invalid[i] += bad.sum()
            self.valid[i] += (~bad).sum()
        return t_end

    def scene_success(self) -> np.ndarray:
        s = self.scene
        steps = np.maximum(self.valid + self.invalid, 1)
        mean_h = (s.h_valid * self.valid + s.h_invalid * self.invalid) / steps
        return np.exp(-s.horizon * mean_h)


class EventPolicy(ChronosPolicy):
    """ChronosPolicy whose VLM batches are the flagged robots (plus any older
    than max_age), so refreshes go where the scene actually changed."""

    def _select(self, sim, t, idle):
        cap = min(self.max_batch, sim.lat.max_vlm_batch)
        need = [i for i in idle if sim.flagged(i, t) or t - sim.plan_t0[i] >= self.max_age]
        need.sort(key=lambda i: sim.plan_t0[i])
        return (need or sorted(idle, key=lambda i: sim.plan_t0[i])[: self.min_batch])[:cap]
