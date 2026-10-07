"""Real-time runtime: the same scheduler driving real work on a wall clock.

`RealtimeSimulator` reuses the simulator's bookkeeping but executes every
launch for real and reads time from `time.perf_counter()`:

  * action batches call `act_fn(batch_size, denoise_steps)`;
  * VLM batches are `LayerSliced` executors: a sequential model run layer by
    layer that yields when its slice budget is spent. This is real
    preemption at layer granularity (a layer can overrun the budget; the
    overrun is measured), for any network expressible as a list of layers.

What it measures that the simulator cannot: the scheduler's own decision
overhead (pure Python) against the control period, timer jitter, and slice
overruns. Default executors sleep for the modeled latency; pass real
callables (e.g. torch modules followed by torch.cuda.synchronize()) to run
models. GPU streams/MPS are not used: one launch at a time, as in the sim.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .sim import Simulator


def precise_sleep(dt: float):
    """Sleep dt seconds with sub-millisecond accuracy (sleep, then spin)."""
    end = time.perf_counter() + dt
    if dt > 0.002:
        time.sleep(dt - 0.001)
    while time.perf_counter() < end:
        pass


class LayerSliced:
    """Run `layers` (callables) on `x` one at a time under time budgets."""

    def __init__(self, layers, x=None):
        self.layers, self.x, self.i = list(layers), x, 0

    @property
    def done(self):
        return self.i >= len(self.layers)

    def step(self, budget: float) -> bool:
        t0 = time.perf_counter()
        while not self.done:
            self.x = self.layers[self.i](self.x)
            self.i += 1
            if time.perf_counter() - t0 >= budget:
                break
        return self.done


class RealtimeSimulator(Simulator):
    def __init__(self, robots, latency, policy, act_fn=None, vlm_factory=None, n_layers=24, **kw):
        super().__init__(robots, latency, policy, **kw)
        self.act_fn = act_fn or (lambda b, k=None: precise_sleep(latency.act(b, k)))
        self.vlm_factory = vlm_factory or (
            lambda b: LayerSliced([lambda x, d=latency.vlm(b) / n_layers: precise_sleep(d)] * n_layers))
        self.overhead, self.overrun = [], []

    def net(self):
        return 0.0                       # real mode: no synthetic network delay

    slice_granularity = 0.0     # measured longest single layer; the policy reserves this much

    def start_vlm(self, rids, t):
        super().start_vlm(rids, t)
        self.vlm_exec = self.vlm_factory(len(rids))

    def run_vlm_slice(self, t, quantum):
        t0 = self._clock()
        done = self.vlm_exec.step(quantum if math.isfinite(quantum) else math.inf)
        t1 = self._clock()
        self._exec += t1 - t0
        if math.isfinite(quantum):
            self.overrun.append(max(0.0, (t1 - t0) - quantum))
        # granularity = longest single layer seen so far (one slice runs >= 1 layer)
        layers = max(self.vlm_exec.i - getattr(self, "_last_i", 0), 1)
        self.slice_granularity = max(self.slice_granularity, (t1 - t0) / layers)
        self._last_i = 0 if done else self.vlm_exec.i
        remaining = 0.0 if done else max(self.vlm_batch[2] - (t1 - t0), 1e-6)
        return self.finish_slice(t, t1, remaining)

    def run_actions(self, t, jobs, k=None):
        t0 = self._clock()
        self.act_fn(len(jobs), k)
        t1 = self._clock()
        self._exec += t1 - t0
        return self.finish_actions(t, t1, jobs, k)

    def run(self, horizon_s: float, record: bool = False):
        self.record = record
        start = time.perf_counter()
        self._clock = lambda: time.perf_counter() - start
        while (t := self._clock()) < horizon_s:
            self._release(t)
            self._exec, tic = 0.0, time.perf_counter()
            t_next = self.policy.step(self, t)
            self.overhead.append(time.perf_counter() - tic - self._exec)
            wait = min(t_next, horizon_s) - self._clock()
            if wait > 0:
                precise_sleep(wait)
        m = self.metrics()
        oh = np.array(self.overhead) * 1e3
        m.update({"decision_ms_p50": float(np.percentile(oh, 50)), "decision_ms_p99": float(np.percentile(oh, 99)),
                  "decision_ms_max": float(oh.max()), "decisions": len(oh),
                  "slice_overrun_ms_max": float(max(self.overrun, default=0.0) * 1e3)})
        return m
