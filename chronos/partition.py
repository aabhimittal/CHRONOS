"""Spatial GPU partitioning (MPS / MIG) instead of time slicing.

The action head gets a fraction `fa` of the GPU and the VLM gets the rest;
both run concurrently, so no VLM slicing is needed and the action path is
never blocked. The price: each side runs on a smaller slice. Latencies scale
as act(B) / fa and vlm(B) / (1 - fa), times `interference` (>= 1) for shared
memory bandwidth and L2. Linear scaling with SM share is optimistic for small
MIG slices; profile real partitions before trusting a crossover point.
"""
from __future__ import annotations

import math
from dataclasses import replace

from .sim import ChronosPolicy, Simulator


class PartitionedSimulator(Simulator):
    def __init__(self, robots, latency, policy: ChronosPolicy, fa=0.3, interference=1.1, **kw):
        s_a, s_v = interference / fa, interference / (1 - fa)
        lat = replace(latency, act_fixed=latency.act_fixed * s_a, act_per=latency.act_per * s_a,
                      vlm_fixed=latency.vlm_fixed * s_v, vlm_per=latency.vlm_per * s_v)
        super().__init__(robots, lat, policy, **kw)
        self.fa = fa

    def run(self, horizon_s: float, record: bool = False):
        self.record = record
        pol = self.policy
        t, act_free, vlm_done = 0.0, 0.0, math.inf
        while t < horizon_s:
            if self.stalls and self.stalls[0][0] <= t:          # a stall freezes both partitions
                s0, dur = self.stalls.pop(0)
                end = max(t, s0 + dur)
                act_free, vlm_done = max(act_free, end), (vlm_done + end - t if math.isfinite(vlm_done) else vlm_done)
                t = end
                continue
            self._release(t)
            if vlm_done <= t + 1e-12:                           # VLM partition finished a batch
                rids, t0s, _ = self.vlm_batch
                for i, t0 in zip(rids, t0s):
                    self.plan_t0[i] = max(self.plan_t0[i], t0)
                self.in_flight[rids] = False
                self.vlm_batch, vlm_done = None, math.inf
            act = self.active(t)
            if self.vlm_batch is None:                          # VLM partition: back-to-back batches
                idle = [i for i in act if not self.in_flight[i]]
                if idle:
                    self.start_vlm(pol._select(self, t, idle), t)
                    vlm_done = t + self.vlm_batch[2]
                    if record:
                        self.trace.append(("vlm", t, vlm_done, len(self.vlm_batch[0])))
            wake = math.inf
            if act_free <= t + 1e-12 and self.pending:          # action partition: lazy EDF
                d = min(d for d, _ in self.pending)
                worst = self.lat.act(len(act)) + self.lat.net_base + self.lat.net_jitter
                if d - t - worst - pol.guard > 1e-9:          # tolerance: avoid a 1e-17 'wait'
                    wake = d - worst - pol.guard
                else:
                    self.pending.sort()
                    jobs, self.pending = self.pending[: pol.max_batch], self.pending[pol.max_batch:]
                    act_free = self.run_actions(t, jobs)
            cands = [x for x in (act_free, vlm_done, wake, self.next_event()) if x > t + 1e-12]
            t = min(cands) if cands else horizon_s
        return self.metrics()
