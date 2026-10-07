"""Discrete-event simulator: one GPU serving a fleet of dual-system robots.

GPU model: a single non-preemptive server. Each launch is either
  * an action batch: the fast head for B robots, cost act(B), or
  * a VLM slice: up to `quantum` seconds of an in-flight VLM refresh batch.
A VLM batch of cost vlm(B) is split into slices (chunked prefill, as in
Sarathi-Serve) so it never blocks the action path for longer than `quantum`.

Robots release one action job per period = chunk / hz with an implicit
deadline at the next release. A job whose deadline passes before launch is
dropped (robot holds its last command); one that finishes after it is late.
Both count as misses.

Plan age ("staleness") at a control step = execution time - capture time of
the observation the active plan was computed from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HIST_EDGES = np.arange(0, 10.0 + 1e-9, 0.01)   # staleness histogram, seconds


@dataclass
class LatencyModel:
    """Batch latency = fixed + per_item * B. Defaults are illustrative
    (roughly a 2-3B VLM + small DiT head on one datacenter GPU); replace with
    profiled numbers before quoting results."""
    vlm_fixed: float = 0.040
    vlm_per: float = 0.006
    act_fixed: float = 0.004
    act_per: float = 0.0003
    slice_overhead: float = 0.001   # per VLM slice: relaunch + KV bookkeeping

    def vlm(self, b): return self.vlm_fixed + self.vlm_per * b
    def act(self, b): return self.act_fixed + self.act_per * b


@dataclass
class Robot:
    task: str
    hz: float = 10.0          # control rate
    chunk: int = 1            # control steps per action job
    phase: float = 0.0        # release offset (s)
    vlm_period: float | None = None   # push-mode refresh period (FIFO baseline)

    @property
    def period(self): return self.chunk / self.hz


@dataclass
class RobotStats:
    hist: np.ndarray = field(default_factory=lambda: np.zeros(len(HIST_EDGES)))
    jobs: int = 0
    late: int = 0
    dropped: int = 0
    held_steps: int = 0


class Simulator:
    def __init__(self, robots, latency: LatencyModel, policy):
        self.robots, self.lat, self.policy = robots, latency, policy
        n = len(robots)
        self.next_rel = np.array([r.phase for r in robots], float)
        self.next_vlm = np.array([r.phase if r.vlm_period else math.inf for r in robots])
        self.plan_t0 = np.zeros(n)                # capture time of active plan
        self.in_flight = np.zeros(n, bool)
        self.pending = []                         # action jobs: (deadline, rid)
        self.vlm_queue = []                       # push-mode requests: (t0, rid)
        self.vlm_batch = None                     # [rids, t0s, remaining_work]
        self.stats = [RobotStats() for _ in robots]

    # --- event plumbing -------------------------------------------------
    def _release(self, t):
        for i, r in enumerate(self.robots):
            while self.next_rel[i] <= t + 1e-12:
                self.pending.append((self.next_rel[i] + r.period, i))
                self.stats[i].jobs += 1
                self.next_rel[i] += r.period
            while self.next_vlm[i] <= t + 1e-12:
                self.vlm_queue.append((self.next_vlm[i], i))
                self.next_vlm[i] += r.vlm_period
        keep = []
        for d, i in self.pending:
            if d <= t:
                self.stats[i].dropped += 1
                self.stats[i].held_steps += self.robots[i].chunk
            else:
                keep.append((d, i))
        self.pending = keep

    def next_event(self):
        return min(self.next_rel.min(), self.next_vlm.min())

    def run_actions(self, t, jobs):
        dur = self.lat.act(len(jobs))
        t_end = t + dur
        for d, i in jobs:
            r, st = self.robots[i], self.stats[i]
            ages = t_end + np.arange(r.chunk) / r.hz - self.plan_t0[i]
            st.hist += np.bincount(np.minimum((ages / 0.01).astype(int), len(HIST_EDGES) - 1),
                                   minlength=len(HIST_EDGES))
            st.late += t_end > d + 1e-12
        return t_end

    def start_vlm(self, rids, t0s):
        self.vlm_batch = [list(rids), list(t0s), self.lat.vlm(len(rids))]
        self.in_flight[list(rids)] = True

    def run_vlm_slice(self, t, quantum):
        rids, t0s, rem = self.vlm_batch
        work = min(rem, quantum)
        t_end = t + work + (self.lat.slice_overhead if math.isfinite(quantum) else 0.0)
        self.vlm_batch[2] = rem - work
        if self.vlm_batch[2] <= 1e-12:
            for i, t0 in zip(rids, t0s):
                self.plan_t0[i] = max(self.plan_t0[i], t0)
            self.in_flight[rids] = False
            self.vlm_batch = None
        return t_end

    # --- main loop ------------------------------------------------------
    def run(self, horizon_s: float):
        t = 0.0
        while t < horizon_s:
            self._release(t)
            t_next = self.policy.step(self, t)
            t = t_next if t_next > t else max(self.next_event(), t + 1e-6)
        return self.metrics(horizon_s)

    def metrics(self, horizon_s):
        jobs = sum(s.jobs for s in self.stats)
        misses = sum(s.late + s.dropped for s in self.stats)
        hist = sum(s.hist for s in self.stats)
        return {"miss_rate": misses / max(jobs, 1),
                "mean_staleness": float((hist * HIST_EDGES).sum() / max(hist.sum(), 1)),
                "p99_staleness": float(HIST_EDGES[np.searchsorted(np.cumsum(hist), 0.99 * hist.sum())]),
                "jobs": jobs}

    def success(self, models: dict, h_miss: float = 0.05) -> np.ndarray:
        """Per-robot predicted episode success under the fitted hazard models."""
        out = []
        for r, st in zip(self.robots, self.stats):
            m = models[r.task]
            steps = st.hist.sum() + st.held_steps
            hz = (m.hazard(HIST_EDGES) * st.hist).sum() + h_miss * st.held_steps
            out.append(math.exp(-m.horizon * hz / max(steps, 1)))
        return np.array(out)


class ChronosPolicy:
    """Deadline-aware scheduler.

    * Action path: EDF with lazy batching. Pending jobs are deferred while the
      earliest deadline still has laxity, so more robots join the batch.
    * VLM path: pull-mode. Frames are captured at batch launch (not at request
      time), stalest robots first (or highest marginal hazard when models are
      given), and run in slices that fit inside the action laxity.
    * Curve-aware membership (models given): a robot joins a VLM batch only if
      refreshing it lowers its hazard, i.e. h(age + L) > h(L), or its age
      exceeds `max_age`. Robots in a flat region of their curve are skipped,
      so batches shrink and sensitive robots are refreshed more often.
    """

    def __init__(self, quantum=0.010, max_batch=64, guard=0.002, models=None,
                 max_age=1.0, min_batch=4):
        self.q, self.max_batch, self.guard, self.models = quantum, max_batch, guard, models
        self.max_age, self.min_batch = max_age, min_batch

    def _gain(self, sim, t, i, lat):
        m = self.models[sim.robots[i].task]
        return float(m.hazard(t - sim.plan_t0[i] + lat) - m.hazard(lat))

    def _select(self, sim, t, idle):
        idle.sort(key=lambda i: sim.plan_t0[i])                     # stalest first
        if not self.models:
            return idle[: self.max_batch]
        lat = sim.lat.vlm(min(len(idle), self.max_batch))
        need = [i for i in idle if self._gain(sim, t, i, lat) > 1e-9
                or t - sim.plan_t0[i] >= self.max_age]
        need.sort(key=lambda i: -self._gain(sim, t, i, lat))
        return (need or idle[: self.min_batch])[: self.max_batch]

    def step(self, sim, t):
        if sim.vlm_batch is None:
            idle = [i for i in range(len(sim.robots)) if not sim.in_flight[i]]
            if idle:
                pick = self._select(sim, t, idle)
                sim.start_vlm(pick, [t] * len(pick))
        worst_case = sim.lat.act(len(sim.robots))
        if sim.pending:
            d = min(d for d, _ in sim.pending)
            laxity = d - t - worst_case - self.guard
            if sim.vlm_batch and laxity >= self.q + sim.lat.slice_overhead:
                return sim.run_vlm_slice(t, self.q)
            if not sim.vlm_batch and laxity > 0:
                return min(sim.next_event(), d - worst_case - self.guard)
            jobs = sorted(sim.pending)[: self.max_batch]
            sim.pending = sorted(sim.pending)[self.max_batch:]
            return sim.run_actions(t, jobs)
        if sim.vlm_batch:
            q = min(self.q, max(sim.next_event() - t, 1e-4))
            return sim.run_vlm_slice(t, q)
        return sim.next_event()


class FifoPolicy:
    """LLM-serving-style baseline: FIFO across request types, greedy batching of
    the head-of-line type, VLM batches run to completion (no slicing), frames
    captured when the robot submits (push mode, needs Robot.vlm_period)."""

    def __init__(self, max_batch=64):
        self.max_batch = max_batch

    def step(self, sim, t):
        heads = []
        if sim.pending:
            heads.append((min(d - sim.robots[i].period for d, i in sim.pending), "act"))
        if sim.vlm_queue:
            heads.append((sim.vlm_queue[0][0], "vlm"))
        if not heads:
            return sim.next_event()
        if min(heads)[1] == "act":
            jobs = sorted(sim.pending)[: self.max_batch]
            sim.pending = sorted(sim.pending)[self.max_batch:]
            return sim.run_actions(t, jobs)
        batch, sim.vlm_queue = sim.vlm_queue[: self.max_batch], sim.vlm_queue[self.max_batch:]
        sim.start_vlm([i for _, i in batch], [t0 for t0, _ in batch])
        return sim.run_vlm_slice(t, math.inf)
