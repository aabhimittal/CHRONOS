"""Discrete-event simulator: one GPU serving a fleet of dual-system robots.

GPU model: a single non-preemptive server. Each launch is either
  * an action batch: the fast head for B robots, cost act(B), or
  * a VLM slice: up to `quantum` seconds of an in-flight VLM refresh batch.
A VLM batch of cost vlm(B) is split into slices (chunked prefill, as in
Sarathi-Serve) so it never blocks the action path for longer than `quantum`.

Robots release one action job per period = chunk / hz with an implicit
deadline at the next release. A job whose deadline passes before launch is
dropped (robot holds its last command); one delivered after it is late. Both
count as misses. Actions reach the robot after a network delay.

Plan age ("staleness") at a control step = execution time - capture time of
the camera frame the active plan was computed from (frames are already
`net_base + U(0, net_jitter)` old when the server sees them).

Industrial perturbations: GPU stalls (`stalls`), robot churn (`Robot.start`/
`stop`), task phases (`Robot.phases`), mixed control rates, network jitter.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields

import numpy as np

HIST_EDGES = np.arange(0, 10.0 + 1e-9, 0.01)   # staleness histogram, seconds


@dataclass
class LatencyModel:
    """Batch latency = fixed + per_item * B. Defaults are illustrative
    (roughly a 2-3B VLM + small DiT head on one datacenter GPU); replace with
    `scripts/profile_latency.py` output before quoting results."""
    vlm_fixed: float = 0.040
    vlm_per: float = 0.006
    act_fixed: float = 0.004
    act_per: float = 0.0003
    slice_overhead: float = 0.001   # per VLM slice: relaunch + KV bookkeeping
    max_vlm_batch: int = 64         # memory cap: KV cache / activations
    net_base: float = 0.0           # one-way robot <-> server latency (s)
    net_jitter: float = 0.0         # extra uniform(0, jitter) per message

    def vlm(self, b): return self.vlm_fixed + self.vlm_per * b
    def act(self, b): return self.act_fixed + self.act_per * b

    @classmethod
    def from_json(cls, text: str) -> "LatencyModel":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in json.loads(text).items() if k in known})


@dataclass
class Robot:
    task: str
    hz: float = 10.0          # control rate
    chunk: int = 1            # control steps per action job
    phase: float = 0.0        # release offset (s)
    vlm_period: float | None = None   # push-mode refresh period (FIFO baseline)
    phases: tuple = ()        # ((task, seconds), ...) work cycle; overrides task
    cycle_offset: float = 0.0
    start: float = 0.0        # joins the fleet
    stop: float = math.inf    # leaves the fleet

    @property
    def period(self): return self.chunk / self.hz

    def task_at(self, t: float) -> str:
        if not self.phases:
            return self.task
        x = (t + self.cycle_offset) % sum(d for _, d in self.phases)
        for task, d in self.phases:
            if x < d:
                return task
            x -= d
        return self.phases[-1][0]

    def tasks(self):
        return {p for p, _ in self.phases} if self.phases else {self.task}


@dataclass
class RobotStats:
    hist: dict = field(default_factory=dict)    # task -> staleness histogram
    held: dict = field(default_factory=dict)    # task -> steps with no fresh action
    jobs: int = 0
    late: int = 0
    dropped: int = 0

    def add(self, task, ages):
        h = self.hist.setdefault(task, np.zeros(len(HIST_EDGES)))
        h += np.bincount(np.minimum((ages / 0.01).astype(int), len(HIST_EDGES) - 1),
                         minlength=len(HIST_EDGES))


class Simulator:
    def __init__(self, robots, latency: LatencyModel, policy, stalls=(), seed=0):
        self.robots, self.lat, self.policy = robots, latency, policy
        self.rng = np.random.default_rng(seed)
        self.stalls = sorted(stalls)               # [(t_start, duration)]
        n = len(robots)
        self.next_rel = np.array([r.start + r.phase for r in robots], float)
        self.next_vlm = np.array([r.start + r.phase if r.vlm_period else math.inf for r in robots])
        self.plan_t0 = np.array([r.start for r in robots], float)   # warm start
        self.in_flight = np.zeros(n, bool)
        self.pending = []                          # action jobs: (deadline, rid)
        self.vlm_queue = []                        # push-mode requests: (t0, rid)
        self.vlm_batch = None                      # [rids, t0s, remaining_work]
        self.stats = [RobotStats() for _ in robots]
        self.trace = []                            # (kind, t0, t1, batch) if record
        self.record = False
        self._all = list(range(n))
        self._static = all(r.start == 0 and r.stop == math.inf for r in robots)

    # --- helpers --------------------------------------------------------
    def net(self):
        j = self.lat.net_jitter
        return self.lat.net_base + (self.rng.uniform(0, j) if j > 0 else 0.0)

    def active(self, t):
        if self._static:
            return self._all
        return [i for i, r in enumerate(self.robots) if r.start <= t < r.stop]

    def next_event(self):
        if not self.robots:
            return math.inf
        return float(min(self.next_rel.min(), self.next_vlm.min()))

    def _release(self, t):
        due = np.flatnonzero((self.next_rel <= t + 1e-12) | (self.next_vlm <= t + 1e-12))
        for i in due:
            r = self.robots[i]
            while self.next_rel[i] <= t + 1e-12:
                if self.next_rel[i] >= r.stop:
                    self.next_rel[i] = math.inf
                    break
                self.pending.append((self.next_rel[i] + r.period, i))
                self.stats[i].jobs += 1
                self.next_rel[i] += r.period
            while self.next_vlm[i] <= t + 1e-12:
                if self.next_vlm[i] >= r.stop:
                    self.next_vlm[i] = math.inf
                    break
                self.vlm_queue.append((self.next_vlm[i] - self.net(), i))
                self.next_vlm[i] += r.vlm_period
        keep = []
        for d, i in self.pending:
            if d <= t:
                st, r = self.stats[i], self.robots[i]
                st.dropped += 1
                task = r.task_at(d)
                st.held[task] = st.held.get(task, 0) + r.chunk
            else:
                keep.append((d, i))
        self.pending = keep

    # --- GPU launches ---------------------------------------------------
    def run_actions(self, t, jobs):
        t_end = t + self.lat.act(len(jobs))
        for d, i in jobs:
            r, st = self.robots[i], self.stats[i]
            arrive = t_end + self.net()
            st.add(r.task_at(arrive), arrive + np.arange(r.chunk) / r.hz - self.plan_t0[i])
            st.late += arrive > d + 1e-12
        if self.record:
            self.trace.append(("act", t, t_end, len(jobs)))
        return t_end

    def start_vlm(self, rids, t):
        t0s = [t - self.net() for _ in rids]
        self.vlm_batch = [list(rids), t0s, self.lat.vlm(len(rids))]
        self.in_flight[list(rids)] = True

    def run_vlm_slice(self, t, quantum):
        rids, t0s, rem = self.vlm_batch
        work = min(rem, quantum)
        t_end = t + work + (self.lat.slice_overhead if math.isfinite(quantum) else 0.0)
        self.vlm_batch[2] = rem - work
        if self.record:
            self.trace.append(("vlm", t, t_end, len(rids)))
        if self.vlm_batch[2] <= 1e-12:
            for i, t0 in zip(rids, t0s):
                self.plan_t0[i] = max(self.plan_t0[i], t0)
            self.in_flight[rids] = False
            self.vlm_batch = None
        return t_end

    # --- main loop ------------------------------------------------------
    def run(self, horizon_s: float, record: bool = False):
        self.record = record
        t = 0.0
        while t < horizon_s:
            if self.stalls and self.stalls[0][0] <= t:
                s0, dur = self.stalls.pop(0)
                if record:
                    self.trace.append(("stall", t, max(t, s0 + dur), 0))
                t = max(t, s0 + dur)
                continue
            self._release(t)
            t_next = self.policy.step(self, t)
            t = t_next if t_next > t else max(self.next_event(), t + 1e-6)
        return self.metrics()

    def metrics(self):
        jobs = sum(s.jobs for s in self.stats)
        misses = sum(s.late + s.dropped for s in self.stats)
        hist = sum((h for s in self.stats for h in s.hist.values()), np.zeros(len(HIST_EDGES)))
        tot = max(hist.sum(), 1)
        return {"miss_rate": misses / max(jobs, 1),
                "mean_staleness": float((hist * HIST_EDGES).sum() / tot),
                "p99_staleness": float(HIST_EDGES[min(np.searchsorted(np.cumsum(hist), 0.99 * tot),
                                                      len(HIST_EDGES) - 1)]),
                "jobs": jobs}

    def success(self, models: dict, h_miss: float = 0.05) -> np.ndarray:
        """Per-robot predicted success. Each task (or phase) the robot visits
        contributes horizon_task * mean step hazard in that task; held steps
        (dropped jobs) carry hazard `h_miss` (assumed, not fitted)."""
        out = []
        for st in self.stats:
            total = 0.0
            for task in set(st.hist) | set(st.held):
                m = models[task]
                h = st.hist.get(task, np.zeros(len(HIST_EDGES)))
                held = st.held.get(task, 0)
                steps = h.sum() + held
                if steps:
                    total += m.horizon * ((m.hazard(HIST_EDGES) * h).sum() + h_miss * held) / steps
            out.append(math.exp(-total))
        return np.array(out)


class ChronosPolicy:
    """Deadline-aware scheduler.

    * Action path: EDF with lazy batching. Pending jobs are deferred while the
      earliest deadline still has laxity, so more robots join the batch.
    * VLM path: pull-mode. Frames are captured at batch launch (not at request
      time) and run in slices that fit inside the action laxity.
    * Curve-aware membership (models given): a robot joins a VLM batch only if
      refreshing it lowers its hazard, i.e. h(age + L) > h(L), or its age
      exceeds `max_age`. Robots in a flat region of their curve are skipped.
    * Phase lookahead (`lookahead=True`): the gain is evaluated under the task
      the robot will be in when the plan lands (t + L), so a robot about to
      enter a critical phase (e.g. grasp) is refreshed *before* it gets there.
    """

    def __init__(self, quantum=0.010, max_batch=64, guard=0.002, models=None,
                 max_age=1.0, min_batch=4, lookahead=True):
        self.q, self.max_batch, self.guard, self.models = quantum, max_batch, guard, models
        self.max_age, self.min_batch, self.lookahead = max_age, min_batch, lookahead

    def _gain(self, sim, t, i, lat):
        """Hazard reduction from refreshing now. With lookahead, the new plan
        is used from t+L until the following one lands (~t+2L), so take the
        worst task over that window."""
        r, age = sim.robots[i], t - sim.plan_t0[i]
        ks = (1, 2) if self.lookahead else (0,)
        return max(float(self.models[r.task_at(t + k * lat)].hazard(age + max(k, 1) * lat)
                         - self.models[r.task_at(t + k * lat)].hazard(max(k, 1) * lat)) for k in ks)

    def _select(self, sim, t, idle):
        cap = min(self.max_batch, sim.lat.max_vlm_batch)
        idle.sort(key=lambda i: sim.plan_t0[i])                     # stalest first
        if not self.models:
            return idle[:cap]
        lat = sim.lat.vlm(min(len(idle), cap))
        gain = {i: self._gain(sim, t, i, lat) for i in idle}
        need = [i for i in idle if gain[i] > 1e-9 or t - sim.plan_t0[i] >= self.max_age]
        need.sort(key=lambda i: -gain[i])
        return (need or idle[: self.min_batch])[:cap]

    def step(self, sim, t):
        act = sim.active(t)
        if sim.vlm_batch is None:
            idle = [i for i in act if not sim.in_flight[i]]
            if idle:
                sim.start_vlm(self._select(sim, t, idle), t)
        worst_case = sim.lat.act(len(act)) + sim.lat.net_base + sim.lat.net_jitter
        if sim.pending:
            d = min(d for d, _ in sim.pending)
            laxity = d - t - worst_case - self.guard
            if sim.vlm_batch and laxity >= self.q + sim.lat.slice_overhead:
                return sim.run_vlm_slice(t, self.q)
            if not sim.vlm_batch and laxity > 0:
                return min(sim.next_event(), d - worst_case - self.guard)
            sim.pending.sort()
            jobs, sim.pending = sim.pending[: self.max_batch], sim.pending[self.max_batch:]
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
            sim.pending.sort()
            jobs, sim.pending = sim.pending[: self.max_batch], sim.pending[self.max_batch:]
            return sim.run_actions(t, jobs)
        cap = min(self.max_batch, sim.lat.max_vlm_batch)
        batch, sim.vlm_queue = sim.vlm_queue[:cap], sim.vlm_queue[cap:]
        sim.vlm_batch = [[i for _, i in batch], [t0 for t0, _ in batch],
                         sim.lat.vlm(len(batch))]
        sim.in_flight[sim.vlm_batch[0]] = True
        return sim.run_vlm_slice(t, math.inf)
