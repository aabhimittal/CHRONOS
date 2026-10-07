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
    act_steps: int = 16             # nominal denoising steps of the action head
    # extra per-control-step hazard when the head runs with fewer steps.
    # ASSUMED placeholder, not fitted: measure success vs K for your head.
    denoise_hazard: dict = field(default_factory=lambda: {16: 0.0, 8: 2e-4, 4: 1e-3, 2: 5e-3})

    def vlm(self, b): return self.vlm_fixed + self.vlm_per * b

    def act(self, b, k=None):
        """Action-head batch latency; cost scales with denoising steps k."""
        return (self.act_fixed + self.act_per * b) * ((k or self.act_steps) / self.act_steps)

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
    phase_sigma: float = 0.0  # lognormal jitter on phase durations (stochastic work cycles)
    seed: int = 0
    _sched: list = field(default=None, repr=False, compare=False)   # realised (start, end, idx)

    @property
    def period(self): return self.chunk / self.hz

    def _realised(self, t):
        """Realised phase schedule (random durations) extended to cover t."""
        if self._sched is None:
            self._sched, self._rng = [], np.random.default_rng(self.seed)
            self._sched.append((-self.cycle_offset, -self.cycle_offset + self._draw(0), 0))
        while self._sched[-1][1] <= t:
            _, end, idx = self._sched[-1]
            nxt = (idx + 1) % len(self.phases)
            self._sched.append((end, end + self._draw(nxt), nxt))
        return self._sched

    def _draw(self, idx):
        return self.phases[idx][1] * float(np.exp(self._rng.normal(0, self.phase_sigma)))

    def phase_state(self, t):
        """(phase index, time already spent in it): what a robot can report."""
        if not self.phase_sigma:
            cyc = sum(d for _, d in self.phases)
            x = (t + self.cycle_offset) % cyc
            for idx, (_, d) in enumerate(self.phases):
                if x < d:
                    return idx, x
                x -= d
            return len(self.phases) - 1, self.phases[-1][1]
        sched = self._realised(t)
        lo = max(0, len(sched) - 64)
        for s0, s1, idx in reversed(sched[lo:]):
            if s0 <= t < s1:
                return idx, t - s0
        return sched[0][2], 0.0

    def predict_task(self, t: float, x: float) -> str:
        """Task expected x seconds from now, from the current phase, time in
        it, and *nominal* durations (no knowledge of the realised schedule)."""
        if not self.phases:
            return self.task
        idx, e = self.phase_state(t)
        rem = max(self.phases[idx][1] - e, 0.0)
        while x >= rem:
            x -= rem
            idx = (idx + 1) % len(self.phases)
            rem = self.phases[idx][1]
        return self.phases[idx][0]

    def task_at(self, t: float) -> str:
        if not self.phases:
            return self.task
        if self.phase_sigma:
            return self.phases[self.phase_state(t)[0]][0]
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
    extra: dict = field(default_factory=dict)   # task -> summed extra hazard (reduced denoising)
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
    def run_actions(self, t, jobs, k=None):
        return self.finish_actions(t, t + self.lat.act(len(jobs), k), jobs, k)

    def finish_actions(self, t, t_end, jobs, k=None):
        """Bookkeeping for an action batch that ran on [t, t_end] (simulated
        or real: chronos.runtime calls this with wall-clock times)."""
        dh = self.lat.denoise_hazard.get(k, 0.0) if k else 0.0
        for d, i in jobs:
            r, st = self.robots[i], self.stats[i]
            arrive = t_end + self.net()
            task = r.task_at(arrive)
            st.add(task, arrive + np.arange(r.chunk) / r.hz - self.plan_t0[i])
            if dh:
                st.extra[task] = st.extra.get(task, 0.0) + dh * r.chunk
            st.late += arrive > d + 1e-12
        if self.record:
            self.trace.append(("act", t, t_end, len(jobs)))
        return t_end

    def start_vlm(self, rids, t):
        t0s = [t - self.net() for _ in rids]
        self.vlm_batch = [list(rids), t0s, self.lat.vlm(len(rids))]
        self.in_flight[list(rids)] = True

    def run_vlm_slice(self, t, quantum):
        rem = self.vlm_batch[2]
        work = min(rem, quantum)
        t_end = t + work + (self.lat.slice_overhead if math.isfinite(quantum) else 0.0)
        return self.finish_slice(t, t_end, rem - work)

    def finish_slice(self, t, t_end, remaining):
        """Bookkeeping for a VLM slice on [t, t_end] leaving `remaining` work;
        the batch's plans land when remaining reaches zero."""
        rids, t0s, _ = self.vlm_batch
        self.vlm_batch[2] = remaining
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
        """Per-robot predicted success: the product over the tasks (phases) a
        robot visits of that task's episode success under its staleness
        histogram (HazardModel.episode_success). Held (dropped) steps carry
        `h_miss` under ruin models (assumed, not fitted)."""
        out = []
        for st in self.stats:
            p = 1.0
            for task in set(st.hist) | set(st.held):
                m = models[task]
                h = st.hist.get(task, np.zeros(len(HIST_EDGES)))
                p *= m.episode_success((m.hazard(HIST_EDGES) * h).sum(), h.sum(),
                                       st.held.get(task, 0), h_miss)
                if st.extra.get(task):                 # reduced-quality actions (elastic head)
                    p *= math.exp(-m.horizon * st.extra[task] / max(h.sum(), 1))
            out.append(p)
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
      True uses the realised schedule (an oracle); "predict" uses only the
      robot's current phase, time in it, and nominal durations.
    """

    def __init__(self, quantum=0.010, max_batch=64, guard=0.002, models=None,
                 max_age=1.0, min_batch=4, lookahead=True, index="greedy", elastic=False):
        self.q, self.max_batch, self.guard, self.models = quantum, max_batch, guard, models
        self.max_age, self.min_batch, self.lookahead = max_age, min_batch, lookahead
        self.index = index            # "greedy" (one-step gain) or "whittle"
        self.elastic = elastic        # cut denoising steps instead of missing a deadline

    def _steps(self, sim, t, jobs):
        """Elastic head: the largest denoising-step count that still meets the
        earliest deadline in the batch (None = nominal)."""
        if not self.elastic:
            return None
        d = jobs[0][0] - sim.lat.net_base - sim.lat.net_jitter
        for k in sorted(sim.lat.denoise_hazard, reverse=True):
            if t + sim.lat.act(len(jobs), k) <= d:
                return None if k == sim.lat.act_steps else k
        return min(sim.lat.denoise_hazard)

    def _whittle(self, sim, t, i, lat):
        """Whittle index of refreshing robot i now (restless-bandit view).

        Refreshing every tau seconds with landing latency L gives average cost
        J(tau) = (int_L^{L+tau} c(u) du + lambda) / tau. The refresh price at
        which tau = a (current plan age) is optimal is
            W(a) = a c(a+L) - int_L^{a+L} c(u) du      (>= 0 for nondecreasing c),
        the continuous analogue of the age-of-information Whittle index. Cost
        c is weighted by the task's exposure (horizon) so tasks are comparable."""
        r, a = sim.robots[i], t - sim.plan_t0[i]
        m = self.models[r.task_at(t + lat if self.lookahead else t)]
        u = np.linspace(lat, a + lat, 32)
        c = m.cost(u) - m.cost(lat)                     # shift so c(L) = 0; W is shift-invariant
        return float(m.horizon * (a * c[-1] - np.trapezoid(c, u)))

    def _gain(self, sim, t, i, lat):
        """Hazard reduction from refreshing now. With lookahead, the new plan
        is used from t+L until the following one lands (~t+2L), so take the
        worst task over that window."""
        return float(self._gains(sim, t, [i], lat)[0])

    def _gains(self, sim, t, idx, lat):
        """Vectorised _gain over robots `idx`: robots are grouped by the task
        they will be in, and each group's curve is evaluated once."""
        idx = np.asarray(idx)
        age = t - sim.plan_t0[idx]
        best = np.full(len(idx), -np.inf)
        # "predict" also scores the current phase (k=0): a phase that overruns
        # its nominal length must not be predicted away, so prediction can only
        # add urgency. Measured: worth it while phase durations are predictable
        # (lognormal sigma <= ~0.3); at sigma 0.6 current-phase-only does better.
        ks = (0, 1, 2) if self.lookahead == "predict" else (1, 2) if self.lookahead else (0,)
        for k in ks:
            if self.lookahead == "predict" and k > 0:
                tasks = np.array([sim.robots[i].predict_task(t, k * lat) for i in idx])
            else:
                tasks = np.array([sim.robots[i].task_at(t + k * lat) for i in idx])
            kk = max(k, 1)
            for task in set(tasks.tolist()):
                sel = tasks == task
                m = self.models[task]
                best[sel] = np.maximum(best[sel], m.cost(age[sel] + kk * lat) - m.cost(kk * lat))
        return best

    def quantum(self, sim):
        """Slice budget the laxity test must reserve: the nominal quantum, or
        the executor's preemption granularity if coarser (real runtimes can
        only yield between layers)."""
        return max(self.q, getattr(sim, "slice_granularity", 0.0))

    def _select(self, sim, t, idle):
        cap = min(self.max_batch, sim.lat.max_vlm_batch)
        idle.sort(key=lambda i: sim.plan_t0[i])                     # stalest first
        if not self.models:
            return idle[:cap]
        lat = sim.lat.vlm(min(len(idle), cap))
        if self.index == "whittle":
            gain = {i: self._whittle(sim, t, i, lat) for i in idle}
        else:
            gain = dict(zip(idle, self._gains(sim, t, idle, lat).tolist()))
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
            if sim.vlm_batch and laxity >= self.quantum(sim) + sim.lat.slice_overhead:
                return sim.run_vlm_slice(t, self.q)
            if not sim.vlm_batch and laxity > 1e-9:          # tolerance: a 1e-17 'wait' would skip the job
                return min(sim.next_event(), d - worst_case - self.guard)
            sim.pending.sort()
            jobs, sim.pending = sim.pending[: self.max_batch], sim.pending[self.max_batch:]
            return sim.run_actions(t, jobs, self._steps(sim, t, jobs))
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
