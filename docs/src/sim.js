// Browser port of chronos/sim.py (ChronosPolicy, FifoPolicy, GPU stalls).
// Kept line-for-line close to the Python; tests/test_demo_parity.py runs both
// on identical fleets and checks the metrics agree.
(function (root) {
  const NB = 1001, BIN = 0.01;             // HIST_EDGES = 0, 0.01, ..., 10.0

  function hazardModel(m) {
    const edges = m.edges.map(e => (e === null || e >= 1e8 ? Infinity : e));
    const hazard = s => {
      let lo = 0, hi = edges.length;         // searchsorted(side="right") - 1
      while (lo < hi) { const mid = (lo + hi) >> 1; if (edges[mid] <= s) lo = mid + 1; else hi = mid; }
      return m.h[Math.min(Math.max(lo - 1, 0), m.h.length - 1)];
    };
    const hEdges = Float64Array.from({ length: NB }, (_, k) => hazard(k * BIN));
    const kind = m.kind || "ruin";
    const fromRate = r => (kind === "ruin" ? Math.exp(-m.horizon * r) : -Math.expm1(-m.horizon * r));
    const cost = s => (kind === "ruin" ? hazard(s) : -hazard(s));
    return { hazard, cost, hEdges, kind, fromRate, horizon: m.horizon, p: s => fromRate(hazard(s)) };
  }

  const Lat = (o = {}) => Object.assign({
    vlm_fixed: 0.040, vlm_per: 0.006, act_fixed: 0.004, act_per: 0.0003,
    slice_overhead: 0.001, max_vlm_batch: 64,
    vlm(b) { return this.vlm_fixed + this.vlm_per * b; },
    act(b) { return this.act_fixed + this.act_per * b; },
  }, o);

  class Sim {
    constructor(robots, lat, policy, stalls = []) {
      Object.assign(this, { robots, lat, policy });
      this.stalls = stalls.slice().sort((a, b) => a[0] - b[0]);
      this.nextRel = robots.map(r => r.phase);
      this.nextVlm = robots.map(r => (r.vlm_period ? r.phase : Infinity));
      this.planT0 = robots.map(() => 0);
      this.inFlight = robots.map(() => false);
      this.pending = []; this.vlmQueue = []; this.vlmBatch = null; this.trace = [];
      this.stats = robots.map(() => ({ hist: new Float64Array(NB), held: 0, jobs: 0, late: 0, dropped: 0 }));
    }
    period(i) { const r = this.robots[i]; return r.chunk / r.hz; }
    nextEvent() {
      let m = Infinity;
      for (let i = 0; i < this.robots.length; i++) m = Math.min(m, this.nextRel[i], this.nextVlm[i]);
      return m;
    }
    release(t) {
      for (let i = 0; i < this.robots.length; i++) {
        const P = this.period(i);
        while (this.nextRel[i] <= t + 1e-12) {
          this.pending.push([this.nextRel[i] + P, i]); this.stats[i].jobs++; this.nextRel[i] += P;
        }
        while (this.nextVlm[i] <= t + 1e-12) {
          this.vlmQueue.push([this.nextVlm[i], i]); this.nextVlm[i] += this.robots[i].vlm_period;
        }
      }
      const keep = [];
      for (const [d, i] of this.pending) {
        if (d <= t) { this.stats[i].dropped++; this.stats[i].held += this.robots[i].chunk; }
        else keep.push([d, i]);
      }
      this.pending = keep;
    }
    sortPending() { this.pending.sort((a, b) => a[0] - b[0] || a[1] - b[1]); }
    runActions(t, jobs) {
      const tEnd = t + this.lat.act(jobs.length);
      for (const [d, i] of jobs) {
        const r = this.robots[i], st = this.stats[i];
        for (let k = 0; k < r.chunk; k++) {
          const age = tEnd + k / r.hz - this.planT0[i];
          st.hist[Math.min(Math.trunc(age / BIN), NB - 1)]++;
        }
        if (tEnd > d + 1e-12) st.late++;
      }
      this.trace.push(["act", t, tEnd, jobs.length]);
      return tEnd;
    }
    startVlm(rids, t0s) {
      this.vlmBatch = [rids, t0s, this.lat.vlm(rids.length)];
      for (const i of rids) this.inFlight[i] = true;
    }
    runVlmSlice(t, q) {
      const [rids, t0s, rem] = this.vlmBatch, work = Math.min(rem, q);
      const tEnd = t + work + (isFinite(q) ? this.lat.slice_overhead : 0);
      this.vlmBatch[2] = rem - work;
      this.trace.push(["vlm", t, tEnd, rids.length]);
      if (this.vlmBatch[2] <= 1e-12) {
        rids.forEach((i, k) => { this.planT0[i] = Math.max(this.planT0[i], t0s[k]); this.inFlight[i] = false; });
        this.vlmBatch = null;
      }
      return tEnd;
    }
    run(horizon) {
      let t = 0;
      while (t < horizon) {
        if (this.stalls.length && this.stalls[0][0] <= t) {
          const [s0, dur] = this.stalls.shift();
          this.trace.push(["stall", t, Math.max(t, s0 + dur), 0]);
          t = Math.max(t, s0 + dur); continue;
        }
        this.release(t);
        const tn = this.policy.step(this, t);
        t = tn > t ? tn : Math.max(this.nextEvent(), t + 1e-6);
      }
      return this.metrics();
    }
    metrics() {
      let jobs = 0, miss = 0, tot = 0, sum = 0;
      const hist = new Float64Array(NB);
      for (const s of this.stats) {
        jobs += s.jobs; miss += s.late + s.dropped;
        for (let k = 0; k < NB; k++) hist[k] += s.hist[k];
      }
      for (let k = 0; k < NB; k++) { tot += hist[k]; sum += hist[k] * k * BIN; }
      let c = 0, p99 = 0;
      for (let k = 0; k < NB; k++) { c += hist[k]; if (c >= 0.99 * tot) { p99 = k * BIN; break; } }
      return { miss_rate: miss / Math.max(jobs, 1), mean_staleness: sum / Math.max(tot, 1), p99_staleness: p99, jobs, hist };
    }
    success(models, hMiss = 0.05) {
      return this.robots.map((r, i) => {
        const m = models[r.task], st = this.stats[i];
        let steps = st.held, hz = m.kind === "ruin" ? hMiss * st.held : 0;
        for (let k = 0; k < NB; k++) { steps += st.hist[k]; hz += m.hEdges[k] * st.hist[k]; }
        return steps ? m.fromRate(hz / steps) : 1;
      });
    }
  }

  class ChronosPolicy {
    constructor(o = {}) {
      Object.assign(this, { q: 0.010, maxBatch: 64, guard: 0.002, models: null, maxAge: 1.0, minBatch: 4 }, o);
    }
    gain(sim, t, i, L) {
      const m = this.models[sim.robots[i].task], age = t - sim.planT0[i];
      let g = -Infinity;
      for (const k of [1, 2]) g = Math.max(g, m.cost(age + k * L) - m.cost(k * L));
      return g;
    }
    select(sim, t, idle) {
      const cap = Math.min(this.maxBatch, sim.lat.max_vlm_batch);
      idle.sort((a, b) => sim.planT0[a] - sim.planT0[b] || a - b);
      if (!this.models) return idle.slice(0, cap);
      const L = sim.lat.vlm(Math.min(idle.length, cap)), g = {};
      for (const i of idle) g[i] = this.gain(sim, t, i, L);
      const need = idle.filter(i => g[i] > 1e-9 || t - sim.planT0[i] >= this.maxAge);
      need.sort((a, b) => g[b] - g[a]);
      return (need.length ? need : idle.slice(0, this.minBatch)).slice(0, cap);
    }
    step(sim, t) {
      const n = sim.robots.length;
      if (sim.vlmBatch === null) {
        const idle = [];
        for (let i = 0; i < n; i++) if (!sim.inFlight[i]) idle.push(i);
        if (idle.length) { const pick = this.select(sim, t, idle); sim.startVlm(pick, pick.map(() => t)); }
      }
      const worst = sim.lat.act(n);
      if (sim.pending.length) {
        let d = Infinity;
        for (const [dd] of sim.pending) d = Math.min(d, dd);
        const lax = d - t - worst - this.guard;
        if (sim.vlmBatch && lax >= this.q + sim.lat.slice_overhead) return sim.runVlmSlice(t, this.q);
        if (!sim.vlmBatch && lax > 1e-9) return Math.min(sim.nextEvent(), d - worst - this.guard);
        sim.sortPending();
        const jobs = sim.pending.slice(0, this.maxBatch);
        sim.pending = sim.pending.slice(this.maxBatch);
        return sim.runActions(t, jobs);
      }
      if (sim.vlmBatch) return sim.runVlmSlice(t, Math.min(this.q, Math.max(sim.nextEvent() - t, 1e-4)));
      return sim.nextEvent();
    }
  }

  class FifoPolicy {
    constructor(o = {}) { this.maxBatch = o.maxBatch || 64; }
    step(sim, t) {
      const heads = [];
      if (sim.pending.length) {
        let r = Infinity;
        for (const [d, i] of sim.pending) r = Math.min(r, d - sim.period(i));
        heads.push([r, "act"]);
      }
      if (sim.vlmQueue.length) heads.push([sim.vlmQueue[0][0], "vlm"]);
      if (!heads.length) return sim.nextEvent();
      heads.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : 1));
      if (heads[0][1] === "act") {
        sim.sortPending();
        const jobs = sim.pending.slice(0, this.maxBatch);
        sim.pending = sim.pending.slice(this.maxBatch);
        return sim.runActions(t, jobs);
      }
      const cap = Math.min(this.maxBatch, sim.lat.max_vlm_batch);
      const batch = sim.vlmQueue.slice(0, cap);
      sim.vlmQueue = sim.vlmQueue.slice(cap);
      sim.startVlm(batch.map(b => b[1]), batch.map(b => b[0]));
      return sim.runVlmSlice(t, Infinity);
    }
  }

  const api = { Sim, ChronosPolicy, FifoPolicy, hazardModel, Lat };
  root.Chronos = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
