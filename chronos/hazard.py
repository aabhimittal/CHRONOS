"""Staleness -> success model. Two forms, chosen per task by cross-validation.

ruin (e.g. carrying, placing): each step executed with plan age s ruins the
episode with probability h(s), h nondecreasing:

    P(success | trace) = exp(-sum_t h(s_t))

opportunity (e.g. catching a moving item): each step offers a chance c(s) to
succeed, c nonincreasing in s; the episode fails only if every chance is
missed:

    P(success | trace) = 1 - exp(-sum_t c(s_t))

They behave differently: under "ruin" the *sum* of staleness exposure
matters; under "opportunity" a few fresh-enough moments are all you need, so
a latency floor that never lets the plan get fresh enough is fatal. The toy
conveyor task is an opportunity task, and the ruin form mispredicts it badly
(scripts/validate_curves.py).

The rate is piecewise-constant over staleness bins and monotone via a
nonnegative increment parametrisation, so in both forms the Bernoulli NLL is
convex: the fit has a unique optimum and needs no tuning.

Because the model consumes per-step staleness *traces*, a curve fitted from
fixed-refresh rollouts can score any scheduler's staleness distribution.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

DEFAULT_EDGES = np.array([0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2, 3, 5, np.inf])


@dataclass
class HazardModel:
    edges: np.ndarray   # (B+1,) seconds; last edge may be inf
    h: np.ndarray       # (B,) per-step hazard, nondecreasing
    horizon: float      # exposure in steps used to turn a per-step rate into success
    task: str = ""
    kind: str = "ruin"  # "ruin" (h = failure hazard) or "opportunity" (h = success rate)

    def hazard(self, s):
        idx = np.searchsorted(self.edges, np.asarray(s, float), side="right") - 1
        return self.h[np.clip(idx, 0, len(self.h) - 1)]

    def from_rate(self, mean_rate):
        """Episode success from the mean per-step rate over the exposure."""
        x = self.horizon * np.asarray(mean_rate, float)
        return np.exp(-x) if self.kind == "ruin" else -np.expm1(-x)

    def p_success(self, s):
        """Success probability when every step runs at constant staleness s."""
        return self.from_rate(self.hazard(s))

    def cost(self, s):
        """Per-step badness, increasing in staleness under both forms; used by
        schedulers to rank refreshes (reduction in cost = benefit)."""
        r = self.hazard(s)
        return r if self.kind == "ruin" else -r

    def episode_success(self, rate_sum, steps, held=0, h_miss=0.05):
        """Score a simulated staleness histogram. Held (dropped) steps add
        hazard h_miss under ruin, and offer no chance under opportunity."""
        total = steps + held
        if total <= 0:
            return 1.0
        extra = h_miss * held if self.kind == "ruin" else 0.0
        return float(self.from_rate((rate_sum + extra) / total))

    def mean_step_hazard(self, s_values, weights):
        w = np.asarray(weights, float)
        return float((self.hazard(s_values) * w).sum() / max(w.sum(), 1e-12))

    def to_json(self) -> str:
        return json.dumps({"task": self.task, "kind": self.kind, "edges": [float(e) for e in self.edges],
                           "h": self.h.tolist(), "horizon": self.horizon})

    @classmethod
    def from_json(cls, text: str) -> "HazardModel":
        d = json.loads(text)
        return cls(np.array(d["edges"], float), np.array(d["h"], float), d["horizon"], d["task"],
                   d.get("kind", "ruin"))


def bin_counts(trace, edges) -> np.ndarray:
    idx = np.searchsorted(edges, np.asarray(trace, float), side="right") - 1
    return np.bincount(np.clip(idx, 0, len(edges) - 2), minlength=len(edges) - 1)


def fit_hazard(traces, successes, edges=DEFAULT_EDGES, task="", ridge=1e-6,
               kind="ruin") -> HazardModel:
    """MLE of a monotone piecewise-constant rate from rollout traces.

    traces: list of per-step staleness arrays (seconds); successes: bools.
    kind="ruin": y = success, rate nondecreasing (h = L @ delta, L lower-tri).
    kind="opportunity": the same likelihood with y = failure and the rate
    nonincreasing (L upper-triangular).
    """
    edges = np.asarray(edges, float)
    N = np.stack([bin_counts(t, edges) for t in traces]).astype(float)   # (E, B)
    y = np.asarray(successes, float)
    B = N.shape[1]
    if kind == "opportunity":
        y = 1 - y
        L = np.triu(np.ones((B, B)))                                      # c_b = sum_{j>=b} delta_j
    else:
        L = np.tril(np.ones((B, B)))                                      # h_b = sum_{j<=b} delta_j

    def nll(delta):
        x = np.maximum(N @ (L @ delta), 1e-10)                            # (E,)
        f = (y * x - (1 - y) * np.log(-np.expm1(-x))).sum() + ridge * delta @ delta
        g_x = y - (1 - y) / np.expm1(x)
        return f, L.T @ (N.T @ g_x) + 2 * ridge * delta

    res = minimize(nll, np.full(B, 1e-3), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * B)
    # exposure: ruin -> mean executed steps; opportunity -> the full episode
    # budget (chances keep coming until the timeout)
    horizon = float(N.sum(1).mean() if kind == "ruin" else N.sum(1).max())
    return HazardModel(edges, L @ res.x, horizon, task, kind)


# --- uncertainty -------------------------------------------------------------

def save_rollouts(path, rollouts):
    """Store ragged staleness traces compactly (npz)."""
    np.savez_compressed(path, lengths=np.array([len(r.staleness) for r in rollouts]),
                        staleness=np.concatenate([r.staleness for r in rollouts]),
                        success=np.array([r.success for r in rollouts]),
                        refresh=np.array([r.refresh_steps for r in rollouts]),
                        latency=np.array([r.latency_steps for r in rollouts]))


def load_rollouts(path):
    """Returns (traces, successes, groups) where groups = (refresh, latency) per episode."""
    d = np.load(path)
    traces = np.split(d["staleness"], np.cumsum(d["lengths"])[:-1])
    return traces, d["success"].astype(bool), list(zip(d["refresh"].tolist(), d["latency"].tolist()))


def bootstrap_fits(traces, successes, groups, n_boot=20, seed=0, **fit_kw):
    """Stratified bootstrap: resample episodes within each (refresh, latency)
    setting, refit. The spread of downstream quantities (capacity) across the
    returned models is the uncertainty that comes from finite rollouts."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(v, len(v)) for v in by.values()])
        out.append(fit_hazard([traces[i] for i in idx], [successes[i] for i in idx], **fit_kw))
    return out


def predict_setting(model: HazardModel, trace) -> float:
    """Success predicted for a staleness pattern, exposed over model.horizon
    steps (the same way the simulator scores a fleet). `trace` must be a full
    pattern for the setting; episode length is never used, so a held-out
    episode's outcome cannot leak into its own prediction."""
    s = np.resize(np.asarray(trace, float), max(int(round(model.horizon)), 1))
    return float(model.from_rate(model.mean_step_hazard(s, np.ones_like(s))))


def cross_validate(traces, successes, groups, **fit_kw) -> list[dict]:
    """Leave-one-setting-out: fit on all other (refresh, latency) settings,
    predict the held-out setting's success rate, compare with what happened."""
    y = np.asarray(successes, float)
    out = []
    for g in sorted(set(groups)):
        hold = np.array([gi == g for gi in groups])
        m = fit_hazard([t for t, h in zip(traces, hold) if not h], y[~hold], **fit_kw)
        full = max((t for t, h in zip(traces, hold) if h), key=len)      # longest = full pattern
        obs, n = y[hold].mean(), int(hold.sum())
        out.append({"setting": [int(v) for v in g], "n": n, "observed": float(obs),
                    "predicted": predict_setting(m, full), "se": float(np.sqrt(max(obs * (1 - obs), 1 / n) / n))})
    return out


def select_kind(traces, successes, groups, **fit_kw):
    """Fit both forms; keep the one with lower leave-one-setting-out error.
    Returns (model fitted on all data, {kind: cv rows})."""
    cv = {k: cross_validate(traces, successes, groups, kind=k, **fit_kw) for k in ("ruin", "opportunity")}
    err = {k: np.mean([abs(r["predicted"] - r["observed"]) for r in rows]) for k, rows in cv.items()}
    best = min(err, key=err.get)
    return fit_hazard(traces, successes, kind=best, **fit_kw), cv
