"""Staleness -> success model.

Each control step executed with plan age s "ruins" the episode with
probability h(s); an episode succeeds iff no step ruins it:

    P(success | trace) = exp(-sum_t h(s_t))

h is piecewise-constant over staleness bins and constrained nondecreasing
(h_b = sum_{j<=b} delta_j, delta >= 0). With bin counts n_i per episode the
log-survival is linear in h, so the Bernoulli NLL is convex in delta: the fit
has a unique optimum and needs no tuning.

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
    horizon: float      # effective episode length in steps, for p(success | s)
    task: str = ""

    def hazard(self, s):
        idx = np.searchsorted(self.edges, np.asarray(s, float), side="right") - 1
        return self.h[np.clip(idx, 0, len(self.h) - 1)]

    def p_success(self, s):
        """Success probability when every step runs at constant staleness s."""
        return np.exp(-self.horizon * self.hazard(s))

    def mean_step_hazard(self, s_values, weights):
        w = np.asarray(weights, float)
        return float((self.hazard(s_values) * w).sum() / max(w.sum(), 1e-12))

    def to_json(self) -> str:
        return json.dumps({"task": self.task, "edges": [float(e) for e in self.edges],
                           "h": self.h.tolist(), "horizon": self.horizon})

    @classmethod
    def from_json(cls, text: str) -> "HazardModel":
        d = json.loads(text)
        return cls(np.array(d["edges"], float), np.array(d["h"], float), d["horizon"], d["task"])


def bin_counts(trace, edges) -> np.ndarray:
    idx = np.searchsorted(edges, np.asarray(trace, float), side="right") - 1
    return np.bincount(np.clip(idx, 0, len(edges) - 2), minlength=len(edges) - 1)


def fit_hazard(traces, successes, edges=DEFAULT_EDGES, task="", ridge=1e-6) -> HazardModel:
    """MLE of a monotone piecewise-constant hazard from rollout traces.

    traces: list of per-step staleness arrays (seconds); successes: bools.
    """
    edges = np.asarray(edges, float)
    N = np.stack([bin_counts(t, edges) for t in traces]).astype(float)   # (E, B)
    y = np.asarray(successes, float)
    B = N.shape[1]
    L = np.tril(np.ones((B, B)))                                          # h = L @ delta

    def nll(delta):
        x = np.maximum(N @ (L @ delta), 1e-10)                            # (E,)
        f = (y * x - (1 - y) * np.log(-np.expm1(-x))).sum() + ridge * delta @ delta
        g_x = y - (1 - y) / np.expm1(x)
        return f, L.T @ (N.T @ g_x) + 2 * ridge * delta

    res = minimize(nll, np.full(B, 1e-3), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * B)
    return HazardModel(edges, L @ res.x, float(N.sum(1).mean()), task)
