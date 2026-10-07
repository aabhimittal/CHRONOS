"""Analyse real VLA staleness rollouts (writes docs/vla_results.json).

For each task: success per refresh/latency setting, the fitted curve and its
held-out error, and the staleness effect *with the starting scene held fixed*
(logistic regression: one intercept per LIBERO init state + a slope on mean
plan age). Scene difficulty dominates small samples, so this paired estimate
is the honest test of whether staleness matters.

    python scripts/vla/analyze_vla.py curves_vla/log_libero_spatial_0.jsonl [more logs ...]
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from chronos.hazard import load_rollouts, select_kind


def age_effect(rows):
    states = sorted({r["ep"] for r in rows})
    E = np.array([[r["ep"] == s for s in states] for r in rows], float)
    age = np.array([np.mean(r["staleness"]) for r in rows])
    y = np.array([r["success"] for r in rows], float)
    p = len(states) + 1

    def nll(w):
        z = E @ w[:-1] + w[-1] * age
        return np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * w[:-1] @ w[:-1]   # ridge: some scenes always succeed

    w = minimize(nll, np.zeros(p), method="BFGS").x
    h, H = 1e-4, np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            ei, ej = np.eye(p)[i] * h, np.eye(p)[j] * h
            H[i, j] = (nll(w + ei + ej) - nll(w + ei - ej) - nll(w - ei + ej) + nll(w - ei - ej)) / (4 * h * h)
    se = float(np.sqrt(np.linalg.inv(H)[-1, -1]))
    b = float(w[-1])
    return {"log_odds_per_s": b, "se": se, "ci95": [b - 1.96 * se, b + 1.96 * se],
            "p": float(2 * (1 - norm.cdf(abs(b / se)))), "odds_ratio_per_s": float(np.exp(b)),
            "episodes_for_p05": int(np.ceil(len(rows) * (2.8 * se / max(abs(b), 1e-9)) ** 2))}


out = {}
for log in sys.argv[1:]:
    log = pathlib.Path(log)
    name = log.stem.removeprefix("log_")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    tab = {}
    for r in rows:
        tab.setdefault((r["k"], r["L"]), []).append(r["success"])
    tr, y, g = load_rollouts(log.parent / f"rollouts_{name}.npz")
    m, cv = select_kind(tr, y, g, task=name)
    (log.parent / f"{name}.json").write_text(m.to_json())
    out[name] = {
        "settings": [{"refresh_s": k / 20, "latency_s": L / 20, "successes": int(sum(v)), "n": len(v)}
                     for (k, L), v in sorted(tab.items())],
        "by_init_state": {str(s): f"{sum(r['success'] for r in rows if r['ep'] == s)}/"
                                  f"{sum(1 for r in rows if r['ep'] == s)}" for s in sorted({r['ep'] for r in rows})},
        "kind": m.kind,
        "heldout_mae": {k: float(np.mean([abs(x["predicted"] - x["observed"]) for x in v])) for k, v in cv.items()},
        "p_success": {str(s): float(m.p_success(s)) for s in (0, 0.5, 1, 1.5, 2, 3)},
        "age_effect_scene_fixed": age_effect(rows),
    }
    e = out[name]["age_effect_scene_fixed"]
    print(f"{name}: {len(rows)} episodes; odds x{e['odds_ratio_per_s']:.2f} per +1 s plan age "
          f"(95% CI log-odds {e['ci95'][0]:.2f}..{e['ci95'][1]:.2f}, p={e['p']:.2f}); "
          f"~{e['episodes_for_p05']} episodes needed for p<0.05; held-out MAE {out[name]['heldout_mae'][m.kind]:.3f}")
pathlib.Path("docs/vla_results.json").write_text(json.dumps(out, indent=1))
