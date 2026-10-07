"""How many rollouts (and GPU-hours) does a trustworthy curve need?

1. Uncertainty vs effort: subsample the saved rollouts to k episodes per
   setting, bootstrap, and report the 90% interval width of the oracle's
   capacity estimate (fast, curve-blind; a proxy for the simulator's).
2. Cost: settings x episodes x steps x seconds/step, per task.

    python scripts/rollout_budget.py --sec-per-step 0.12 --tasks 10 --steps 300
"""
import argparse
import pathlib

import numpy as np

from chronos.hazard import fit_hazard, load_rollouts
from chronos.oracle import fast_capacity
from chronos.sim import Robot

ap = argparse.ArgumentParser()
ap.add_argument("--sec-per-step", type=float, default=0.12,
                help="wall time per env step incl. policy inference (e.g. ~0.1 s for a 3B VLA on one GPU)")
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--steps", type=float, default=300, help="mean executed steps per episode")
ap.add_argument("--settings", type=int, default=30, help="refresh x latency grid size")
ap.add_argument("--boot", type=int, default=30)
args = ap.parse_args()

data = {p.stem.removeprefix("rollouts_"): load_rollouts(p) for p in sorted(pathlib.Path("curves").glob("rollouts_*.npz"))}
tasks = sorted(data)
rng = np.random.default_rng(0)
print("episodes/setting   capacity 90% interval (oracle, toy curves)   GPU-hours for your grid")
for k in (10, 20, 40, 60):
    caps = []
    for _ in range(args.boot):
        models = {}
        for t in tasks:
            tr, y, g = data[t]
            by = {}
            for i, gi in enumerate(g):
                by.setdefault(gi, []).append(i)
            idx = np.concatenate([rng.choice(v, k) for v in by.values()])
            models[t] = fit_hazard([tr[i] for i in idx], [y[i] for i in idx], kind="ruin")
        caps.append(fast_capacity([Robot(t) for t in tasks], models))
    lo, med, hi = np.percentile(caps, [5, 50, 95])
    hours = args.tasks * args.settings * k * args.steps * args.sec_per_step / 3600
    print(f"{k:>16}   {med:5.0f}  [{lo:.0f}, {hi:.0f}]  width {hi - lo:4.0f}{'':18}{hours:8.0f}")
