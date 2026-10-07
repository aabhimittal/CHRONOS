"""Bootstrap confidence intervals on robots-per-GPU.

Capacity is a step function of the fitted curve, so finite rollouts can move
it by many robots. This refits every curve on stratified bootstrap resamples
of the saved rollouts and recomputes capacity for each scheduler.

    python scripts/capacity_ci.py --boot 20       # ~5 min; writes docs/ci.json
"""
import argparse
import json
import pathlib

import numpy as np

from chronos.admission import SLO, capacity
from chronos.hazard import bootstrap_fits, load_rollouts
from chronos.oracle import fast_capacity
from chronos.sim import ChronosPolicy, FifoPolicy, Robot

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=20)
ap.add_argument("--curves", default="curves")
ap.add_argument("--out", default="docs/ci.json")
args = ap.parse_args()

fits = {}
for p in sorted(pathlib.Path(args.curves).glob("rollouts_*.npz")):
    task = p.stem.removeprefix("rollouts_")
    fits[task] = bootstrap_fits(*load_rollouts(p), n_boot=args.boot, task=task)
tasks = sorted(fits)
tpl = lambda **kw: [Robot(t, **kw) for t in tasks]
slo = SLO()

res = {"CHRONOS (curve-aware)": [], "CHRONOS (curve-blind)": [], "Oracle (closed form)": [],
       "FIFO batching (0.5 s)": []}
for b in range(args.boot):
    models = {t: fits[t][b] for t in tasks}
    est = fast_capacity(tpl(), models, slo=slo)
    seed = max(1, est // 2)
    res["Oracle (closed form)"].append(est)
    res["CHRONOS (curve-aware)"].append(capacity(lambda: ChronosPolicy(models=models, max_age=slo.s_max),
                                                 tpl(), models, slo=slo, n_start=seed)[0])
    res["CHRONOS (curve-blind)"].append(capacity(ChronosPolicy, tpl(), models, slo=slo, n_start=seed)[0])
    res["FIFO batching (0.5 s)"].append(capacity(FifoPolicy, tpl(vlm_period=0.5), models, slo=slo)[0])
    print(b, {k: v[-1] for k, v in res.items()}, flush=True)

summary = {k: {"median": float(np.median(v)), "lo": float(np.percentile(v, 5)),
               "hi": float(np.percentile(v, 95)), "samples": v} for k, v in res.items()}
pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(f"{k:<26} {v['median']:.0f}  [{v['lo']:.0f}, {v['hi']:.0f}]  (90% bootstrap interval)")
