"""Run the scheduler on a real clock and report its own overhead.

    python scripts/runtime_bench.py            # sleep-based executors, ~20 s
    python scripts/runtime_bench.py --numpy    # VLM = real numpy layers (CPU compute)
"""
import argparse
import json
import pathlib

import numpy as np

from chronos.admission import make_fleet
from chronos.hazard import HazardModel
from chronos.runtime import LayerSliced, RealtimeSimulator
from chronos.sim import ChronosPolicy, LatencyModel, Robot, Simulator

ap = argparse.ArgumentParser()
ap.add_argument("--numpy", action="store_true")
ap.add_argument("--horizon", type=float, default=4.0)
args = ap.parse_args()

models = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path("curves").glob("*.json")}
lat = LatencyModel()
tpl = [Robot(t) for t in sorted(models)]
kw = {}
if args.numpy:
    W = np.random.default_rng(0).standard_normal((256, 256)) / 16
    kw["vlm_factory"] = lambda b: LayerSliced([lambda x: np.tanh(x @ W)] * 60, np.ones((b * 8, 256)))
rows = []
for n in (8, 32, 64):
    sim = RealtimeSimulator(make_fleet(tpl, n), lat, ChronosPolicy(models=models), **kw)
    r = sim.run(args.horizon)
    ref = Simulator(make_fleet(tpl, n), lat, ChronosPolicy(models=models)).run(args.horizon)
    rows.append({"n": n, "miss_rate": r["miss_rate"], "sim_miss_rate": ref["miss_rate"],
                 "mean_staleness": r["mean_staleness"], "sim_mean_staleness": ref["mean_staleness"],
                 **{k: r[k] for k in ("decision_ms_p50", "decision_ms_p99", "decision_ms_max", "slice_overrun_ms_max")}})
    print({k: round(float(v), 4) for k, v in rows[-1].items()}, flush=True)
pathlib.Path("docs/runtime.json").write_text(json.dumps(rows, indent=1, default=float))
