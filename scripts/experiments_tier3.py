"""Regenerate the Tier 3 results (writes docs/tier3.json). ~4 min.

Phase prediction vs oracle lookahead, online admission, fitted h_miss,
cheapest GPU mix. (Runtime overhead: scripts/runtime_bench.py.)
"""
import json
import pathlib
from dataclasses import replace

import numpy as np

from chronos.admission import capacity
from chronos.hazard import HazardModel, fit_hazard
from chronos.online import admit, arrivals, fleet
from chronos.oracle import fast_capacity
from chronos.placement import cheapest_fleet
from chronos.rollout import ConveyorReach, OraclePlanner, sweep
from chronos.sim import ChronosPolicy, LatencyModel, Robot, Simulator

lat = LatencyModel()
models = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path("curves").glob("*.json")}
tasks = sorted(models)
R = {}

R["phase_prediction"] = []
for sig in (0.0, 0.3, 0.6):
    ph = [Robot("shelf", phases=(("shelf", 4.0), ("conveyor", 2.0)), phase_sigma=sig)]
    row = {"sigma": sig}
    for name, la in (("oracle lookahead", True), ("predicted (robust)", "predict"), ("current phase only", False)):
        row[name] = capacity(lambda: ChronosPolicy(models=models, lookahead=la), ph, models, horizon_s=24, n_start=20)[0]
    R["phase_prediction"].append(row)
    print(row, flush=True)

tpl = [Robot(t) for t in tasks]
cal = capacity(lambda: ChronosPolicy(models=models), tpl, models, n_start=30)[0] / fast_capacity(tpl, models)
R["online"] = {"calibration": cal, "rows": []}
for rate in (2.0, 3.0):
    req = arrivals(rate, 30.0, 60.0, tasks, seed=1)
    for pol, c in (("admit all", 1.0), ("oracle", 1.0), ("oracle, calibrated", cal)):
        adm = admit(req, models, lat, policy="all" if pol == "admit all" else "oracle", calibration=c)
        sim = Simulator(fleet(adm), lat, ChronosPolicy(models=models))
        r = sim.run(60.0)
        s = sim.success(models)
        live = [sum(1 for a in adm if a[0] <= t < a[1]) for t in np.arange(0, 60, 0.5)]
        R["online"]["rows"].append({"offered": rate * 30, "policy": pol, "admitted": len(adm), "requests": len(req),
                                    "peak_live": max(live), "miss_rate": float(r["miss_rate"]),
                                    "success": float(s.mean()), "success_p5": float(np.quantile(s, 0.05))})
        print(R["online"]["rows"][-1], flush=True)

R["h_miss"] = {}
for task, sp in (("shelf", 0.05), ("conveyor", 0.15)):
    r = sweep(ConveyorReach(sp), OraclePlanner(), [1, 4, 16], [0, 2, 5], 30, drop_grid=(0.0, 0.2, 0.5))
    R["h_miss"][task] = fit_hazard([x.staleness for x in r], [x.success for x in r], held=[x.held for x in r]).h_miss
print("h_miss", R["h_miss"], flush=True)

s = lambda f: replace(lat, vlm_fixed=lat.vlm_fixed * f, vlm_per=lat.vlm_per * f, act_fixed=lat.act_fixed * f, act_per=lat.act_per * f)
types = {"big (1.0x latency, $1.00/h)": (lat, 1.0), "mid (1.8x, $0.45/h)": (s(1.8), 0.45), "small (3.5x, $0.25/h)": (s(3.5), 0.25)}
R["gpu_mix"] = cheapest_fleet({"shelf": 300, "conveyor": 100}, {t: Robot(t) for t in tasks}, models, types)
print("gpu_mix", R["gpu_mix"], flush=True)
pathlib.Path("docs/tier3.json").write_text(json.dumps(R, indent=1, default=float))
