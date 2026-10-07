"""Regenerate the Tier 2 results (writes docs/tier2.json). ~5 min.

Whittle vs greedy refresh, elastic action head, spatial partitioning,
event-triggered refresh. The speed governor has its own script.
"""
import functools
import json
import pathlib

import numpy as np

from chronos.admission import capacity, make_fleet
from chronos.events import EventPolicy, EventSimulator, SceneModel
from chronos.hazard import DEFAULT_EDGES as E, HazardModel
from chronos.partition import PartitionedSimulator
from chronos.sim import ChronosPolicy, LatencyModel, Robot, Simulator

lat = LatencyModel()
models = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path("curves").glob("*.json")}
mix = [Robot(t) for t in sorted(models)]
R = {}

# Whittle vs greedy: toy fleet, and a fleet mixing a gradual and a cliff-shaped curve
mid = (E[:-1] + np.minimum(E[1:], 6)) / 2
shapes = {"grad": HazardModel(E, 4e-3 * mid, 60), "cliff": HazardModel(E, np.where(E[:-1] < 0.75, 1e-4, 3e-2), 60)}
R["whittle"] = {}
for name, tpl, m in (("toy mix", mix, models), ("gradual + cliff", [Robot("grad"), Robot("cliff")], shapes)):
    R["whittle"][name] = {idx: capacity(lambda: ChronosPolicy(models=m, index=idx), tpl, m, horizon_s=12, n_start=8)[0]
                          for idx in ("greedy", "whittle")}
print("whittle", R["whittle"], flush=True)

# elastic action head at 30 Hz past the action-path limit
R["elastic"] = []
for n in (80, 100, 120, 160):
    row = {"n": n, "act_ms": lat.act(n) * 1e3}
    for el in (False, True):
        sim = Simulator(make_fleet([Robot(t, hz=30) for t in sorted(models)], n), lat,
                        ChronosPolicy(models=models, elastic=el))
        r = sim.run(6)
        row["elastic" if el else "fixed"] = {"miss_rate": float(r["miss_rate"]), "success": float(sim.success(models).mean())}
    R["elastic"].append(row)
print("elastic", R["elastic"], flush=True)

# spatial partitioning vs time slicing
pf = lambda: ChronosPolicy(models=models)
R["partition"] = {"time-sliced": capacity(pf, mix, models, horizon_s=12, n_start=25)[0]}
for intf in (1.0, 1.15):
    for fa in (0.2, 0.3, 0.4):
        S = functools.partial(PartitionedSimulator, fa=fa, interference=intf)
        R["partition"][f"fa={fa} interference={intf}"] = capacity(pf, mix, models, horizon_s=12, n_start=5, sim_cls=S)[0]
print("partition", R["partition"], flush=True)

# event-triggered vs age-based refresh
R["events"] = []
for rate in (0.1, 0.3, 1.0):
    for n in (32, 64, 96):
        sc = SceneModel(rate=rate)
        row = {"rate": rate, "n": n}
        for name, pol in (("age (curve-blind)", ChronosPolicy()),
                          ("age (curve-aware)", ChronosPolicy(models={"scene": sc.expected_curve()}, max_age=2.0)),
                          ("event-triggered", EventPolicy(max_age=2.0))):
            s = EventSimulator(make_fleet([Robot("scene")], n), lat, pol, sc, 10.0)
            s.run(10.0)
            row[name] = float(s.scene_success().mean())
        R["events"].append(row)
print("events", R["events"], flush=True)
pathlib.Path("docs/tier2.json").write_text(json.dumps(R, indent=1))
