"""Items/hour per GPU with a fixed conveyor speed vs a speed governor.

    python scripts/speed_governor.py        # ~3 min; writes docs/speed.json
"""
import json
import pathlib

import numpy as np

from chronos.hazard import HazardModel, select_kind
from chronos.rollout import ConveyorReach, OraclePlanner, sweep
from chronos.sim import Robot
from chronos.speed import evaluate_speeds, governed

BASE = 0.15                       # nominal conveyor speed (units/s), as in fit_curves.py
LEVELS = (0.5, 0.75, 1.0)         # speed multipliers the facility may choose
out = pathlib.Path("curves/speed")
out.mkdir(parents=True, exist_ok=True)
family = {"conveyor": {}}
for v in LEVELS:
    f = out / f"conveyor_v{v:.2f}.json"
    if not f.exists():
        r = sweep(ConveyorReach(BASE * v), OraclePlanner(), [1, 2, 4, 8, 16, 32], [0, 1, 2, 3, 5], 40)
        m, _ = select_kind([x.staleness for x in r], [x.success for x in r],
                           [(x.refresh_steps, x.latency_steps) for x in r], task="conveyor")
        f.write_text(m.to_json())
    family["conveyor"][v] = HazardModel.from_json(f.read_text())

base = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path("curves").glob("*.json")}
tpl = [Robot(t) for t in sorted(base)]
rows = []
for n in range(20, 121, 10):
    r = evaluate_speeds(tpl, family, base, n)
    g, fixed = governed(r), next(x for x in r if x["v"] == 1.0)
    rows.append({"n": n, "fixed": fixed["throughput"] if fixed["feasible"] else None,
                 "governed": g["throughput"] if g else None, "v": g["v"] if g else None})
    print(rows[-1], flush=True)
best_fixed = max((r["fixed"] or 0) for r in rows)
best_gov = max((r["governed"] or 0) for r in rows)
print(f"items/hour per GPU (relative units): fixed speed {best_fixed:.1f}, governed {best_gov:.1f} "
      f"({best_gov / best_fixed - 1:+.0%})")
pathlib.Path("docs/speed.json").write_text(json.dumps({"rows": rows, "best_fixed": best_fixed,
                                                       "best_governed": best_gov}, indent=1))
