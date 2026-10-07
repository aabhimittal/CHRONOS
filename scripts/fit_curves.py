"""Fit p(success | staleness) per task from staleness-injection rollouts.

Default: toy ConveyorReach tasks (pipeline test only). For LIBERO/SimplerEnv,
implement chronos.rollout.Env + DualSystemPolicy for your model and pass them
to `sweep` the same way.
"""
import argparse
import pathlib

import numpy as np

from chronos.hazard import fit_hazard
from chronos.rollout import ConveyorReach, OraclePlanner, sweep

TOY_TASKS = {"shelf": 0.05, "conveyor": 0.15}   # target speed, units/s

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--out", default="curves")
args = ap.parse_args()

out = pathlib.Path(args.out)
out.mkdir(exist_ok=True)
probe = [0, .1, .2, .3, .5, 1, 2, 3]
print("task      " + " ".join(f"s={s:<4}" for s in probe))
for task, speed in TOY_TASKS.items():
    rolls = sweep(ConveyorReach(speed), OraclePlanner(), [1, 2, 4, 8, 16, 32], [0, 1, 3], args.episodes)
    m = fit_hazard([r.staleness for r in rolls], [r.success for r in rolls], task=task)
    (out / f"{task}.json").write_text(m.to_json())
    print(f"{task:<9} " + " ".join(f"{p:.3f} " for p in m.p_success(probe)))
