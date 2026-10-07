"""Fit p(success | staleness) per task from staleness-injection rollouts.

Default: toy ConveyorReach tasks (pipeline test only). For LIBERO/SimplerEnv,
implement chronos.rollout.Env + DualSystemPolicy for your model and pass them
to `sweep` the same way.
"""
import argparse
import pathlib

import numpy as np

from chronos.hazard import save_rollouts, select_kind
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
    # latency grid must cover the plan latencies the scheduler produces
    # (0.05-0.5 s here); an earlier {0, 1, 3}-step grid left the model
    # extrapolating exactly where capacity is decided
    rolls = sweep(ConveyorReach(speed), OraclePlanner(), [1, 2, 4, 8, 16, 32], [0, 1, 2, 3, 5], args.episodes)
    m, _ = select_kind([r.staleness for r in rolls], [r.success for r in rolls],
                       [(r.refresh_steps, r.latency_steps) for r in rolls], task=task)
    (out / f"{task}.json").write_text(m.to_json())
    save_rollouts(out / f"rollouts_{task}.npz", rolls)     # for bootstrap CIs / validation
    print(f"{task:<9} " + " ".join(f"{p:.3f} " for p in m.p_success(probe)) + f"  [{m.kind}]")
