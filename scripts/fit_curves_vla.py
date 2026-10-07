"""Fit staleness curves for a real VLA in LIBERO (run on a GPU machine).

    python scripts/fit_curves_vla.py --suite libero_10 --tasks 0,1,2 \\
        --policy mypkg.policies:make_pi0 --episodes 50 --out curves_vla/
    python scripts/fit_curves_vla.py --dry-run          # wiring check, no LIBERO/GPU

--policy names a zero-argument factory returning a chronos.vla.ChunkPolicy or
SplitPolicy. Refresh/latency grids are in control steps at LIBERO's 20 Hz;
the latency grid must cover the plan latencies your fleet will see, or the
fit extrapolates where capacity is decided (see validate_curves.py).
"""
import argparse
import importlib
import pathlib

import numpy as np

from chronos.hazard import cross_validate, save_rollouts, select_kind
from chronos.rollout import sweep

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--tasks", default="0")
ap.add_argument("--policy", help="module:factory")
ap.add_argument("--refresh", default="1,2,4,8,16,32")
ap.add_argument("--latency", default="0,2,4,6,10")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--out", default="curves_vla")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

R = [int(x) for x in args.refresh.split(",")]
L = [int(x) for x in args.latency.split(",")]
out = pathlib.Path(args.out)
out.mkdir(exist_ok=True)

if args.dry_run:
    from chronos.rollout import ConveyorReach
    from chronos.vla import ChunkPolicy
    make_env = lambda task_id: ConveyorReach(0.15)
    chunk = lambda obs: np.repeat((obs["target"] - obs["agent"])[None] / 8, 8, 0)
    make_policy = lambda: ChunkPolicy(chunk)
    R, L, args.episodes = [1, 4, 16], [0, 2], 10
else:
    from chronos.vla import LiberoEnv
    mod, fn = args.policy.split(":")
    make_policy = getattr(importlib.import_module(mod), fn)
    make_env = lambda task_id: LiberoEnv(args.suite, task_id)

policy = make_policy()
for task_id in [int(t) for t in args.tasks.split(",")]:
    env = make_env(task_id)
    name = f"{args.suite}_{task_id}" if not args.dry_run else f"dryrun_{task_id}"
    print(f"{name}: {len(R) * len(L) * args.episodes} episodes x <= {env.horizon} steps")
    rolls = sweep(env, policy, R, L, args.episodes)
    traces, ys = [r.staleness for r in rolls], [r.success for r in rolls]
    groups = [(r.refresh_steps, r.latency_steps) for r in rolls]
    m, cv = select_kind(traces, ys, groups, task=name)
    (out / f"{name}.json").write_text(m.to_json())
    save_rollouts(out / f"rollouts_{name}.npz", rolls)
    err = np.mean([abs(r["predicted"] - r["observed"]) for r in cv[m.kind]])
    print(f"  kind={m.kind}  held-out MAE={err:.3f}  success@fresh={m.p_success(0):.2f}  @1s={m.p_success(1.0):.2f}")
