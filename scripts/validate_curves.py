"""Is the hazard model trustworthy? Leave-one-setting-out calibration.

For each (refresh period, latency) setting, fit on all other settings and
predict this one's success rate. If |predicted - observed| is routinely
beyond ~2 standard errors, the model's independence assumption is wrong for
this task and capacity numbers built on it should not be quoted.

    python scripts/validate_curves.py        # writes docs/calibration.json
"""
import json
import pathlib

import numpy as np

from chronos.admission import SLO
from chronos.hazard import cross_validate, load_rollouts

out = {}
for p in sorted(pathlib.Path("curves").glob("rollouts_*.npz")):
    task = p.stem.removeprefix("rollouts_")
    traces, ys, groups = load_rollouts(p)
    rows = cross_validate(traces, ys, groups, kind=json.loads((p.parent / f"{task}.json").read_text())["kind"])
    # operating region: settings whose mean plan age an admitted fleet can have
    mean_age = {tuple(g): np.mean(max((t for t, gg in zip(traces, groups) if gg == g), key=len))
                for g in set(groups)}
    op = [r for r in rows if mean_age[tuple(r["setting"])] <= SLO().s_max]
    err = np.array([r["predicted"] - r["observed"] for r in rows])
    z = np.array([abs(e) / max(r["se"], 1e-3) for e, r in zip(err, rows)])
    op_err = np.array([abs(r["predicted"] - r["observed"]) for r in op])
    out[task] = {"rows": rows, "mae": float(np.abs(err).mean()), "max_abs": float(np.abs(err).max()),
                 "frac_within_2se": float((z <= 2).mean()),
                 "operating_mae": float(op_err.mean()), "operating_max_abs": float(op_err.max())}
    print(f"\n{task}: MAE {out[task]['mae']:.3f}, max |err| {out[task]['max_abs']:.3f}, "
          f"{out[task]['frac_within_2se']:.0%} of settings within 2 SE; operating region "
          f"(mean age <= {SLO().s_max} s): MAE {out[task]['operating_mae']:.3f}, max {out[task]['operating_max_abs']:.3f}")
    print("  refresh  latency   observed  predicted")
    for r in rows:
        print(f"  {r['setting'][0]:>7}  {r['setting'][1]:>7}   {r['observed']:.3f}     {r['predicted']:.3f}")
pathlib.Path("docs/calibration.json").write_text(json.dumps(out, indent=1))
