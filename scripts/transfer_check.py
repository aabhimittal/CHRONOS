"""Does a staleness curve transfer? Fit on rollouts A, predict rollouts B.

    python scripts/transfer_check.py curves_vla/rollouts_libero_spatial_0.npz curves_vla/rollouts_libero_spatial_1.npz

The intended use is A = simulation, B = the same task on hardware. If the
predicted-vs-observed error on B is much larger than A's own held-out error
(scripts/validate_curves.py), the curve does not transfer and capacity numbers
built on it should not be used for the real fleet.
"""
import sys

import numpy as np

from chronos.hazard import cross_validate, load_rollouts, select_kind, transfer_check

a, b = sys.argv[1], sys.argv[2]
ta, ya, ga = load_rollouts(a)
tb, yb, gb = load_rollouts(b)
model, cv = select_kind(ta, ya, ga)
own = np.mean([abs(r["predicted"] - r["observed"]) for r in cv[model.kind]])
rows = transfer_check(model, tb, yb, gb)
err = np.mean([abs(r["predicted"] - r["observed"]) for r in rows])
print(f"fit on {a} ({model.kind}); its own held-out MAE {own:.3f}")
print(f"applied to {b}: MAE {err:.3f}  ->  {'transfers' if err <= 2 * own + 0.05 else 'does NOT transfer'}")
print("  refresh latency  observed predicted   n")
for r in rows:
    print(f"  {r['setting'][0]:>7} {r['setting'][1]:>7}   {r['observed']:.3f}    {r['predicted']:.3f}  {r['n']:>3}")
