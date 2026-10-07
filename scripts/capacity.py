"""Headline metric: robots served per GPU at a fixed success-rate loss."""
import argparse
import json
import pathlib

from chronos.admission import SLO, capacity
from chronos.hazard import HazardModel
from chronos.oracle import fast_capacity
from chronos.sim import ChronosPolicy, FifoPolicy, LatencyModel, Robot

ap = argparse.ArgumentParser()
ap.add_argument("--curves", default="curves")
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--delta", type=float, default=0.02)
ap.add_argument("--s-max", type=float, default=1.0)
ap.add_argument("--hz", type=float, default=10.0)
ap.add_argument("--chunk", type=int, default=1)
ap.add_argument("--horizon", type=float, default=20.0, help="simulated seconds per N")
ap.add_argument("--latency", default="{}", help='JSON overrides, e.g. {"vlm_per":0.01}')
ap.add_argument("--latency-file", help="LatencyModel JSON from scripts/profile_latency.py")
args = ap.parse_args()

models = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path(args.curves).glob("*.json")}
base = json.loads(open(args.latency_file).read()) if args.latency_file else {}
lat = LatencyModel(**{**base, **json.loads(args.latency)})
slo = SLO(args.eps, args.s_max, args.delta)
tpl = lambda **kw: [Robot(t, hz=args.hz, chunk=args.chunk, **kw) for t in sorted(models)]

rows = []
est = fast_capacity(tpl(), models, lat, slo)
seed = max(1, int(0.5 * est))
n, ref, _ = capacity(lambda: ChronosPolicy(models=models, max_age=args.s_max), tpl(), models, lat, slo,
                     args.horizon, n_start=seed)
rows.append(("chronos (EDF + sliced VLM, curve-aware batches)", n))
n, _, _ = capacity(lambda: ChronosPolicy(), tpl(), models, lat, slo, args.horizon, n_start=seed)
rows.append(("chronos (curve-blind, refresh all)", n))
rows.append(("oracle prediction (closed form, curve-blind)", est))
best = max((capacity(FifoPolicy, tpl(vlm_period=p), models, lat, slo, args.horizon)[0], p)
           for p in (0.1, 0.2, 0.5, 1.0, 2.0))
rows.append((f"fifo batching (best refresh period {best[1]}s)", best[0]))
rows.append(("dedicated (1 robot / GPU)", 1))

print(f"SLO: miss<={args.eps}, E[s]<={args.s_max}s, success loss<={args.delta} "
      f"(dedicated success {ref:.3f}); tasks={sorted(models)}; {args.hz}Hz x{args.chunk}")
print(f"{'policy':<48}robots/GPU")
for name, n in rows:
    print(f"{name:<48}{n}")
