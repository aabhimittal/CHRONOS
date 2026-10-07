"""Profile batch latency of your VLM and action head; writes a LatencyModel JSON.

Plug in real models by editing `load_fns` (e.g. GR00T N1's Eagle-2 backbone
and DiT head under torch.no_grad()). `--dummy` uses numpy matmuls so the
script runs anywhere; its numbers are meaningless except as a smoke test.

    python scripts/profile_latency.py --dummy --out latency.json
    python scripts/capacity.py --latency-file latency.json
"""
import argparse
import json

import numpy as np

from chronos.latency import fit_affine, time_fn


def load_fns(dummy: bool):
    if dummy:
        w_v, w_a = np.random.rand(512, 512), np.random.rand(128, 128)
        vlm = lambda b: [np.random.rand(b * 16, 512) @ w_v for _ in range(8)]
        act = lambda b: np.random.rand(b * 4, 128) @ w_a
        return vlm, act, lambda: None
    raise SystemExit("edit load_fns() to wrap your VLM / action head, or pass --dummy")


ap = argparse.ArgumentParser()
ap.add_argument("--dummy", action="store_true")
ap.add_argument("--batches", default="1,2,4,8,16,32,64")
ap.add_argument("--max-vlm-batch", type=int, default=64, help="largest batch that fits in memory")
ap.add_argument("--out", default="latency.json")
args = ap.parse_args()

bs = [int(b) for b in args.batches.split(",")]
vlm, act, sync = load_fns(args.dummy)
vf, vp = fit_affine(bs, time_fn(vlm, bs, sync=sync))
af, ap_ = fit_affine(bs, time_fn(act, bs, sync=sync))
out = {"vlm_fixed": vf, "vlm_per": vp, "act_fixed": af, "act_per": ap_,
       "max_vlm_batch": args.max_vlm_batch}
open(args.out, "w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
