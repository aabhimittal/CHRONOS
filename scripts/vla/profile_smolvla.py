"""Measure SmolVLA latency and split it into VLM vs action-head cost.

SmolVLA inference = one VLM prefix pass (images + text + state -> KV cache)
+ K flow-matching steps of the action expert over that cache. Timing it at
batch sizes B and step counts K and fitting

    t(B, K) = (a + b B)  +  K (c + d B)
               VLM part      action-head part

separates the two without touching model internals. Writes a LatencyModel
JSON (CPU numbers here; the same script on a GPU gives the real ones).

    /home/user/venv-vla/bin/python -u scripts/vla/profile_smolvla.py --out docs/latency_smolvla_cpu.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--steps", default="2,5,10")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default="docs/latency_smolvla_cpu.json")
args = ap.parse_args()
torch.set_num_threads(os.cpu_count())

repo = "HuggingFaceVLA/smolvla_libero"
pcfg = PreTrainedConfig.from_pretrained(repo)
pcfg.pretrained_path, pcfg.device = repo, "cpu"
env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
venv = next(iter(next(iter(make_env(env_cfg, n_envs=1).values())).values()))
policy = make_policy(cfg=pcfg, env_cfg=env_cfg).eval()
pre, _ = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=repo,
                                  preprocessor_overrides={"device_processor": {"device": "cpu"}})
env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
obs, _ = venv.reset(seed=0)
o = preprocess_observation(obs)
o["task"] = list(venv.call("task_description"))
one = pre(env_pre(o))


def batch(b):
    def rep(v):
        if torch.is_tensor(v):
            return v.repeat(b, *[1] * (v.dim() - 1))
        return v * b if isinstance(v, list) else v           # task strings: one per robot; None etc. unchanged
    return {k: rep(v) for k, v in one.items()}


rows = []
with torch.inference_mode():
    policy.predict_action_chunk(one)                                   # warm-up
    for b in [int(x) for x in args.batches.split(",")]:
        x = batch(b)
        for k in [int(x) for x in args.steps.split(",")]:
            policy.config.num_steps = policy.model.config.num_steps = k
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                policy.predict_action_chunk(x)
                ts.append(time.perf_counter() - t0)
            rows.append({"B": b, "K": k, "s": float(np.median(ts))})
            print(rows[-1], flush=True)

B = np.array([r["B"] for r in rows], float)
K = np.array([r["K"] for r in rows], float)
y = np.array([r["s"] for r in rows])
a, b, c, d = np.linalg.lstsq(np.stack([np.ones_like(B), B, K, K * B], 1), y, rcond=None)[0]
k0 = 10
lat = {"vlm_fixed": max(a, 0), "vlm_per": max(b, 0), "act_fixed": max(c * k0, 0), "act_per": max(d * k0, 0),
       "act_steps": k0, "device": "cpu", "model": repo, "measurements": rows}
pred = (a + b * B) + K * (c + d * B)
lat["fit_max_rel_err"] = float(np.max(np.abs(pred - y) / y))
open(args.out, "w").write(json.dumps(lat, indent=1))
print(f"VLM: {a * 1e3:.0f} ms + {b * 1e3:.0f} ms/robot;  action head per step: {c * 1e3:.1f} ms + {d * 1e3:.1f} ms/robot "
      f"(K={k0}: {c * k0 * 1e3:.0f} + {d * k0 * 1e3:.0f} ms/robot); fit max rel err {lat['fit_max_rel_err']:.1%}")
