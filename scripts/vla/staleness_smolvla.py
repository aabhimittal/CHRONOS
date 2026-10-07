"""Real staleness curves: SmolVLA (HuggingFaceVLA/smolvla_libero) in LIBERO, on CPU.

Staleness here = open-loop execution of an action chunk: the chunk predicted
from the frame captured at step c is executed at step t with index t - c.
A refresh every `k` steps whose result lands `L` steps after capture is the
same experiment as chronos.rollout.run_episode on the toy task, so the output
feeds the unchanged fitting / validation / bootstrap code.

Run with the VLA environment (LeRobot + LIBERO + torch):
    /home/user/venv-vla/bin/python -u scripts/vla/staleness_smolvla.py \\
        --suite libero_spatial --tasks 0 --refresh 10,25,50 --latency 0,5 --episodes 6
Writes curves_vla/rollouts_<suite>_<task>.npz incrementally (resumable).
"""
import argparse
import json
import os
import pathlib
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import torch

from chronos.hazard import save_rollouts
from chronos.rollout import Rollout, run_episode
from chronos.vla import ChunkPolicy

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", default="0")
ap.add_argument("--refresh", default="10,25,50")
ap.add_argument("--latency", default="0,5")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--num-steps", type=int, default=10, help="flow-matching denoising steps")
ap.add_argument("--max-steps", type=int, default=0, help="override episode length (0 = suite default)")
ap.add_argument("--out", default="curves_vla")
args = ap.parse_args()
torch.set_num_threads(os.cpu_count())

pcfg = PreTrainedConfig.from_pretrained("HuggingFaceVLA/smolvla_libero")
pcfg.pretrained_path, pcfg.device, pcfg.num_steps = "HuggingFaceVLA/smolvla_libero", "cpu", args.num_steps


class LiberoTask:
    """chronos.rollout.Env over one LeRobot LIBERO task (one vector sub-env)."""

    def __init__(self, suite, task_id, policy):
        self.cfg = LiberoCfg(task=suite, task_ids=[task_id])
        self.venv = next(iter(next(iter(make_env(self.cfg, n_envs=1).values())).values()))
        self.hz = float(self.cfg.fps)
        self.horizon = args.max_steps or int(self.venv.call("_max_episode_steps")[0])
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=pcfg, pretrained_path=pcfg.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.cfg, policy_cfg=pcfg)
        self.policy = policy

    def _batch(self, obs):
        o = preprocess_observation(obs)
        o["task"] = list(self.venv.call("task_description"))
        return self.pre(self.env_pre(o))

    def reset(self, seed):
        obs, _ = self.venv.reset(seed=seed)
        self.policy.reset()
        return obs

    def infer(self, obs):
        """Raw env obs -> (chunk, action_dim) env-space actions."""
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self._batch(obs))[0]       # (H, D) normalised
        acts = [self.env_post({"action": self.post(chunk[i:i + 1])})["action"] for i in range(len(chunk))]
        return np.concatenate([a.cpu().numpy() for a in acts])

    def step(self, action):
        obs, _, term, trunc, info = self.venv.step(np.asarray(action)[None])
        ok = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
        return obs, ok


policy = make_policy(cfg=pcfg, env_cfg=LiberoCfg(task=args.suite, task_ids=[0])).eval()
out = pathlib.Path(args.out)
out.mkdir(exist_ok=True)
for task_id in [int(t) for t in args.tasks.split(",")]:
    env = LiberoTask(args.suite, task_id, policy)
    chunker = ChunkPolicy(env.infer)
    name = f"{args.suite}_{task_id}" + (f"_K{args.num_steps}" if args.num_steps != 10 else "")
    path = out / f"rollouts_{name}.npz"
    log = out / f"log_{name}.jsonl"
    done = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    rolls = [Rollout(np.array(d["staleness"]), d["success"], d["k"], d["L"]) for d in done]
    seen = {(d["k"], d["L"], d["ep"]) for d in done}
    for k in [int(x) for x in args.refresh.split(",")]:
        for L in [int(x) for x in args.latency.split(",")]:
            for ep in range(args.episodes):
                if (k, L, ep) in seen:
                    continue
                t0 = time.time()
                r = run_episode(env, chunker, k, L, seed=ep)
                rolls.append(r)
                with log.open("a") as f:
                    f.write(json.dumps({"k": k, "L": L, "ep": ep, "success": bool(r.success),
                                        "staleness": r.staleness.round(4).tolist()}) + "\n")
                save_rollouts(path, rolls)
                print(f"{name} k={k} L={L} ep={ep} success={r.success} steps={len(r.staleness)} "
                      f"{time.time() - t0:.0f}s", flush=True)
