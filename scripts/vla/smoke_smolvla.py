"""Smoke test: SmolVLA (LIBERO checkpoint) in LIBERO on CPU via LeRobot. Run with the VLA venv."""
import os
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors

torch.set_num_threads(4)
env_cfg = LiberoEnv(task="libero_spatial", task_ids=[0])
envs = make_env(env_cfg, n_envs=1)
venv = next(iter(next(iter(envs.values())).values()))
pcfg = PreTrainedConfig.from_pretrained("lerobot/smolvla_libero")
pcfg.pretrained_path = "lerobot/smolvla_libero"
pcfg.device = "cpu"
print("chunk_size", pcfg.chunk_size, "n_action_steps", pcfg.n_action_steps, "num_steps", pcfg.num_steps)
policy = make_policy(cfg=pcfg, env_cfg=env_cfg).eval()
pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=pcfg.pretrained_path,
                                     preprocessor_overrides={"device_processor": {"device": "cpu"}})
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
obs, _ = venv.reset(seed=0)
t0 = time.time()
o = preprocess_observation(obs)
o["task"] = list(venv.call("task_description"))
o = pre(env_pre(o))
with torch.inference_mode():
    chunk = policy.predict_action_chunk(o)
print("chunk", tuple(chunk.shape), "inference s", round(time.time() - t0, 2))
