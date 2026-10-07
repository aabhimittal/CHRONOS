"""Adapters for real VLA policies and simulators (run on a GPU machine).

Two ways a real model maps onto the staleness harness (chronos.rollout):

  ChunkPolicy  - any model that predicts an action *chunk* from one
                 observation (pi0, OpenVLA-OFT, GR00T N1 `get_action`).
                 plan = the chunk; act = the entry for "steps since the frame
                 was captured". Staleness here = open-loop chunk execution.
  SplitPolicy  - a true dual-system model with separable halves: a slow
                 encoder (VLM) and a fast decoder (action head) that runs
                 every step on the *current* observation plus the cached plan.
                 Staleness here = age of the cached VLM features.

These are different quantities. Report which one a curve measures.

LiberoEnv wraps a LIBERO task with the setup sequence used by OpenVLA's
LIBERO evaluation (benchmark -> task -> init states -> OffScreenRenderEnv,
settle with no-op steps). It is import-guarded and NOT exercised in this
repository's CI (no GPU/MuJoCo rendering here); tests cover the adapters with
a fake env that has the same interface.
"""
from __future__ import annotations

import os

import numpy as np

# Max episode steps per LIBERO suite (as used by OpenVLA's LIBERO eval)
LIBERO_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
                    "libero_10": 520, "libero_90": 400}


class ChunkPolicy:
    """`infer(obs) -> (H, action_dim)` chunk predicted from one observation."""
    wants_age = True

    def __init__(self, infer):
        self.infer = infer

    def plan(self, obs):
        return np.asarray(self.infer(obs))

    def act(self, obs, plan, age_steps):
        # entry k of the chunk was meant for k steps after capture; past the end,
        # hold the last action (what a real controller does when a chunk runs out)
        return plan[min(age_steps, len(plan) - 1)]


class SplitPolicy:
    """`encode(obs) -> features` (slow, VLM); `decode(obs, features) -> action` (fast)."""
    wants_age = False

    def __init__(self, encode, decode):
        self.encode, self.decode = encode, decode

    def plan(self, obs):
        return self.encode(obs)

    def act(self, obs, plan):
        return self.decode(obs, plan)


class LiberoEnv:
    """chronos.rollout.Env over one LIBERO task. Seeds index the task's fixed
    initial states. obs gets an "instruction" key with the task language."""
    hz = 20.0

    def __init__(self, suite="libero_10", task_id=0, resolution=256, settle_steps=10):
        from libero.libero import benchmark, get_libero_path          # noqa: import-guarded
        from libero.libero.envs import OffScreenRenderEnv
        ts = benchmark.get_benchmark_dict()[suite]()
        self.task = ts.get_task(task_id)
        self.init_states = ts.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
        self.env.seed(0)
        self.horizon = LIBERO_MAX_STEPS[suite]
        self.settle_steps = settle_steps

    def _wrap(self, obs):
        return {**obs, "instruction": self.task.language}

    def reset(self, seed):
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[seed % len(self.init_states)])
        for _ in range(self.settle_steps):                      # let objects settle
            obs, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        return self._wrap(obs)

    def step(self, action):
        obs, _, done, _ = self.env.step(np.asarray(action).tolist())
        return self._wrap(obs), bool(done)
