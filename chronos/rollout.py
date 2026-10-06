"""Staleness-injection rollout harness.

Wraps any dual-system policy (slow `plan`, fast `act`) and forces the plan to
be refreshed every `refresh_steps` steps, arriving `latency_steps` after its
observation was captured. Records the per-step plan age, which is what
`fit_hazard` consumes.

Mapping to real models:
  * GR00T N1: plan = Eagle-2 VLM embedding, act = DiT action head.
  * pi0 / OpenVLA have no separate planner; the analogue of staleness is
    executing an action chunk open-loop, i.e. plan = predicted chunk and
    act = index into it. Same harness, different meaning: say which you ran.

`ConveyorReach` is a toy closed-loop task (a moving target) used only to test
the pipeline end to end. Its curves say nothing about real VLAs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Env(Protocol):
    hz: float
    horizon: int
    def reset(self, seed: int) -> Any: ...
    def step(self, action) -> tuple[Any, bool]: ...   # (obs, success)


class DualSystemPolicy(Protocol):
    def plan(self, obs) -> Any: ...
    def act(self, obs, plan) -> Any: ...


@dataclass
class Rollout:
    staleness: np.ndarray   # (T,) plan age in seconds at each executed step
    success: bool
    refresh_steps: int
    latency_steps: int


def run_episode(env: Env, policy: DualSystemPolicy, refresh_steps: int,
                latency_steps: int, seed: int) -> Rollout:
    obs = env.reset(seed)
    history = [obs]
    plan, plan_step = policy.plan(obs), 0     # warm start: fresh plan at t=0
    pending = []                              # (ready_step, capture_step, plan)
    ages, success = [], False
    for t in range(env.horizon):
        if t > 0 and t % refresh_steps == 0:
            pending.append((t + latency_steps, t, policy.plan(history[t])))
        while pending and pending[0][0] <= t:
            _, plan_step, plan = pending.pop(0)
        ages.append((t - plan_step) / env.hz)
        obs, success = env.step(policy.act(obs, plan))
        history.append(obs)
        if success:
            break
    return Rollout(np.array(ages), success, refresh_steps, latency_steps)


def sweep(env: Env, policy: DualSystemPolicy, refresh_grid, latency_grid,
          episodes: int, seed: int = 0) -> list[Rollout]:
    out, s = [], seed
    for k in refresh_grid:
        for lat in latency_grid:
            for _ in range(episodes):
                out.append(run_episode(env, policy, k, lat, s))
                s += 1
    return out


class ConveyorReach:
    """Agent must touch a target drifting at `speed` (units/s) in a unit box."""

    def __init__(self, speed: float, hz: float = 10.0, horizon: int = 150,
                 radius: float = 0.05, agent_speed: float = 0.5):
        self.speed, self.hz, self.horizon, self.radius = speed, hz, horizon, radius
        self.max_step = agent_speed / hz

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.target = rng.uniform(0.1, 0.9, 2)
        ang = rng.uniform(0, 2 * np.pi)
        self.vel = self.speed / self.hz * np.array([np.cos(ang), np.sin(ang)])
        self.agent = np.array([0.5, 0.5])
        return self._obs()

    def _obs(self):
        return {"agent": self.agent.copy(), "target": self.target.copy()}

    def step(self, action):
        a = np.asarray(action, float)
        n = np.linalg.norm(a)
        self.agent = self.agent + (a if n <= self.max_step else a * self.max_step / n)
        self.target = self.target + self.vel
        for i in range(2):                     # bounce off walls
            if not 0 <= self.target[i] <= 1:
                self.vel[i] *= -1
                self.target[i] = np.clip(self.target[i], 0, 1)
        return self._obs(), bool(np.linalg.norm(self.agent - self.target) < self.radius)


class OraclePlanner:
    """'VLM' = perfect target localisation; 'action head' = go to planned point."""

    def plan(self, obs):
        return obs["target"]

    def act(self, obs, plan):
        return plan - obs["agent"]
