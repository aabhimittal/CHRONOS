"""Tier 3: real-time runtime, phase prediction, online admission, fitted h_miss, GPU mix."""
from dataclasses import replace

import numpy as np
import pytest

from chronos.admission import make_fleet
from chronos.hazard import DEFAULT_EDGES as E, HazardModel, fit_hazard
from chronos.online import admit, arrivals, fleet
from chronos.placement import cheapest_fleet
from chronos.runtime import LayerSliced, RealtimeSimulator
from chronos.sim import ChronosPolicy, LatencyModel, Robot, Simulator

LAT = LatencyModel()
KNEE = HazardModel(E, np.where(E[:-1] < 0.75, 1e-4, 1e-2), 50)
FLAT = HazardModel(E, np.full(len(E) - 1, 1e-5), 50)


def test_layer_sliced_matches_unsliced_and_yields():
    W = np.random.default_rng(0).standard_normal((32, 32)) / 6
    layers = [lambda x: np.tanh(x @ W)] * 40
    ref = np.ones((4, 32))
    for f in layers:
        ref = f(ref)
    ex, steps = LayerSliced(layers, np.ones((4, 32))), 0
    while not ex.step(1e-9):           # tiny budget: one layer per slice
        steps += 1
    assert steps == 39 and np.allclose(ex.x, ref)


def test_realtime_runtime_meets_deadlines_and_reports_overhead():
    m = {"t": KNEE}
    r = RealtimeSimulator(make_fleet([Robot("t")], 8), LAT, ChronosPolicy(models=m)).run(1.5)
    assert r["miss_rate"] <= 0.05                       # wall clock on shared CI runners: loose
    assert 0 < r["decision_ms_p50"] < 5 and r["decisions"] > 10


def test_phase_predictor_uses_nominal_durations_only():
    r = Robot("a", phases=(("a", 4.0), ("b", 2.0)), phase_sigma=0.5, seed=3)
    idx, e = r.phase_state(1.0)
    assert r.predict_task(1.0, 0.0) == r.phases[idx][0]
    assert r.predict_task(1.0, r.phases[idx][1] - e + 0.01) != r.phases[idx][0]
    # realised durations differ from nominal (stochastic cycle)
    s = r._realised(30.0)
    assert any(abs((e1 - s0) - r.phases[i][1]) > 0.1 for s0, e1, i in s[1:])


def test_predict_lookahead_never_drops_current_phase_urgency():
    m = {"a": FLAT, "b": KNEE}
    pol = ChronosPolicy(models=m, lookahead="predict")
    sim = Simulator([Robot("a", phases=(("b", 2.0), ("a", 4.0)), phase_sigma=0.4, seed=1)], LAT, pol)
    sim.plan_t0[:] = -2.0                                # stale plan
    t = 1.9                                              # near the nominal end of critical phase "b"
    cur = KNEE.cost(2.0 + 1.9 + 0.05) - KNEE.cost(0.05)
    assert pol._gains(sim, t, [0], 0.05)[0] >= cur - 1e-12


def test_online_admission_protects_the_tail_under_overload():
    models = {"t": KNEE, "flat": FLAT}
    req = arrivals(6.0, 30.0, 30.0, ["t", "flat"], seed=2)          # offered load ~ 2x capacity
    out = {}
    for pol in ("all", "oracle"):
        adm = admit(req, models, LAT, policy=pol, calibration=1.5)
        sim = Simulator(fleet(adm), LAT, ChronosPolicy(models=models))
        sim.run(30.0)
        out[pol] = (len(adm), np.quantile(sim.success(models), 0.05))
    assert out["oracle"][0] < out["all"][0]
    assert out["oracle"][1] > out["all"][1] + 0.1                     # tail protected


def test_h_miss_is_recovered_from_dropped_action_rollouts():
    rng = np.random.default_rng(0)
    true_h, h_miss = 1e-3, 2e-2
    traces, ys, held = [], [], []
    for _ in range(3000):
        T, k = 100, int(rng.integers(0, 40))
        traces.append(np.full(T, 0.2))
        held.append(k)
        ys.append(rng.random() < np.exp(-(true_h * T + h_miss * k)))
    m = fit_hazard(traces, ys, held=held)
    assert m.h_miss == pytest.approx(h_miss, rel=0.25)


def test_gpu_mix_never_costs_more_than_a_single_type():
    models = {"t": KNEE, "flat": FLAT}
    s = lambda f: replace(LAT, vlm_fixed=LAT.vlm_fixed * f, vlm_per=LAT.vlm_per * f,
                          act_fixed=LAT.act_fixed * f, act_per=LAT.act_per * f)
    r = cheapest_fleet({"t": 30, "flat": 30}, {"t": Robot("t"), "flat": Robot("flat")}, models,
                       {"big": (LAT, 1.0), "small": (s(2.0), 0.4)}, horizon_s=5)
    assert r["cost"] <= min(r["single_type_cost"].values()) + 1e-9
    assert sum(r["capacity"][k] * v for k, v in r["plan"].items()) >= 60
