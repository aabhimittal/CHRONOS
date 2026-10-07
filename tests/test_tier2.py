"""Tier 2: Whittle index, elastic action head, spatial partitioning, speed governor, events."""
import functools

import numpy as np
import pytest

from chronos.admission import capacity, make_fleet
from chronos.events import EventPolicy, EventSimulator, SceneModel
from chronos.hazard import DEFAULT_EDGES as E, HazardModel
from chronos.partition import PartitionedSimulator
from chronos.sim import ChronosPolicy, LatencyModel, Robot, Simulator
from chronos.speed import evaluate_speeds, governed

LAT = LatencyModel()
MID = (E[:-1] + np.minimum(E[1:], 6)) / 2
CLIFF = HazardModel(E, np.where(E[:-1] < 0.75, 0.0, 3e-2), 60)


class _Sim:                                   # minimal stand-in for Simulator in index tests
    def __init__(self, age, model):
        self.robots = [Robot("m")]
        self.plan_t0 = np.array([-age])
        self.models = {"m": model}


def whittle(age, model, L=0.1):
    pol = ChronosPolicy(models={"m": model}, lookahead=False, index="whittle")
    return pol._whittle(_Sim(age, model), 0.0, 0, L)


def test_whittle_linear_cost_grows_quadratically():
    """c(u) = k u  =>  W(a) = k a^2 / 2 (times horizon)."""
    k = 1e-3
    lin = HazardModel(np.linspace(0, 10, 2001).tolist() + [np.inf], k * np.linspace(0, 10, 2001), 1.0)
    for a in (0.5, 1.0, 2.0):
        assert whittle(a, lin) == pytest.approx(k * a * a / 2, rel=0.03)


def test_whittle_saturates_past_a_cliff():
    """Documented failure mode: past the knee W = C (knee - L) is constant,
    so cliff robots stop gaining urgency (greedy is the default for this reason)."""
    w = [whittle(a, CLIFF) for a in (1.0, 1.5, 2.5)]
    assert w[0] == pytest.approx(w[1], rel=0.05) and w[1] == pytest.approx(w[2], rel=0.05)
    assert w[0] == pytest.approx(60 * 3e-2 * (0.75 - 0.1), rel=0.05)


def test_elastic_head_removes_overload_misses():
    m = {"t": CLIFF}
    fleet = lambda: make_fleet([Robot("t", hz=30)], 120)       # act(120) = 40 ms > 33 ms period
    fixed = Simulator(fleet(), LAT, ChronosPolicy(models=m)).run(3)
    sim = Simulator(fleet(), LAT, ChronosPolicy(models=m, elastic=True))
    el = sim.run(3)
    assert fixed["miss_rate"] > 0.5 and el["miss_rate"] == 0
    assert any(st.extra for st in sim.stats)                    # quality cost is accounted for


def test_elastic_is_inert_when_deadlines_fit():
    m = {"t": CLIFF}
    sim = Simulator(make_fleet([Robot("t")], 20), LAT, ChronosPolicy(models=m, elastic=True))
    sim.run(3)
    assert not any(st.extra for st in sim.stats)


def test_partitioned_sim_meets_deadlines_and_refreshes():
    m = {"t": CLIFF}
    sim = PartitionedSimulator(make_fleet([Robot("t")], 10), LAT, ChronosPolicy(models=m), fa=0.3)
    r = sim.run(5)
    assert r["miss_rate"] == 0 and r["mean_staleness"] < 1.0


def test_time_slicing_beats_partitioning_at_default_latencies():
    m = {"t": CLIFF, "f": HazardModel(E, np.full(len(E) - 1, 1e-5), 60)}
    tpl, pf = [Robot("t"), Robot("f")], lambda: ChronosPolicy(models=m)
    sliced = capacity(pf, tpl, m, horizon_s=6, n_start=20)[0]
    part = capacity(pf, tpl, m, horizon_s=6, n_start=5,
                    sim_cls=functools.partial(PartitionedSimulator, fa=0.3, interference=1.15))[0]
    assert sliced > part > 0


def test_speed_governor_only_picks_feasible_levels():
    slow = HazardModel(E, np.where(E[:-1] < 1.5, 1e-4, 2e-2), 60)
    fast = CLIFF
    family = {"t": {0.5: slow, 1.0: fast}}
    rows = evaluate_speeds([Robot("t")], family, {"t": fast}, 40, horizon_s=5)
    g = governed(rows)
    assert g is None or g["feasible"]
    assert {r["v"] for r in rows} == {0.5, 1.0}


def test_expected_curve_matches_event_model():
    sc = SceneModel(rate=0.5)
    c = sc.expected_curve()
    assert c.hazard(0.01) == pytest.approx(sc.h_valid, abs=1e-3)
    assert c.hazard(5.5) == pytest.approx(sc.h_valid + (sc.h_invalid - sc.h_valid) * (1 - np.exp(-0.5 * 5.5)), rel=0.05)


def test_event_trigger_beats_age_based():
    sc = SceneModel(rate=0.3)
    run = lambda pol: EventSimulator(make_fleet([Robot("scene")], 48), LAT, pol, sc, 6.0)
    a, e = run(ChronosPolicy()), run(EventPolicy(max_age=2.0))
    a.run(6), e.run(6)
    assert e.scene_success().mean() > a.scene_success().mean() + 0.05
