"""Industrial edge cases: stalls, bursts, churn, mixed rates, overload, jitter."""
import math

import numpy as np
import pytest

from chronos.admission import make_fleet
from chronos.hazard import DEFAULT_EDGES, HazardModel
from chronos.latency import fit_affine
from chronos.placement import gpus_needed, proportional_pattern
from chronos.sim import ChronosPolicy, FifoPolicy, LatencyModel, Robot, Simulator

H = HazardModel(DEFAULT_EDGES, np.where(DEFAULT_EDGES[:-1] < 0.3, 1e-4, 1e-2), 50)
FLAT = HazardModel(DEFAULT_EDGES, np.full(len(DEFAULT_EDGES) - 1, 1e-5), 50)
MODELS = {"t": H, "flat": FLAT}
LAT = LatencyModel()


def run(robots, policy=None, lat=LAT, horizon=5.0, **kw):
    sim = Simulator(robots, lat, policy or ChronosPolicy(), **kw)
    return sim, sim.run(horizon, record=True)


def test_empty_fleet():
    _, m = run([])
    assert m["jobs"] == 0 and m["miss_rate"] == 0


def test_long_horizon_job_count_exact():
    sim, m = run([Robot("t")], horizon=120.0)
    assert abs(m["jobs"] - 1200) <= 1 and m["miss_rate"] == 0


def test_gpu_stall_misses_bounded_and_recovers():
    """A 300 ms driver/thermal stall drops ~3 periods of jobs, then recovers."""
    fleet = make_fleet([Robot("t")], 16)
    sim, m = run(fleet, stalls=[(2.0, 0.3)], horizon=6.0)
    misses = sum(s.late + s.dropped for s in sim.stats)
    assert 0 < misses <= 16 * 5
    sim2, _ = run(make_fleet([Robot("t")], 16), stalls=[(2.0, 0.3)], horizon=4.0)
    tail = sum(s.late + s.dropped for s in sim.stats) - sum(s.late + s.dropped for s in sim2.stats)
    assert tail == 0                                   # no misses in [4, 6): recovered


def test_stall_at_time_zero():
    _, m = run([Robot("t")], stalls=[(0.0, 0.25)], horizon=2.0)
    assert m["jobs"] > 0


def test_aligned_burst_meets_deadlines():
    """All robots release at the same instant (synchronised PLC clock)."""
    sim, m = run(make_fleet([Robot("t")], 48, aligned=True))
    assert m["miss_rate"] == 0


def test_churn_join_and_leave():
    fleet = [Robot("t", phase=0.01 * i, start=2.0 if i % 2 else 0.0,
                   stop=3.0 if i % 3 == 0 else math.inf) for i in range(24)]
    sim, m = run(fleet)
    assert m["miss_rate"] == 0
    for r, st in zip(fleet, sim.stats):
        alive = min(r.stop, 5.0) - r.start
        assert abs(st.jobs - alive / r.period) <= 2      # no jobs outside [start, stop)
    joined = [i for i, r in enumerate(fleet) if r.start == 2.0]
    assert all(sim.plan_t0[i] > 2.0 for i in joined)      # newcomers got refreshed


def test_mixed_control_rates():
    tpl = [Robot("t", hz=5), Robot("t", hz=10), Robot("t", hz=30)]
    sim, m = run(make_fleet(tpl, 18))
    assert m["miss_rate"] == 0
    fast = [s for r, s in zip(sim.robots, sim.stats) if r.hz == 30]
    assert all(s.jobs >= 140 for s in fast)


def test_vlm_overload_degrades_gracefully():
    """Far past capacity, CHRONOS keeps every deadline and lets staleness grow;
    FIFO starts dropping actions."""
    _, c = run(make_fleet([Robot("t")], 160), horizon=4.0)
    _, f = run(make_fleet([Robot("t", vlm_period=0.5)], 160), FifoPolicy(), horizon=4.0)
    assert c["miss_rate"] == 0 and c["mean_staleness"] > 1.0
    assert f["miss_rate"] > 0.5


def test_action_path_infeasible_terminates():
    """act(N) > period: misses are unavoidable; sim must still finish."""
    lat = LatencyModel(act_per=0.004)                     # act(40) = 0.164 s > 0.1 s
    _, m = run(make_fleet([Robot("t")], 40), lat=lat, horizon=3.0)
    assert m["miss_rate"] > 0.3


def test_network_jitter_absorbed_in_laxity():
    lat = LatencyModel(net_base=0.005, net_jitter=0.010)
    _, m0 = run(make_fleet([Robot("t")], 24))
    _, m = run(make_fleet([Robot("t")], 24), lat=lat)
    assert m["miss_rate"] == 0
    assert m["mean_staleness"] > m0["mean_staleness"] + 0.005


def test_memory_cap_respected():
    lat = LatencyModel(max_vlm_batch=8)
    sim, _ = run(make_fleet([Robot("t")], 40), lat=lat, horizon=3.0)
    assert max(b for k, _, _, b in sim.trace if k == "vlm") <= 8
    sim, _ = run(make_fleet([Robot("t", vlm_period=0.5)], 40), FifoPolicy(), lat=lat, horizon=3.0)
    assert max(b for k, _, _, b in sim.trace if k == "vlm") <= 8


def test_vlm_slices_never_exceed_quantum():
    sim, _ = run(make_fleet([Robot("t")], 32), ChronosPolicy(quantum=0.010))
    assert max(t1 - t0 for k, t0, t1, _ in sim.trace if k == "vlm") <= 0.010 + LAT.slice_overhead + 1e-9


def test_phase_lookahead_beats_current_phase():
    # knee (0.75 s) must sit above the VLM latency, else even a fresh plan is
    # past it, every refresh gain is zero and lookahead has nothing to act on
    knee = HazardModel(DEFAULT_EDGES, np.where(DEFAULT_EDGES[:-1] < 0.75, 1e-4, 1e-2), 50)
    models = {"t": knee, "flat": FLAT}
    tpl = [Robot("flat", phases=(("flat", 4.0), ("t", 2.0)))]
    s = {}
    for name, la in (("ahead", True), ("now", False)):
        sim = Simulator(make_fleet(tpl, 56), LAT, ChronosPolicy(models=models, lookahead=la))
        sim.run(12.0)
        s[name] = sim.success(models).mean()
    assert s["ahead"] > s["now"]


def test_proportional_pattern_and_placement():
    pat = proportional_pattern({"a": 300, "b": 100})
    assert pat.count("a") == 15 and pat.count("b") == 5
    assert pat[:4].count("a") == 3
    tpl = {"t": Robot("t"), "flat": Robot("flat")}
    for strat in ("mixed", "segregated"):
        g = gpus_needed({"t": 40, "flat": 40}, tpl, MODELS, strat, horizon_s=5.0)["gpus"]
        assert 1 <= g < math.inf


def test_latency_fit_recovers_affine():
    b = np.array([1, 2, 4, 8, 16, 32])
    f, p = fit_affine(b, 0.03 + 0.005 * b + np.random.default_rng(0).normal(0, 1e-4, 6))
    assert f == pytest.approx(0.03, abs=2e-3) and p == pytest.approx(0.005, abs=2e-4)
