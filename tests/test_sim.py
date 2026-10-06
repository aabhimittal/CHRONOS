import numpy as np

from chronos.admission import evaluate
from chronos.hazard import DEFAULT_EDGES, HazardModel
from chronos.sim import ChronosPolicy, FifoPolicy, LatencyModel, Robot

MODELS = {"t": HazardModel(DEFAULT_EDGES, np.where(DEFAULT_EDGES[:-1] < 0.3, 1e-4, 1e-2), 50)}
LAT = LatencyModel()


def test_single_robot_meets_deadlines():
    m = evaluate(ChronosPolicy, [Robot("t")], 1, LAT, MODELS, 5)
    assert m["miss_rate"] == 0 and m["mean_staleness"] < 0.15


def test_chronos_keeps_deadlines_where_fifo_blocks():
    # atomic VLM batch of 40 (~0.28s) exceeds the 0.1s action period
    c = evaluate(ChronosPolicy, [Robot("t")], 40, LAT, MODELS, 5)
    f = evaluate(FifoPolicy, [Robot("t", vlm_period=0.5)], 40, LAT, MODELS, 5)
    assert c["miss_rate"] < 0.01 < f["miss_rate"]


def test_staleness_grows_with_fleet():
    s = [evaluate(ChronosPolicy, [Robot("t")], n, LAT, MODELS, 5)["mean_staleness"] for n in (2, 16, 64)]
    assert s[0] < s[1] < s[2]


def test_curve_aware_batches_skip_insensitive_robots():
    flat = HazardModel(DEFAULT_EDGES, np.full(len(DEFAULT_EDGES) - 1, 1e-5), 50)
    models = {"t": MODELS["t"], "flat": flat}
    fleet = [Robot("t"), Robot("flat")]
    aware = evaluate(lambda: ChronosPolicy(models=models), fleet, 48, LAT, models, 5)
    blind = evaluate(ChronosPolicy, fleet, 48, LAT, models, 5)
    assert aware["success"] > blind["success"]
