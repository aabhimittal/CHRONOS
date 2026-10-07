import numpy as np
import pytest

from chronos.admission import evaluate
from chronos.hazard import DEFAULT_EDGES, HazardModel
from chronos.oracle import predict
from chronos.sim import ChronosPolicy, LatencyModel, Robot

M = {"t": HazardModel(DEFAULT_EDGES, np.zeros(len(DEFAULT_EDGES) - 1), 10)}
LAT = LatencyModel()


@pytest.mark.parametrize("hz,chunk", [(5, 1), (10, 1), (20, 4), (30, 1)])
@pytest.mark.parametrize("n", [4, 16, 32, 64, 100])
def test_oracle_tracks_simulator(hz, chunk, n):
    p = predict(n, LAT, hz, chunk)
    if not p["schedulable"]:
        pytest.skip("oracle declines; sufficient test only")
    m = evaluate(ChronosPolicy, [Robot("t", hz=hz, chunk=chunk)], n, LAT, M, 8)
    assert m["miss_rate"] == 0                                   # schedulable => no misses
    assert p["mean_staleness"] == pytest.approx(m["mean_staleness"], rel=0.15)
