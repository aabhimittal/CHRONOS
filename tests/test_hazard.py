import numpy as np

from chronos.hazard import DEFAULT_EDGES, HazardModel, fit_hazard


def test_recovers_monotone_hazard():
    rng = np.random.default_rng(0)
    true = HazardModel(DEFAULT_EDGES, np.where(DEFAULT_EDGES[:-1] < 0.5, 1e-3, 2e-2), 50)
    traces, ys = [], []
    for _ in range(3000):
        k = rng.choice([1, 3, 8, 15]) / 10
        tr = np.arange(50) % (k * 10) / 10 + rng.uniform(0, 0.3)
        traces.append(tr)
        ys.append(rng.random() < np.exp(-true.hazard(tr).sum()))
    fit = fit_hazard(traces, ys)
    assert np.all(np.diff(fit.h) >= -1e-12)
    assert abs(fit.p_success(0.1) - true.p_success(0.1)) < 0.05
    assert abs(fit.p_success(1.0) - true.p_success(1.0)) < 0.1


def test_json_roundtrip():
    m = HazardModel(DEFAULT_EDGES, np.linspace(0, 1, len(DEFAULT_EDGES) - 1), 10.0, "x")
    m2 = HazardModel.from_json(m.to_json())
    assert np.allclose(m.h, m2.h) and m2.task == "x" and np.isinf(m2.edges[-1])
