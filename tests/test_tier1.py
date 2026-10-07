"""Tier 1: per-robot SLO, bootstrap uncertainty, held-out calibration, model form."""
import numpy as np
import pytest

from chronos.admission import SLO, capacity, evaluate, feasible, reference
from chronos.hazard import (DEFAULT_EDGES, HazardModel, bootstrap_fits, cross_validate,
                            fit_hazard, predict_setting, select_kind)
from chronos.sim import ChronosPolicy, LatencyModel, Robot

EDGES = DEFAULT_EDGES
RUIN = HazardModel(EDGES, np.where(EDGES[:-1] < 0.5, 1e-4, 2e-2), 60, kind="ruin")
OPP = HazardModel(EDGES, np.where(EDGES[:-1] < 0.3, 5e-2, 1e-3), 150, kind="opportunity")


def synth(true, n_per=60, seed=0):
    """Rollouts from a known model over a refresh x latency grid."""
    rng = np.random.default_rng(seed)
    traces, ys, groups = [], [], []
    for k in (1, 4, 16, 32):
        for lat in (0, 2, 5):
            pattern = (np.arange(150) % k + lat) / 10
            for _ in range(n_per):
                rate = true.hazard(pattern)
                if true.kind == "ruin":                 # first ruin step ends it as a failure
                    hit = rng.random(150) < rate
                    T = 150 if not hit.any() else int(np.argmax(hit)) + 1
                    y = not hit.any()
                else:                                   # first catch ends it as a success
                    hit = rng.random(150) < rate
                    T = int(np.argmax(hit)) + 1 if hit.any() else 150
                    y = bool(hit.any())
                traces.append(pattern[:T]); ys.append(y); groups.append((k, lat))
    return traces, ys, groups


@pytest.mark.parametrize("true", [RUIN, OPP], ids=["ruin", "opportunity"])
def test_select_kind_recovers_generating_form(true):
    tr, y, g = synth(true)
    m, cv = select_kind(tr, y, g)
    err = {k: np.mean([abs(r["predicted"] - r["observed"]) for r in rows]) for k, rows in cv.items()}
    assert m.kind == true.kind and err[true.kind] < err[{"ruin": "opportunity", "opportunity": "ruin"}[true.kind]]
    assert err[true.kind] < 0.12     # leave-one-setting-out on 12 settings: some bins lose most evidence


def test_opportunity_fit_is_monotone_nonincreasing():
    tr, y, g = synth(OPP)
    m = fit_hazard(tr, y, kind="opportunity")
    assert np.all(np.diff(m.h) <= 1e-12) and m.p_success(0.1) > m.p_success(2.0)


def test_prediction_ignores_episode_length():
    """A held-out setting's prediction must not depend on how long its
    episodes ran (short = succeeded early would leak the label)."""
    pattern = (np.arange(150) % 8) / 10
    assert predict_setting(RUIN, pattern) == predict_setting(RUIN, np.concatenate([pattern, pattern]))


def test_cross_validate_reports_every_setting():
    tr, y, g = synth(RUIN, n_per=20)
    rows = cross_validate(tr, y, g)
    assert len(rows) == len(set(g)) and all(0 <= r["predicted"] <= 1 for r in rows)


def test_bootstrap_is_stratified_and_varies():
    tr, y, g = synth(RUIN, n_per=20)
    fits = bootstrap_fits(tr, y, g, n_boot=5)
    assert len(fits) == 5
    assert np.std([f.p_success(1.0) for f in fits]) > 0          # resampling changes the fit


def test_per_robot_slo_catches_starved_minority():
    """A mean that passes can hide a tail that fails."""
    m = {"miss_rate": 0.0, "mean_staleness": 0.5, "success": 0.99,
         "loss_per_robot": np.r_[np.zeros(90), np.full(10, -0.2)]}
    assert not feasible(dict(m), 1.0, SLO())                         # 10% of robots lose 20 pts
    assert feasible(dict(m), 1.0, SLO(delta_tail=1.0))               # mean-only SLO would pass


def test_tighter_per_robot_budget_never_raises_capacity():
    models = {"t": RUIN, "flat": HazardModel(EDGES, np.full(len(EDGES) - 1, 1e-5), 60)}
    tpl = [Robot("t"), Robot("flat")]
    pf = lambda: ChronosPolicy(models=models)
    loose = capacity(pf, tpl, models, slo=SLO(delta_tail=1.0), horizon_s=6, n_start=10)[0]
    tight = capacity(pf, tpl, models, slo=SLO(delta_tail=0.01), horizon_s=6, n_start=10)[0]
    assert tight <= loose


def test_reference_is_per_template():
    models = {"t": RUIN, "flat": HazardModel(EDGES, np.full(len(EDGES) - 1, 1e-5), 60)}
    ref = reference([Robot("t"), Robot("flat")], models, LatencyModel(), 4)
    assert ref.shape == (2,) and ref[1] >= ref[0]


def test_transfer_check_flags_a_different_environment():
    same = synth(RUIN, n_per=30, seed=1)
    other = synth(HazardModel(EDGES, np.where(EDGES[:-1] < 0.2, 1e-4, 4e-2), 60), n_per=30, seed=2)
    m = fit_hazard(*synth(RUIN, n_per=30)[:2])
    from chronos.hazard import transfer_check
    err = lambda rows: np.mean([abs(r["predicted"] - r["observed"]) for r in rows])
    assert err(transfer_check(m, *same)) < err(transfer_check(m, *other))
