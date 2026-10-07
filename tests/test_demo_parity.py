"""The web demo's JS simulator must agree with the Python ground truth."""
import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from chronos.admission import make_fleet
from chronos.hazard import HazardModel
from chronos.sim import ChronosPolicy, FifoPolicy, LatencyModel, Robot, Simulator

ROOT = pathlib.Path(__file__).resolve().parents[1]
NODE = """
const C = require(process.argv[1]); const cfg = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const models = {}; for (const [k, m] of Object.entries(cfg.models)) models[k] = C.hazardModel(m);
const pol = cfg.policy === 'fifo' ? new C.FifoPolicy() : new C.ChronosPolicy({models: cfg.aware ? models : null});
const sim = new C.Sim(cfg.robots, C.Lat(), pol, cfg.stalls);
const m = sim.run(cfg.horizon); const s = sim.success(models);
console.log(JSON.stringify({miss_rate: m.miss_rate, mean_staleness: m.mean_staleness, jobs: m.jobs,
                            success: s.reduce((a, b) => a + b, 0) / s.length}));
"""

CASES = [("chronos", False, 24, []), ("chronos", True, 48, []), ("fifo", False, 24, []),
         ("chronos", True, 32, [[2.0, 0.3]]), ("fifo", False, 40, [[1.0, 0.2]])]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("policy,aware,n,stalls", CASES)
def test_js_matches_python(policy, aware, n, stalls):
    models = {p.stem: HazardModel.from_json(p.read_text()) for p in (ROOT / "curves").glob("*.json")}
    tpl = [Robot(t, vlm_period=0.5 if policy == "fifo" else None) for t in sorted(models)]
    fleet = make_fleet(tpl, n, seed=3)
    pol = FifoPolicy() if policy == "fifo" else ChronosPolicy(models=models if aware else None)
    sim = Simulator(fleet, LatencyModel(), pol, stalls=[tuple(s) for s in stalls])
    py = sim.run(6.0)
    py["success"] = float(sim.success(models).mean())
    cfg = {"policy": policy, "aware": aware, "horizon": 6.0, "stalls": stalls,
           "robots": [{"task": r.task, "hz": r.hz, "chunk": r.chunk, "phase": r.phase,
                       "vlm_period": r.vlm_period} for r in fleet],
           "models": {k: {"edges": [min(e, 1e9) for e in m.edges.tolist()], "h": m.h.tolist(),
                          "horizon": m.horizon} for k, m in models.items()}}
    out = subprocess.run(["node", "-e", NODE, str(ROOT / "docs/src/sim.js")], input=json.dumps(cfg),
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    assert js["jobs"] == py["jobs"]
    for k in ("miss_rate", "mean_staleness", "success"):
        assert js[k] == pytest.approx(float(py[k]), rel=1e-6, abs=1e-9), k
