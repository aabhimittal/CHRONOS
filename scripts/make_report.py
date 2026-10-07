"""Regenerate every number and figure in the README and the web demo.

    python scripts/make_report.py          # ~2-3 min on a laptop CPU
Writes docs/results.json and docs/figures/*.svg.
"""
import json
import pathlib

import numpy as np

from chronos.admission import SLO, capacity, evaluate
from chronos.hazard import HazardModel
from chronos.oracle import fast_capacity, predict
from chronos.placement import gpus_needed
from chronos.sim import ChronosPolicy, FifoPolicy, LatencyModel, Robot

OUT = pathlib.Path("docs")
models = {p.stem: HazardModel.from_json(p.read_text()) for p in pathlib.Path("curves").glob("*.json")}
lat, slo = LatencyModel(), SLO()
tasks = sorted(models)
tpl = lambda **kw: [Robot(t, **kw) for t in tasks]
aware = lambda: ChronosPolicy(models=models, max_age=slo.s_max)
R = {"slo": vars(slo), "latency": vars(lat)}

# 1. curves
s_grid = np.round(np.arange(0, 3.01, 0.05), 3)
R["curves"] = {t: {"s": s_grid.tolist(), "p": models[t].p_success(s_grid).round(4).tolist()} for t in tasks}

# 2. sweep over N
Ns = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80, 96]
pols = {"CHRONOS (curve-aware)": (aware, tpl()), "CHRONOS (curve-blind)": (ChronosPolicy, tpl()),
        "FIFO batching": (FifoPolicy, tpl(vlm_period=0.5))}
R["sweep"] = {name: [evaluate(pf, t, n, lat, models, 10.0) for n in Ns] for name, (pf, t) in pols.items()}
for v in R["sweep"].values():
    for m in v:
        m["miss_rate"] = float(m["miss_rate"])
R["sweep_N"] = Ns

# 3. headline capacity
est = fast_capacity(tpl(), models, lat, slo)
seed = max(1, est // 2)
cap = {"CHRONOS (curve-aware)": capacity(aware, tpl(), models, lat, slo, n_start=seed)[0],
       "CHRONOS (curve-blind)": capacity(ChronosPolicy, tpl(), models, lat, slo, n_start=seed)[0],
       "Oracle (closed form)": est,
       "FIFO batching (best period)": max(capacity(FifoPolicy, tpl(vlm_period=p), models, lat, slo)[0]
                                          for p in (0.2, 0.5, 1.0)),
       "Dedicated": 1}
R["capacity"] = cap

# 4. phase-aware ablation: 4 s transit (insensitive) + 2 s grasp at a conveyor
ph = [Robot("shelf", phases=(("shelf", 4.0), ("conveyor", 2.0)))]
R["phase"] = {name: capacity(pf, ph, models, lat, slo, horizon_s=24, n_start=20)[0] for name, pf in {
    "phase lookahead": aware,
    "current phase only": lambda: ChronosPolicy(models=models, max_age=slo.s_max, lookahead=False),
    "curve-blind": ChronosPolicy}.items()}

# 5. oracle vs simulator
pts = []
for hz, ch in ((5, 1), (10, 1), (20, 4), (30, 1)):
    for n in (4, 16, 32, 64, 100):
        p = predict(n, lat, hz, ch)
        if p["schedulable"]:
            m = evaluate(ChronosPolicy, [Robot(tasks[0], hz=hz, chunk=ch)], n, lat, models, 8.0)
            pts.append({"hz": hz, "chunk": ch, "n": n, "sim": m["mean_staleness"], "oracle": p["mean_staleness"]})
R["oracle"] = pts

# 6. multi-GPU placement for a 400-robot fleet
counts = {"shelf": 300, "conveyor": 100}
tmpl = {t: Robot(t) for t in tasks}
R["placement"] = {"fleet": counts, **{s: gpus_needed(counts, tmpl, models, s, lat, slo)["gpus"]
                                      for s in ("mixed", "segregated")}}

(OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
print(json.dumps({k: R[k] for k in ("capacity", "phase", "placement")}, indent=1))

# ---------- figures ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})
C = {"CHRONOS (curve-aware)": "#2a7de1", "CHRONOS (curve-blind)": "#8fb8ef", "FIFO batching": "#e0703a"}

fig, ax = plt.subplots(figsize=(5, 3.2))
for t, c in zip(tasks, ("#e0703a", "#2a7de1")):
    ax.step(s_grid, R["curves"][t]["p"], where="post", label=t, color=c, lw=2)
ax.set(xlabel="plan staleness s (seconds)", ylabel="p(success | s)", ylim=(0, 1.05),
       title="Fitted staleness curves (toy tasks)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "figures/curves.svg"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for name, v in R["sweep"].items():
    for ax, key in zip(axs, ("success", "miss_rate", "mean_staleness")):
        ax.plot(Ns, [m[key] for m in v], "-o", ms=3, color=C[name], label=name, lw=2)
axs[0].axhline(R["sweep"]["CHRONOS (curve-aware)"][0]["success"] - slo.delta, ls=":", c="gray")
axs[1].axhline(slo.eps, ls=":", c="gray")
for ax, t in zip(axs, ("mean success", "deadline-miss rate", "mean staleness (s)")):
    ax.set(xlabel="robots on one GPU", title=t)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "figures/sweep.svg"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4, 3.4))
o = np.array([[p["sim"], p["oracle"]] for p in pts])
ax.plot([0, o.max() * 1.05], [0, o.max() * 1.05], c="gray", ls=":")
ax.scatter(o[:, 0], o[:, 1], c="#2a7de1", s=18)
ax.set(xlabel="simulated E[s] (s)", ylabel="oracle E[s] (s)", title="Closed-form oracle vs simulator")
fig.tight_layout(); fig.savefig(OUT / "figures/oracle.svg"); plt.close(fig)
print("figures written")
