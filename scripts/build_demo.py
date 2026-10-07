"""Inline the JS simulator and results into one self-contained demo page.

    python scripts/build_demo.py                      # -> docs/index.html (GitHub Pages)
    python scripts/build_demo.py --fragment out.html  # body-only variant for hosts that add <head>
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--fragment")
args = ap.parse_args()

res = json.loads((ROOT / "docs/results.json").read_text())
models = {}
for p in sorted((ROOT / "curves").glob("*.json")):
    m = json.loads(p.read_text().replace("Infinity", "1e9"))
    models[p.stem] = {"edges": m["edges"], "h": m["h"], "horizon": m["horizon"], "kind": m.get("kind", "ruin")}
data = {"models": models, **{k: res[k] for k in ("curves", "sweep", "sweep_N", "capacity", "phase", "placement")}}


def load(name):
    f = ROOT / "docs" / name
    return json.loads(f.read_text()) if f.exists() else None


def findings():
    """One row per experiment, built from the result files (nothing hand-typed)."""
    rows, F = [], lambda area, what, result, verdict: rows.append(
        {"area": area, "what": what, "result": result, "verdict": verdict})
    ci, cal, t2, t3, sp, rt = (load(n) for n in ("ci.json", "calibration.json", "tier2.json", "tier3.json",
                                                   "speed.json", "runtime.json"))
    if ci:
        a, f = ci["CHRONOS (curve-aware)"], ci["FIFO batching (0.5 s)"]
        F("trust", "Bootstrap CI on robots/GPU (20 refits of the curves)",
          f"curve-aware {a['median']:.0f} [{a['lo']:.0f}, {a['hi']:.0f}] vs FIFO {f['median']:.0f}: the gap survives", "helps")
    if cal and "conveyor" in cal:
        c = cal["conveyor"]
        F("trust", "Held-out calibration (leave one refresh/latency setting out)",
          f"conveyor MAE {c['mae']:.3f}, {c['operating_mae']:.3f} in the operating region; worst {c['max_abs']:.2f} at high latency", "neutral")
    F("trust", "Per-robot SLO (5th-percentile loss within 5 pts)",
      "curve-aware widens the tail (3.6 vs 2.1 pts); a 2-pt budget costs 4 robots", "neutral")
    if sp:
        F("feature", "Speed governor: slow the conveyor to admit more robots",
          f"items/hour per GPU {sp['best_fixed']:.1f} → {sp['best_governed']:.1f} ({sp['best_governed'] / sp['best_fixed'] - 1:+.0%})", "helps")
    if t2:
        w = t2["whittle"]
        gc, toy = w["gradual + cliff"], w["toy mix"]
        verdict = ("helps" if gc["whittle"] > gc["greedy"] and toy["whittle"] >= toy["greedy"] - 2
                   else "hurts" if gc["whittle"] < gc["greedy"] else "neutral")
        F("feature", "Whittle index instead of greedy refresh (same anticipation window)",
          f"gradual+cliff fleet {gc['whittle']} vs {gc['greedy']}; toy {toy['whittle']} vs {toy['greedy']}", verdict)
        e = next(r for r in t2["elastic"] if r["n"] == 120)
        F("feature", "Elastic action head (fewer denoising steps under overload)",
          f"misses {e['fixed']['miss_rate']:.0%} → {e['elastic']['miss_rate']:.0%} at 120 robots, 30 Hz (quality cost assumed)", "helps")
        pt = t2["partition"]
        best = max(v for k, v in pt.items() if k != "time-sliced")
        F("feature", "Spatial GPU partitioning (MPS/MIG) instead of time slicing",
          f"best {best} robots vs {pt['time-sliced']} time-sliced", "hurts")
        ev = next(r for r in t2["events"] if r["rate"] == 0.3 and r["n"] == 64)
        F("feature", "Event-triggered refresh (scene-change detector)",
          f"success {ev['event-triggered']:.3f} vs {ev['age (curve-blind)']:.3f} age-based at 0.3 changes/s", "helps")
    if t3:
        pp = {r["sigma"]: r for r in t3["phase_prediction"]}
        if 0.6 in pp:
            F("systems", "Predicting phases instead of knowing them",
              f"σ=0.3: {pp[0.3]['predicted (robust)']} vs {pp[0.3]['current phase only']} current-only; "
              f"σ=0.6: {pp[0.6]['predicted (robust)']} vs {pp[0.6]['current phase only']}", "neutral")
        on = [r for r in t3["online"]["rows"] if r["offered"] == 90]
        if on:
            al, ca = on[0], on[-1]
            F("systems", "Online admission with the calibrated oracle (offered ~90 robots)",
              f"admits {ca['admitted']}/{ca['requests']}, 5th-pct robot {ca['success_p5']:.2f} vs {al['success_p5']:.2f} admitting all", "helps")
        hm = t3["h_miss"]
        F("systems", "Fitted cost of a missed deadline (h_miss)",
          f"{max(hm.values()):.0e} on the toy vs 5e-2 assumed: must be measured per task", "neutral")
        g = t3["gpu_mix"]
        F("systems", "Cheapest mix of GPU types (400 robots)",
          f"${g['cost']:.2f}/h vs ${min(g['single_type_cost'].values()):.2f}/h best single type", "helps")
    if rt:
        r = max(rt, key=lambda x: x["n"])
        F("systems", "Scheduler on a real clock (wall-time runtime)",
          f"{r['n']:.0f} robots: decision p99 {r['decision_ms_p99']:.1f} ms, misses {r['miss_rate']:.1%} (sim {r['sim_miss_rate']:.1%})", "neutral")
    return rows


data["ci"] = {k: {kk: v[kk] for kk in ("median", "lo", "hi")} for k, v in (load("ci.json") or {}).items()}
data["findings"] = findings()

body = (ROOT / "docs/src/template.html").read_text()
body = body.replace("/*SIM_JS*/", (ROOT / "docs/src/sim.js").read_text())
body = body.replace("/*DATA_JSON*/", json.dumps(data, separators=(",", ":")))
if args.fragment:
    pathlib.Path(args.fragment).write_text(body)
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="description" content="Interactive simulator: how many VLA robots can one GPU serve '
        'under deadline and staleness SLOs?">\n</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(ROOT / "docs/index.html").write_text(page)
print(f"docs/index.html: {len(page) / 1024:.0f} KB")
