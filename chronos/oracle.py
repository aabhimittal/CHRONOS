"""Closed-form admission oracle for ChronosPolicy (curve-blind mode).

The simulator is the ground truth but costs ~10-100 ms per fleet size; an
online admission controller needs an answer when a robot asks to join. This
module predicts the steady state in microseconds.

Action path (EDF with non-preemptive blocking). Every period P the action
head serves the whole active fleet in roughly one batch, and any job can be
blocked by at most one VLM slice already on the GPU. The classic
non-preemptive EDF bound gives the sufficient test

    act(N) + net + guard + (quantum + slice_overhead) <= P.

VLM path. Between two action batches the scheduler fits
k = floor((P - act(N) - net - guard) / (q + o)) slices, so the VLM gets
eff = k q / (k (q + o) + act(N)) of the GPU (slicing is quantised: at high
control rates k drops to 1 and throughput falls off a cliff). Robots are
split into ceil(N/cap) rounds; round j of size B_j takes C_j = vlm(B_j)/eff
and each robot is refreshed every R = sum_j C_j. Its plan age sweeps
uniformly over [C, C + R] (plus network and intra-chunk offsets), which
gives E[s] and, via the hazard model, success.

Accuracy vs the simulator is tested in tests/test_oracle.py. The oracle
models curve-blind refresh, so for curve-aware scheduling it is a
conservative (lower-bound) capacity estimate.
"""
from __future__ import annotations

import math

import numpy as np

from .sim import LatencyModel


def predict(n: int, lat: LatencyModel, hz=10.0, chunk=1, quantum=0.010, guard=0.002,
            max_batch=64) -> dict:
    P = chunk / hz
    net = lat.net_base + lat.net_jitter
    blocking = quantum + lat.slice_overhead
    schedulable = lat.act(n) + net + guard + blocking <= P
    u = lat.act(n) / P
    k = max(math.floor((P - lat.act(n) - net - guard) / blocking), 0)
    eff = k * quantum / (k * blocking + lat.act(n)) if k else 1e-9
    cap = max(1, min(max_batch, lat.max_vlm_batch))
    rounds = [cap] * (n // cap) + ([n % cap] if n % cap else [])
    Cs = [lat.vlm(b) / eff for b in rounds] or [lat.vlm(1) / eff]
    C, R = float(np.mean(Cs)), float(sum(Cs))
    lo = C + lat.net_base + lat.net_jitter / 2 + lat.act(n) / 2
    hi = lo + R + (chunk - 1) / hz
    return {"schedulable": schedulable, "util_action": u, "batch_time": C,
            "refresh_interval": R, "s_lo": lo, "s_hi": hi, "mean_staleness": (lo + hi) / 2}


def predict_success(pred: dict, model, h_miss=0.05, schedulable_penalty=True) -> float:
    s = np.linspace(pred["s_lo"], pred["s_hi"], 64)
    p = float(model.from_rate(model.hazard(s).mean()))
    if pred["schedulable"] or not schedulable_penalty:
        return p
    return p * float(model.from_rate(h_miss)) if model.kind == "ruin" else 0.0


def fast_capacity(templates, models, lat=LatencyModel(), slo=None, ref=None, n_max=1024, **kw):
    """Largest N (mixed fleet cycling through `templates`) the oracle admits."""
    from .admission import SLO
    slo = slo or SLO()
    hz, chunk = templates[0].hz, templates[0].chunk
    ref_t = {t.task: predict_success(predict(1, lat, hz, chunk, **kw), models[t.task]) for t in templates}
    ref = ref if ref is not None else float(np.mean([ref_t[t.task] for t in templates]))
    best = 0
    for n in range(1, n_max + 1):
        p = predict(n, lat, hz, chunk, **kw)
        per = [predict_success(p, models[templates[i % len(templates)].task]) for i in range(n)]
        refs = [ref_t[templates[i % len(templates)].task] for i in range(n)]
        loss_tail = float(np.quantile(np.subtract(per, refs), slo.tail_q))
        if (p["schedulable"] and p["mean_staleness"] <= slo.s_max and np.mean(per) >= ref - slo.delta
                and loss_tail >= -slo.delta_tail):
            best = n
        elif n > best + 8:
            break
    return best
