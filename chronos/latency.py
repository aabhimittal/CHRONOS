"""Fit the affine batch-latency model from profiled timings."""
from __future__ import annotations

import time

import numpy as np


def fit_affine(batch_sizes, seconds) -> tuple[float, float]:
    """Least-squares `fixed + per * B`, clamped non-negative. Returns (fixed, per)."""
    B = np.asarray(batch_sizes, float)
    A = np.stack([np.ones_like(B), B], 1)
    fixed, per = np.linalg.lstsq(A, np.asarray(seconds, float), rcond=None)[0]
    return max(float(fixed), 0.0), max(float(per), 0.0)


def time_fn(fn, batch_sizes, warmup=2, iters=5, sync=lambda: None):
    """Median wall time of fn(B) per batch size. Pass sync=torch.cuda.synchronize on GPU."""
    out = []
    for b in batch_sizes:
        for _ in range(warmup):
            fn(b)
        sync()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(b)
            sync()
            ts.append(time.perf_counter() - t0)
        out.append(float(np.median(ts)))
    return out
