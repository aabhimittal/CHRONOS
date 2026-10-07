# CHRONOS

Deadline-aware serving for dual-system vision-language-action (VLA) policies.

LLM serving optimizes throughput. Robot control has deadlines: a late action
isn't a slow response, it's a dropped item. CHRONOS asks how many robots one GPU
can serve if the objective is **robots per GPU, subject to a deadline-miss bound
and a success-rate loss bound**, and what scheduler gets there.

> **Status: simulation only.** No robot hardware, and no real VLA has been
> profiled or rolled out yet. Every number below comes from a toy task and
> illustrative latency constants. It shows the pipeline works and which way
> the effects point. It is **not** a fleet result.

## Idea

Dual-system VLAs (GR00T N1: Eagle-2 VLM + DiT action head) split into a slow
semantic path and a fast action path. The action head runs every control tick
off the *latest* plan. Two quantities decide fleet capacity:

1. **Staleness** `s`: the age of the observation behind the active plan.
   The project depends on the curve `p(success | s)`, fitted per task.
2. **Scheduling.** Batch VLM refreshes across the fleet, run EDF on the action
   path, and admit robots while `P(miss) <= eps`, `E[s] <= s_max` and
   `success >= dedicated - delta`.

## Mechanism

| Component | What it does | Where |
|---|---|---|
| Staleness-injection harness | Wraps any `plan`/`act` policy and forces a refresh every *k* steps with *L* steps of latency. Records the plan age at each step. | `chronos/rollout.py` |
| Hazard model | `P(success) = exp(-Σ_t h(s_t))`, with `h` piecewise-constant and monotone. The NLL is convex, so the fit has a unique optimum. It scores whole staleness **traces**, so a curve fitted from fixed-period rollouts can score any scheduler. | `chronos/hazard.py` |
| GPU simulator | One non-preemptive GPU running action batches and VLM work. | `chronos/sim.py` |
| `ChronosPolicy` | EDF with lazy batching on actions. VLM work is cut into slices of at most 10 ms (Sarathi-style), so it never blocks a deadline. Frames are captured when a batch launches (pull mode). Batch membership is **curve-aware**: a robot sitting in a flat part of its curve is skipped. | `chronos/sim.py` |
| `FifoPolicy` | LLM-serving baseline. FIFO order, greedy batching, atomic VLM batches, robots push frames on a fixed period. That period is tuned per run, so the baseline is not a strawman. | `chronos/sim.py` |
| Admission | Scans N upward and records the largest N that meets the SLO. | `chronos/admission.py` |

## Run

```bash
pip install -e .[dev] && pytest -q
python scripts/fit_curves.py          # toy rollouts -> curves/*.json
python scripts/capacity.py            # headline table
python scripts/capacity.py --chunk 4 --hz 20 --latency '{"vlm_per":0.01}'
```

## Toy results (pipeline check, not a claim)

SLO: miss ≤ 1%, E[s] ≤ 1 s, success loss ≤ 2 points. 10 Hz, one action per job.

| Policy | Robots/GPU | Binding constraint |
|---|---|---|
| CHRONOS, curve-aware batches | 55 | success (staleness) |
| CHRONOS, curve-blind | 43 | success (staleness) |
| FIFO batching, best period | 18 | deadline misses (1.9% at N=19); success still 0.995 |
| Dedicated | 1 | — |

How to read this:
- **FIFO and CHRONOS fail for different reasons.** FIFO hits the deadline wall:
  an atomic VLM batch outlasts the action period. Most of its staleness budget
  goes unused. Slicing the VLM work turns a deadline problem into a staleness
  problem, which degrades gradually instead of all at once.
- **The curve-aware gain (+28%) is inflated by the toy.** The toy "shelf" task
  is flat in staleness, so skipping it costs nothing. Real tasks will be less
  bimodal. If every task's curve has the same knee, this gain disappears and
  the scheduler reduces to ordinary batching.
- **Dedicated = 1 is a weak baseline.** The comparison that matters is CHRONOS
  vs. tuned FIFO. Quote that ratio, not the comparison against dedicated.
- **Capacity moves in steps.** `h` is piecewise-constant, so the predicted
  success has steps too, and capacity jumps when a robot's typical staleness
  crosses a bin edge. Use finer bins once you have enough rollouts.

## Known weaknesses / what's not modeled

- **Sim-to-real.** Curves fitted in LIBERO or SimplerEnv may not transfer to
  hardware. The toy env (`ConveyorReach`) only exists to test the pipeline.
- **No real VLA adapter yet.** A LIBERO/SimplerEnv `Env` and a GR00T/π0
  `DualSystemPolicy` still need writing. For π0 and OpenVLA there is no
  separate planner, so "staleness" means open-loop action-chunk execution.
  That is a different quantity, and results must say which one was measured.
- **Latency model.** `fixed + per_item·B` with made-up constants. Profile real
  batch latencies (including the cost of VLM slicing and KV/prefix memory
  limits on B) before quoting anything.
- **GPU model.** One non-preemptive stream. MPS/MIG spatial sharing, multiple
  GPUs, and network jitter are not modeled.
- **Hazard independence.** The model assumes each step's risk depends only on
  that step's staleness. Correlated failures, for example a stale plan during
  a grasp, are missed. Deadline misses use a fixed hazard `h_miss` per held
  step, which is assumed, not fitted.
- **Rollout budget.** One curve takes about (refresh grid × latency grid ×
  episodes) rollouts per task. For a 7B VLA in LIBERO, budget that in GPU-hours
  before starting.

## Layout

```
chronos/hazard.py     staleness -> success model + convex MLE fit
chronos/rollout.py    staleness-injection harness, toy env
chronos/sim.py        GPU simulator, CHRONOS + FIFO policies
chronos/admission.py  SLO + capacity search
scripts/              fit_curves.py, capacity.py
tests/                pytest
```
