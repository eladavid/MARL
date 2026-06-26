# Remote handoff — n=4 Drone experiment (MAC-REINFORCE paper)

This package runs the **drone-patrol** numerical experiment for the MAC-REINFORCE
paper on a strong remote machine. It is a one-click run; results are figures + a
text summary. Written so another agent (or a human) can pick it up cold.

## TL;DR — how to run
```bash
# from the MARL repo root (branch: feature/projected_gd_with_batch_variance_control)
git pull
POOL_Q=50000 NCORES=30 bash remote/run_all.sh     # tune to the box's physical cores
# results in results_remote/:
#   fig_drone_strategies.pdf, SUMMARY.txt              (the figure + numbers)
#   part_H1_sched_*.pkl, part_H0_sched_*.pkl           (pools, resumable)
#   random_search.json                                 (the "0 of 1e6" baseline, reproducible)
# env knobs: POOL_Q (H=1 pool size), H0_Q, RAND_TOTAL (random-search count), EPISODES, NCORES
```
This run **pins p** (does a large pool find the optimum at all? local got 0/10,600) and
saves every number in the figure as a reproducible artifact. Expectation, per our
analysis: still ~0 optima — the optimum requires individual *sacrifice* and has a
vanishing basin; β/temperature is a *selection* knob and cannot make the *generator*
emit a candidate it never produces. A bigger pool tightens the bound (e.g. 0/50,000),
it does not change the conclusion.

## What the paper is about (1 paragraph)
Cooperative MARL as a finite-horizon **Markov potential game** with **decoupled,
deterministic local dynamics** (`s_i' = a_i`) and a shared objective. Independent
per-agent REINFORCE ("MAC-REINFORCE") from a fixed start tracks projected ascent on
a shared potential Φ (no communication), but converges only to *a* Nash equilibrium.
A **stochastic Nash-hopping meta-algorithm** (Algorithm 2) then concentrates on the
**global** potential maximizer. Two obstacles: (1) *representation* — memoryless
policies can't express the optimal joint cycle; a history buffer H fixes it; (2)
*selection* — independent learning traps in suboptimal NEs; the meta-algorithm selects
the global optimum.

## The drone game (this experiment)
Instantiates the paper's §2.1.1 example. **n=4 drones, 4 nodes** = 1 charger (R0) +
3 mission posts (M1,M2,M3); per-agent state `(node, battery)`, battery `B=4`,
deterministic dynamics, **frozen sub-policy at empty battery forces a reload** (lives
in the transition, keeps decoupling). Reward: `g` = coverage − congestion (limited
charger capacity → stacking penalized), `u_i` = battery aversion.
- **Optimum (vectorized VI, `drone_vi_vec.py`): Φ\*=84.69**, a clean **period-4
  rotating patrol** — all 3 posts covered every step while the 4 drones take turns at
  the single charger (the charging "hole" rotates ag0→ag1→ag2→ag3).
- **H\*=1**: a 1-step history buffer *exactly represents* the optimum (verified by
  construction). **H=0 (memoryless) cannot** — no clock to phase-lock.

## What's already established (local runs) — so you know what to expect
- **Independent learning robustly FAILS to find the optimum.** Across **0/10,600**
  PSGA runs (fixed + scheduled batch) and **0/1,000,000** random-deterministic
  ("early-hardening") policies, the global optimum was **never** found. It is 1 of
  ~**10^135** behaviorally-distinct deterministic policies (H=1).
- **Why:** PSGA converges to a *suboptimal NE* — it learns the right structure
  (rotating patrol) but can't perfectly **phase-lock** the staggered reload without
  communication; battery phases drift until two drones reload on the same step,
  dropping coverage. Caps at ~**0.91 Φ\***. Not undertraining (frozen at 20k episodes),
  not a finite-horizon artifact (longer T is *worse*).
- **Strategy comparison (Φ/Φ\*)** measured locally:
  - H=0 memoryless ceiling: ~**0.22**  (representation gap)
  - single PSGA H=1 (mean): **0.68**  (a lottery: worst seeds ~0.03–0.43)
  - random-deterministic search (1e6): **0.79**
  - **MAC-REINFORCE + meta-algorithm (cold β): ~0.89**  (concentrates monotonically 0.70→0.89)
  - VI global optimum: **1.000** (undiscoverable here)
- **Takeaway the figure makes:** the meta-algorithm **dominates all other strategies**
  and extracts the **best discoverable** policy; the residual 0.89→1.0 gap *quantifies
  how hard this realistic instance is*. (NOTE: do **not** claim the meta reaches the
  *global* optimum on the drone game — it doesn't; that claim belongs to the abstract
  graded game, where the optimum IS discoverable. Here the honest claim is "best
  discoverable + dominates other strategies + the optimum is a needle.")

## What this remote run is FOR
1. **Pin p** — does a *large* pool (POOL_Q=24k–50k, scheduled batch) find the optimum
   *at all*? Local got 0/10,600. If still 0 at 50k → p < 1/50000, a very robust
   "independent learning cannot find it" statement. If a few appear → we have p and
   can quantify the meta's reach time (~1/p).
2. Re-confirm the **H=0 ceiling** and **meta concentration** at scale, and emit the
   publication figure.

## Outputs
- `results_remote/part_H1_sched_*.pkl` — H=1 pool candidate stats (incremental, resumable)
- `results_remote/part_H0_sched_*.pkl` — H=0 memoryless pool
- `results_remote/fig_drone_strategies.pdf` — the comparison figure
- `results_remote/SUMMARY.txt` — the numbers above, recomputed at scale

## CRITICAL gotchas (learned the hard way)
1. **Thread pinning is mandatory.** `torch.set_num_threads(1)` ALONE is insufficient —
   BLAS/MKL ignore it and spawn ~7 threads/worker → on a 12-core box, 12 workers gave
   **load 80+** and a 3× slowdown. `run_all.sh` exports
   `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS=1` AND the worker sets them before
   importing numpy/torch. **Keep `NCORES` ≤ physical cores.**
2. **Calibrate under load, not in isolation** (a single worker runs ~2× faster than it
   does under full parallelism — don't size the run from a solo timing).
3. **Chunked saves**: workers save every `--chunk` candidates, so a kill/restart loses
   at most one chunk and progress is visible in the `log_*.log` files.
4. **Per-candidate cost** (pinned, single core): ~3–4 s fixed batch, ~6–7 s scheduled
   batch (avg batch ≈63), at 6000 episodes. So POOL_Q=24000 on ~30 cores ≈ 30–60 min.

## Code map (all in the MARL repo)
- `drone_game.py` — game + slow exact VI (small n). `drone_vi_vec.py` — vectorized VI
  (validated == slow VI at n=3; feasible to n=4, ~7s).
- `drone_train.py` — single-system vectorized MAC-REINFORCE (drone). `drone_pool.py` —
  candidate-vectorized pool trainer (train many systems at once).
- `meta_algorithm.py` — **the validated Algorithm-2 accept rule** (`accepts`). The
  analysis replays the pool through THIS (faithful: the meta's candidates are exactly
  random-restart PSGA, i.e. the pool).
- `remote/pool_worker.py`, `remote/analyze.py`, `remote/run_all.sh` — this package.

## If you want to go further
- **Bigger pools** (POOL_Q=100k) to push the p bound lower.
- **Larger n** (n=5 drones, 4 missions): needs `drone_vi_vec.py` at n=5 — feasible but
  heavier (joint states 20^5 ≈ 3.2M); the optimum stays a closed-form-free VI reference.
- The **abstract graded game** (separate, already done locally) carries the
  "meta reaches the TRUE global optimum" result (p≈6% there). Not part of this run.
