# Running the Nash-hopping meta-algorithm experiment (Fig 4)

Faithful paper Algorithm 2 (Stochastically-Stable Nash Hopping), made tractable by
**MAC-REINFORCE parallelization**. Produces `ν^β(Π★)` vs temperature `β` — the
fraction of time the meta-algorithm's incumbent sits at the global optimum — showing
concentration on the optimum as `β→0` (Theorem 35).

## Idea (why parallel generation is valid)
Algorithm 2's candidate at each epoch is drawn from a **random restart** `θ_rand ~ U[Θ]`,
**independently of the current incumbent**. The incumbent only enters the *accept decision*.
So we can **generate the per-epoch candidate pairs in parallel** and **replay them through the
two-stage accept rule sequentially** — this equals running Algorithm 2 one epoch at a time,
*in distribution*. Each slot produces a pair:
- `jump  = harden(θ_rand)`                      — the early-stop candidate (no training)
- `cand  = harden(MAC-REINFORCE(θ_rand))`       — the PSGA candidate

## Files
- `meta_algorithm.py`   — env builder, hardened eval (`Ḡ, Ūᵢ, Φ`), the per-agent accept rule.
- `meta_parallel.py`    — parallel candidate generation (`ProcessPoolExecutor`).
- `run_meta_experiment.py` — **the CLI** (two phases: `generate`, `select`, or `all`).
- `congestion_game/`, `claude_parallelized/parallel_simulation.py` — the toy game + MAC-REINFORCE (`train`).

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install torch numpy matplotlib tqdm psutil
# (only if you also want TensorBoard live-logging:) pip install "setuptools<81" tensorboard
```
Python 3.10–3.12, CPU-only is fine (tiny tabular game). Run **from the repo root** so
`congestion_game` and `claude_parallelized` are importable.

## Run

**Two phases** (recommended on a cluster — generate the expensive part once, sweep cheaply):
```bash
# 1) GENERATE candidate pairs in parallel (the expensive part).  --workers = #cores.
python run_meta_experiment.py generate \
       --inits 122 022 222 111 \
       --K 200 --sub-episodes 400 --sub-batch 32 \
       --workers 32 --save-every 5 --out results_meta

# 2) SELECT: replay through the two-stage rule, sweep beta, many seeds -> CSV + figure (seconds).
python run_meta_experiment.py select \
       --inits 122 022 222 111 \
       --epochs 2000 --seeds 50 --out results_meta
```
Or everything at once (small/local): `python run_meta_experiment.py all --inits 122 --K 50 --sub-episodes 400 --sub-batch 32`

## Recommended parameters & runtime (calibrated)
Per-candidate cost is **linear**: `≈ 0.55 ms × (sub_episodes × sub_batch × 16)` ≈ **`sub_episodes × sub_batch × 9 µs`** on one core. Measured points: `400 ep × 32 batch ≈ 108 s/candidate` and reaches the optimum **~29% of the time** (`PSGA @opt`).

| profile | `--K` `--sub-episodes` `--sub-batch` | `PSGA @opt` | s/candidate (1 core) | wall for 1 init |
|---|---|---|---|---|
| **default (validated)** | `200  400  32` | ~0.29 | ~108 s | ~12 min @32 cores · ~45 min @8 |
| **harder / faster** | `200  250  16` | lower (verify >0) | ~35 s | ~4 min @32 · ~15 min @8 |

- **`--K`**: 100–200 is plenty — the pool is only a *sample* of the candidate distribution (selection resamples with replacement). `K=1000` is 5–10× overkill (≈ a *day* per init — don't).
- **`--sub-episodes` / `--sub-batch`**: must keep `PSGA @opt > 0` (printed live during `generate`). `400/32 → 0.29`; **120 episodes → ~0** (too short, chain can't move). Lower `sub_episodes` to get a *harder* (lower-baseline) operating point — just watch the printed `@opt`.
- **`--epochs` / `--seeds`** (select): cheap (seconds) — scale freely for tighter error bars.

## Checkpointing & resume (safe to interrupt)
`generate` **saves incrementally** (every `--save-every` candidates, atomic write) and is **resumable**: each candidate is keyed by seed, so re-running the *same command* loads what's done and generates only the rest. A SLURM timeout / Ctrl-C loses at most `--save-every` candidates. To extend a pool, just re-run with a larger `--K`.

### Key knobs
- `--inits` : initial states as digit strings, e.g. `122` = `(1,2,2)`. (Trap-prone inits make the rescue visible; the optimum is representable at the default `H=1`.)
- `--K` : candidate pairs generated per init (the parallel work). Larger = lower-variance candidate distribution.
- `--sub-episodes`, `--sub-batch` : the MAC-REINFORCE subroutine budget. **Must be large enough that some candidates reach the optimum** (`PSGA @opt > 0`, printed during `generate`). **Calibrated:** fixed batch 32 with **400 episodes already gives `PSGA @opt ≈ 0.29`** on init `(1,2,2)` (mean ratio 0.806) — healthy. Use **400–800** episodes. (120 episodes is too short — `@opt ≈ 0`, the chain can't move.) If `generate` reports `PSGA @opt = 0.00`, increase the budget.
- `--epochs`, `--seeds` : selection-chain length and number of seeds (error bars). Cheap — scale freely.

## Outputs (in `--out`)
- `pairs_<init>.pkl`  — cached candidate pairs per init (so you can re-`select` without re-generating).
- `nu_beta.csv`       — `init, beta, pool_at_opt, nu_mean, nu_std`.
- `fig4_nu_beta.png`  — `ν^β(Π★)` vs `β`, one curve per init (dotted line = `pool@opt`, the no-selection / plain-PG baseline).

**What to expect:** each curve rises from `≈ pool@opt` at large `β` (no selection) to `≈1.0` as `β→0` — the meta-algorithm concentrating on the global optimum.

## Cluster notes
- Pure CPU, embarrassingly parallel in `generate`; set `--workers` to the allocated cores.
- `generate` cost ≈ `K / workers × (one MAC-REINFORCE subroutine)`. Each subroutine is a small tabular run (`sub_episodes × sub_batch × 16` steps).
- Example SLURM:
  ```bash
  #SBATCH -c 32
  #SBATCH --mem=8G
  #SBATCH -t 01:00:00
  python run_meta_experiment.py generate --inits 122 022 222 111 --K 200 \
         --sub-episodes 400 --sub-batch 32 --workers $SLURM_CPUS_PER_TASK --save-every 5 --out results_meta
  python run_meta_experiment.py select --inits 122 022 222 111 --epochs 2000 --seeds 50 --out results_meta
  ```
  (Resumable — if the job hits the time limit, just resubmit the same command and it continues.)
- Then copy back `results_meta/` (the `.pkl`s + `nu_beta.csv` + `fig4_nu_beta.png`).
