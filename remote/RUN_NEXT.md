# What to run next on the remote — abstract-game scaling

**One job left.** This is the only remaining experiment. It answers the referee
question *"you claim scaling in the number of agents — does Nash-hopping still
concentrate on the global optimum as agents grow?"* by computing the concentration
curve at **k = 5, 6, 7, 8** on the abstract graded congestion game (which has a
**closed-form optimum**, so we know Φ* exactly at any number of agents).

## The command
```bash
# from the MARL repo root, after: git pull
KS="5 6 7 8" Q=4000 EPISODES=5000 NCORES=30 bash remote/run_abstract.sh
```
- `KS`        agent counts to test (default "5 6 7 8")
- `Q`         candidate policies trained per k (default 3000; **4000+ recommended** — higher k has rarer optima, need a bigger pool to contain a few)
- `EPISODES`  PSGA episodes per candidate (5000 is enough; plateau ~8000)
- `NCORES`    parallel workers — **set to the box's physical core count**

## What it produces (in `results_abstract/`)
- `fig_abstract_scaling.pdf` — the curve: ν^β vs β, one line per k. **The result we want: each line rises toward ~1.0 as β cools** (hopping concentrates on the optimum), holding as k grows.
- `SUMMARY.txt` — per k: pool size, optimum-rate p, best ratio, peak ν^β.
- `abs_k{k}_*.pkl` — the pools (resumable; a killed run resumes from checkpoints).

## What to expect
- As **k grows**, the optimum gets **rarer** (p shrinks) — that's the selection problem hardening with agents, and it's expected.
- The analysis **automatically uses long enough chains** (epochs scaled to 1/p) so the concentration is real, not a short-chain artifact. (This was the lesson from the drone run, already baked in.)
- **Success looks like:** every k's curve reaching high ν^β (≈0.9–1.0) at cold β. If a k shows `optfrac=0 (no optimum in pool)`, **rerun that k with larger Q** — the optimum is just rarer than the pool sampled.

## Time / resources
- Thread-pinned (no machine-screaming); keep `NCORES ≤ physical cores`.
- ~`Q × EPISODES` work per k, parallelized. At Q=4000, EPISODES=5000, ~30 cores:
  roughly **20–45 min per k**, so ~1.5–3 h for all four. Resumable if interrupted.

## After it finishes
Push `results_abstract/` (or just the `.pkl` + `SUMMARY.txt` + pdf) back, and we
plug the curve into the paper's scaling figure. That completes the experiments;
everything after is figure regeneration (at γ=1) + writing the section.

---
### Note (already handled, FYI)
- Everything runs at **γ=1** (undiscounted, matches the paper) — `git pull` brings it.
- The analysis avoids the infinite-horizon VI that hangs at γ=1; it uses the
  closed-form optimum and the validated `meta_algorithm.accepts` rule.
