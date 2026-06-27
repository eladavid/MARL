# Abstract-game scaling run — status & per-k episode budgets

Goal: does Nash-hopping concentrate on the global optimum as agent-count k grows?
Pools built per k (graded congestion game, rho=0.7, closed-form Phi*), to be replayed
through `meta_algorithm.accepts` by `remote/abstract_analyze.py`.

## KEY FINDING: the training budget must scale with k (it's undertraining, not rarity)
At a fixed 5K episodes the optimum becomes unreachable as k grows (k=6 @ 5K: 0 optima,
whole distribution capped at 0.805). Giving more episodes lifts the whole distribution
and the optimum becomes abundant again. So episodes(k) must grow.

## Pools in this directory (gamma=1.0)
| k | episodes | candidates | optfrac (>=0.99) | mean | notes |
|---|---|---|---|---|---|
| 5 | 5,000  | 4,000 | 6.4%  | 0.82  | complete |
| 6 | 25,000 | 2,400 | 25.3% | 0.968 | early-stopped (optfrac positive & stable; 607 optima — plenty) |
| 7 | TBD (~40K?) | pending | — | — | size from k=5->k=6 scaling (~3-4x per agent) |
| 8 | TBD (~60K?) | pending | — | — | — |

(Note: k=6 @ 5,000 episodes gave 0 optima — that pool was discarded; 25K is the real budget.)

## Reproduce / continue
- Build a pool:   `PY=<py> KS="7" Q=4000 EPISODES=40000 CHUNK=50 NCORES=16 bash remote/run_abstract.sh`
  (resilient: resumes from checkpoint, supervises crashes — see remote/run_abstract.sh)
- Analyze + figure: `python remote/abstract_analyze.py --dir results_abstract --ks 5,6,7,8`
- `abs_k{k}_*.pkl` hold per-candidate (Gbar,Ubar,Phi) stats + ratios + opt + episodes.

## Why this lives here / machine note
Built on the flaky i9-14900K box (see HARDWARE_FAULT_REPORT). k=6's 25K-episode pool is
~5h there; k=7/k=8 at higher episodes are many hours each — prefer a stable machine for
the heavy remainder. Pushing pools to git so they're pullable/safe against a box crash.
