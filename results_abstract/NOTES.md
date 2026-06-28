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
| 6 | 25,000 | 2,400 | 25.3% | 0.968 | early-stopped (607 optima) |
| 7 | 45,000 | 1,600 | 31%   | 0.852 | early-stopped (502 optima) |
| 8 | 65,000 | 400   | 42.5% | 0.934 | early-stopped (170 optima; ~440s/cand on flaky box -> 1600 would be ~6h, not worth it) |

**Scaling law (measured): episode budget is ADDITIVE, ~+20K per agent** — k=5:5K, k=6:25K,
k=7:45K, k=8:65K. The optimum is reached at every k once the budget clears its threshold;
optfrac stays healthy (6–31%) across k → Nash-hopping concentration scales with agents.
(Note: k=6 @ 5,000 episodes gave 0 optima — undertrained, discarded; 25K is the real budget.)

## Learning-curve probe (remote/abstract_curve.py) — IMPORTANT for how to REPORT
Ran k=5 to 20K episodes, Q=100, logging optfrac/mean every 2K (see `curve_k5_Q100.png/.json`):

| ep | 2K | 4K | 6K | 8K | 10K | 12K | 16K | 20K |
|---|---|---|---|---|---|---|---|---|
| optfrac | 0 | .03 | .12 | .33 | **.45** | .47 | .45 | .42 |
| mean    | .71 | .77 | .88 | .94 | .97 | .97 | .98 | .98 |

**The optfrac-rises-with-k trend (6%→25%→31%→42% across k=5..8) is a CONFOUND, fully
episode-driven — NOT a property of k.** Proof: k=5 *itself* climbs 6% (@5K) → ~45% (@10K),
i.e. it reaches the SAME plateau as k=8@65K. So when you train each k enough, optfrac is
roughly **k-independent (~42–47%)**. Every pool above was trained at a different point on
its S-curve, which is the whole reason optfrac looked like it varied with k.

Shape: a smooth **sigmoid** (slow start → rapid 4–12K transition → plateau ~12K). Threshold
(optfrac first >0) ≈ 4–5K for k=5; thresholds scale ~**+12K/agent** (gentler than the +20K
plateau-ish budget we used — i.e. k=6/7/8 were over-trained vs what the meta needs).

### Reporting implications
- DON'T claim "optimum-rate increases with k" — it's the budget, not k.
- Frame the result as: *with budget scaled to k, Nash-hopping concentrates at every k.*
- The meta only needs optfrac>0 (drone worked at p=4e-5), so train MINIMALLY (to threshold)
  + cheap meta-replay. Training to the plateau wastes compute — the meta exists to avoid that.
- For a clean nu^beta(k) figure, either document per-k budget in the caption, or re-train each
  k to a matched operating point (its threshold).

## Reproduce / continue
- Build a pool:   `PY=<py> KS="7" Q=4000 EPISODES=40000 CHUNK=50 NCORES=16 bash remote/run_abstract.sh`
  (resilient: resumes from checkpoint, supervises crashes — see remote/run_abstract.sh)
- Analyze + figure: `python remote/abstract_analyze.py --dir results_abstract --ks 5,6,7,8`
- `abs_k{k}_*.pkl` hold per-candidate (Gbar,Ubar,Phi) stats + ratios + opt + episodes.

## Why this lives here / machine note
Built on the flaky i9-14900K box (see HARDWARE_FAULT_REPORT). k=6's 25K-episode pool is
~5h there; k=7/k=8 at higher episodes are many hours each — prefer a stable machine for
the heavy remainder. Pushing pools to git so they're pullable/safe against a box crash.
