"""Faithful Algorithm 2, made tractable by MAC-REINFORCE parallelization.

Candidate generation is incumbent-independent (theta_rand ~ U[Theta]), so we
generate the per-slot pairs IN PARALLEL and replay them through the two-stage
accept rule SEQUENTIALLY -- equal to running Algorithm 2 one epoch at a time, in
distribution, but cheap.

Each parallel slot k produces a pair:
  jump_k = harden(theta_rand_k)                       # early-stop candidate (no PSGA)
  cand_k = harden(MAC-REINFORCE(theta_rand_k))        # PSGA candidate
Both as (Gbar, [Ubar_i], Phi). Sequential replay: try jump (stage 2); if rejected,
run the PSGA candidate (stage 3/4).
"""
import sys, os, random as pyrandom, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
from meta_algorithm import make_env, eval_hardened, accepts, optimal_phi, randomize, N, S, A, T, GAMMA
from parallel_simulation import train
from vectorized_train import train_vectorized


def gen_candidate(args):
    """One parallel slot: random restart -> (jump_stat, psga_cand_stat).
    Uses the vectorized trainer by default (~16x faster, validated equal in
    distribution by test_vectorized.py); pass vectorized=False for the original."""
    seed, init, sub_episodes, sub_batch, lr = args[:5]
    vectorized = args[5] if len(args) > 5 else True
    pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ecg = make_env(init)
    randomize(ecg)                                   # theta_rand ~ U[Theta]
    jump = eval_hardened(ecg)                         # harden(theta_rand)
    if vectorized:
        pols = [ag.policy_func for ag in ecg.agents]  # MAC-REINFORCE from theta_rand (vectorized)
        train_vectorized(pols, tuple(init), sub_episodes, sub_batch,
                         S=S, A=A, H=1, T=T, gamma=GAMMA, lr=lr, use_baseline=True)
    else:
        train(ecg, sub_episodes, batch_size=sub_batch, use_baseline=True,
              learning_rate=lr, grow_batch=False)
    cand = eval_hardened(ecg)
    return jump, cand


def generate_pairs(init, K, sub_episodes, sub_batch, lr=1e-3, workers=None):
    args = [(s, init, sub_episodes, sub_batch, lr) for s in range(K)]
    pairs = []
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
        for r in ex.map(gen_candidate, args):
            pairs.append(r)
    return pairs


if __name__ == "__main__":
    # CALIBRATION: small parallel batch -> per-candidate optimal fractions + timing
    init = (1, 2, 2)
    K, sub_episodes, sub_batch = 24, 400, 32
    opt = optimal_phi(init)
    print(f"init {init} | optimal Phi {opt:.2f} | generating K={K} pairs "
          f"(sub_episodes={sub_episodes}, batch={sub_batch}) ...", flush=True)
    t0 = time.time()
    pairs = generate_pairs(init, K, sub_episodes, sub_batch)
    dt = time.time() - t0
    jr = [j[2] / opt for j, c in pairs]
    cr = [c[2] / opt for j, c in pairs]
    print(f"done in {dt:.0f}s ({dt/K:.1f}s/candidate parallelized)", flush=True)
    print(f"JUMP  (harden random)  : mean ratio {np.mean(jr):.3f} | @opt {np.mean([r>=0.99 for r in jr]):.2f} "
          f"| max {max(jr):.3f}", flush=True)
    print(f"PSGA  (after training) : mean ratio {np.mean(cr):.3f} | @opt {np.mean([r>=0.99 for r in cr]):.2f} "
          f"| max {max(cr):.3f}", flush=True)
    print(f"=> PSGA optimal fraction {np.mean([r>=0.99 for r in cr]):.2f} "
          f"(need >0 for the meta-algorithm to have something to select)", flush=True)
