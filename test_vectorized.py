"""Validity tests for vectorized_train.py vs the original sequential pipeline.

T1  batched g/u  == scalar g_func/make_u_i               (exact)
T2  vectorized REINFORCE gradient == transparent naive-loop reference, on identical
    actions, recomputing log-probs from the SAME policies                 (exact, autograd)
T3  end-to-end: vectorized train reaches the optimum at the same rate as the original
    env-based train(), and is faster                                       (statistical)

Run:  python test_vectorized.py
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np, torch
from congestion_game.policies import DirectTabularPolicy
from congestion_game.reward_functions import g_func, make_u_i
from meta_algorithm import make_env, eval_hardened
from parallel_simulation import train
import vectorized_train as V

N, S, A, H, T = 3, 3, 3, 1, 16
GAMMA, LR = 0.99, 1e-3


def t1_rewards():
    print("[T1] batched rewards == scalar rewards ...", flush=True)
    torch.manual_seed(0)
    u = make_u_i(S)
    maxbad = 0.0
    for _ in range(200):
        acts = torch.randint(0, A, (N,))
        states = torch.randint(0, S, (N,))
        g_s = float(g_func(acts, N))
        g_b = float(V.g_batched(acts.unsqueeze(0), N, A)[0])
        maxbad = max(maxbad, abs(g_s - g_b))
        for i in range(N):
            us = float(u(int(states[i]), int(acts[i])))
            ub = float(V.u_batched(states[i:i+1], acts[i:i+1], S)[0])
            maxbad = max(maxbad, abs(us - ub))
    assert maxbad < 1e-6, f"reward mismatch {maxbad}"
    print(f"     OK (max abs diff {maxbad:.2e})", flush=True)


def _naive_loss(policies, init, fixed_actions, B, use_baseline=True):
    """Transparent reference: per-rollout reward-to-go REINFORCE, mean baseline,
    log-probs recomputed from `policies`, ORIGINAL g_func/make_u_i rewards."""
    u = make_u_i(S)
    all_lp, all_ret = [[] for _ in range(N)], [[] for _ in range(N)]
    for b in range(B):
        state = [int(s) for s in init]; buf = [[0] * H for _ in range(N)]
        lp_bt, rew_bt = [[] for _ in range(N)], [[] for _ in range(N)]
        for t in range(T):
            acts = []
            for i in range(N):
                aug = torch.tensor(buf[i] + [state[i]], dtype=torch.long)
                probs = policies[i](aug)
                a = fixed_actions[t, b, i]
                lp_bt[i].append(torch.distributions.Categorical(probs).log_prob(a))
                acts.append(int(a))
            g = float(g_func(torch.tensor(acts), N))
            for i in range(N):
                rew_bt[i].append(g + float(u(state[i], acts[i])))
            for i in range(N):
                buf[i] = (buf[i] + [state[i]])[-H:] if H > 0 else []
                state[i] = acts[i]
        for i in range(N):
            all_ret[i].append(V.disc_returns(torch.tensor(rew_bt[i]), GAMMA))
            all_lp[i].append(torch.stack(lp_bt[i]))
    loss = 0.0
    for i in range(N):
        base = torch.stack(all_ret[i]).mean(0) if use_baseline else 0.0
        li = 0.0
        for lp, R in zip(all_lp[i], all_ret[i]):
            li = li + (-(lp * (R - base)).sum())
        loss = loss + li / B
    return loss


def t2_gradient():
    print("[T2] vectorized gradient == naive-loop reference (identical actions) ...", flush=True)
    B = 16
    torch.manual_seed(1)
    init = (1, 2, 2)
    pols = [DirectTabularPolicy([S] * (H + 1), A) for _ in range(N)]
    fixed = torch.randint(0, A, (T, B, N))               # identical actions for both paths

    # vectorized grad
    for p in pols: p.zero_grad(set_to_none=True)
    logps, rews, _ = V.vec_rollout(pols, init, S, A, H, T, B, fixed_actions=fixed)
    V.reinforce_loss(logps, rews, GAMMA, B, use_baseline=True).backward()
    g_vec = [p.probs_table.grad.clone() for p in pols]

    # naive grad
    for p in pols: p.zero_grad(set_to_none=True)
    _naive_loss(pols, init, fixed, B, use_baseline=True).backward()
    g_naive = [p.probs_table.grad.clone() for p in pols]

    worst = max(float((gv - gn).abs().max()) for gv, gn in zip(g_vec, g_naive))
    assert worst < 1e-4, f"gradient mismatch {worst}"
    print(f"     OK (max abs grad diff {worst:.2e})", flush=True)


def _atopt_original(seeds, episodes, batch, init):
    out, t0 = [], time.time()
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        ecg = make_env(init)
        train(ecg, episodes, batch_size=batch, use_baseline=True, learning_rate=LR, grow_batch=False)
        out.append(eval_hardened(ecg)[2])
    return out, time.time() - t0


def _atopt_vectorized(seeds, episodes, batch, init):
    out, t0 = [], time.time()
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        ecg = make_env(init)
        pols = [ag.policy_func for ag in ecg.agents]
        V.train_vectorized(pols, init, episodes, batch, S, A, H, T, GAMMA, LR, True)
        out.append(eval_hardened(ecg)[2])
    return out, time.time() - t0


def t3_end_to_end():
    print("[T3] end-to-end: optimum-rate match + speed (this runs the slow original) ...", flush=True)
    init = (1, 2, 2); seeds = list(range(10)); episodes, batch = 300, 16
    opt = __import__("meta_algorithm").optimal_phi(init)
    o, to = _atopt_original(seeds, episodes, batch, init)
    v, tv = _atopt_vectorized(seeds, episodes, batch, init)
    fo = np.mean([x / opt >= 0.99 for x in o]); fv = np.mean([x / opt >= 0.99 for x in v])
    mo, mv = np.mean([x / opt for x in o]), np.mean([x / opt for x in v])
    print(f"     original : @opt {fo:.2f}  mean ratio {mo:.3f}  ({to:.0f}s for {len(seeds)} runs)", flush=True)
    print(f"     vectorized: @opt {fv:.2f}  mean ratio {mv:.3f}  ({tv:.0f}s)  speedup x{to/max(tv,1e-9):.1f}", flush=True)
    # statistical equivalence to the original (M=10): match @opt fraction and mean ratio
    assert abs(fv - fo) <= 0.15, f"@opt fractions diverge: orig {fo:.2f} vs vec {fv:.2f}"
    assert abs(mv - mo) <= 0.15, f"mean-ratio drift: orig {mo:.3f} vs vec {mv:.3f}"
    assert tv < to, f"vectorized should be faster (orig {to:.0f}s vs vec {tv:.0f}s)"
    print("     OK (optimum-rate matches the original, and faster)", flush=True)


if __name__ == "__main__":
    t1_rewards()
    t2_gradient()
    t3_end_to_end()
    print("\nALL TESTS PASSED", flush=True)
