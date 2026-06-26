"""Mission 1: does a REST-IN-PLACE u_i change the optimum / make traps stickier?

Runs the validated vectorized MAC-REINFORCE (PSGA) from a fixed trap-ish init over
many seeds, under two private rewards:
  - ORIGINAL  u: s==0 stay, else move  (move-rewarding -> optimum is a CYCLE, opt/step=15)
  - REST       u: reward a==s for all s (stay/min-energy -> optimum is STATIC, opt/step=19)
For each run we harden (argmax) and measure mean Phi/step of the deterministic rollout,
normalized by the per-step optimum. Reach-rate = fraction reaching the optimum.
"""
import sys, os, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np, torch
import vectorized_train as vt
from vectorized_train import train_vectorized
from meta_algorithm import make_env, randomize, S, A, T, GAMMA

INIT = (1, 2, 2); H = 1; M = 40; EPISODES = 800; BATCH = 32; LR = 1e-3
_orig_u = vt.u_batched

def u_rest_batched(s, a, Sdim):
    s = s.float(); a = a.float(); d = (s - a).abs() / Sdim
    return 3.0 * (1.0 - d)            # rewards a==s everywhere (stay)

def greedy_phi(pols, init):
    curr = torch.stack([torch.full((1,), int(x), dtype=torch.long) for x in init])  # (N,1)
    buf = torch.zeros(len(pols), 1, H, dtype=torch.long); tot = 0.0
    with torch.no_grad():
        for _ in range(T):
            acts = []
            for i, p in enumerate(pols):
                aug = torch.cat([buf[i], curr[i].unsqueeze(1)], dim=1)
                probs = p(aug).view(-1, A)                 # -> (1,A) robustly
                acts.append(torch.argmax(probs, dim=1))
            actions = torch.stack(acts, dim=1)                      # (1,N)
            g = vt.g_batched(actions, len(pols), A)
            us = sum(vt.u_batched(curr[i], actions[:, i], S) for i in range(len(pols)))
            tot += float((g + us)[0])
            buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)
            curr = actions.t().contiguous()
    return tot / T

def run(label, opt_step):
    ratios = []
    for seed in range(M):
        pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        ecg = make_env(INIT, H=H); randomize(ecg)
        pols = [ag.policy_func for ag in ecg.agents]
        train_vectorized(pols, INIT, EPISODES, BATCH, S, A, H, T, GAMMA, LR, True)
        ratios.append(greedy_phi(pols, INIT) / opt_step)
    r = np.array(ratios)
    print(f"{label:24s}: reach(>=0.99)={int((r>=0.99).sum()):2d}/{M}  "
          f"mean={r.mean():.3f}  median={np.median(r):.3f}  "
          f"stalled(<0.95)={int((r<0.95).sum()):2d}/{M}", flush=True)
    return r

if __name__ == "__main__":
    print(f"init {INIT}, {M} seeds, {EPISODES} ep, batch {BATCH}\n")
    vt.u_batched = _orig_u
    run("ORIGINAL u (move)", 15.0)
    vt.u_batched = u_rest_batched
    run("REST-IN-PLACE u (stay)", 19.0)
