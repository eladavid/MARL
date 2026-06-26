"""Build + cache the k=5 graded pool ONCE (sim_figs/pool_stats_k5.pkl). All k=5 figure
scripts replay this cache (instant). Also logs per-candidate POTENTIAL TRAJECTORIES for
the fig_psga-style 'individual runs trap-to-trap' panel."""
import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
from scaled_pool import rollout as prollout, _strides
import scaled_pool
from graded_game import make_g_torch, closed_form, C as MOVE_C, W as COV_W

K, RHO, Q, EP, H = 5, 0.7, 250, 5000, 1
T, GAMMA = 16, 0.99
R_MAX = COV_W + MOVE_C
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_figs")
os.makedirs(HERE, exist_ok=True)
scaled_pool.g_gen = make_g_torch(RHO)

def main():
    m = K; strides_t = torch.tensor(_strides([m + 1] * H + [m])); S = (m + 1) ** H * m
    opt = closed_form(K, RHO); t0 = time.time()
    torch.manual_seed(0)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(m)).sample((Q, K, S)))
    optim = torch.optim.SGD([P], lr=1e-3)
    traj = np.zeros((Q, EP // 100))          # logged discounted-Phi/opt every 100 episodes
    for ep in range(EP):
        logps, rews, _ = prollout(P, Q, K, m, H, 32, 0.1, strides_t)
        rets = disc_returns(rews.detach(), GAMMA); adv = rets - rets.mean(dim=3, keepdim=True)
        loss = -(logps * adv).sum() / 32
        optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            P.copy_(project_onto_simplex(P.view(Q * K * S, m)).view(Q, K, S, m))
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                _, _, pots = prollout(P, Q, K, m, H, 1, 0.1, strides_t, greedy=True)
                traj[:, ep // 100] = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy() / opt
            print(f"  ep {ep+1}/{EP}  mean@now={traj[:, ep//100].mean():.3f} "
                  f"opt-frac={np.mean(traj[:, ep//100]>=0.99):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    # final hardened stats (Gbar, Ubar_vec, Phi) normalized as eval_hardened
    with torch.no_grad():
        _, rews, pots = prollout(P, Q, K, m, H, 1, 0.1, strides_t, greedy=True)
        w = (GAMMA ** torch.arange(T)).view(T, 1)
        rt = rews.squeeze(-1); pt = pots.squeeze(-1)
        g_t = (rt.sum(2) - pt) / (K - 1); u_t = rt - g_t.unsqueeze(2)
        G = (g_t * w).sum(0).numpy(); U = (u_t * w.unsqueeze(2)).sum(0).numpy()
        Phi = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy()
    norm = T * R_MAX
    stats = [(float(G[q] / norm), (U[q] / norm), float(Phi[q])) for q in range(Q)]
    pickle.dump(stats, open(os.path.join(HERE, "pool_stats_k5.pkl"), "wb"))
    pickle.dump({"traj": traj, "opt": opt, "ratios": (Phi / opt).tolist()},
                open(os.path.join(HERE, "pool_traj_k5.pkl"), "wb"))
    print(f"\nSAVED pool_stats_k5.pkl + pool_traj_k5.pkl | optfrac={np.mean(Phi/opt>=0.99):.3f} "
          f"mean={np.mean(Phi/opt):.3f} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
