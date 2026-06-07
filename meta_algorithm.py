"""Stochastically-Stable Nash Hopping (paper Algorithm 2), implemented fresh from
the .tex spec (Def 11 + Alg 2). NOT using policies.py's decision logic.

Distributed two-stage selection:
  each agent i draws z_i ~ Bernoulli(p_i^beta),  p_i^beta = exp((1/beta)(Delta_i - off)),
  Delta_i = (1/N)(Gbar_b - Gbar_a) + (Ubar_b[i] - Ubar_a[i]),  adopt iff all z_i = 1.
  off = (1 + 1/N) is the faithful per-agent offset (guarantees p_i<=1).
  off = 0 ("reduced resistance", clamp p_i<=1) is the analysis path: same stochastically
  -stable set, but the chain actually mixes at small beta (the common (N+1) that would
  otherwise freeze joint adoption only ever appears as the PRODUCT of the N decisions).
"""
import sys, os, itertools, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))

import numpy as np
import torch
from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.episodic_congestion_game import EpisodicCongestionGame
from congestion_game.policies import DirectTabularPolicy
from congestion_game.reward_functions import g_func, make_u_i, make_potential_func
from parallel_simulation import train, find_joint_optimum

N, S, A = 3, 3, 3
T, GAMMA, R_MAX = 16, 0.99, 14.0          # R_MAX so Gbar,Ubar in (0,1); doesn't change argmax
EXPLORE = 0.1                              # alpha-greedy inside the MAC-REINFORCE subroutine


def make_env(init_states, H=1, exploration_rate=EXPLORE):
    agents = []
    for i in range(N):
        pol = DirectTabularPolicy([S] * (H + 1), A, exploration_rate=exploration_rate)
        agents.append(EpisodicAgent(S, A, policy_func=pol, init_state=init_states[i]))
    return EpisodicCongestionGame(agents, A, g_func,
                                  [make_u_i(S) for _ in range(N)], H, T, False, GAMMA)


def eval_hardened(ecg):
    """Discounted Gbar, [Ubar_i], Phi(disc) of the HARDENED (argmax) policy from s0."""
    ecg.reset()
    _, _, rewards, potentials = ecg.do_episode(is_inference=True)
    G = 0.0; U = [0.0] * N; Phi = 0.0; disc = 1.0
    for t in range(len(potentials)):
        pot = float(potentials[t])
        r = [float(rewards[t][i]) for i in range(N)]
        g_t = (sum(r) - pot) / (N - 1)            # rewards[i]=g+u_i, pot=g+sum(u)
        G += disc * g_t
        for i in range(N): U[i] += disc * (r[i] - g_t)
        Phi += disc * pot
        disc *= GAMMA
    Gbar = G / (T * R_MAX); Ubar = [u / (T * R_MAX) for u in U]
    return Gbar, Ubar, Phi


def accepts(stat_b, stat_a, beta, reduced):
    """All-N distributed accept test of candidate b over incumbent a."""
    Gb, Ub, _ = stat_b; Ga, Ua, _ = stat_a
    off = 0.0 if reduced else (1.0 + 1.0 / N)
    for i in range(N):
        Delta_i = (Gb - Ga) / N + (Ub[i] - Ua[i])
        p_i = np.exp((Delta_i - off) / beta)
        if reduced: p_i = min(1.0, p_i)
        if pyrandom.random() >= p_i:           # z_i = 0  -> reject (need ALL accept)
            return False
    return True


def randomize(ecg):
    for ag in ecg.agents: ag.policy_func.reset_parameters()   # theta ~ U[Theta] (Dirichlet)

def snapshot(ecg): return [ag.get_params() for ag in ecg.agents]
def restore(ecg, snap):
    for ag, p in zip(ecg.agents, snap): ag.set_params(p)


def optimal_phi(init_states):
    Phi = make_potential_func(S)
    opt = find_joint_optimum(N, S, A, Phi, gamma=GAMMA)
    s = tuple(init_states); tot = 0.0; disc = 1.0
    for _ in range(T):
        a = tuple(int(x) for x in opt[s]); tot += disc * float(Phi(torch.tensor(s), torch.tensor(a)))
        disc *= GAMMA; s = a
    return tot


def run_hopping(init_states, beta, K, sub_episodes=120, sub_batch=16, lr=1e-3,
                reduced=True, seed=0, verbose=False, log_dir=None):
    pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    ecg = make_env(init_states)
    opt_phi = optimal_phi(init_states)
    gstep = 0   # cumulative subroutine training episodes = shared x-axis

    def log_hop(inc_r, prop_r, acc, early_stop, psga_ran):
        if writer:
            writer.add_scalar("incumbent/hard_ratio", inc_r, gstep)
            if prop_r is not None:
                writer.add_scalar("proposal/hard_ratio", prop_r, gstep)
            writer.add_scalar("accepted", 1.0 if acc else 0.0, gstep)
            writer.add_scalar("events/early_stop_jump", 1.0 if early_stop else 0.0, gstep)
            writer.add_scalar("events/psga_ran", 1.0 if psga_ran else 0.0, gstep)
            writer.add_scalar("optimum/ref", 1.0, gstep); writer.flush()

    # initial incumbent: MAC-REINFORCE from a random start, harden
    randomize(ecg)
    train(ecg, sub_episodes, batch_size=sub_batch, use_baseline=True, learning_rate=lr, grow_batch=False,
          tb_writer=writer, tb_tag="pg/soft_ratio", tb_step0=gstep, tb_norm=opt_phi)
    gstep += sub_episodes
    inc_stat = eval_hardened(ecg)
    ratios = [inc_stat[2] / opt_phi]
    log_hop(ratios[0], None, True, early_stop=False, psga_ran=True)   # initial incumbent = a PSGA run

    for k in range(1, K + 1):
        randomize(ecg); theta_rand = snapshot(ecg)      # theta_rand ~ U[Theta]
        jump_stat = eval_hardened(ecg)                  # pi_jump = harden(theta_rand)
        accepted, stage, prop_ratio = False, "none", jump_stat[2] / opt_phi
        jump_accepted = accepts(jump_stat, inc_stat, beta, reduced)
        if jump_accepted:                                # STAGE 2: hardening early-stop (no PSGA)
            inc_stat = jump_stat; accepted, stage = True, "jump"
            gstep += 1
        else:                                            # STAGE 3: PSGA (MAC-REINFORCE) runs
            restore(ecg, theta_rand)
            train(ecg, sub_episodes, batch_size=sub_batch, use_baseline=True, learning_rate=lr,
                  grow_batch=False, tb_writer=writer, tb_tag="pg/soft_ratio", tb_step0=gstep, tb_norm=opt_phi)
            gstep += sub_episodes
            cand_stat = eval_hardened(ecg); prop_ratio = cand_stat[2] / opt_phi
            if accepts(cand_stat, inc_stat, beta, reduced):
                inc_stat = cand_stat; accepted, stage = True, "candidate"
        ratios.append(inc_stat[2] / opt_phi)
        log_hop(ratios[-1], prop_ratio, accepted, early_stop=jump_accepted, psga_ran=not jump_accepted)
        if verbose and (k % 5 == 0 or k == K):
            print(f"  epoch {k:3d} (step {gstep}): incumbent {ratios[-1]:.3f} | proposal {prop_ratio:.3f} "
                  f"| {'ACCEPT('+stage+')' if accepted else 'reject'}", flush=True)
    if writer: writer.close()
    return ratios, opt_phi


if __name__ == "__main__":
    # quick validation: trap-prone init, reduced resistance, small beta -> should climb to optimum
    init = (1, 2, 2)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                           f"hop_init{''.join(map(str,init))}_beta0.05_seed1")
    print(f"init {init} | optimal Phi = {optimal_phi(init):.2f} | logging to {log_dir}", flush=True)
    ratios, opt = run_hopping(init, beta=0.05, K=40, reduced=True, seed=1, verbose=True, log_dir=log_dir)
    print(f"\nstart ratio {ratios[0]:.3f} -> final ratio {ratios[-1]:.3f} "
          f"| max {max(ratios):.3f} | frac@opt(>=0.99) {np.mean([r>=0.99 for r in ratios]):.2f}", flush=True)
