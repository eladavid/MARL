"""Vectorized finite-horizon VI for the drone game (general n). Loops over the |A|^n
joint actions but vectorizes the Bellman backup over all |S|^n joint states with numpy.
Accumulates the max online (never materializes the full (S, A) table).

Validated against drone_game.vi at n=3 (must match Phi*). Enables n=4 (the richer
'4 drones, 1 charger, 3 missions' game) where the pure-Python VI is impractical.
"""
import itertools, time, numpy as np
import drone_game as dg


def vi_vec(n, s0, verbose=False):
    R, M, NODES, B, BATT = dg.R, dg.M, dg.NODES, dg.B, dg.BATT
    T, GAMMA = dg.T, dg.GAMMA
    Rmask = np.array([1 if v in R else 0 for v in range(NODES)], bool)
    Mmask = np.array([1 if v in M else 0 for v in range(NODES)], bool)
    nM = len(M); S = (NODES * BATT) ** n

    # decode every joint state -> per-agent node[S,n], batt[S,n]
    locs = np.arange(NODES * BATT)
    loc_node = locs // BATT; loc_batt = locs % BATT          # per-agent local decode
    # joint index = sum_i local_i * (NODES*BATT)^i
    base = NODES * BATT
    idxs = np.arange(S)
    node = np.empty((S, n), np.int64); batt = np.empty((S, n), np.int64)
    tmp = idxs.copy()
    for i in range(n):
        li = tmp % base; tmp //= base
        node[:, i] = loc_node[li]; batt[:, i] = loc_batt[li]

    actions = list(itertools.product(range(NODES), repeat=n))
    FR = dg.FROZEN_RELOAD

    # precompute, per action, the realized reward Phi_a[S] and next-state index NX_a[S]
    def reward_and_next(a):
        a = np.array(a)                                       # (n,)
        empty = (batt == 0)                                   # (S,n)
        ea = np.where(empty, FR, a[None, :])                  # executed action (frozen reload at b=0)
        ea_isR = Rmask[ea]                                    # (S,n)
        nb = np.where(ea_isR, B, np.clip(batt - 1, 0, None))  # next battery
        # g(ea): coverage of distinct missions - congestion (stacking)
        onehot = (ea[:, :, None] == np.arange(NODES)[None, None, :])  # (S,n,NODES)
        counts = onehot.sum(1)                                # (S,NODES)
        coverage = (counts[:, Mmask] > 0).sum(1)              # (S,)
        cong = np.clip(counts - 1, 0, None).sum(1)            # (S,)
        g = dg.W_COV * coverage / nM - dg.W_CONG * cong
        u = -dg.C_BATT * (1 - nb / B)                         # (S,n)
        Phi_a = g + u.sum(1)                                  # (S,)
        # next joint index from (ea node, nb)
        loc = ea * BATT + nb                                  # (S,n) per-agent next local idx
        nx = np.zeros(S, np.int64); mult = 1
        for i in range(n):
            nx += loc[:, i] * mult; mult *= base
        return Phi_a, nx

    PA = np.empty((len(actions), S), np.float32)
    NX = np.empty((len(actions), S), np.int64)
    t0 = time.time()
    for ai, a in enumerate(actions):
        PA[ai], NX[ai] = reward_and_next(a)
    if verbose: print(f"  precompute {len(actions)} actions: {time.time()-t0:.1f}s", flush=True)

    V = np.zeros(S, np.float32); argA = np.empty((T, S), np.int32)
    for t in range(T - 1, -1, -1):
        best = np.full(S, -1e18, np.float32); barg = np.zeros(S, np.int32)
        for ai in range(len(actions)):
            q = PA[ai] + GAMMA * V[NX[ai]]
            m = q > best; best = np.where(m, q, best); barg = np.where(m, ai, barg)
        V = best; argA[t] = barg
    # roll out optimal trajectory from s0
    s0idx = 0; mult = 1
    for i in range(n):
        s0idx += (s0[i][0] * BATT + s0[i][1]) * mult; mult *= base
    s = s0idx; traj = []
    for t in range(T):
        a = actions[argA[t, s]]
        ea = tuple(dg.frozen_action(batt[s, i], a[i]) for i in range(n))
        traj.append(ea); s = int(NX[argA[t, s], s])
    return float(V[s0idx]), traj


if __name__ == "__main__":
    # 1) validate against the slow pure-Python VI at n=3 (current config)
    n = 3; s0 = tuple([(0, dg.B)] * n)
    t0 = time.time(); star_v, traj_v = vi_vec(n, s0, verbose=True); tv = time.time() - t0
    star_s, _ = dg.vi(n, s0)
    print(f"VALIDATE n=3: vec Phi*={star_v:.4f}  slow Phi*={star_s:.4f}  "
          f"match={abs(star_v-star_s)<1e-3}  (vec {tv:.1f}s)")
