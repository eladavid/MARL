"""Tunable-sharpness coverage game (widen the optimal basin).

g(a) = W * [ (1-rho) * coverage/m  +  rho * 1[full coverage] ]
  coverage = # distinct occupied slots.  rho=1 -> hard cliff (current needle);
  rho=0 -> fully graded ramp (wide basin). u = move-reward (unchanged).
Optimum = full-coverage rotation -> Phi*/step = W + n*C (closed form, rho-independent,
as long as full coverage is reachable). This module: numpy refs (VI, closed form,
best-static, H0 enumeration) + torch g for training, all parameterized by rho."""
import itertools, numpy as np, torch
from scale_cong import GAMMA, T
W, C = 10.0, 3.0

# ---------- numpy references ----------
def g_np(action, n, m, rho):
    a = np.asarray(action); cov = len(set(a.tolist()))
    return W * ((1 - rho) * cov / m + rho * (1.0 if cov == m else 0.0))

def phi_np(state, action, n, m, rho):
    u = C * np.sum(np.asarray(state) != np.asarray(action))
    return g_np(action, n, m, rho) + u

def closed_form(n, rho):
    """full-coverage rotation from collided start: 1 transient + W + n*C per step."""
    m = n; s0 = tuple([0] * n); perm = tuple(range(n)); cur = perm; traj = [perm]
    for t in range(1, T):
        cur = tuple((x + 1) % n for x in cur); traj.append(cur)
    s = s0; V = 0.0
    for t, a in enumerate(traj):
        V += (GAMMA ** t) * phi_np(s, a, n, m, rho); s = tuple(a)
    return V

def vi(n, rho):
    m = n; states = list(itertools.product(range(m), repeat=n)); idx = {s: k for k, s in enumerate(states)}
    Phi = np.array([[phi_np(s, a, n, m, rho) for a in states] for s in states])
    V = np.zeros(len(states)); argA = np.zeros((T, len(states)), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = Phi + GAMMA * V[np.newaxis, :]; argA[t] = Q.argmax(1); V = Q.max(1)
    s0 = tuple([0] * n); s = s0; traj = []
    for t in range(T):
        a = states[argA[t, idx[s]]]; traj.append(a); s = a
    return V[idx[s0]], traj

def best_static(n, rho):
    m = n; s0 = tuple([0] * n); best = -1e9
    for a in itertools.product(range(m), repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            V += (GAMMA ** t) * phi_np(s, a, n, m, rho); s = a
        best = max(best, V)
    return best

def enum_H0(n, rho):
    m = n; s0 = tuple([0] * n); per = list(itertools.product(range(m), repeat=m)); best = -1e9
    for prof in itertools.product(per, repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(prof[i][s[i]] for i in range(n)); V += (GAMMA ** t) * phi_np(s, a, n, m, rho); s = a
        best = max(best, V)
    return best

# ---------- torch g for training (matches g_np) ----------
def make_g_torch(rho):
    def g_t(actions, N, A):
        counts = torch.nn.functional.one_hot(actions, A).sum(1).float()   # (B,A)
        cov = (counts > 0).sum(1).float()                                 # (B,) distinct slots
        full = (cov == A).float()
        return W * ((1 - rho) * cov / A + rho * full)
    return g_t

if __name__ == "__main__":
    print(f"graded coverage game (W={W}, C={C}): structure vs sharpness rho\n")
    for n in [3, 4]:
        print(f"--- k={n} ---")
        for rho in [0.0, 0.3, 0.6, 1.0]:
            star, traj = vi(n, rho); cf = closed_form(n, rho); stat = best_static(n, rho)
            h0 = enum_H0(n, rho)
            period = "static" if traj[-1] == traj[-2] else f"cyc{sum(1 for _ in [1])}"
            cyc = traj[-1] != traj[-2]
            print(f"  rho={rho:.1f}: Phi*={star:7.2f} cf={cf:7.2f} | "
                  f"trap_ratio={stat/star:.3f}  H0gap={1-h0/star:.3f}  "
                  f"{'CYCLIC' if cyc else 'STATIC'} opt (tail {traj[-2:]})")
        print()
