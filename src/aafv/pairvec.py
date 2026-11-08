"""P5.2 - P5.6 aggregation (numpy, no GPU): from per-comparison scores to branch vectors, availability and reliability.
Shapes: N pairs, up to KMAX shared targets. Missing comparisons are masked, never imputed."""
import numpy as np
from .gap import targets

KMAX = 7

def target_arrays(a1, a2, grid_ages, k_max=KMAX):
    N = len(a1); t_fwd = np.zeros(N, int); t_bwd = np.zeros(N, int); T = np.zeros((N, k_max), int); K = np.zeros(N, int)
    for i in range(N):
        f, b, tt, k = targets(int(a1[i]), int(a2[i]), grid_ages, k_max)
        t_fwd[i], t_bwd[i], K[i] = f, b, k; T[i, :k] = tt
    return t_fwd, t_bwd, T, K

def fit_standardizer(S):
    """S: (n, 5) comparison scores of one branch over train_fit. Returns (mean, std)."""
    return S.mean(0), np.maximum(S.std(0), 1e-6)

def branch_vector(Sf, Sb, Ssh, mf, mb, msh, std):
    """Sf, Sb: (N,5) raw scores of fwd and bwd; Ssh: (N,K,5); m*: availability masks (N,), (N,), (N,K); std: (mean, sd).
    Returns x (N,5), avail (N,), K_used (N,), spread (N,) from s_arc (column 0), used_sh mask (N,K)."""
    mu, sd = std
    Zf, Zb, Zs = (Sf - mu) / sd, (Sb - mu) / sd, (Ssh - mu) / sd
    avail = mf & mb
    used = msh & avail[:, None]
    K_used = used.sum(1)
    base = 0.5 * (Zf + Zb)
    shared = (Zs * used[..., None]).sum(1) / np.maximum(K_used, 1)[:, None]
    x = np.where((K_used >= 1)[:, None], 0.5 * base + 0.5 * shared, base)
    x[~avail] = 0.0
    arcs = np.concatenate([Sf[:, :1], Sb[:, :1], Ssh[:, :, 0]], axis=1)                 # raw cosines
    ok = np.concatenate([avail[:, None], avail[:, None], used], axis=1)
    cnt = ok.sum(1)
    mean = (arcs * ok).sum(1) / np.maximum(cnt, 1)
    sd_ = np.sqrt((((arcs - mean[:, None]) ** 2) * ok).sum(1) / np.maximum(cnt, 1))      # population std
    spread = np.where(cnt >= 3, sd_, np.abs(arcs[:, 0] - arcs[:, 1]))
    spread[~avail] = 0.0
    return x, avail, K_used, spread, used

def r_target(arc_to_orig, est_age, t, bias_pair, q_gen, eps, w=(1 / 3, 1 / 3, 1 / 3)):
    Ri = np.maximum(0.0, arc_to_orig)
    e = est_age - bias_pair - t
    Ra = 1.0 - np.minimum(1.0, np.maximum(0.0, np.abs(e) - eps) / (2.0 * eps))
    return w[0] * Ri + w[1] * Ra + w[2] * q_gen

def reliability(Rt_fwd, Rt_bwd, Rt_sh1, Rt_sh2, used, avail, spread, sigma_max):
    """Rt_*: per generated face reliability (fwd: face1 at t_fwd, bwd: face2 at t_bwd, sh1/sh2: (N,K) shared). Returns R_branch (N,)."""
    k = used.sum(1)
    r1 = (Rt_fwd + (Rt_sh1 * used).sum(1)) / (1 + k)
    r2 = (Rt_bwd + (Rt_sh2 * used).sum(1)) / (1 + k)
    Rfaces = np.sqrt(np.maximum(r1, 0) * np.maximum(r2, 0))
    Rc = 1.0 - np.minimum(1.0, spread / sigma_max)
    R = 0.7 * Rfaces + 0.3 * Rc
    R[~avail] = 0.0
    return R
