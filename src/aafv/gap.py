import math

def g(a1, a2, a_max=80):
    return min(1.0, abs(a1 - a2) / a_max)

def gap_bin(gap, edges=(0, 5, 10, 15, 20, 30)):
    gap = abs(gap)
    idx = 0
    for i, e in enumerate(edges):
        if gap >= e:
            idx = i
    return f"G{idx + 1}"

def grid(min_age, max_age, step=5):
    lo = int(math.floor(min_age / step) * step)
    hi = int(math.ceil(max_age / step) * step)
    return list(range(lo, hi + 1, step))

def snap(a, grid_ages):
    # nearest grid age; ties go to the higher age
    best = None
    for t in grid_ages:
        d = abs(t - a)
        if best is None or d < best[0] or (d == best[0] and t > best[1]):
            best = (d, t)
    return best[1]

def targets(a1, a2, grid_ages, k_max=7):
    assert a1 <= a2
    t_fwd, t_bwd = snap(a2, grid_ages), snap(a1, grid_ages)
    T = [t for t in grid_ages if a1 < t < a2]
    if len(T) > k_max:
        idx = [round(i * (len(T) - 1) / (k_max - 1)) for i in range(k_max)]
        T = [T[i] for i in idx]
    return t_fwd, t_bwd, T, len(T)
