import numpy as np

def margin(imp_sorted_desc):
    v = np.asarray(imp_sorted_desc)
    spread = v[0] - v[min(4, len(v) - 1)]
    return max(0.001, 0.5 * spread)

def thresholds(p_impostor_pass, p_same):
    """T_accept above the highest passing impostor; T_reject below the lowest same-person P."""
    imp = np.sort(np.asarray(p_impostor_pass))[::-1]
    t_acc = float(imp[0] + margin(imp))
    same = np.sort(np.asarray(p_same))
    t_rej = float(same[0] - margin(same[::-1]))
    warn = None
    if t_rej >= t_acc:  # lowest same exceeds highest impostor: perfect separation
        t_acc = t_rej = float((imp[0] + same[0]) / 2)
        warn = "WARNING: perfect separation on calibration; data may be too easy"
    return t_acc, t_rej, warn

def zone(p, t_acc, t_rej):
    return np.where(p >= t_acc, "accept", np.where(p < t_rej, "reject", "review"))
