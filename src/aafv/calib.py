"""P7 calibrators (monotonic in the score) and the P8 guard function. No torch."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
EPS = 1e-6

class Calibrator:
    """kind: 'platt' (on S), 'isotonic' (on S), 'platt_g' (on S and g)."""
    def __init__(self, kind): self.kind = kind
    def fit(self, S, y, g=None):
        if self.kind == "isotonic": self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(S, y)
        else: self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(S, g), y)
        return self
    def _x(self, S, g): return np.c_[S, g] if self.kind == "platt_g" else np.asarray(S).reshape(-1, 1)
    def predict(self, S, g=None):
        p = self.m.predict(np.asarray(S)) if self.kind == "isotonic" else self.m.predict_proba(self._x(S, g))[:, 1]
        return np.clip(p, 0.0, 1.0)

def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS); y = np.asarray(y); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def guard_pass(Q, R_sam, avail_sam, P_dir, P_sam, q_min, r_min, rho):
    """Q >= Q_MIN; direct branch P_d >= RHO; if SAM is available: R_sam >= R_MIN and P_s >= RHO."""
    sam_ok = (avail_sam == 0) | ((R_sam >= r_min) & (P_sam >= rho))
    return (Q >= q_min) & (P_dir >= rho) & sam_ok
