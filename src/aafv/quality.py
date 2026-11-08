"""P1.3 image-quality measures (C2) on the aligned crop. Raw measures first; normalization constants are fitted on the train split only."""
import numpy as np, cv2

MEASURES = ["sharp", "illum", "contrast", "noise", "detail"]

def raw_measures(rgb, face_px=None):
    """rgb uint8 HxWx3. Returns dict of raw numbers.
    sharp = variance of the Laplacian; illum = 1 - |mean - 128| / 128; contrast = std of gray;
    noise = Immerkaer (1996) sigma estimate (lower is better); detail = side of the detected face box in source pixels (resolution)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = g.shape
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    illum = 1.0 - abs(g.mean() - 128.0) / 128.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(g, -1, k)[1:-1, 1:-1]
    noise = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
    return {"sharp": float(lap), "illum": float(illum), "contrast": float(g.std()), "noise": float(noise),
            "detail": float(face_px if face_px is not None else min(h, w))}

def fit_norm(df_train, lo_pct=5, hi_pct=95):
    """Per-measure (lo, hi) from the train split; returns {measure: [lo, hi]}."""
    return {m: [float(np.percentile(df_train[m], lo_pct)), float(np.percentile(df_train[m], hi_pct))] for m in MEASURES}

def normalize(df, norm):
    """Map each measure to 0..1 (1 = good). Noise is inverted. Q = mean of the five."""
    out = df.copy()
    for m in MEASURES:
        lo, hi = norm[m]
        z = np.clip((df[m] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        out["q_" + m] = 1.0 - z if m == "noise" else z
    out["Q"] = out[["q_" + m for m in MEASURES]].mean(axis=1)
    return out
