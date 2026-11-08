"""SSIM identical in definition to skimage.metrics.structural_similarity defaults (gray, 7x7 uniform window, sample covariance,
K1=0.01, K2=0.03, borders cropped), implemented with cv2 for speed and so it can run in worker processes."""
import numpy as np, cv2

def ssim_gray(x, y, data_range=255.0, win=7):
    x = x.astype(np.float64); y = y.astype(np.float64)
    f = lambda a: cv2.boxFilter(a, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)
    ux, uy = f(x), f(y)
    NP = win * win; cov_norm = NP / (NP - 1)
    vx = cov_norm * (f(x * x) - ux * ux); vy = cov_norm * (f(y * y) - uy * uy); vxy = cov_norm * (f(x * y) - ux * uy)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    p = (win - 1) // 2
    return float(S[p:-p, p:-p].mean())

def ssim_rgb(a, b):
    """a, b: uint8 RGB HxWx3."""
    return ssim_gray(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY))
