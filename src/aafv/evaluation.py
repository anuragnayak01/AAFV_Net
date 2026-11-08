import numpy as np
from scipy.stats import beta

def zero_error_bound(n, alpha=0.05):
    """Upper bound on false-accept rate when 0 of n person-disjoint pairs were accepted."""
    return 1 - alpha ** (1.0 / n)

def clopper_pearson_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))
