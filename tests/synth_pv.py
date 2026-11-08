"""Synthetic pair_vectors for an end-to-end smoke test of P6-P9 (NOT real data). Usage: python tests/synth_pv.py OUTDIR"""
import sys, numpy as np, pandas as pd
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(0)
C = ["arc", "eff", "vgg", "lpips", "ssim"]
def make(split, n_same, n_imp, sets):
    rows = []; n = n_same + n_imp; y = np.r_[np.ones(n_same), np.zeros(n_imp)].astype(int)
    gap = rng.integers(0, 40, n); gb = pd.cut(gap, [-1, 5, 10, 15, 20, 30, 100], labels=["G1", "G2", "G3", "G4", "G5", "G6"]).astype(str)
    d = pd.DataFrame(dict(pair_id=[f"{split}{i}" for i in range(n)], label=y, gap=gap, gap_bin=gb, gender=rng.choice(["M", "F"], n),
        set_name=np.where(y == 1, "same", rng.choice(sets, n)), dataset="MORPH", img_id_1=[f"a{i}" for i in range(n)], img_id_2=[f"b{i}" for i in range(n)],
        person_id_1=[f"{split}p{i % (n // 5)}" for i in range(n)], person_id_2=[f"{split}q{i}" for i in range(n)]))
    d["weight"] = 1.0; d["g"] = np.minimum(1, gap / 80); d["Q"] = rng.uniform(0.2, 0.9, n); d["enhanced_pair"] = 0; d["K_used"] = 3
    av = (rng.random(n) > 0.02).astype(int); d["avail_sam"] = av; d["avail_diff"] = 0; d["avail_multi"] = av; d["R_sam"] = rng.uniform(0.2, 0.9, n) * av; d["R_diff"] = 0.0
    d["s_spread"] = 0.1; d["f_spread"] = 0.0
    for j, c in enumerate(C):
        d["d_" + c] = rng.normal(0, 1, n) + (1.6 - 0.02 * gap) * y; d["s_" + c] = (rng.normal(0, 1, n) + (1.3 * y)) * av; d["f_" + c] = 0.0
    return d
make("train_fit", 6000, 18000, ["train_diff_random", "train_diff_hard"]).to_csv(out / "pair_vectors_train_fit.csv", index=False)
make("train_val", 1000, 3000, ["train_diff_random"]).to_csv(out / "pair_vectors_train_val.csv", index=False)
make("calibration", 3000, 30000, ["D_random", "D_all_random", "D_hard"]).to_csv(out / "pair_vectors_calibration.csv", index=False)
make("test", 3000, 30000, ["D_random", "D_all_random", "D_hard"]).to_csv(out / "pair_vectors_test.csv", index=False)
print("synthetic written to", out)
