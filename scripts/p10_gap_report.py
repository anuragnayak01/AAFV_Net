"""Gap-stratified comparison of the frozen ablation variants on the CALIBRATION split (no test data).
For each age-gap bin: AUC and true-accept rate at a fixed false-accept rate, same-person pairs vs different-person pairs (D_all_random + D_hard) of the SAME bin.
Pair-level bootstrap 95% intervals (same-person pairs are correlated within a person, so the intervals are optimistic).
  python scripts/p10_gap_report.py --p5q artifacts/p5q --out artifacts/ablations [--variants full,arcface_only,...]"""
import sys, argparse, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd, torch
import p10_ablations as A
import __main__; __main__.Scorer = A.Scorer          # the frozen pickles were written by p10_ablations running as __main__

ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--shm", default="/dev/shm/aafv/abl")
ap.add_argument("--variants", default="full,arcface_only,direct5_logreg,direct_only_mlp,sam_only_mlp,agg_no_shared_targets"); ap.add_argument("--fprs", default="0.001,0.01")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; fprs = [float(x) for x in a.fprs.split(",")]; rng = np.random.default_rng(42)
ca = A.load_frames(a, ["calibration"])["calibration"]; rows = []
def stats(s, imp, fpr):
    imp = np.sort(imp); u = np.searchsorted(imp, s, side="left") / len(imp); thr = np.quantile(imp, 1 - fpr)
    return float(u.mean()), float((s >= thr).mean()), u, thr
for n in a.variants.split(","):
    fz = pickle.load(open(Path(a.out) / "variants" / n / "frozen.pkl", "rb")); T = A.transform(ca, A.V[n], a.shm, "calibration"); S, W = fz["scorer"](T, dev); P = fz["dec"]["cal"].predict(S)
    for b in ["G1", "G2", "G3", "G4", "G5", "G6"]:
        m = (ca.gap_bin == b).to_numpy(); s = P[m & (ca.label == 1).to_numpy()]; imp = P[m & (ca.label == 0).to_numpy() & ca.set_name.isin(["D_all_random", "D_hard"]).to_numpy()]
        r = dict(variant=n, bin=b, n_same=len(s), n_diff=len(imp))
        if len(s) >= 20 and len(imp) >= 1000:
            auc, _, u, _ = stats(s, imp, 0.001); bt = [u[rng.integers(0, len(u), len(u))].mean() for _ in range(300)]; r["AUC"] = f"{auc:.4f} [{np.percentile(bt, 2.5):.4f},{np.percentile(bt, 97.5):.4f}]"
            for f in fprs:
                _, tpr, _, thr = stats(s, imp, f); bt = [(s[rng.integers(0, len(s), len(s))] >= thr).mean() for _ in range(300)]; r[f"TPR@FPR{f:g}"] = f"{tpr:.3f} [{np.percentile(bt, 2.5):.3f},{np.percentile(bt, 97.5):.3f}]"
        else: r["AUC"] = "n too small"
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "gap_report_calibration.csv", index=False)
for col in [c for c in df.columns if c in ("AUC",) or c.startswith("TPR")]:
    print(f"\n== {col}  (rows = gap bin, columns = variant)"); print(df.pivot(index="bin", columns="variant", values=col).to_string())
print("\npairs per bin:\n", df[df.variant == df.variant.iloc[0]][["bin", "n_same", "n_diff"]].to_string(index=False))
