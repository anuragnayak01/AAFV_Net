"""Complete decision comparison on the CALIBRATION split (no test data): accept AND reject side, same guarantees for every model.
Accept side: the frozen zero-false-accept threshold of each variant. Reject side: T_reject = beta-quantile of same-person scores (false-reject target beta).
  python scripts/p10_decision_table.py --p5q artifacts/p5q --out artifacts/ablations [--variants ...] [--betas 0.005,0.01,0.02]"""
import sys, argparse, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd, torch
import p10_ablations as A
import __main__; __main__.Scorer = A.Scorer

ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--shm", default="/dev/shm/aafv/abl")
ap.add_argument("--variants", default="full,arcface_only,arcface_only_noguard,direct5_logreg,direct_only_mlp,agg_no_shared_targets"); ap.add_argument("--betas", default="0.005,0.01,0.02")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; ca = A.load_frames(a, ["calibration"])["calibration"]; rows = []
same = (ca.label == 1).to_numpy(); lowgap = ca.gap_bin.isin(["G1", "G2", "G3"]).to_numpy()
for n in a.variants.split(","):
    fz = pickle.load(open(Path(a.out) / "variants" / n / "frozen.pkl", "rb")); T = A.transform(ca, A.V[n], a.shm, "calibration"); S, W = fz["scorer"](T, dev); P0, _ = A.decide(fz["dec"], T, S, W)
    for beta in [float(b) for b in a.betas.split(",")]:
        dec = dict(fz["dec"], t_rej=float(np.quantile(P0[same], beta))); P, z = A.decide(dec, T, S, W)
        sh = lambda m, k: float((z[m] == k).mean()) if m.sum() else np.nan
        r = dict(variant=n, beta=beta, T_accept=round(dec["t_acc"], 3), T_reject=round(dec["t_rej"], 4), same_acc=sh(same, "accept"), same_rev=sh(same, "review"), same_rej=sh(same, "reject"),
                 same_acc_G1G3=sh(same & lowgap, "accept"), same_acc_G4G6=sh(same & ~lowgap, "accept"))
        for st in ("D_random", "D_all_random", "D_hard"):
            m = (ca.set_name == st).to_numpy(); r[f"{st}_FA"] = int((z[m] == "accept").sum()); r[f"{st}_rej"] = sh(m, "reject"); r[f"{st}_rev"] = sh(m, "review")
        r["automation_balanced"] = 0.5 * (r["same_acc"] + r["same_rej"]) + 0.5 * (r["D_all_random_rej"] + (r["D_all_random_FA"] / max((ca.set_name == "D_all_random").sum(), 1)))
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "decision_table_calibration.csv", index=False)
pd.set_option("display.width", 250)
for beta, d in df.groupby("beta"):
    print(f"\n=== false-reject target beta = {beta}  (share of each pair type in each zone; FA must be 0)")
    print(d[["variant", "T_accept", "T_reject", "same_acc", "same_rev", "same_rej", "same_acc_G4G6", "D_random_rej", "D_all_random_rej", "D_hard_rej", "D_hard_FA", "automation_balanced"]].round(3).to_string(index=False))
