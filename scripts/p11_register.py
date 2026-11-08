"""Registers baseline scores (scripts/p11_*.py output) as frozen variants of the ablation machinery, with the SAME decision rule as every other variant
(Platt on train_val, guard = quality + direct-branch probability, thresholds on calibration, hashes in ablation_freeze.json).
  python scripts/p11_register.py --p5q artifacts/p5q --abl artifacts/ablations --baselines siamese_cnn,siamese_vit,vlm [--scores artifacts/baselines]"""
import sys, os, json, argparse, pickle, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, pandas as pd
from aafv import config
import p10_ablations as A

ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--abl", required=True); ap.add_argument("--baselines", required=True)
ap.add_argument("--scores", default="artifacts/baselines"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); abl = Path(a.abl); frz = abl / "ablation_freeze.json"; h = json.load(open(frz)) if frz.exists() else {}; rows = []
fr = A.load_frames(a, ["train_fit", "train_val", "calibration"])
for b in a.baselines.split(","):
    name = "baseline_" + b; f = Path(a.scores) / b / "scores.npz"
    if not f.exists(): print("no scores for", b, "->", f); continue
    z = np.load(f); sc = A.Scorer("pre", {"scores": dict(zip(z["pair_id"].astype(str), z["logit"].astype(float)))}); v = A.V[name]
    T = {s: A.transform(fr[s], v, "", s) for s in fr}
    for s in ("train_val", "calibration"):
        T[s] = T[s][T[s].pair_id.isin(sc.p["scores"])].reset_index(drop=True)                 # baselines that scored only a subset (VLM) are evaluated on that subset
        print(f"{b}: {s} pairs scored: {len(T[s])} (same-person {int(T[s].label.sum())})")
    Sv, Wv = sc(T["train_val"], "cpu"); Sc, Wc = sc(T["calibration"], "cpu"); dec = A.fit_decision(v, cfg, T["train_fit"], T["train_val"], T["calibration"], Sv, Wv, Sc, Wc)
    P, zn = A.decide(dec, T["calibration"], Sc, Wc); res = dict(variant=name, group="external", T_accept=dec["t_acc"], RHO=dec["rho"], **A.summarize(T["calibration"], P, zn)); rows.append(res)
    od = abl / "variants" / name; od.mkdir(parents=True, exist_ok=True); json.dump(res, open(od / "calibration.json", "w")); pk = od / "frozen.pkl"
    import __main__; __main__.Scorer = A.Scorer
    pickle.dump(dict(scorer=sc, dec=dec), open(pk, "wb")); h[name] = hashlib.sha256(open(pk, "rb").read()).hexdigest()
    print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res.items() if k in ("T_accept", "same_accept_all", "same_accept_G4-G6", "FA_D_random", "AUC_D_hard", "TPR@FPR1e-3_D_hard")})
json.dump(h, open(frz, "w"), indent=1)
if rows:
    cp = abl / "ablation_calibration.csv"; old = pd.read_csv(cp) if cp.exists() else pd.DataFrame(columns=["variant"])
    pd.concat([old[~old.variant.isin([r["variant"] for r in rows])], pd.DataFrame(rows)], ignore_index=True).to_csv(cp, index=False)
print("registered:", [r["variant"] for r in rows])
