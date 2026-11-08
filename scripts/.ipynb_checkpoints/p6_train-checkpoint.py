"""P6: train the fusion model (5 seeds x small grid). Reads ONLY train_fit and train_val pair vectors (never calibration/test).
Usage: python scripts/p6_train.py --p5 artifacts/p5 --out artifacts/models [--max_epochs 100]"""
import sys, json, argparse, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from aafv import config, fusion, calib

ap = argparse.ArgumentParser(); ap.add_argument("--p5", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max_epochs", type=int, default=100); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = pd.read_csv(Path(a.p5) / "pair_vectors_train_fit.csv", dtype={"person_id_1": str, "person_id_2": str})
va = pd.read_csv(Path(a.p5) / "pair_vectors_train_val.csv", dtype={"person_id_1": str, "person_id_2": str})
print(f"train_fit {len(tr)} pairs ({int(tr.label.sum())} same) | train_val {len(va)} ({int(va.label.sum())} same) | device {dev}")
s = tr[tr.label == 1].gap_bin.value_counts().reindex(["G1", "G2", "G3", "G4", "G5", "G6"], fill_value=0); print("train_fit same-person pairs by bin:", s.to_dict())
for b in ("G4", "G5", "G6"):
    if s[b] < 30: print(f"WARNING: {b} has only {s[b]} training same-person pairs (not fabricated)")

grid = list(itertools.product([1e-3, 3e-3], [1e-4, 1e-2])); results = {}
for lr, wd in grid:
    runs = []
    for seed in range(cfg["SEEDS"]):
        m, h, sk = fusion.train_one(tr, va, 1000 * seed + cfg["SEED"], lr, wd, cfg["P_MASK"], cfg["MLP_SIZES"], dev, a.max_epochs)
        runs.append((m, h, sk, min(x["val_logloss"] for x in h)))
    results[(lr, wd)] = runs
    print(f"lr={lr} wd={wd}: val log loss per seed {[round(r[3], 4) for r in runs]}  skipped batches {sum(r[2] for r in runs)}", flush=True)
best = min(results, key=lambda k: np.mean([r[3] for r in results[k]])); runs = results[best]
ll = np.array([r[3] for r in runs]); print(f"chosen lr={best[0]} wd={best[1]} (by train_val log loss only); mean {ll.mean():.4f}")
print("GATE seeds stable (spread within 10%):", bool((ll.max() - ll.min()) / ll.mean() <= 0.10))
for i, (m, h, sk, _) in enumerate(runs):
    fusion.save_model(m, out / f"fusion_seed{i}.pt", dict(sizes=list(cfg["MLP_SIZES"]), lr=best[0], wd=best[1], seed=i, config_hash=config.config_hash(cfg)))
json.dump({f"seed{i}": r[1] for i, r in enumerate(runs)} | {"chosen": dict(lr=best[0], wd=best[1])}, open(out / "p6_training_curves.json", "w"))
S, _ = fusion.predict([r[0] for r in runs], va, dev); p = 1 / (1 + np.exp(-S))
rows = [dict(bin="ALL", n=len(va), same=int(va.label.sum()), auc=roc_auc_score(va.label, S), logloss=calib.logloss(va.label, p))]
for b, d in va.groupby("gap_bin"):
    ix = d.index.to_numpy()
    if d.label.nunique() == 2: rows.append(dict(bin=b, n=len(d), same=int(d.label.sum()), auc=roc_auc_score(d.label, S[ix]), logloss=calib.logloss(d.label, p[ix])))
print(pd.DataFrame(rows).round(4).to_string(index=False))
config.evidence("P6", cfg, seeds=len(runs), chosen=best, models=str(out))
