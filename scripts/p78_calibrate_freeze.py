"""P7 + P8: calibrator, guards, thresholds, audit sheets, and (with --freeze) the frozen thresholds.json.
Reads train_fit, train_val, calibration pair vectors. NEVER the test split.
Step 1 (no flag): fit, choose RHO, write draft + audit contact sheets. Look at the sheets, write label corrections if any.
Step 2 (--freeze): apply artifacts/audit/label_corrections.csv (pair_id,new_label,reason), recompute and freeze.
Usage: python scripts/p78_calibrate_freeze.py --p5 artifacts/p5 --models artifacts/models --out artifacts [--crops /dev/shm/aafv/crops] [--freeze]"""
import sys, json, argparse, pickle, hashlib, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from aafv import config, fusion, calib, thresholds

ap = argparse.ArgumentParser()
for k in ("p5", "models", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--crops", default="/dev/shm/aafv/crops"); ap.add_argument("--freeze", action="store_true"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); dev = "cuda" if torch.cuda.is_available() else "cpu"
for d in ("thresholds", "audit", "models"): (out / d).mkdir(parents=True, exist_ok=True)
rd = lambda s: pd.read_csv(Path(a.p5) / f"pair_vectors_{s}.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str})
tf, tv, ca = rd("train_fit"), rd("train_val"), rd("calibration")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
mp = sorted(Path(a.models).glob("fusion_seed*.pt")); models = [fusion.load_model(p, dev) for p in mp]; print(f"{len(models)} models")
Sv, Wv = fusion.predict(models, tv, dev); Sc, Wc = fusion.predict(models, ca, dev)

# ---- audit corrections (freeze step only)
if a.freeze and (out / "audit/label_corrections.csv").exists():
    cor = pd.read_csv(out / "audit/label_corrections.csv", dtype={"pair_id": str}); m = dict(zip(cor.pair_id, cor.new_label))
    ca["label"] = [m.get(str(p), l) for p, l in zip(ca.pair_id.astype(str), ca.label)]; print(f"applied {len(cor)} label corrections")

# ---- P7: choose calibrator on train_val by person-level CV log loss
yv = tv.label.to_numpy(); cv = {}
for kind in ("platt", "isotonic"):
    ll = []
    for tri, tei in GroupKFold(5).split(Sv, yv, tv.person_id_1):
        c = calib.Calibrator(kind).fit(Sv[tri], yv[tri]); ll.append(calib.logloss(yv[tei], c.predict(Sv[tei])))
    cv[kind] = float(np.mean(ll))
kind = min(cv, key=cv.get); print("calibrator CV log loss:", {k: round(v, 4) for k, v in cv.items()}, "->", kind)
cal = calib.Calibrator(kind).fit(Sv, yv)

def reliability(P, d):
    """Prior-matched check per gap bin: the calibration impostors of a bin are re-weighted so that the same:different ratio equals the ratio of train_val in THAT bin
    (the calibrated P carries that bin prior, because g is a model input). Bins with fewer than 20 same-person pairs in train_val are skipped, not guessed."""
    tvr = tv.groupby("gap_bin").label.agg(["sum", "count"]); rows = []
    for b, g in d.assign(P=P).groupby("gap_bin"):
        n1, n0 = int((g.label == 1).sum()), int((g.label == 0).sum()); t1, tc = tvr.loc[b]
        if n1 == 0 or n0 == 0 or t1 < 20: rows.append(dict(bin=b, n=len(g), same=n1, mean_P=np.nan, observed=np.nan, note="skipped: <20 same-person pairs in train_val")); continue
        rate = t1 / tc; w = np.where(g.label == 1, 1.0, (1 - rate) / rate * n1 / n0)
        rows.append(dict(bin=b, n=len(g), same=n1, mean_P=float(np.average(g.P, weights=w)), observed=float(np.average(g.label, weights=w)), note=""))
    return pd.DataFrame(rows)
rel = reliability(cal.predict(Sc, ca.g.to_numpy()), ca); rel["abs_dev"] = (rel.mean_P - rel.observed).abs()
print(rel.round(3).to_string(index=False))
big = rel[(rel.n >= 50) & rel.abs_dev.notna()]
if len(big) and big.abs_dev.max() > 0.10:
    print("max deviation > 0.10: refit Platt with g as an extra input"); cal = calib.Calibrator("platt_g").fit(Sv, yv, tv.g.to_numpy()); kind = "platt_g"
    rel = reliability(cal.predict(Sc, ca.g.to_numpy()), ca); print(rel.round(3).to_string(index=False))
rel.to_csv(out / "audit/reliability_by_bin.csv", index=False)

# ---- P8: branch Platt (train_val), guards, RHO, thresholds
pdl = LogisticRegression(C=1e6, max_iter=1000).fit(Wv[:, [0]], yv)
ps_mask = tv.avail_sam.to_numpy() == 1; psl = LogisticRegression(C=1e6, max_iter=1000).fit(Wv[ps_mask][:, [1]], yv[ps_mask])
Pc = cal.predict(Sc, ca.g.to_numpy()); Pd = pdl.predict_proba(Wc[:, [0]])[:, 1]; Ps = psl.predict_proba(Wc[:, [1]])[:, 1]
q_min = float(np.percentile(tf.Q, 5)); r_min = float(np.percentile(tf.R_sam[tf.avail_sam == 1], 5)); print(f"Q_MIN {q_min:.4f}  R_MIN(sam) {r_min:.4f}")
same = ca.label.to_numpy() == 1; best = None; table = []
for rho in cfg["RHO_OPTIONS"]:
    ok = calib.guard_pass(ca.Q.to_numpy(), ca.R_sam.to_numpy(), ca.avail_sam.to_numpy(), Pd, Ps, q_min, r_min, rho)
    ta, tr_, warn = thresholds.thresholds(Pc[(~same) & ok], Pc[same])
    z = thresholds.zone(Pc, ta, tr_); z = np.where((z == "accept") & ~ok, "review", z)
    err = int(((z == "accept") & ~same).sum()); rev = float((z[same] == "review").mean())
    table.append(dict(RHO=rho, T_accept=ta, T_reject=tr_, errors=err, same_accepted=float((z[same] == "accept").mean()), same_review=rev, imp_passing_guards=int(((~same) & ok).sum())))
    if err == 0 and (best is None or rev < best[0]): best = (rev, rho, ta, tr_, warn, ok)
print(pd.DataFrame(table).round(4).to_string(index=False))
assert best is not None, "no RHO gave zero calibration errors"
_, rho, t_acc, t_rej, warn, ok = best
if warn: print(warn)
top = ca.assign(P=Pc)[~same].nlargest(8, "P")[["pair_id", "set_name", "person_id_1", "person_id_2", "gap", "P"]]; print("highest-P different-person calibration pairs:\n", top.round(4).to_string(index=False))
print("different-person pairs with P > 0.5:", int((Pc[~same] > 0.5).sum()), "| > 0.9:", int((Pc[~same] > 0.9).sum()))
if t_acc >= 1.0: print("NOTE: T_accept >= 1, so nothing can be auto-accepted: some different-person pair scores ~1.0 (check the audit sheet for label errors)")
zone = np.where((thresholds.zone(Pc, t_acc, t_rej) == "accept") & ~ok, "review", thresholds.zone(Pc, t_acc, t_rej))
print(f"CHOSEN RHO={rho}  T_accept={t_acc:.6f}  T_reject={t_rej:.6f}")
print("calibration zones: same-person", pd.Series(zone[same]).value_counts().to_dict(), "| different-person", pd.Series(zone[~same]).value_counts().to_dict())
print("calibration auto-accept errors:", int(((zone == "accept") & ~same).sum()))
ca["zone"] = zone; bins = ca.groupby("gap_bin").zone.value_counts(normalize=True).unstack(fill_value=0).round(3); print("zone share by gap bin (all calibration pairs):\n", bins.to_string())
bins.to_csv(out / "audit/zone_share_by_bin.csv")

# ---- P8.5 audit sheets (draft step)
def sheet(d, path, title):
    from PIL import Image, ImageDraw
    c = Path(a.crops); n = len(d); cols = 4; cell = (2 * 112 + 8, 112 + 26); W = cols * cell[0]; H = ((n + cols - 1) // cols) * cell[1]
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    for i, r in enumerate(d.itertuples()):
        x, y = (i % cols) * cell[0], (i // cols) * cell[1]
        for j, iid in enumerate((r.img_id_1, r.img_id_2)):
            f = c / f"{iid}.png"
            if f.exists(): im.paste(Image.open(f).convert("RGB").resize((112, 112)), (x + j * 116, y + 24))
        dr.text((x + 2, y + 2), f"{i+1}. P={r.P:.3f} gap={r.gap}y id={r.pair_id}", fill="black")
    im.save(path, quality=85); print("wrote", path, "|", title)
ca["P"] = Pc
if not a.freeze:
    sheet(ca[~same].nlargest(50, "P"), out / "audit/audit_different_top50.jpg", "different-person pairs with the HIGHEST P (are they really different people?)")
    sheet(ca[same].nsmallest(50, "P"), out / "audit/audit_same_bottom50.jpg", "same-person pairs with the LOWEST P (are they really the same person?)")
    ca[~same].nlargest(50, "P")[["pair_id", "img_id_1", "img_id_2", "person_id_1", "person_id_2", "gap", "P"]].to_csv(out / "audit/audit_different_top50.csv", index=False)
    ca[same].nsmallest(50, "P")[["pair_id", "img_id_1", "img_id_2", "person_id_1", "person_id_2", "gap", "P"]].to_csv(out / "audit/audit_same_bottom50.csv", index=False)
    with open(out / "models/calibrator_draft.pkl", "wb") as f: pickle.dump(dict(cal=cal, pd=pdl, ps=psl), f)
    print("DRAFT only. Review the two sheets; list any wrong labels in artifacts/audit/label_corrections.csv (pair_id,new_label,reason); then run again with --freeze.")
    sys.exit(0)

# ---- P8.7 freeze
cp = out / "models/calibrator.pkl"
with open(cp, "wb") as f: pickle.dump(dict(cal=cal, pd=pdl, ps=psl), f)
th = dict(T_accept=t_acc, T_reject=t_rej, Q_MIN=q_min, R_MIN_sam=r_min, RHO=rho, calibrator_kind=kind, calibrator_file=str(cp), calibrator_sha256=sha(cp),
          model_sha256={p.name: sha(p) for p in mp}, config_hash=config.config_hash(cfg), n_calibration_pairs=len(ca), n_same=int(same.sum()),
          created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), looks_max=cfg["LOOKS_MAX"], stage="A (SAM branch only, no FG-NET out-of-fold data)")
tp = out / "thresholds/thresholds.json"
if tp.exists(): os.chmod(tp, 0o644)
json.dump(th, open(tp, "w"), indent=1); os.chmod(tp, 0o444)
h = sha(tp); open(out / "thresholds/thresholds.sha256", "w").write(h + "\n"); os.chmod(out / "thresholds/thresholds.sha256", 0o444)
print("FROZEN thresholds.json SHA-256:", h); config.evidence("P8", cfg, T_accept=t_acc, T_reject=t_rej, RHO=rho)
