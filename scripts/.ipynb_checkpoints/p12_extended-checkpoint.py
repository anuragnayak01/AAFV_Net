"""Extended evaluation of every frozen variant, one script for: (1) model-neutral hard negatives, (2) person-level bootstrap CIs, (3) the final TEST look,
(4) an external split (FG-NET) scored with the SAME frozen thresholds. Nothing is retrained or re-tuned here.

Neutral hard sets: among random different-person pairs (D_random + D_all_random) of the evaluated split, the top --top_frac by the logit of a miner model that is NOT part of
our pipeline: Siamese ViT -> D_by_vit, Siamese CNN -> D_by_cnn, mean rank of both -> D_by_both. A method is never scored on a set mined by itself (shown as n/a).
Thresholds: T_accept, RHO, guards frozen on calibration; T_reject(beta) = beta-quantile of calibration same-person probabilities (as in p10_decision_table.py).

  calibration, all pairs : python scripts/p12_extended.py --p5q artifacts/p5q --abl artifacts/ablations --split calibration
  common subset          : ... --subset artifacts/baselines/subset_pair_ids.txt --tag _subset
  FINAL TEST (uses a look): ... --split test --i_understand_this_uses_a_look
  FG-NET                  : ... --extra_vectors artifacts/p5_fg/pair_vectors_fgnet.csv   (calibration frozen thresholds applied to FG-NET)"""
import sys, os, json, argparse, pickle, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--abl", required=True); ap.add_argument("--split", default="calibration", choices=["calibration", "test"])
ap.add_argument("--variants", default="full,arcface_only,direct_only_mlp,agg_no_shared_targets,baseline_siamese_cnn,baseline_siamese_vit,baseline_vlm"); ap.add_argument("--betas", default="0.005,0.01,0.02")
ap.add_argument("--scores", default="artifacts/baselines"); ap.add_argument("--shm", default="/dev/shm/aafv/abl"); ap.add_argument("--top_frac", type=float, default=0.01); ap.add_argument("--boot", type=int, default=500)
ap.add_argument("--subset", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--i_understand_this_uses_a_look", action="store_true"); ap.add_argument("--test_log", default="artifacts/q/logs/test_access.log")
ap.add_argument("--extra_vectors", default=""); ap.add_argument("--extra_pairs", default=""); ap.add_argument("--extra_min_age", type=int, default=0); ap.add_argument("--extra_name", default="fgnet"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args()
if a.subset: os.environ["AAFV_SUBSET"] = a.subset
import numpy as np, pandas as pd, torch
from aafv import config, evaluation
import p10_ablations as A
import __main__; __main__.Scorer = A.Scorer
cfg = config.load(a.params); abl = Path(a.abl); dev = "cuda" if torch.cuda.is_available() else "cpu"; betas = [float(b) for b in a.betas.split(",")]; rng = np.random.default_rng(7)
names = [n for n in a.variants.split(",") if (abl / "variants" / n / "frozen.pkl").exists()]; print("variants:", names)
if a.split == "test" or a.extra_vectors:
    h = json.load(open(abl / "ablation_freeze.json"))
    for n in names: assert hashlib.sha256(open(abl / "variants" / n / "frozen.pkl", "rb").read()).hexdigest() == h[n], f"{n} changed since freeze"
if a.split == "test":
    assert a.i_understand_this_uses_a_look, "the test split needs --i_understand_this_uses_a_look"
    log = Path(a.test_log); used = len(log.read_text().splitlines()) if log.exists() else 0; assert used < cfg["LOOKS_MAX"], f"{used} looks already used"
    log.parent.mkdir(parents=True, exist_ok=True); open(log, "a").write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} look={used+1} p12_extended\n")
cal = A.load_frames(a, ["calibration"])["calibration"]; tgt = cal if a.split == "calibration" else A.load_frames(a, ["test"])["test"]
MINERS = {"vit": "siamese_vit", "cnn": "siamese_cnn"}; SELF = {"D_by_vit": {"baseline_siamese_vit"}, "D_by_cnn": {"baseline_siamese_cnn"}, "D_by_both": {"baseline_siamese_vit", "baseline_siamese_cnn"}}

def neutral_sets(df):
    """Adds boolean columns D_by_vit / D_by_cnn / D_by_both (random impostors of df with the highest miner scores)."""
    pool = ((df.label == 0) & df.set_name.isin(["D_random", "D_all_random"])).to_numpy(); rk = {}
    for k, b in MINERS.items():
        f = Path(a.scores) / b / "scores.npz"; df["D_by_" + k] = False
        if not f.exists(): print("no miner scores:", b); continue
        z = np.load(f); sc = pd.Series(z["logit"].astype(float), index=z["pair_id"].astype(str)); s = df.pair_id.map(sc).to_numpy(); ok = pool & np.isfinite(s)
        thr = np.quantile(s[ok], 1 - a.top_frac); df["D_by_" + k] = ok & (s >= thr); rk[k] = pd.Series(np.where(ok, s, -np.inf)).rank(pct=True).to_numpy()
    if len(rk) == 2:
        m = (rk["vit"] + rk["cnn"]) / 2; ok = pool & np.isfinite(m); df["D_by_both"] = ok & (m >= np.quantile(m[ok], 1 - a.top_frac))
    else: df["D_by_both"] = False
    return df
tgt = neutral_sets(tgt)
if a.split == "calibration": cal = tgt
SETS = ["D_random", "D_all_random", "D_hard", "D_by_vit", "D_by_cnn", "D_by_both"]
def setmask(df, s): return df[s].to_numpy() if s.startswith("D_by") else (df.set_name == s).to_numpy()

def scored(n, df, split, scores_dict=None):
    """Frozen variant n applied to df. Returns (S, W, covered_mask). Baselines may cover only part of df (VLM)."""
    fz = pickle.load(open(abl / "variants" / n / "frozen.pkl", "rb")); sc = fz["scorer"] if scores_dict is None else A.Scorer("pre", {"scores": scores_dict}); cov = np.ones(len(df), bool)
    if sc.kind == "pre": cov = df.pair_id.isin(sc.p["scores"]).to_numpy()
    d = df[cov].reset_index(drop=True); T = A.transform(d, A.V[n], a.shm, split); S, W = sc(T, dev); return fz, T, S, W, cov

def cluster_ci(num_mask, den_mask, cl):
    """Cluster bootstrap (resample persons) of mean(num | den)."""
    if den_mask.sum() == 0: return np.nan, np.nan, np.nan
    t = pd.DataFrame({"c": cl[den_mask], "x": num_mask[den_mask].astype(float)}).groupby("c").x.agg(["sum", "count"]); s, c = t["sum"].to_numpy(), t["count"].to_numpy()
    ix = rng.integers(0, len(s), (a.boot, len(s))); r = s[ix].sum(1) / c[ix].sum(1); return s.sum() / c.sum(), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

def summarize(df, z, variant, extra_sets=None):
    r = {}; same = (df.label == 1).to_numpy(); cl = df.person_id_1.to_numpy(); lowgap = df.gap_bin.isin(["G1", "G2", "G3"]).to_numpy(); r["n_pairs"] = len(df)
    for k, m in (("same_acc", z == "accept"), ("same_rev", z == "review"), ("same_rej", z == "reject")):
        v, lo, hi = cluster_ci(m, same, cl); r[k] = v; r[k + "_lo"] = lo; r[k + "_hi"] = hi
    r["same_acc_G1G3"] = float((z[same & lowgap] == "accept").mean()) if (same & lowgap).any() else np.nan
    for b in sorted(df.gap_bin.unique()):
        mb = same & (df.gap_bin == b).to_numpy(); r[f"same_acc_{b}"] = float((z[mb] == "accept").mean()) if mb.sum() else np.nan; r[f"n_same_{b}"] = int(mb.sum())
    for s in (SETS if extra_sets is None else extra_sets):
        if s in SELF and variant in SELF[s]: r[f"{s}_n"] = 0; continue                 # never scored on a set it mined
        m = setmask(df, s) if (s in df.columns or s in set(df.set_name)) else np.zeros(len(df), bool); r[f"{s}_n"] = int(m.sum())
        if m.sum() == 0: continue
        fa = int((z[m] == "accept").sum()); r[f"{s}_FA"] = fa; r[f"{s}_FA_bound95"] = evaluation.zero_error_bound(int(m.sum()), 0.05) if fa == 0 else np.nan
        v, lo, hi = cluster_ci(z == "reject", m, cl); r[f"{s}_rej"] = v; r[f"{s}_rej_lo"] = lo; r[f"{s}_rej_hi"] = hi; r[f"{s}_rev"] = float((z[m] == "review").mean())
    return r

rows = []; extra = pd.read_csv(a.extra_vectors, dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str}) if a.extra_vectors else None
if extra is not None and a.extra_min_age:            # in-support cut: keep pairs whose YOUNGER image is at least --extra_min_age (the training range starts at 16); ages come from the pair table
    pt = pd.read_csv(a.extra_pairs, usecols=["pair_id", "a1"], dtype={"pair_id": str}); keep = set(pt[pt.a1 >= a.extra_min_age].pair_id); n0 = len(extra)
    extra = extra[extra.pair_id.isin(keep)].reset_index(drop=True); print(f"in-support cut (min age {a.extra_min_age}): {len(extra)} of {n0} pairs, same-person {int(extra.label.sum())}")
if extra is not None: extra["_row"] = np.arange(len(extra))
for n in names:
    fz, Tc, Sc, Wc, cov_c = scored(n, cal, "calibration"); P0, _ = A.decide(fz["dec"], Tc, Sc, Wc); samec = (Tc.label == 1).to_numpy()
    if a.split == "calibration": Tt, St, Wt, covt = Tc, Sc, Wc, cov_c
    else: _, Tt, St, Wt, covt = scored(n, tgt, "test")
    for beta in betas:
        dec = dict(fz["dec"], t_rej=float(np.quantile(P0[samec], beta))); _, z = A.decide(dec, Tt, St, Wt)
        rows.append(dict(variant=n, beta=beta, split=a.split, T_accept=dec["t_acc"], T_reject=dec["t_rej"], **summarize(Tt, z, n)))
        if extra is not None and A.V[n]["agg"] == "default":
            if A.V[n]["kind"] == "pre":
                f = Path(a.scores) / n.replace("baseline_", "") / f"scores_{Path(a.extra_vectors).stem.replace('pair_vectors_', 'pairs_')}.npz"
                if not f.exists(): print("no external scores for", n, f); continue
                z_ = np.load(f); _, Te, Se, We, _ = scored(n, extra, "ext", dict(zip(z_["pair_id"].astype(str), z_["logit"].astype(float))))
            else: Te = A.transform(extra, A.V[n], a.shm, "ext"); Se, We = fz["scorer"](Te, dev)
            _, ze = A.decide(dec, Te, Se, We); rows.append(dict(variant=n, beta=beta, split=a.extra_name, T_accept=dec["t_acc"], T_reject=dec["t_rej"], **summarize(Te, ze, n, extra_sets=["FG_random"])))
df = pd.DataFrame(rows); out = abl / f"extended_{a.split}{a.tag}.csv"; df.to_csv(out, index=False); print("saved", out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
for (split, beta), d in df.groupby(["split", "beta"]):
    if beta != 0.01: continue
    c = [x for x in ["variant", "same_acc", "same_acc_lo", "same_acc_hi", "same_acc_G1G3", "D_hard_rej", "D_by_vit_rej", "D_by_cnn_rej", "D_by_both_rej", "D_by_both_n", "D_all_random_FA", "FG_random_FA", "FG_random_rej"] if x in d.columns]
    print(f"\n=== {split}, beta = {beta}"); print(d[c].round(3).to_string(index=False))