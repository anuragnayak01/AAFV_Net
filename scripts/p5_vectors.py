"""P5.4 - P5.6: standardise (train_fit only), build branch vectors, availability and reliability, write pair_vectors_<split>.csv.
Stage A: SAM branch only; the diffusion branch is always unavailable (zeros, avail 0); no enhancers (enhanced_pair = 0)."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from aafv import config, pairvec

ap = argparse.ArgumentParser()
for k in ("pairs", "p2", "p5"): ap.add_argument("--" + k, required=True)
ap.add_argument("--splits", default="train_fit,train_val,calibration,test"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
ap.add_argument("--std_json", default=""); ap.add_argument("--sigma_max", type=float, default=0.0)   # external split: reuse the frozen train_fit standardizers and SIGMA_MAX
a = ap.parse_args(); cfg = config.load(a.params); p5 = Path(a.p5)
faces = pd.read_csv(p5 / "faces.csv", dtype={"img_id": str}); est = pd.read_csv(Path(a.p2) / "age_est.csv", dtype={"img_id": str})
Qimg = pd.read_csv(Path(a.p2) / "quality.csv", dtype={"img_id": str}).set_index("img_id").Q
err = json.load(open(Path(a.p2) / "age_error.json")); eps = err["eps"]
bias_img = (est.set_index("img_id").est_age - est.set_index("img_id").label_age)
f_est, f_q, f_arc = faces.est_age.to_numpy(), faces.q.to_numpy(), faces.arc_to_orig.to_numpy()
cols = ["arc", "eff", "vgg", "lpips", "ssim"]

R = {s: np.load(p5 / f"raw_{s}.npz") for s in a.splits.split(",")}
# ---- standardizers from train_fit only (or loaded from --std_json for an external split)
if a.std_json:
    js = json.load(open(a.std_json)); std = {b: (np.array([js[b][c]["mean"] for c in cols]), np.array([js[b][c]["std"] for c in cols])) for b in ("direct", "sam")}
else:
    tr = R["train_fit"]
    S_dir_tr = tr["S_dir"]
    sam_all = np.concatenate([tr["S_fwd"], tr["S_bwd"], tr["S_sh"].reshape(-1, 5)]); sam_all = sam_all[np.isfinite(sam_all).all(1)]
    std = {"direct": pairvec.fit_standardizer(S_dir_tr), "sam": pairvec.fit_standardizer(sam_all)}
    json.dump({b: {c: {"mean": float(m), "std": float(sd)} for c, m, sd in zip(cols, *v)} for b, v in std.items()} | {"n_pairs": int(len(S_dir_tr))},
              open(p5 / "standardizers.json", "w"), indent=1)

def build(sp):
    r = R[sp]; P = pd.read_csv(Path(a.pairs) / f"pairs_{sp}.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str})
    N = len(P); Sf, Sb, Ssh = r["S_fwd"], r["S_bwd"], r["S_sh"]
    mf, mb, msh = np.isfinite(Sf[:, 0]), np.isfinite(Sb[:, 0]), np.isfinite(Ssh[:, :, 0])
    x, avail, K_used, spread, used = pairvec.branch_vector(np.nan_to_num(Sf), np.nan_to_num(Sb), np.nan_to_num(Ssh), mf, mb, msh, std["sam"])
    xd = (r["S_dir"] - std["direct"][0]) / std["direct"][1]
    bias_pair = 0.5 * (P.img_id_1.map(bias_img).to_numpy() + P.img_id_2.map(bias_img).to_numpy())
    def rt(g, t):
        g0 = np.maximum(g, 0); return pairvec.r_target(f_arc[g0], f_est[g0], t, bias_pair if t.ndim == 1 else bias_pair[:, None], f_q[g0], eps)
    Rt_f, Rt_b = rt(r["g_fwd"], r["tf"]), rt(r["g_bwd"], r["tb"])
    Rt_1, Rt_2 = rt(r["g_sh1"], r["T"]), rt(r["g_sh2"], r["T"])
    return P, xd, x, avail, K_used, spread, (Rt_f, Rt_b, Rt_1, Rt_2, used)

built = {sp: build(sp) for sp in R}
if a.std_json:
    assert a.sigma_max > 0, "--sigma_max required with --std_json (value stored in standardizers.json by a normal run)"; SIGMA_MAX = a.sigma_max; share = None
else:
    sig = built["train_fit"][5][built["train_fit"][3]]; SIGMA_MAX = float(np.percentile(sig, 95)); print("SIGMA_MAX", SIGMA_MAX)
    tp = pd.concat([built[s][0] for s in ("train_fit", "train_val") if s in built]); share = tp.gap_bin.value_counts(normalize=True)
    jj = json.load(open(p5 / "standardizers.json")); jj["sigma_max"] = SIGMA_MAX; json.dump(jj, open(p5 / "standardizers.json", "w"), indent=1)
for sp, (P, xd, x, avail, K_used, spread, rr) in built.items():
    Rb = pairvec.reliability(rr[0], rr[1], rr[2], rr[3], rr[4], avail, spread, SIGMA_MAX)
    o = P[["pair_id", "label", "gap", "gap_bin", "gender", "set_name", "dataset", "img_id_1", "img_id_2", "person_id_1", "person_id_2"]].copy()
    w = 1.0 / P.gap_bin.map(share).to_numpy() if sp.startswith("train") else np.ones(len(P))
    o["weight"] = w / w.mean() if sp.startswith("train") else w
    o["g"] = np.minimum(1.0, P.gap.to_numpy() / cfg["A_MAX"]); o["Q"] = 0.5 * (P.img_id_1.map(Qimg).to_numpy() + P.img_id_2.map(Qimg).to_numpy())
    o["enhanced_pair"] = 0; o["K_used"] = K_used; o["avail_sam"] = avail.astype(int); o["avail_diff"] = 0; o["avail_multi"] = (K_used >= 1).astype(int)
    o["R_sam"] = Rb; o["R_diff"] = 0.0; o["s_spread"] = spread; o["f_spread"] = 0.0
    for j, c in enumerate(cols):
        o["d_" + c] = xd[:, j]; o["s_" + c] = x[:, j]; o["f_" + c] = 0.0
    o.to_csv(p5 / f"pair_vectors_{sp}.csv", index=False)
    num = o[[c for c in o.columns if c[:2] in ("d_", "s_") or c in ("g", "Q", "R_sam")]]
    print(f"{sp}: {len(o)} pairs | finite {bool(np.isfinite(num.to_numpy()).all())} | g,Q,R in 0..1: {bool(((o[['g','Q','R_sam']] >= 0) & (o[['g','Q','R_sam']] <= 1)).all().all())} | SAM unavailable {1 - o.avail_sam.mean():.3%}")
    print("   unavailable by gap bin:", (1 - o.groupby("gap_bin").avail_sam.mean()).round(3).to_dict())
config.evidence("P5", cfg, sigma_max=SIGMA_MAX, eps=eps)
