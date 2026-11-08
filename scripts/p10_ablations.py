"""P10 ablations: every variant is re-trained on train_fit (early stopping on train_val), calibrated on train_val, thresholds set on the
calibration split by the SAME rule as the main system. Test split is untouched until `--phase test` (one extra look, logged).
Uses /dev/shm for staging and 8 workers (one per GPU slice; the last GPU instance is shared as in P4).
  python scripts/p10_ablations.py --phase calib --p5 artifacts/p5 --p5q artifacts/p5q --main_models artifacts/q/models --out artifacts/ablations
  python scripts/p10_ablations.py --phase test  ... --i_understand_this_uses_a_look
"""
import sys, os, re, json, argparse, subprocess, pickle, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from aafv import config, calib, thresholds, evaluation

COLS = ["arc", "eff", "vgg", "lpips", "ssim"]
D, S_ = ["d_" + c for c in COLS], ["s_" + c for c in COLS]
# ---------------------------------------------------------------- variants
V = {}
def add(name, group, **kw): V[name] = {**dict(group=group, kind="mlp", drop_direct=False, drop_sam=False, keep=None, ctx="full", p_mask=None, weights=True, agg="default", guard="full"), **kw}
add("full", "main")
add("arcface_only", "baseline", kind="arc", drop_sam=True, guard="qd")
add("arcface_only_noguard", "baseline", kind="arc", drop_sam=True, guard="none")
add("direct5_logreg", "baseline", kind="logreg", drop_sam=True, guard="qd")
add("direct_only_mlp", "branch", drop_sam=True, guard="qd")
add("sam_only_mlp", "branch", drop_direct=True, guard="qsam")
for c in COLS: add("drop_" + c, "scores", drop_cols=[c])
add("arc_only_scores", "scores", keep=["arc"])
add("fusion_logreg_no_mlp", "fusion", kind="logreg")
add("no_context", "fusion", ctx="none")
add("no_reliability", "fusion", ctx="no_R")
add("no_branch_masking", "fusion", p_mask=0.0)
add("no_pair_weights", "fusion", weights=False)
add("agg_fwd_only", "aggregation", agg="fwd_only")
add("agg_no_shared_targets", "aggregation", agg="no_shared")
add("agg_shared_only", "aggregation", agg="shared_only")
add("guard_none", "guards", guard="none")
add("guard_quality_only", "guards", guard="q")
add("guard_quality_reliability", "guards", guard="qr")
for _n, _g in (("baseline_siamese_cnn", "external"), ("baseline_siamese_vit", "external"), ("baseline_vlm", "external")): add(_n, _g, kind="pre", drop_sam=True, guard="qd")
for v in V.values(): v.setdefault("drop_cols", [])
GUARD = {"full": (1, 1, 1, 1), "q": (1, 0, 0, 0), "qr": (1, 1, 0, 0), "none": (0, 0, 0, 0), "qd": (1, 0, 1, 0), "qsam": (1, 1, 0, 1)}   # use Q, R, P_dir, P_sam

def devices():
    try: u = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))
    except Exception: u = []
    if not u: return [os.environ.get("CUDA_VISIBLE_DEVICES", "")] * 8        # no MIG: every worker uses whatever is visible (CPU in tests)
    return u[:-1] + [u[-1]] * (8 - len(u) + 1)

# ---------------------------------------------------------------- data
def load_frames(a, splits):
    rd = lambda root, s: pd.read_csv(Path(root) / f"pair_vectors_{s}.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str})
    out = {s: rd(a.p5q, s) for s in splits}
    for d_ in out.values(): d_["_row"] = np.arange(len(d_))          # row in the unfiltered table, so staged aggregation arrays stay aligned under --subset
    sub = os.environ.get("AAFV_SUBSET")                      # optional common subset of pair ids for train_val / calibration / test (used when a baseline scored only a subset)
    if sub:
        keep = set(Path(sub).read_text().split())
        for s_ in out:
            if s_ != "train_fit": out[s_] = out[s_][out[s_].pair_id.isin(keep)].reset_index(drop=True)
    return out

def agg_arrays(raw, std, mode):
    Sf, Sb, Ssh = raw["S_fwd"], raw["S_bwd"], raw["S_sh"]; mu, sd = std
    mf, mb, msh = np.isfinite(Sf[:, 0]), np.isfinite(Sb[:, 0]), np.isfinite(Ssh[:, :, 0])
    Zf, Zb, Zs = (np.nan_to_num(Sf) - mu) / sd, (np.nan_to_num(Sb) - mu) / sd, (np.nan_to_num(Ssh) - mu) / sd
    avail = mf if mode == "fwd_only" else (mf & mb); used = msh & avail[:, None]; K = used.sum(1)
    base = Zf if mode == "fwd_only" else 0.5 * (Zf + Zb); shared = (Zs * used[..., None]).sum(1) / np.maximum(K, 1)[:, None]
    x = base if mode in ("fwd_only", "no_shared") else np.where((K >= 1)[:, None], shared, base)
    x = x.copy(); x[~avail] = 0.0; return x.astype(np.float32)

def stage(a):
    """Parent: write aggregation arrays (aligned to the filtered pair tables) to /dev/shm."""
    sh = Path(a.shm); sh.mkdir(parents=True, exist_ok=True)
    if (sh / "staged.ok").exists(): return
    st = json.load(open(Path(a.p5) / "standardizers.json")); std = (np.array([st["sam"][c]["mean"] for c in COLS]), np.array([st["sam"][c]["std"] for c in COLS]))
    for s in a.splits.split(","):
        full = pd.read_csv(Path(a.p5) / f"pair_vectors_{s}.csv", usecols=["pair_id"], dtype={"pair_id": str}); q = pd.read_csv(Path(a.p5q) / f"pair_vectors_{s}.csv", usecols=["pair_id"], dtype={"pair_id": str})
        pos = pd.Series(np.arange(len(full)), index=full.pair_id.values)[q.pair_id.values].to_numpy(); raw = np.load(Path(a.p5) / f"raw_{s}.npz")
        for mode in ("fwd_only", "no_shared", "shared_only"): np.save(sh / f"agg_{mode}_{s}.npy", agg_arrays(raw, std, mode)[pos])
    (sh / "staged.ok").write_text("ok")

def transform(df, v, agg_dir, split):
    df = df.copy(); df["gQ"], df["gR"], df["gA"] = df.Q.to_numpy(), df.R_sam.to_numpy(), df.avail_sam.to_numpy()
    if v["agg"] != "default": df[S_] = np.load(Path(agg_dir) / f"agg_{v['agg']}_{split}.npy")[df["_row"].to_numpy()]
    if v["drop_direct"]: df[D] = 0.0
    if v["drop_sam"]: df[S_] = 0.0; df["R_sam"] = 0.0; df["avail_sam"] = 0; df["avail_multi"] = 0; df["gA"] = 0
    drop = set(v["drop_cols"]) | (set(COLS) - set(v["keep"]) if v["keep"] else set())
    for c in drop: df["d_" + c] = 0.0; df["s_" + c] = 0.0
    if v["ctx"] == "none":
        for c in ["g", "Q", "enhanced_pair", "R_sam", "R_diff", "avail_sam", "avail_diff", "avail_multi"]: df[c] = 0.0
    if v["ctx"] == "no_R": df["R_sam"] = 0.0
    return df

# ---------------------------------------------------------------- scorers
class Scorer:
    def __init__(self, kind, payload): self.kind, self.p = kind, payload
    def __call__(self, df, dev):
        if self.kind == "pre":                                  # externally computed score (baseline models): looked up by pair id
            S = df.pair_id.map(self.p["scores"]).to_numpy(np.float64); assert np.isfinite(S).all(), f"{int((~np.isfinite(S)).sum())} pairs have no baseline score: use the subset file (--subset)"
            W = np.zeros((len(df), 3), np.float32); W[:, 0] = S; return S, W
        if self.kind == "mlp":
            from aafv import fusion
            return fusion.predict([fusion.load_model_from_state(s, dev) for s in self.p], df, dev)
        X = self._x(df); S = self.p["m"].decision_function(X); c = self.p["m"].coef_[0]; W = np.zeros((len(df), 3), np.float32)
        if self.kind == "arc": W[:, 0] = S
        else: W[:, 0] = df[D].to_numpy() @ c[:5]; W[:, 1] = df[S_].to_numpy() @ c[5:10]
        return S, W
    def _x(self, df): return df[["d_arc"]].to_numpy() if self.kind == "arc" else df[D + S_ + ["g", "Q", "avail_sam"]].to_numpy()

def fit_scorer(v, tr, va, cfg, lr, wd, seeds, dev, max_epochs):
    w = tr.weight.to_numpy() if v["weights"] else np.ones(len(tr))
    if v["kind"] in ("arc", "logreg"):
        sc = Scorer(v["kind"], {}); X = sc._x(tr); m = LogisticRegression(C=1.0, max_iter=500).fit(X, tr.label, sample_weight=w); sc.p = {"m": m}; return sc
    from aafv import fusion
    tr = tr.assign(weight=w); pm = cfg["P_MASK"] if v["p_mask"] is None else v["p_mask"]
    if v["drop_sam"]: pm = 0.0
    states = []
    for s in range(seeds):
        m, _, _ = fusion.train_one(tr, va, 1000 * s + cfg["SEED"], lr, wd, pm, cfg["MLP_SIZES"], dev, max_epochs); states.append({k: x.cpu() for k, x in m.state_dict().items()})
    return Scorer("mlp", states)

# ---------------------------------------------------------------- decision (same rule as P8)
def pass_guard(df, Pd, Ps, g, qmin, rmin, rho):
    uq, ur, upd, ups = GUARD[g]; ok = np.ones(len(df), bool)
    if uq: ok &= df.gQ.to_numpy() >= qmin
    if upd: ok &= Pd >= rho
    r_ok = (df.gR.to_numpy() >= rmin) if ur else np.ones(len(df), bool); s_ok = (Ps >= rho) if ups else np.ones(len(df), bool)
    return ok & ((df.gA.to_numpy() == 0) | (r_ok & s_ok))

def fit_decision(v, cfg, tr, va, ca, Sv, Wv, Sc, Wc):
    yv = va.label.to_numpy(); cal = calib.Calibrator("platt").fit(Sv, yv)
    pdl = LogisticRegression(C=1e6, max_iter=1000).fit(Wv[:, [0]], yv); m = va.avail_sam.to_numpy() == 1
    psl = LogisticRegression(C=1e6, max_iter=1000).fit(Wv[m][:, [1]], yv[m]) if m.sum() > 50 and Wv[m][:, 1].std() > 0 else None
    Pc = cal.predict(Sc); Pd = pdl.predict_proba(Wc[:, [0]])[:, 1]; Ps = psl.predict_proba(Wc[:, [1]])[:, 1] if psl is not None else np.ones(len(ca))
    qmin = float(np.percentile(tr.gQ, 5)); r = tr.gR[tr.gA == 1]; rmin = float(np.percentile(r, 5)) if len(r) else 0.0
    same = ca.label.to_numpy() == 1; best = None
    rhos = cfg["RHO_OPTIONS"] if (GUARD[v["guard"]][2] or GUARD[v["guard"]][3]) else [0.0]
    for rho in rhos:
        ok = pass_guard(ca, Pd, Ps, v["guard"], qmin, rmin, rho); imp = Pc[(~same) & ok]
        if len(imp) == 0: ta, tr_ = 2.0, float(np.sort(Pc[same])[0] - 0.001)
        else: ta, tr_, _ = thresholds.thresholds(imp, Pc[same])
        z = np.where((thresholds.zone(Pc, ta, tr_) == "accept") & ~ok, "review", thresholds.zone(Pc, ta, tr_)); acc = float((z[same] == "accept").mean())
        if best is None or acc > best[0]: best = (acc, rho, ta, tr_)
    return dict(cal=cal, pdl=pdl, psl=psl, qmin=qmin, rmin=rmin, rho=best[1], t_acc=best[2], t_rej=best[3], guard=v["guard"])

def decide(dec, df, S, W):
    P = dec["cal"].predict(S); Pd = dec["pdl"].predict_proba(W[:, [0]])[:, 1]; Ps = dec["psl"].predict_proba(W[:, [1]])[:, 1] if dec["psl"] is not None else np.ones(len(df))
    ok = pass_guard(df, Pd, Ps, dec["guard"], dec["qmin"], dec["rmin"], dec["rho"]); z = thresholds.zone(P, dec["t_acc"], dec["t_rej"]); return P, np.where((z == "accept") & ~ok, "review", z)

def summarize(df, P, z):
    same = df.label.to_numpy() == 1; df = df.assign(P=P, z=z, same=same); out = {}
    for name, d in (("all", df), ("G1-G3", df[df.gap_bin.isin(["G1", "G2", "G3"])]), ("G4-G6", df[df.gap_bin.isin(["G4", "G5", "G6"])])): s = d[d.same]; out[f"same_accept_{name}"] = float((s.z == "accept").mean()) if len(s) else np.nan
    for st in ("D_random", "D_all_random", "D_hard"):
        d = df[df.same | (df.set_name == st)]; imp = df[df.set_name == st]; out[f"FA_{st}"] = int((imp.z == "accept").sum()); out[f"N_{st}"] = len(imp)
        if len(imp) and d.same.any(): out[f"AUC_{st}"] = float(roc_auc_score(d.same, d.P)); f, t, _ = roc_curve(d.same, d.P); out[f"TPR@FPR1e-3_{st}"] = float(t[f <= 1e-3].max()) if (f <= 1e-3).any() else 0.0
    return out

# ---------------------------------------------------------------- worker
def run_variant(name, a, cfg):
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"; v = V[name]; sh = Path(a.shm); od = Path(a.out) / "variants" / name; od.mkdir(parents=True, exist_ok=True)
    fr = load_frames(a, ["train_fit", "train_val", "calibration"]); T = {s: transform(fr[s], v, sh, s) for s in fr}
    sc = fit_scorer(v, T["train_fit"], T["train_val"], cfg, a.lr, a.wd, a.seeds, dev, a.max_epochs)
    Sv, Wv = sc(T["train_val"], dev); Sc, Wc = sc(T["calibration"], dev); dec = fit_decision(v, cfg, T["train_fit"], T["train_val"], T["calibration"], Sv, Wv, Sc, Wc)
    P, z = decide(dec, T["calibration"], Sc, Wc); res = dict(variant=name, group=v["group"], T_accept=dec["t_acc"], RHO=dec["rho"], **summarize(T["calibration"], P, z))
    bad = {k: res[k] for k in ("FA_D_random", "FA_D_all_random", "FA_D_hard") if res[k] != 0}
    if bad: print(f"WARNING {name}: calibration false accepts should be 0 by construction: {bad}", flush=True)
    pickle.dump(dict(scorer=sc, dec=dec), open(od / "frozen.pkl", "wb")); json.dump(res, open(od / "calibration.json", "w")); print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in res.items()}, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="calib"); ap.add_argument("--p5", required=True); ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--main_models", required=True)
    ap.add_argument("--shm", default="/dev/shm/aafv/abl"); ap.add_argument("--splits", default="train_fit,train_val,calibration,test"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max_epochs", type=int, default=100); ap.add_argument("--worker", type=int, default=-1); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--only", default="")
    ap.add_argument("--subset", default=""); ap.add_argument("--i_understand_this_uses_a_look", action="store_true"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml")); ap.add_argument("--test_log", default="artifacts/q/logs/test_access.log")
    a = ap.parse_args(); cfg = config.load(a.params); Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.subset: os.environ["AAFV_SUBSET"] = a.subset
    meta = __import__("torch").load(next(Path(a.main_models).glob("fusion_seed0.pt")), weights_only=False)["meta"]; a.lr, a.wd = meta["lr"], meta["wd"]
    names = [n for n in V if V[n]["kind"] != "pre" and (not a.only or n in a.only.split(","))]
    if a.phase == "calib":
        if a.worker >= 0:
            for n in names[a.worker::a.workers]: run_variant(n, a, cfg)
            return
        stage(a); devs = devices(); procs = []
        for i in range(a.workers):
            cmd = [sys.executable, __file__, "--phase", "calib", "--worker", str(i), "--workers", str(a.workers), "--p5", a.p5, "--p5q", a.p5q, "--out", a.out, "--main_models", a.main_models, "--shm", a.shm, "--seeds", str(a.seeds), "--max_epochs", str(a.max_epochs), "--only", a.only]
            procs.append(subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": devs[i]}, stdout=open(Path(a.out) / f"worker{i}.log", "w"), stderr=subprocess.STDOUT))
        for p in procs: p.wait()
        rows = [json.load(open(Path(a.out) / "variants" / n / "calibration.json")) for n in names if (Path(a.out) / "variants" / n / "calibration.json").exists()]
        df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "ablation_calibration.csv", index=False); print(df.round(4).to_string(index=False))
        h = {n: hashlib.sha256(open(Path(a.out) / "variants" / n / "frozen.pkl", "rb").read()).hexdigest() for n in names if (Path(a.out) / "variants" / n / "frozen.pkl").exists()}
        json.dump(h, open(Path(a.out) / "ablation_freeze.json", "w"), indent=1); print("variants frozen:", len(h)); return
    # ---- test phase: one logged look for every frozen variant at once
    assert a.i_understand_this_uses_a_look, "pass --i_understand_this_uses_a_look"
    h = json.load(open(Path(a.out) / "ablation_freeze.json")); import torch
    for n, s in h.items(): assert hashlib.sha256(open(Path(a.out) / "variants" / n / "frozen.pkl", "rb").read()).hexdigest() == s, f"{n} changed since freeze"
    log = Path(a.test_log); used = len(log.read_text().splitlines()) if log.exists() else 0; assert used < cfg["LOOKS_MAX"], f"{used} looks already used"
    log.parent.mkdir(parents=True, exist_ok=True); open(log, "a").write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} look={used+1} ablations\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; te0 = load_frames(a, ["test"])["test"]; rows = []
    for n in h:
        v = V[n]; fz = pickle.load(open(Path(a.out) / "variants" / n / "frozen.pkl", "rb")); te = transform(te0, v, a.shm, "test"); S, W = fz["scorer"](te, dev); P, z = decide(fz["dec"], te, S, W)
        r = dict(variant=n, group=v["group"], **summarize(te, P, z)); sm = te0.assign(acc=(z == "accept"))[te0.label == 1]; g = sm.groupby("person_id_1").acc.agg(["sum", "count"]); ix = np.random.default_rng(42).integers(0, len(g), (1000, len(g)))
        b = g["sum"].to_numpy()[ix].sum(1) / g["count"].to_numpy()[ix].sum(1); r["same_accept_CI95"] = f"{np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f}"; r["FA_random_bound"] = evaluation.zero_error_bound(r["N_D_random"], 0.05 / cfg["LOOKS_MAX"]) if r["FA_D_random"] == 0 else np.nan; rows.append(r); print(n, flush=True)
    df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "ablation_test.csv", index=False); print(df.round(4).to_string(index=False))

if __name__ == "__main__": main()