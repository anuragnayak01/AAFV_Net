"""P9: the one-look test. Refuses to run unless thresholds.json matches its recorded SHA-256 and a look is available.
Usage: python scripts/p9_test.py --p5 artifacts/p5 --models artifacts/models --out artifacts --i_understand_this_uses_a_look"""
import sys, json, argparse, pickle, hashlib, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from aafv import config, fusion, calib, thresholds, evaluation

ap = argparse.ArgumentParser()
for k in ("p5", "models", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--i_understand_this_uses_a_look", action="store_true"); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); dev = "cuda" if torch.cuda.is_available() else "cpu"
assert a.i_understand_this_uses_a_look, "pass --i_understand_this_uses_a_look: this spends one of the test looks"
tp = out / "thresholds/thresholds.json"; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert tp.exists() and sha(tp) == (out / "thresholds/thresholds.sha256").read_text().strip(), "thresholds.json missing or changed since freeze: refusing to run"
th = json.load(open(tp)); assert th["config_hash"] == config.config_hash(cfg), "params.yaml changed since freeze"
mp = sorted(Path(a.models).glob("fusion_seed*.pt")); assert {p.name: sha(p) for p in mp} == th["model_sha256"], "model files changed since freeze"
assert sha(th["calibrator_file"]) == th["calibrator_sha256"], "calibrator changed since freeze"
log = out / "logs/test_access.log"; log.parent.mkdir(parents=True, exist_ok=True)
used = len(log.read_text().splitlines()) if log.exists() else 0; assert used < cfg["LOOKS_MAX"], f"{used} looks already used (LOOKS_MAX={cfg['LOOKS_MAX']})"
look = used + 1; open(log, "a").write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} look={look} thresholds_sha256={sha(tp)}\n")
alpha = 0.05 / cfg["LOOKS_MAX"]; print(f"LOOK {look} of {cfg['LOOKS_MAX']} | alpha per look {alpha}")

te = pd.read_csv(Path(a.p5) / "pair_vectors_test.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str})
models = [fusion.load_model(p, dev) for p in mp]; pk = pickle.load(open(th["calibrator_file"], "rb"))
S, W = fusion.predict(models, te, dev); P = pk["cal"].predict(S, te.g.to_numpy())
Pd = pk["pd"].predict_proba(W[:, [0]])[:, 1]; Ps = pk["ps"].predict_proba(W[:, [1]])[:, 1]
ok = calib.guard_pass(te.Q.to_numpy(), te.R_sam.to_numpy(), te.avail_sam.to_numpy(), Pd, Ps, th["Q_MIN"], th["R_MIN_sam"], th["RHO"])
z = thresholds.zone(P, th["T_accept"], th["T_reject"]); z = np.where((z == "accept") & ~ok, "review", z); te["zone"] = z; te["P"] = P
te["set"] = np.where(te.label == 1, "same", te.set_name)
rep = {"look": look, "alpha_per_look": alpha}
tab = te.groupby("set").zone.value_counts().unstack(fill_value=0).reindex(columns=["reject", "review", "accept"], fill_value=0); print(tab.to_string()); rep["zones_by_set"] = tab.to_dict("index")
by_bin = te.groupby(["set", "gap_bin"]).zone.value_counts().unstack(fill_value=0).reindex(columns=["reject", "review", "accept"], fill_value=0); print(by_bin.to_string())
by_gen = te.groupby(["set", "gender"]).zone.value_counts().unstack(fill_value=0).reindex(columns=["reject", "review", "accept"], fill_value=0); print(by_gen.to_string())
dr = te[te.set == "D_random"]; N = len(dr); fa = int((dr.zone == "accept").sum()); rep["N_D_random"] = N; rep["FA_random"] = fa
print(f"\nFA_random = {fa} of N = {N}")
if fa == 0:
    b = evaluation.zero_error_bound(N, alpha); rep["bound"] = b; print(f"CLAIM: 0 of {N} person-disjoint different-person pairs auto-accepted -> false-accept rate < {b:.5f} (about 1 in {int(1/b)}) at {100*(1-alpha):.1f}% confidence")
else:
    rep["upper_clopper_pearson"] = evaluation.clopper_pearson_upper(fa, N, alpha); print("FA_random > 0: the claim is NOT made. Follow 06 section 4. Do not change any threshold.")
for s in ("D_all_random", "D_hard"):
    d = te[te.set == s]; rep[f"FA_{s}"] = int((d.zone == "accept").sum()); print(f"{s}: {len(d)} pairs, auto-accepted {rep[f'FA_{s}']}")
sm = te[te.set == "same"]; fr = sm[sm.zone == "reject"].groupby("gap_bin").agg(false_rejects=("pair_id", "size"), people=("person_id_1", "nunique")); print("false rejects (same person auto-rejected) by bin:\n", fr.to_string() if len(fr) else "none")
rep["false_rejects"] = fr.to_dict("index")
def boot(d, flag, n=1000, seed=42):
    g = d.assign(f=flag).groupby("person_id_1").f.agg(["sum", "count"]); s_, c_ = g["sum"].to_numpy(), g["count"].to_numpy()
    ix = np.random.default_rng(seed).integers(0, len(g), (n, len(g))); r = s_[ix].sum(1) / c_[ix].sum(1); return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
da = te[te.set == "D_all_random"]
rep["boot_FA_all_random_95"] = boot(da, (da.zone == "accept").to_numpy()); rep["boot_review_same_95"] = boot(sm, (sm.zone == "review").to_numpy()); rep["boot_accept_same_95"] = boot(sm, (sm.zone == "accept").to_numpy())
print("person-level bootstrap 95% intervals:", {k: [round(x, 5) for x in v] for k, v in rep.items() if k.startswith("boot")})
print(f"same-person: auto-accepted {(sm.zone=='accept').mean():.3%}, review {(sm.zone=='review').mean():.3%}, auto-rejected {(sm.zone=='reject').mean():.3%}")
(out / "report").mkdir(exist_ok=True); json.dump(rep, open(out / f"report/test_report_look{look}.json", "w"), indent=1, default=str)
te.drop(columns=["P"]).assign(P=P)[["pair_id", "set", "label", "gap_bin", "gender", "P", "zone"]].to_csv(out / f"report/test_pair_zones_look{look}.csv", index=False)
config.evidence("P9", cfg, look=look, FA_random=fa)
