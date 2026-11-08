"""P2 + P3.1 (GPU, one slice): quality normalization, ArcFace embeddings of the aligned crops, DEX age estimates, age error summary.
Writes artifacts/p2/{quality.csv, quality_norm.json, arc_orig.npy, arc_ids.csv, age_est.csv, age_error.json}."""
import sys, os, re, json, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))[0]
import numpy as np, pandas as pd, cv2, torch
from aafv import config, pairs, quality
import c4_eval_age

ap = argparse.ArgumentParser()
for k in ("split_csv", "p1_csv", "crops", "out", "sam_dir", "dex"): ap.add_argument("--" + k, required=True)
ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

sp = pd.read_csv(a.split_csv, dtype=str)
p1 = pd.read_csv(a.p1_csv); p1 = p1[p1.ok == 1].reset_index(drop=True)
lab = {r.filename: (int(r.age), r.new_split, r.person_id) for r in sp.itertuples()}
for f in p1.img_id:
    if f not in lab: lab[f] = (int(re.match(r"\d+A(\d+)", f).group(1)), "fgnet", "fg" + re.match(r"(\d+)A", f).group(1))
p1["label_age"] = p1.img_id.map(lambda f: lab[f][0]); p1["split0"] = p1.img_id.map(lambda f: lab[f][1]); p1["person_id"] = p1.img_id.map(lambda f: lab[f][2])
val_people = pairs.train_subsplit(sp[sp.new_split == "train"].person_id.unique(), cfg["TRAIN_VAL_FRACTION"], cfg["SEED"])
p1["split"] = np.where(p1.split0 == "train", np.where(p1.person_id.isin(val_people), "train_val", "train_fit"), p1.split0)

# quality: normalization fitted on train_fit only
norm = quality.fit_norm(p1[p1.split == "train_fit"]); json.dump(norm, open(out / "quality_norm.json", "w"), indent=1)
qd = quality.normalize(p1, norm); qd[["img_id", "split", "Q"] + ["q_" + m for m in quality.MEASURES]].to_csv(out / "quality.csv", index=False)
print("Q: mean %.3f, 5th pct %.3f" % (qd.Q.mean(), qd.Q.quantile(0.05)), flush=True)

from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0); rec = app.models["recognition"]
print("ArcFace providers in use:", rec.session.get_providers(), flush=True)
net = c4_eval_age.load_dex(a.sam_dir, a.dex, "cuda")

ids = p1.img_id.tolist(); B = 128; embs = np.zeros((len(ids), 512), np.float16); ages = np.zeros(len(ids), np.float32)
def load(i): return cv2.imread(str(Path(a.crops) / (i + ".png")))
with ThreadPoolExecutor(16) as ex:
    for s in range(0, len(ids), B):
        ims = list(ex.map(load, ids[s:s + B]))
        f = rec.get_feat([cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA) for im in ims])
        embs[s:s + B] = (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float16)
        ages[s:s + B] = c4_eval_age.dex_age(net, [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in ims], "cuda")
        if (s // B) % 50 == 0: print(f"  {s}/{len(ids)}", flush=True)
np.save(out / "arc_orig.npy", embs); pd.DataFrame({"img_id": ids}).to_csv(out / "arc_ids.csv", index=False)
est = pd.DataFrame({"img_id": ids, "est_age": ages, "label_age": p1.label_age.values, "split": p1.split.values})
est.to_csv(out / "age_est.csv", index=False)

v = est[est.split == "train_val"].copy(); v["err"] = v.est_age - v.label_age
bias = float(v.err.mean()); eps = float(min(cfg["EPS_LIMIT"], (v.err - bias).abs().quantile(0.75)))
bands = pd.cut(v.label_age, [-1, 14, 19, 24, 29, 34, 39, 49, 100])
res = {"bias_global": bias, "mae_global": float(v.err.abs().mean()), "eps": eps, "n_images": int(len(v)),
       "mae_by_band": {str(k): float(g.abs().mean()) for k, g in v.groupby(bands, observed=True).err},
       "bias_by_band": {str(k): float(g.mean()) for k, g in v.groupby(bands, observed=True).err},
       "eps_rule": "min(EPS_LIMIT, 75th percentile of |error - bias_global| on train_val)"}
json.dump(res, open(out / "age_error.json", "w"), indent=1); print(json.dumps(res, indent=1))
config.evidence("P2", cfg, n_images=len(ids), out=str(out))
