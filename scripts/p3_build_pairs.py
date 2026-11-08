"""P3.2 pair builder (MORPH). Usage: p3_build_pairs.py SPLIT_CSV OUT_DIR [--usable usable.csv]"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/scripts", 1)[0] + "/src")
from aafv import config, pairs
ap = argparse.ArgumentParser(); ap.add_argument("split_csv"); ap.add_argument("out_dir")
ap.add_argument("--usable"); ap.add_argument("--emb"); ap.add_argument("--emb_ids"); ap.add_argument("--n_all_random", type=int); ap.add_argument("--params", default=__file__.rsplit("/scripts", 1)[0] + "/configs/params.yaml")
ap.add_argument("--skip_all_random", action="store_true")
a = ap.parse_args(); cfg = config.load(a.params); rng = np.random.default_rng(cfg["SEED"])
df = pd.read_csv(a.split_csv, dtype={"identity": str, "person_id": str})
df = df[df.status != "quarantined"] if "status" in df else df
df = df.rename(columns={"gender_from_filename": "gender", "new_split": "split0"})
df["img_id"] = df["filename"]
val = pairs.train_subsplit(df[df.split0 == "train"].person_id.unique(), cfg["TRAIN_VAL_FRACTION"], cfg["SEED"])   # before any filter
if a.usable:
    u = pd.read_csv(a.usable); ok = set(u[u.ok == 1]["img_id"]); n0 = len(df); df = df[df.img_id.isin(ok)]
    print("dropped unusable:", n0 - len(df))
emb_of = {}
if a.emb:
    E = np.load(a.emb).astype(np.float32); eid = pd.read_csv(a.emb_ids)["img_id"].tolist(); emb_of = {i: k for k, i in enumerate(eid)}
df["split"] = np.where(df.split0 == "train", np.where(df.person_id.isin(val), "train_val", "train_fit"), df.split0)
assert df.groupby("person_id").split.nunique().max() == 1
import os; os.makedirs(a.out_dir, exist_ok=True)
df[["person_id", "split"]].drop_duplicates().to_csv(f"{a.out_dir}/train_subsplit.csv", index=False)
for sp in ["train_fit", "train_val", "calibration", "test"]:
    d = df[df.split == sp]; cap = cfg["CAP_SAME"] if sp.startswith("train") else None
    s = pairs.same_pairs(d, cap, rng)
    parts = [s]
    n_rand = a.n_all_random or cfg["RANDOM_DIFF_PAIRS"]
    hard = pairs.hard_negatives(d, E[[emb_of[i] for i in d.img_id]], cfg["K_HARD"]) if emb_of else None
    if sp in ("calibration", "test"):
        parts.append(pairs.d_random(d, rng))
        if not a.skip_all_random: parts.append(pairs.d_all_random(d, n_rand, rng))
        if hard is not None: parts.append(hard)
    else:
        n_diff = cfg["DIFF_PER_SAME"] * len(s)
        parts.append(pairs.d_all_random(d, n_diff // 2, rng).assign(set_name="train_diff_random"))
        if hard is not None: parts.append(hard.sample(n=min(len(hard), n_diff // 2), random_state=cfg["SEED"]).assign(set_name="train_diff_hard"))
    p = pairs.finalize(pd.concat(parts, ignore_index=True), sp, "morph")
    p.to_csv(f"{a.out_dir}/pairs_{sp}.csv", index=False)
    print(f"--- {sp}: people={d.person_id.nunique()} images={len(d)}")
    print(p.groupby(["set_name", "gap_bin"]).size().unstack(fill_value=0).to_string())
config.evidence("P3", cfg, out=a.out_dir)
