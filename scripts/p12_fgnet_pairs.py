"""FG-NET external large-gap evaluation: pair table (schema of pairs_<split>.csv, split name 'fgnet'). No gender rule (gender not in the image names).
Same-person: every image pair of a person (a1 <= a2). Impostors: --n_imp random pairs of different people (set FG_random). Images are the aligned crops that P1 accepted.
  python scripts/p12_fgnet_pairs.py --p1_csv artifacts/p1/detect_quality.csv --out artifacts/pairs_fg [--n_imp 20000]
Also writes <out>/fg_ids.txt (all image ids used) for p5_faces --only_list."""
import sys, re, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from aafv import pairs, config
ap = argparse.ArgumentParser(); ap.add_argument("--p1_csv", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n_imp", type=int, default=20000)
ap.add_argument("--params", default=str(ROOT / "configs/params.yaml")); a = ap.parse_args(); cfg = config.load(a.params); rng = np.random.default_rng(cfg["SEED"])
p1 = pd.read_csv(a.p1_csv); p1 = p1[p1.ok == 1]
rx = re.compile(r"^(\d+)A(\d+)", re.I); rows = []
for f in p1.img_id:
    m = rx.match(str(f))
    if m and "." in str(f) and not re.match(r"^\d{5}_", str(f)): rows.append((f, "fg" + m.group(1), int(m.group(2))))   # FG-NET names look like 001A02.JPG; MORPH names look like 00013_00M19.JPG
d = pd.DataFrame(rows, columns=["img_id", "person_id", "age"]); d["gender"] = "U"
print(f"{len(d)} FG-NET images, {d.person_id.nunique()} people, ages {d.age.min()}-{d.age.max()}")
same = pairs.same_pairs(d, cap=None, rng=rng)
v = d[["img_id", "person_id", "age"]].to_numpy(); i = rng.integers(0, len(v), a.n_imp * 2); j = rng.integers(0, len(v), a.n_imp * 2); keep = v[i, 1] != v[j, 1]; imp = []
for x, y in zip(v[i[keep]][:a.n_imp], v[j[keep]][:a.n_imp]):
    if x[2] > y[2]: x, y = y, x
    imp.append((x[0], y[0], x[1], y[1], int(x[2]), int(y[2]), "U"))
imp = pd.DataFrame(imp, columns=["img_id_1", "img_id_2", "person_id_1", "person_id_2", "a1", "a2", "gender"]).assign(set_name="FG_random", label=0)
p = pairs.finalize(pd.concat([same, imp], ignore_index=True), "fgnet", "fgnet"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
p.to_csv(out / "pairs_fgnet.csv", index=False); (out / "fg_ids.txt").write_text("\n".join(d.img_id)); print(p.groupby(["set_name", "gap_bin"]).size().unstack(fill_value=0).to_string())
