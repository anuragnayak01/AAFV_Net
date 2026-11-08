"""Quarantine duplicate identities (label-free rule in src/aafv/dupids.py) and write filtered pair tables.
Usage: python scripts/p_quarantine.py --split_csv F --p2 DIR --p5 artifacts/p5 --out_p5 artifacts/p5q --audit artifacts/audit"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from aafv import dupids
ap = argparse.ArgumentParser()
for k in ("split_csv", "p2", "p5", "out_p5", "audit"): ap.add_argument("--" + k, required=True)
ap.add_argument("--extra_people", default="", help="comma-separated person ids quarantined after visual audit (D30)")
a = ap.parse_args(); Path(a.out_p5).mkdir(parents=True, exist_ok=True); Path(a.audit).mkdir(parents=True, exist_ok=True)
sp = pd.read_csv(a.split_csv, dtype=str); sp["img_id"] = sp.filename.astype(str)
ids = pd.read_csv(Path(a.p2) / "arc_ids.csv").img_id.astype(str); E = np.load(Path(a.p2) / "arc_orig.npy").astype(np.float32)
m = ids.to_frame().merge(sp[["img_id", "person_id", "new_split"]], on="img_id", how="left"); ok = m.person_id.notna().to_numpy(); m = m[ok].reset_index(drop=True); E = E[ok]
E /= np.linalg.norm(E, axis=1, keepdims=True); up, inv = np.unique(m.person_id.to_numpy(), return_inverse=True); ix = {p: i for i, p in enumerate(up)}
C = np.zeros((len(up), E.shape[1]), np.float32); np.add.at(C, inv, E); C /= np.linalg.norm(C, axis=1, keepdims=True)
q, rows = dupids.quarantine_people(list(up), lambda x, y: C[ix[x]] @ C[ix[y]])
q |= {x for x in a.extra_people.split(",") if x}
pd.DataFrame(rows, columns=["short", "long", "cos", "reason"]).to_csv(Path(a.audit) / "quarantine_pairs.csv", index=False)
split_of = m.groupby("person_id").new_split.first()
pd.DataFrame({"person_id": sorted(q), "split": [split_of[p] for p in sorted(q)]}).to_csv(Path(a.audit) / "quarantine_people.csv", index=False)
print(f"{len(rows)} duplicate-ID pairs -> {len(q)} people quarantined of {len(up)} ({100*len(q)/len(up):.1f}%)"); print(pd.Series([split_of[p] for p in q]).value_counts().to_dict())
for sname in ("train_fit", "train_val", "calibration", "test"):
    d = pd.read_csv(Path(a.p5) / f"pair_vectors_{sname}.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str})
    bad = d.person_id_1.isin(q) | d.person_id_2.isin(q); k = d[~bad]; k.to_csv(Path(a.out_p5) / f"pair_vectors_{sname}.csv", index=False)
    print(f"{sname}: {len(d)} -> {len(k)} pairs (dropped {int(bad.sum())}); same {int(d.label.sum())} -> {int(k.label.sum())}; D_random {int((d.set_name=='D_random').sum())} -> {int((k.set_name=='D_random').sum())}")
