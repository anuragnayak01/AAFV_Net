import numpy as np, pandas as pd
from .gap import gap_bin

def same_pairs(df, cap=None, rng=None):
    rows = []
    for pid, grp in df.groupby("person_id"):
        g_ = grp.sort_values("age")
        ids = g_["img_id"].tolist(); ages = g_["age"].tolist(); gen = g_["gender"].iloc[0]
        cand = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                gp = ages[j] - ages[i]
                cand.setdefault(gap_bin(gp), []).append((ids[i], ids[j], pid, ages[i], ages[j], gen))
        for b, lst in cand.items():
            if cap is not None and len(lst) > cap:
                sel = rng.choice(len(lst), cap, replace=False)
                lst = [lst[k] for k in sel]
            rows += lst
    return _frame(rows, "same", 1)

def _frame(rows, set_name, label):
    out = pd.DataFrame(rows, columns=["img_id_1", "img_id_2", "person_id", "a1", "a2", "gender"])
    out["person_id_1"] = out["person_id"]; out["person_id_2"] = out["person_id"]
    return out.drop(columns="person_id").assign(set_name=set_name, label=label)

def d_random(df, rng):
    """Person-disjoint matching, same gender, one random image per person."""
    rows = []
    people = df.groupby("person_id").agg(gender=("gender", "first")).reset_index()
    for gen, grp in people.groupby("gender"):
        ids = grp["person_id"].to_numpy().copy(); rng.shuffle(ids)
        for k in range(0, len(ids) - 1, 2):
            pa, pb = ids[k], ids[k + 1]
            ia = df[df.person_id == pa].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
            ib = df[df.person_id == pb].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
            if ia.age > ib.age: ia, ib = ib, ia
            rows.append((ia.img_id, ib.img_id, ia.person_id, ib.person_id, int(ia.age), int(ib.age), gen))
    out = pd.DataFrame(rows, columns=["img_id_1", "img_id_2", "person_id_1", "person_id_2", "a1", "a2", "gender"])
    return out.assign(set_name="D_random", label=0)

def d_all_random(df, n, rng):
    rows = []
    for gen, grp in df.groupby("gender"):
        m = int(round(n * len(grp) / len(df)))
        v = grp[["img_id", "person_id", "age"]].to_numpy()
        i = rng.integers(0, len(v), m); j = rng.integers(0, len(v), m)
        keep = v[i, 1] != v[j, 1]
        for a, b in zip(v[i[keep]], v[j[keep]]):
            if a[2] > b[2]: a, b = b, a
            rows.append((a[0], b[0], a[1], b[1], int(a[2]), int(b[2]), gen))
    out = pd.DataFrame(rows, columns=["img_id_1", "img_id_2", "person_id_1", "person_id_2", "a1", "a2", "gender"])
    return out.assign(set_name="D_all_random", label=0)

def finalize(p, split, dataset):
    p = p.copy()
    p["gap"] = p["a2"] - p["a1"]
    assert (p["gap"] >= 0).all()
    p["gap_bin"] = p["gap"].map(gap_bin)
    assert ((p.person_id_1 == p.person_id_2).astype(int) == p.label).all()
    p["split"] = split; p["dataset"] = dataset; p["weight"] = 1.0
    p.insert(0, "pair_id", [f"{split}:{i}" for i in range(len(p))])
    return p


def train_subsplit(person_ids, frac, seed):
    """Persons of train that go to train_val. Depends only on the sorted person list, never on image filters."""
    ids = np.array(sorted(set(person_ids))); rng = np.random.default_rng(seed); rng.shuffle(ids)
    return set(ids[: int(round(frac * len(ids)))])

def hard_negatives(df, emb, k, chunk=2000):
    """For each image, its k most ArcFace-similar images of a DIFFERENT person with the SAME gender, inside df.
    df: img_id, person_id, age, gender (row order == emb row order, emb L2-normalized). Returns an ordered pair table (a1 <= a2)."""
    pid = df["person_id"].to_numpy(); gen = df["gender"].to_numpy(); ids = df["img_id"].to_numpy(); age = df["age"].to_numpy()
    E = emb.astype(np.float32); rows = set()
    for s0 in range(0, len(df), chunk):
        sim = E[s0:s0 + chunk] @ E.T
        bad = (pid[s0:s0 + chunk, None] == pid[None, :]) | (gen[s0:s0 + chunk, None] != gen[None, :])
        sim[bad] = -2.0
        kk = min(k, sim.shape[1] - 1)
        top = np.argpartition(-sim, kk, axis=1)[:, :kk]
        for i, row in enumerate(top):
            for j in row:
                if sim[i, j] > -2.0:
                    a, b = s0 + i, int(j)
                    if age[a] > age[b]: a, b = b, a
                    rows.add((ids[a], ids[b], pid[a], pid[b], int(age[a]), int(age[b]), gen[a]))
    out = pd.DataFrame(sorted(rows), columns=["img_id_1", "img_id_2", "person_id_1", "person_id_2", "a1", "a2", "gender"])
    return out.assign(set_name="D_hard", label=0)
