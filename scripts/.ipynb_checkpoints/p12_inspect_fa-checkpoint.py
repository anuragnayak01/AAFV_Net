"""List (and draw) the false accepts of one frozen variant on an external vector table, with the same frozen thresholds as p12_extended.py.
  python scripts/p12_inspect_fa.py --p5q artifacts/p5q --abl artifacts/ablations --extra_vectors artifacts/p5_fg/pair_vectors_fgnet.csv --png artifacts/audit_fg_fa.png"""
import sys, argparse, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--p5q", required=True); ap.add_argument("--abl", required=True); ap.add_argument("--extra_vectors", required=True)
ap.add_argument("--variant", default="full"); ap.add_argument("--beta", type=float, default=0.01); ap.add_argument("--shm", default="/dev/shm/aafv/abl"); ap.add_argument("--crops", default="/dev/shm/aafv/crops"); ap.add_argument("--png", default="")
a = ap.parse_args()
import numpy as np, pandas as pd, torch
import p10_ablations as A
import __main__; __main__.Scorer = A.Scorer
dev = "cuda" if torch.cuda.is_available() else "cpu"; n = a.variant; fz = pickle.load(open(Path(a.abl) / "variants" / n / "frozen.pkl", "rb"))
cal = A.load_frames(a, ["calibration"])["calibration"]; Tc = A.transform(cal, A.V[n], a.shm, "calibration"); Sc, Wc = fz["scorer"](Tc, dev); P0, _ = A.decide(fz["dec"], Tc, Sc, Wc)
dec = dict(fz["dec"], t_rej=float(np.quantile(P0[(Tc.label == 1).to_numpy()], a.beta)))
ex = pd.read_csv(a.extra_vectors, dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str}); ex["_row"] = np.arange(len(ex))
Te = A.transform(ex, A.V[n], a.shm, "ext"); Se, We = fz["scorer"](Te, dev); P, z = A.decide(dec, Te, Se, We); Te["P"] = P; Te["zone"] = z
fa = Te[(Te.label == 0) & (Te.zone == "accept")]; print(f"{len(fa)} false accepts, T_accept {dec['t_acc']:.4f}"); pd.set_option("display.width", 250)
print(fa[["pair_id", "img_id_1", "img_id_2", "person_id_1", "person_id_2", "gap", "Q", "R_sam", "P"]].round(4).to_string(index=False))
if a.png and len(fa):
    import cv2
    rows = []
    for r in fa.itertuples():
        im = [cv2.imread(str(Path(a.crops) / f"{i}.png")) for i in (r.img_id_1, r.img_id_2)]; im = [cv2.resize(x, (200, 200)) for x in im]
        t = np.hstack(im).copy(); cv2.putText(t, f"{r.img_id_1} | {r.img_id_2} P={r.P:.3f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1); rows.append(t)
    cv2.imwrite(a.png, np.vstack(rows)); print("wrote", a.png)