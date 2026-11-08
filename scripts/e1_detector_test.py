"""E1 detector test (02 section 5): dlib frontal detector + 68 landmarks on all FG-NET images and 300 random MORPH images (seed 42).
Writes: e1_results.csv, e1_contact_sheet.jpg (20 images with landmarks, for checking by eye). Prints detection counts."""
import sys, os, bz2, hashlib, urllib.request, csv, random
from pathlib import Path
import numpy as np, cv2, dlib, pandas as pd

def run(fgnet_dir, morph_dir, split_csv, out_dir, seed=42, n_morph=300):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dat = out / "shape_predictor_68_face_landmarks.dat"
    if not dat.exists():
        urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", str(dat) + ".bz2")
        dat.write_bytes(bz2.BZ2File(str(dat) + ".bz2").read()); os.remove(str(dat) + ".bz2")
    print("predictor sha256:", hashlib.sha256(dat.read_bytes()).hexdigest())
    det, sp = dlib.get_frontal_face_detector(), dlib.shape_predictor(str(dat))
    fg = sorted(p for p in Path(fgnet_dir).rglob("*") if p.is_file())
    mp = {p.name: p for p in Path(morph_dir).rglob("*") if p.is_file()}
    names = sorted(pd.read_csv(split_csv, dtype=str)["filename"]); random.Random(seed).shuffle(names)
    mo = [mp[n] for n in names[:n_morph]]
    rows, shots = [], []
    for ds, files in (("fgnet", fg), ("morph", mo)):
        for k, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None: rows.append((ds, f.name, "unreadable", 0, 0, 0)); continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            d = det(rgb, 1)
            status = "none" if len(d) == 0 else ("one" if len(d) == 1 else "several")
            area = max([(r.width() * r.height()) for r in d], default=0)
            rows.append((ds, f.name, status, len(d), img.shape[0], img.shape[1]))
            if len(shots) < 20 and k % (max(1, len(files) // 10)) == 0 and len(d) > 0:
                r = max(d, key=lambda r: r.width() * r.height()); shp = sp(rgb, r)
                vis = img.copy()
                for i in range(68): cv2.circle(vis, (shp.part(i).x, shp.part(i).y), 2, (0, 255, 0), -1)
                cv2.rectangle(vis, (r.left(), r.top()), (r.right(), r.bottom()), (0, 0, 255), 2)
                shots.append(cv2.resize(vis, (200, 200)))
    df = pd.DataFrame(rows, columns=["dataset", "file", "status", "n_faces", "h", "w"]); df.to_csv(out / "e1_results.csv", index=False)
    if shots:
        while len(shots) % 5: shots.append(np.zeros((200, 200, 3), np.uint8))
        sheet = np.vstack([np.hstack(shots[i:i + 5]) for i in range(0, len(shots), 5)]); cv2.imwrite(str(out / "e1_contact_sheet.jpg"), sheet)
    print(df.groupby(["dataset", "status"]).size().unstack(fill_value=0).to_string())
    print(df.groupby("dataset").apply(lambda g: (g.status == "one").mean()).rename("share exactly one face").to_string())
    return df
