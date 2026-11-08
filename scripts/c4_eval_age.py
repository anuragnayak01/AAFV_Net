"""C4 selection, candidate 'DEX VGG age classifier' (the one SAM uses in its aging loss; research/evaluation-only per 09).
Evaluated on OUR aligned 256x256 crops (dlib chain + align.py), with SAM's own preprocessing: RGB in [-1, 1], bilinear to 224,
age = softmax expectation over classes 0..100. Writes artifacts/env/c4/dex_eval.csv and prints MAE/bias by age band."""
import os, re, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, PIL.Image, dlib, torch, torch.nn.functional as F

def load_dex(sam_dir, weights, device):
    import types
    src = (Path(sam_dir) / "models" / "dex_vgg.py").read_text()
    bad = "out['p5'].view(out['p5'].size(0), -1)"
    assert bad in src, "dex_vgg.py changed; check the view() call"
    src = src.replace(bad, "out['p5'].reshape(out['p5'].size(0), -1)")   # torch>=2: pool output may not be contiguous
    m = types.ModuleType("dex_vgg"); exec(compile(src, "dex_vgg_patched", "exec"), m.__dict__)
    net = m.VGG()
    ck = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.replace("-", "_"): v for k, v in ck.items()})
    return net.to(device).eval()

@torch.no_grad()
def dex_age(net, imgs, device):
    x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) / 255.0 for i in imgs])).permute(0, 3, 1, 2)
    x = ((x - 0.5) / 0.5).to(device)
    x = F.interpolate(x, size=(224, 224), mode="bilinear")
    p = F.softmax(net(x)["fc8"], dim=1)
    return (p * torch.arange(p.shape[1], device=device)).sum(1).cpu().numpy()

def run(sam_dir, weights, predictor, items, out_csv, src_dir, device="cuda", batch=32):
    sys.path.insert(0, src_dir)
    from aafv import detect, align
    det, sp = dlib.get_frontal_face_detector(), dlib.shape_predictor(predictor)
    net = load_dex(sam_dir, weights, device)
    rows, buf, meta = [], [], []
    def flush():
        if buf:
            for m, a in zip(meta, dex_age(net, buf, device)): rows.append((*m, float(a)))
            buf.clear(); meta.clear()
    miss = 0
    for ds, path, age in items:
        im = PIL.Image.open(path).convert("RGB"); rgb = np.asarray(im)
        box, how, n = detect.detect_chain(rgb, lambda a, u: det(a, u), None)
        if box is None: miss += 1; rows.append((ds, Path(path).name, age, "none", np.nan)); continue
        shp = sp(rgb, dlib.rectangle(*box)); lm = np.array([[q.x, q.y] for q in shp.parts()])
        buf.append(align.align_face(im, lm)); meta.append((ds, Path(path).name, age, how))
        if len(buf) == batch: flush()
    flush()
    df = pd.DataFrame(rows, columns=["dataset", "file", "age", "stage", "est_age"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True); df.to_csv(out_csv, index=False)
    df["err"] = df.est_age - df.age
    bands = pd.cut(df.age, [-1, 4, 9, 14, 19, 29, 39, 49, 100], labels=["0-4", "5-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50+"])
    print("no face (all stages):", miss)
    print(df.groupby(["dataset", bands], observed=True).err.agg(n="count", MAE=lambda e: e.abs().mean(), bias="mean", sd="std").round(2).to_string())
    print("overall MAE:", df.groupby("dataset").err.apply(lambda e: round(e.abs().mean(), 2)).to_dict())
    return df
