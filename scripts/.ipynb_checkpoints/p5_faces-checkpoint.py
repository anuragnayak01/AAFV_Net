"""P5.1: features of every face version (originals + generated). Uses the big GPU instance. Writes to /dev/shm/aafv/p5 and artifacts/p5.
Outputs: faces.csv (face_id, kind, img_id, target_age, est_age, q, arc_to_orig), feats_{arc,eff,vgg}.npy (float16, L2-normalised), faces_u8.npy (uint8 N,256,256,3)."""
import sys, os, re, json, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))[-1]
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F, timm, torchvision
from aafv import quality, gen_worker
import c4_eval_age

ap = argparse.ArgumentParser()
for k in ("p1_csv", "p2", "crops", "gen", "manifest", "sam_dir", "dex", "shm", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--only_list", default="")          # optional text file of img_ids: build faces only for these originals (FG-NET run)
a = ap.parse_args(); shm = Path(a.shm); shm.mkdir(parents=True, exist_ok=True); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

p1 = pd.read_csv(a.p1_csv); p1 = p1[p1.ok == 1].reset_index(drop=True)
ids = pd.read_csv(Path(a.p2) / "arc_ids.csv").img_id.tolist(); assert ids == p1.img_id.tolist()
est = pd.read_csv(Path(a.p2) / "age_est.csv"); norm = json.load(open(Path(a.p2) / "quality_norm.json"))
detail = dict(zip(p1.img_id, p1.detail)); arc_orig = np.load(Path(a.p2) / "arc_orig.npy").astype(np.float32)

m = pd.read_csv(a.manifest, dtype={"img_id": str}); m = m[m.generator == "sam"]
only = set(Path(a.only_list).read_text().split()) if a.only_list else None; bad_gen = []
rows = [("orig|" + i, "orig", i, -1, str(Path(a.crops) / f"{i}.png")) for i in ids if only is None or i in only]
for r in m.itertuples(index=False):
    p = gen_worker.gen_path(a.gen, r.shard, r.img_id, r.target_age)
    if p.exists() and (only is None or r.img_id in only):
        if only is not None:                         # external run: skip generated faces that are unreadable or not 256x256 (reported below)
            try:
                from PIL import Image
                with Image.open(p) as im_: ok_ = im_.size == (256, 256)
            except Exception: ok_ = False
            if not ok_: bad_gen.append(str(p)); continue
        rows.append((f"{r.img_id}|sam|{r.target_age}", "sam", r.img_id, int(r.target_age), str(p)))
faces = pd.DataFrame(rows, columns=["face_id", "kind", "img_id", "target_age", "path"]); N = len(faces)
if bad_gen: print(f"skipped {len(bad_gen)} invalid generated faces, e.g. {bad_gen[:3]}")
print(f"{N} faces ({(faces.kind == 'orig').sum()} originals, {(faces.kind == 'sam').sum()} SAM)", flush=True)

U8 = np.lib.format.open_memmap(shm / "faces_u8.npy", mode="w+", dtype=np.uint8, shape=(N, 256, 256, 3))
def load(i):
    im = cv2.imread(faces.path[i]); assert im is not None and im.shape[:2] == (256, 256), faces.path[i]
    U8[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
with ThreadPoolExecutor(16) as ex: list(ex.map(load, range(N)))
U8.flush(); print("images loaded", flush=True)

dev = "cuda"
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"]); app.prepare(ctx_id=0)
rec = app.models["recognition"]
eff = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).to(dev).eval()
vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").to(dev).eval()
dex = c4_eval_age.load_dex(a.sam_dir, a.dex, dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
arc = np.zeros((N, 512), np.float16); fe = np.zeros((N, 1280), np.float16); fv = np.zeros((N, 4096), np.float16); ages = np.zeros(N, np.float32)
B = 128
with torch.no_grad():
    for s in range(0, N, B):
        rgb = np.asarray(U8[s:s + B]); x = torch.from_numpy(rgb).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
        f = rec.get_feat([cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (112, 112), interpolation=cv2.INTER_AREA) for im in rgb])
        arc[s:s + B] = (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float16)
        fe[s:s + B] = F.normalize(eff(x224 - 0.5), dim=1).cpu().numpy().astype(np.float16)        # paper: inputs scaled to [-0.5, 0.5]
        v = vgg.avgpool(vgg.features((x224 - mean) / std)).flatten(1); v = vgg.classifier[:6](v)  # first 6 FC-stage layers, 4096-d
        fv[s:s + B] = F.normalize(v, dim=1).cpu().numpy().astype(np.float16)
        xd = (x - 0.5) / 0.5
        ages[s:s + B] = c4_eval_age.dex_age(dex, [r for r in rgb], dev)
        if (s // B) % 100 == 0: print(f"  {s}/{N}", flush=True)
np.save(shm / "feats_arc.npy", arc); np.save(shm / "feats_eff.npy", fe); np.save(shm / "feats_vgg.npy", fv)

# quality of every face (generated faces inherit the original's face size for the 'detail' measure)
raws = [quality.raw_measures(np.asarray(U8[i]), face_px=detail[faces.img_id[i]]) for i in range(N)]
Q = quality.normalize(pd.DataFrame(raws), norm).Q.to_numpy()
pos = {i: k for k, i in enumerate(ids)}
orig_idx = faces.img_id.map(pos).to_numpy()
arc_to_orig = np.where(faces.kind == "orig", 1.0, (arc.astype(np.float32) * arc_orig[orig_idx]).sum(1))
faces["est_age"] = ages; faces["q"] = Q; faces["arc_to_orig"] = arc_to_orig
faces.drop(columns="path").to_csv(out / "faces.csv", index=False)
g = faces[faces.kind == "sam"]
print("SAM faces: mean arc_to_orig %.3f, mean |est_age - target| %.2f, mean q %.3f" % (g.arc_to_orig.mean(), (g.est_age - g.target_age).abs().mean(), g.q.mean()))
print("done")