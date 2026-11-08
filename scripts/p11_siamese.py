"""P11 baseline: Siamese network on the ORIGINAL face pair (no aged faces, no age gap, no guards beyond quality). Trained on train_fit, early stopping on train_val log loss.
  simple NN :  --name siamese_cnn --arch resnet18 --img 128
  ViT       :  --name siamese_vit --arch vit_base_patch16_224.augreg_in21k_ft_in1k --img 224 --lr 2e-5 --epochs 3
Writes artifacts/baselines/<name>/scores.npz (pair_id, logit for train_val, calibration, test pairs; no labels are used for scoring) and training_log.json.
Images: aligned crops (PNG, 256x256) read from --crops (default /dev/shm/aafv/crops)."""
import sys, os, re, json, argparse, time, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    try: os.environ["CUDA_VISIBLE_DEVICES"] = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))[-1]
    except Exception: pass
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, cv2
from aafv import calib

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--arch", default="resnet18"); ap.add_argument("--img", type=int, default=128); ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--crops", default="/dev/shm/aafv/crops"); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--bs", type=int, default=128); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--pretrained", type=int, default=1); ap.add_argument("--limit_pairs", type=int, default=0); ap.add_argument("--extra", default=""); ap.add_argument("--load", type=int, default=0)   # 1 = skip training, load <out>/<name>/model.pt   # comma list of pair tables (pair_id,img_id_1,img_id_2) scored too, e.g. artifacts/pairs_fg/pairs_fgnet.csv
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"; od = Path(a.out) / a.name; od.mkdir(parents=True, exist_ok=True)
rd = lambda s: pd.read_csv(Path(a.p5q) / f"pair_vectors_{s}.csv", dtype={"img_id_1": str, "img_id_2": str, "pair_id": str}, usecols=["pair_id", "img_id_1", "img_id_2", "label", "weight"])
fr = {s: rd(s) for s in ("train_fit", "train_val", "calibration", "test")}
if a.limit_pairs: fr = {s: d.sample(min(len(d), a.limit_pairs), random_state=0).reset_index(drop=True) for s, d in fr.items()}
extra_tabs = {Path(p).stem: pd.read_csv(p, dtype={"img_id_1": str, "img_id_2": str, "pair_id": str}, usecols=["pair_id", "img_id_1", "img_id_2"]) for p in a.extra.split(",") if p}
ids = sorted(set().union(*[set(d.img_id_1) | set(d.img_id_2) for d in list(fr.values()) + list(extra_tabs.values())])); ix = {k: i for i, k in enumerate(ids)}; print(f"{len(ids)} images, device {dev}", flush=True)
def load(k):
    im = cv2.imread(str(Path(a.crops) / f"{k}.png"))
    if im is None: im = cv2.imread(str(Path(a.crops) / f"{Path(k).stem}.png"))
    assert im is not None, k; return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (a.img, a.img), interpolation=cv2.INTER_AREA)
with ThreadPoolExecutor(16) as ex: IM = np.stack(list(ex.map(load, ids)))              # (n, img, img, 3) uint8 in RAM
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def batch(rows, aug=False):
    x = torch.from_numpy(IM[rows]).to(dev).permute(0, 3, 1, 2).float() / 255.0
    if aug: flip = torch.rand(len(rows), device=dev) < 0.5; x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    return (x - MEAN) / STD

class Siam(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(a.arch, pretrained=bool(a.pretrained), num_classes=0, **({"img_size": a.img} if a.arch.startswith("vit") else {}))
        d = s.b.num_features; s.head = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def pair(s, ea, eb): return s.head(torch.cat([(ea - eb).abs(), ea * eb], 1)).squeeze(1)
    def forward(s, xa, xb): return s.pair(s.b(xa), s.b(xb))
m = Siam().to(dev); hp = list(m.head.parameters()); bp = list(m.b.parameters())
opt = torch.optim.AdamW([{"params": bp, "lr": a.lr}, {"params": hp, "lr": a.lr * 5}], weight_decay=0.01); bce = nn.BCEWithLogitsLoss(reduction="none")

def rows_of(d): return d.img_id_1.map(ix).to_numpy(), d.img_id_2.map(ix).to_numpy()
@torch.no_grad()
def embed_all():
    m.eval(); E = []
    for i in range(0, len(ids), 256):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): E.append(m.b(batch(np.arange(i, min(i + 256, len(ids))))).float())
    return torch.cat(E)
@torch.no_grad()
def score(d, E):
    r1, r2 = rows_of(d); out = []
    for i in range(0, len(d), 65536): out.append(m.pair(E[r1[i:i+65536]], E[r2[i:i+65536]]).float().cpu())
    return torch.cat(out).numpy()
tr, va = fr["train_fit"], fr["train_val"]; r1, r2 = rows_of(tr); y = torch.tensor(tr.label.to_numpy(np.float32), device=dev); w = torch.tensor(tr.weight.to_numpy(np.float32), device=dev)
best, state, bad, log = 1e9, None, 0, []
if a.load: state = torch.load(od / 'model.pt', map_location=dev); log = json.load(open(od / 'training_log.json'))['log']; best = json.load(open(od / 'training_log.json'))['best_val_logloss']
else:
    for ep in range(a.epochs):
        m.train(); perm = rng.permutation(len(tr)); t0 = time.time(); tl = 0.0; nb = 0
        for i in range(0, len(perm), a.bs):
            p = perm[i:i + a.bs]
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"): s = m(batch(r1[p], True), batch(r2[p], True))
            loss = (w[p] * bce(s.float(), y[p])).mean()
            if not torch.isfinite(loss): raise RuntimeError("NaN loss")
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
        Sv = score(va, embed_all()); vl = calib.logloss(va.label.to_numpy(), 1 / (1 + np.exp(-Sv))); log.append(dict(epoch=ep, train_loss=tl / nb, val_logloss=vl, seconds=time.time() - t0)); print(log[-1], flush=True)
        if vl < best - 1e-4: best, bad, state = vl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
m.load_state_dict(state); torch.save(state, od / "model.pt"); E = embed_all(); pid, lg = [], []
for s_ in ("train_val", "calibration", "test"): pid += fr[s_].pair_id.tolist(); lg.append(score(fr[s_], E))
np.savez(od / "scores.npz", pair_id=np.array(pid), logit=np.concatenate(lg))
if not a.load: json.dump(dict(args=vars(a), log=log, best_val_logloss=best), open(od / "training_log.json", "w"), indent=1)
for k, d in extra_tabs.items(): np.savez(od / f"scores_{k}.npz", pair_id=d.pair_id.to_numpy().astype(str), logit=score(d, E)); print("extra scored:", k, len(d))
print(f"saved {od/'scores.npz'} | best val log loss {best:.4f}")