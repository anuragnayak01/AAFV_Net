"""P5.2 + P5.3: five raw scores for every comparison of every pair (direct, SAM forward/backward, SAM shared targets).
Columns: [arc, eff, vgg, lpips, ssim]. Writes artifacts/p5/raw_<split>.npz. GPU for LPIPS, CPU pool for SSIM."""
import sys, os, re, argparse, subprocess, time
from pathlib import Path
from multiprocessing import get_context
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))[-1]
import numpy as np, pandas as pd, torch, lpips
from aafv import config, gap, pairvec, ssim as ssim_mod

ap = argparse.ArgumentParser()
for k in ("pairs", "faces", "shm", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--splits", default="train_fit,train_val,calibration,test"); ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--limit_pairs", type=int, default=0); ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); shm = Path(a.shm)

faces = pd.read_csv(a.faces, dtype={"img_id": str})
orig = {r.img_id: i for i, r in enumerate(faces.itertuples()) if r.kind == "orig"}
gen = {(r.img_id, int(r.target_age)): i for i, r in enumerate(faces.itertuples()) if r.kind == "sam"}
F = {k: np.load(shm / f"feats_{k}.npy", mmap_mode="r") for k in ("arc", "eff", "vgg")}
U8 = np.load(shm / "faces_u8.npy", mmap_mode="r")
allp = {s: pd.read_csv(Path(a.pairs) / f"pairs_{s}.csv", dtype={"person_id_1": str, "person_id_2": str, "img_id_1": str, "img_id_2": str}) for s in a.splits.split(",")}
grid = gap.grid(min(p.a1.min() for p in allp.values()), max(p.a2.max() for p in allp.values()), cfg["GRID_STEP"])
print("grid", grid, flush=True)

def _ssim_chunk(xy):
    return [ssim_mod.ssim_rgb(np.asarray(U8[x]), np.asarray(U8[y])) for x, y in xy]

def score_pairs_idx(X, Y, pool, net):
    """X, Y: int arrays (n,) of face rows, all valid. Returns (n,5)."""
    n = len(X); S = np.zeros((n, 5), np.float32)
    for s in range(0, n, 50000):
        sl = slice(s, s + 50000)
        for j, k in enumerate(("arc", "eff", "vgg")):
            c = (F[k][X[sl]].astype(np.float32) * F[k][Y[sl]].astype(np.float32)).sum(1)
            S[sl, j] = c if k == "arc" else (c + 1) / 2
    with torch.no_grad():
        for s in range(0, n, 256):
            x = torch.from_numpy(np.asarray(U8[X[s:s + 256]])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            y = torch.from_numpy(np.asarray(U8[Y[s:s + 256]])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            d = net(x, y).flatten().cpu().numpy(); S[s:s + 256, 3] = 1 - np.minimum(d, 1.0)
    xy = list(zip(X.tolist(), Y.tolist())); chunks = [xy[i:i + 2000] for i in range(0, n, 2000)]
    S[:, 4] = np.concatenate([np.array(r, np.float32) for r in pool.map(_ssim_chunk, chunks)]) if chunks else []
    return S

with get_context("fork").Pool(a.workers) as pool:        # forked before CUDA is initialised in this process
    net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    for sp, P in allp.items():
        t0 = time.time()
        if a.limit_pairs: P = P.head(a.limit_pairs)
        N = len(P); tf, tb, T, K = pairvec.target_arrays(P.a1.to_numpy(), P.a2.to_numpy(), grid, cfg["K_MAX"])
        i1, i2 = P.img_id_1.tolist(), P.img_id_2.tolist()
        o1 = np.array([orig[i] for i in i1]); o2 = np.array([orig[i] for i in i2])
        gf = np.array([gen.get((i1[n], int(tf[n])), -1) for n in range(N)]); gb = np.array([gen.get((i2[n], int(tb[n])), -1) for n in range(N)])
        gs1 = -np.ones((N, 7), int); gs2 = -np.ones((N, 7), int)
        for n in range(N):
            for k in range(K[n]):
                gs1[n, k] = gen.get((i1[n], int(T[n, k])), -1); gs2[n, k] = gen.get((i2[n], int(T[n, k])), -1)
        X = np.concatenate([o1, gf, o1, gs1.ravel()]); Y = np.concatenate([o2, o2, gb, gs2.ravel()])
        valid = (X >= 0) & (Y >= 0)
        S = np.full((len(X), 5), np.nan, np.float32); S[valid] = score_pairs_idx(X[valid], Y[valid], pool, net)
        S_dir, S_f, S_b = S[:N], S[N:2 * N], S[2 * N:3 * N]; S_sh = S[3 * N:].reshape(N, 7, 5)
        np.savez(out / f"raw_{sp}.npz", S_dir=S_dir, S_fwd=S_f, S_bwd=S_b, S_sh=S_sh, g_fwd=gf, g_bwd=gb, g_sh1=gs1, g_sh2=gs2,
                 o1=o1, o2=o2, tf=tf, tb=tb, T=T, K=K)
        print(f"{sp}: {N} pairs, {int(valid.sum())} comparisons scored, fwd+bwd available {(np.isfinite(S_f[:,0]) & np.isfinite(S_b[:,0])).mean():.3f}, {time.time()-t0:.0f}s", flush=True)
print("done")
