"""P4 (SAM branch): build the work list from the pair tables, run 8 workers (one per slice / shared big instance), write status.
Usage: python scripts/p4_sam.py --pairs artifacts/pairs --crops /dev/shm/aafv/crops --gen /dev/shm/aafv/gen --sam third_party/SAM --out artifacts/gen [--limit N]
       python scripts/p4_sam.py --worker SHARD ...   (internal)"""
import sys, os, re, argparse, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, numpy as np
from aafv import config, gap, manifest, gen_worker

ap = argparse.ArgumentParser()
ap.add_argument("--pairs"); ap.add_argument("--crops"); ap.add_argument("--gen"); ap.add_argument("--sam"); ap.add_argument("--out")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--worker", type=int, default=-1)
ap.add_argument("--params", default=str(ROOT / "configs/params.yaml"))
a = ap.parse_args(); cfg = config.load(a.params); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

def devices():
    u = re.findall(r"MIG-[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True))
    return u[:-1] + [u[-1]] * (8 - len(u) + 1)          # slices, then the big instance shared by the remaining workers

def load_manifest():
    return pd.read_csv(out / "manifest.csv", dtype={"img_id": str})

if a.worker >= 0:
    m = load_manifest(); shard = a.worker
    def runner(age, in_dir, exp):
        cmd = [sys.executable, str(ROOT / "scripts/run_sam_inference.py"), a.sam, "--exp_dir", str(exp),
               "--checkpoint_path", "pretrained_models/sam_ffhq_aging.pt", "--data_path", str(in_dir),
               "--test_batch_size", "16", "--test_workers", "4", "--resize_outputs", "--target_age", str(age)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError((r.stdout + r.stderr)[-400:])
    st = gen_worker.run_worker(m, shard, a.crops, a.gen, f"/dev/shm/aafv/sam_tmp/w{shard}", runner)
    print(f"[shard {shard}] finished: {st}", flush=True); sys.exit(0)

# ---- driver
P = pd.concat([pd.read_csv(p, dtype={"person_id_1": str, "person_id_2": str}) for p in sorted(Path(a.pairs).glob("pairs_*.csv"))])
grid = gap.grid(P.a1.min(), P.a2.max(), cfg["GRID_STEP"]); print("grid:", grid)
m = manifest.build_manifest(P, grid, cfg["K_MAX"], generators=("sam",))
if a.limit: m = m.sort_values("item_id").head(a.limit)
m.to_csv(out / "manifest.csv", index=False)
print(f"manifest: {len(m)} SAM items, {m.img_id.nunique()} images, shards {m.shard.value_counts().sort_index().to_dict()}", flush=True)
devs = devices(); t0 = time.time(); procs = []
for s in range(8):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": devs[s]}
    procs.append(subprocess.Popen([sys.executable, __file__, "--worker", str(s), "--pairs", a.pairs, "--crops", a.crops, "--gen", a.gen,
                                   "--sam", a.sam, "--out", a.out], env=env, stdout=open(out / f"worker{s}.log", "w"), stderr=subprocess.STDOUT))
for p in procs: p.wait()
m["status"] = [("done" if gen_worker.gen_path(a.gen, r.shard, r.img_id, r.target_age).exists() else "failed") for r in m.itertuples(index=False)]
m.to_csv(out / "manifest.csv", index=False)
print(m.status.value_counts().to_dict(), f"| {time.time()-t0:.0f}s")
print("failure rate by target age:", m.assign(f=(m.status == "failed")).groupby("target_age").f.mean().round(3).to_dict())
sys.exit(1 if (a.limit and (m.status == "failed").any()) else 0)
