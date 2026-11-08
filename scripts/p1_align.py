"""P1: detect (dlib chain, then buffalo_l box fallback), align to 256x256, measure quality. Resumable. CPU, multiprocess.
Usage: python scripts/p1_align.py --split_csv morph_person_split_v5.csv --morph_dir DIR --fgnet_dir DIR --predictor DAT --out artifacts/p1 --crops /dev/shm/aafv/crops"""
import sys, os, argparse, time
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, PIL.Image
from aafv import detect, align, quality

_G = {}
def _init(predictor):
    import dlib
    _G["det"] = dlib.get_frontal_face_detector(); _G["sp"] = dlib.shape_predictor(predictor); _G["dlib"] = dlib

def process_one(job):
    img_id, path, crops = job
    try:
        im = PIL.Image.open(path).convert("RGB"); rgb = np.asarray(im)
        box, how, n = detect.detect_chain(rgb, lambda a, u: _G["det"](a, u), None)
        if box is None: return dict(img_id=img_id, ok=0, stage="none", n=0)
        shp = _G["sp"](rgb, _G["dlib"].rectangle(*box)); lm = np.array([[q.x, q.y] for q in shp.parts()])
        a = align.align_face(im, lm); a.save(Path(crops) / (img_id + ".png"))
        r = quality.raw_measures(np.asarray(a), face_px=min(box[2] - box[0], box[3] - box[1]))
        return dict(img_id=img_id, ok=1, stage=how, n=n, **r)
    except Exception as e:
        return dict(img_id=img_id, ok=0, stage="error:" + type(e).__name__, n=0)

def main():
    ap = argparse.ArgumentParser()
    for k in ("split_csv", "morph_dir", "fgnet_dir", "predictor", "out", "crops"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out, crops = Path(a.out), Path(a.crops); out.mkdir(parents=True, exist_ok=True); crops.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(a.split_csv, dtype=str)
    mm = {p.name: p for p in Path(a.morph_dir).rglob("*") if p.is_file()}
    fg = {p.name: p for p in Path(a.fgnet_dir).iterdir() if p.is_file()}
    jobs = [(f, str(mm[f]), str(crops)) for f in sp.filename] + [(f, str(p), str(crops)) for f, p in fg.items()]
    done = {}
    res_csv = out / "detect_quality.csv"
    if res_csv.exists():
        old = pd.read_csv(res_csv); done = {r.img_id for r in old[old.ok == 1].itertuples()}
        jobs = [j for j in jobs if j[0] not in done]; rows = old[old.ok == 1].to_dict("records")
    else: rows = []
    if a.limit: jobs = jobs[:a.limit]
    print(f"{len(jobs)} images to process, {len(done)} already done", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.predictor,)) as pool:
        for i, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=32), 1):
            rows.append(r)
            if i % 5000 == 0: print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    fails = [r for r in rows if not r.get("ok")]
    if fails:                                    # stage 3 of the chain (D-18): buffalo_l detector box + dlib landmarks, CPU
        from insightface.app import FaceAnalysis
        import cv2
        _init(a.predictor)
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
        paths = {j[0]: j[1] for j in jobs}
        for r in fails:
            try:
                bgr = cv2.imread(paths[r["img_id"]]); f = app.get(bgr)
                if not f: continue
                b = max(f, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])); box = [int(v) for v in b.bbox]
                im = PIL.Image.open(paths[r["img_id"]]).convert("RGB"); rgb = np.asarray(im)
                shp = _G["sp"](rgb, _G["dlib"].rectangle(*box)); lm = np.array([[q.x, q.y] for q in shp.parts()])
                al = align.align_face(im, lm); al.save(crops / (r["img_id"] + ".png"))
                q = quality.raw_measures(np.asarray(al), face_px=min(box[2] - box[0], box[3] - box[1]))
                iid = [k for k in paths if k == r["img_id"]][0]
                r.clear(); r.update(dict(img_id=iid, ok=1, stage="scrfd_box", n=len(f), **q))
            except Exception:
                continue
    df = pd.DataFrame(rows); df.to_csv(res_csv, index=False)
    print("stages:", df.stage.value_counts().to_dict()); print("ok:", int(df.ok.sum()), "of", len(df), f"| {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
