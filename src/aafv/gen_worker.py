"""P4.3 worker logic for the SAM branch (restartable). Generation itself is injected as `runner`, so the loop is testable without a GPU."""
import os, shutil
from pathlib import Path

def gen_path(root, shard, img_id, age):
    return Path(root) / f"sam_w{shard}" / "inference_results" / str(age) / f"{img_id}.png"

def pending_by_age(m, root, shard):
    """{target age: [img_id, ...]} for this shard's SAM items whose output file does not exist yet."""
    sub = m[(m.generator == "sam") & (m.shard == shard)]
    out = {}
    for r in sub.itertuples(index=False):
        if not gen_path(root, shard, r.img_id, r.target_age).exists():
            out.setdefault(int(r.target_age), []).append(r.img_id)
    return out

def prepare_dir(tmp_root, age, img_ids, crops):
    d = Path(tmp_root) / str(age)
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    for i in img_ids:
        os.symlink(Path(crops) / f"{i}.png", d / f"{i}.png")
    return d

def run_worker(m, shard, crops, root, tmp_root, runner, retries=2, log=print):
    """runner(age, in_dir, exp_dir) must write exp_dir/inference_results/<age>/<img_id>.png. Returns counts."""
    exp = Path(root) / f"sam_w{shard}"; exp.mkdir(parents=True, exist_ok=True)
    stats = {"done_now": 0, "missing_after": 0}
    for attempt in range(retries + 1):
        todo = pending_by_age(m, root, shard)
        if not todo: break
        for age in sorted(todo):
            d = prepare_dir(tmp_root, age, todo[age], crops)
            try: runner(age, d, exp)
            except Exception as e: log(f"[shard {shard}] age {age} attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}")
            shutil.rmtree(d, ignore_errors=True)
        log(f"[shard {shard}] attempt {attempt}: remaining ages {sorted(pending_by_age(m, root, shard))}")
    left = pending_by_age(m, root, shard)
    stats["missing_after"] = sum(len(v) for v in left.values())
    return stats
