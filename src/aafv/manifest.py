"""P4.1 / P4.2: which (image, generator, target age) items must be generated, and which worker takes each."""
import zlib
import pandas as pd
from .gap import grid as make_grid, snap, targets

N_SHARDS = 8

def shard_of(img_id, n=N_SHARDS):
    return zlib.crc32(str(img_id).encode()) % n          # fixed hash: every restart agrees

def needs_from_pairs(pairs, grid_ages, k_max=7):
    """Return {img_id: set(target ages)}. Image 1 is aged to snap(a2), image 2 is de-aged to snap(a1),
    and both are taken to every intermediate grid age in T (shared targets)."""
    need = {}
    for r in pairs[["img_id_1", "img_id_2", "a1", "a2"]].itertuples(index=False):
        t_fwd, t_bwd, T, _ = targets(int(r.a1), int(r.a2), grid_ages, k_max)
        need.setdefault(r.img_id_1, set()).update([t_fwd, *T])
        need.setdefault(r.img_id_2, set()).update([t_bwd, *T])
    return need

def build_manifest(pairs, grid_ages, k_max=7, generators=("sam", "diffusion"), diffusion_images=None):
    """diffusion_images: optional set of img_ids the diffusion branch is allowed to run on (scope, see R-27)."""
    need = needs_from_pairs(pairs, grid_ages, k_max)
    rows = []
    for img, ages in need.items():
        for gen in generators:
            if gen == "diffusion" and diffusion_images is not None and img not in diffusion_images:
                continue
            for t in sorted(ages):
                rows.append((f"{img}|{gen}|{t}", img, gen, int(t), shard_of(img), "pending", "", float("nan"), -1))
    return pd.DataFrame(rows, columns=["item_id", "img_id", "generator", "target_age", "shard", "status", "reason", "seconds", "worker"])

def budget_hours(m, n_workers=8, sam_s_per_item=0.06, dif_inversion_s=92.0, dif_s_per_age=26.0):
    """Wall-clock estimate from measured/assumed speeds (all inputs are printed by the caller). SAM speed is a forward-pass figure
    from 4 images; the diffusion figures come from one image (01 D23)."""
    sam = (m.generator == "sam").sum() * sam_s_per_item / n_workers / 3600
    d = m[m.generator == "diffusion"]
    imgs = d.img_id.nunique()
    dif = (imgs * dif_inversion_s + len(d) * dif_s_per_age) / n_workers / 3600
    return {"sam_hours": sam, "diffusion_hours": dif, "diffusion_images": imgs, "sam_items": int((m.generator == "sam").sum()),
            "diffusion_items": len(d)}
