import os
import cv2
import numpy as np, pandas as pd, pytest
from aafv import gap, pairs

def test_g():
    assert gap.g(10, 50) == 0.5 and gap.g(0, 100) == 1.0

def test_bins():
    assert [gap.gap_bin(x) for x in (0, 4, 5, 14, 15, 29, 30, 60)] == ["G1","G1","G2","G3","G4","G5","G6","G6"]

def test_grid_snap_targets():
    G = gap.grid(16, 47)
    assert G[0] == 15 and G[-1] == 50
    assert gap.snap(17.5, G) == 20 and gap.snap(22, G) == 20
    tf, tb, T, K = gap.targets(18, 43, G)
    assert (tf, tb, T, K) == (45, 20, [20, 25, 30, 35, 40], 5)
    assert gap.targets(20, 24, G)[3] == 0
    assert gap.targets(0, 80, gap.grid(0, 80))[3] == 7

def _df():
    rows = []
    for p in range(40):
        for k, a in enumerate([20 + p % 5, 30, 55]):
            rows.append((f"i{p}_{k}", f"p{p}", a, "M" if p % 4 else "F"))
    return pd.DataFrame(rows, columns=["img_id", "person_id", "age", "gender"])

def test_pairs():
    df = _df(); rng = np.random.default_rng(1)
    s = pairs.finalize(pairs.same_pairs(df), "t", "x")
    assert len(s) == 40 * 3 and (s.label == 1).all()
    c = pairs.same_pairs(df, cap=1, rng=rng)
    assert len(c) <= len(pairs.same_pairs(df))
    d = pairs.finalize(pairs.d_random(df, rng), "t", "x")
    ids = pd.concat([d.person_id_1, d.person_id_2])
    assert ids.is_unique and (d.label == 0).all()
    a = pairs.finalize(pairs.d_all_random(df, 500, rng), "t", "x")
    assert (a.person_id_1 != a.person_id_2).all()

from aafv import evaluation, thresholds
def test_bound():
    assert round(1 / evaluation.zero_error_bound(1343, 0.05)) == 449
    assert round(1 / evaluation.zero_error_bound(1343, 0.025)) == 365
    assert abs(evaluation.clopper_pearson_upper(0, 1343) - evaluation.zero_error_bound(1343)) < 1e-9

def test_thresholds():
    rng = np.random.default_rng(0)
    imp = rng.beta(2, 8, 500); same = rng.beta(6, 3, 500)
    ta, tr, w = thresholds.thresholds(imp, same)
    assert ta > imp.max() and tr < same.min() or w
    z = thresholds.zone(np.array([ta + .01, (ta + tr) / 2, tr - .01]), ta, tr)
    if ta > tr: assert list(z) == ["accept", "review", "reject"]
    ta2, tr2, w2 = thresholds.thresholds([0.1, 0.2], [0.8, 0.9])
    assert w2 and ta2 == tr2

from aafv import detect
class _R:
    def __init__(s, l, t, r, b): s.l, s.t, s.r, s.b = l, t, r, b
    def left(s): return s.l
    def top(s): return s.t
    def right(s): return s.r
    def bottom(s): return s.b

def test_detect_chain():
    big, small = _R(0, 0, 100, 100), _R(0, 0, 10, 10)
    assert detect.detect_chain(None, lambda im, u: [small, big] if u == 1 else [], None) == ((0, 0, 100, 100), "dlib_up1", 2)
    assert detect.detect_chain(None, lambda im, u: [big] if u == 2 else [], None)[1] == "dlib_up2"
    assert detect.detect_chain(None, lambda im, u: [], lambda im: [(5, 5, 50, 50, 0.6)]) == ((5, 5, 50, 50), "scrfd_box", 1)
    assert detect.detect_chain(None, lambda im, u: [], lambda im: []) == (None, "none", 0)

from aafv import align
import PIL.Image
def _fake_lm(cx, cy, s):
    lm = np.zeros((68, 2))
    lm[36:42] = [cx - s, cy]; lm[42:48] = [cx + s, cy]
    lm[48:60] = [cx, cy + 1.5 * s]; lm[48] = [cx - 0.6 * s, cy + 1.5 * s]; lm[54] = [cx + 0.6 * s, cy + 1.5 * s]
    return lm

def test_align_shapes_and_edges():
    img = PIL.Image.fromarray((np.random.rand(180, 180, 3) * 255).astype("uint8"))
    for cx, cy in ((90, 80), (20, 20), (160, 170)):   # centre, and faces near edges (padding branch)
        out = align.align_face(img, _fake_lm(cx, cy, 20))
        assert out.size == (256, 256) and out.mode == "RGB"
    big = PIL.Image.fromarray((np.random.rand(1200, 1000, 3) * 255).astype("uint8"))   # shrink branch
    assert align.align_face(big, _fake_lm(500, 500, 150)).size == (256, 256)

def test_align_puts_eyes_at_same_place():
    img = PIL.Image.fromarray(np.zeros((400, 400, 3), "uint8"))
    for (cx, cy, s) in ((200, 200, 30), (180, 220, 45)):
        a = np.asarray(align.align_face(img, _fake_lm(cx, cy, s)))
        assert a.shape == (256, 256, 3)

from aafv import manifest
def test_manifest():
    G = gap.grid(15, 50)
    pr = pd.DataFrame({"img_id_1": ["a", "c"], "img_id_2": ["b", "b"], "a1": [18, 30], "a2": [43, 30]})
    need = manifest.needs_from_pairs(pr, G)
    assert need["a"] == {45, 20, 25, 30, 35, 40}
    assert need["b"] == {20, 30, 25, 35, 40} | {30}
    assert need["c"] == {30}
    m = manifest.build_manifest(pr, G, diffusion_images={"a"})
    assert set(m[m.generator == "diffusion"].img_id) == {"a"}
    assert m.item_id.is_unique and (m.shard == m.img_id.map(manifest.shard_of)).all()
    assert manifest.shard_of("x") == manifest.shard_of("x") and 0 <= manifest.shard_of("x") < 8
    b = manifest.budget_hours(m)
    assert b["diffusion_images"] == 1

from aafv import quality
def test_quality_orders_sharp_and_noise():
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64)) * 255).astype("uint8"); base = cv2.GaussianBlur(base, (0, 0), 3)
    sharp = np.stack([base] * 3, -1); blur = cv2.GaussianBlur(sharp, (0, 0), 3)
    noisy = np.clip(sharp.astype(float) + rng.normal(0, 20, sharp.shape), 0, 255).astype("uint8")
    a, b, c = quality.raw_measures(sharp), quality.raw_measures(blur), quality.raw_measures(noisy)
    assert a["sharp"] > b["sharp"] and c["noise"] > a["noise"]
    dark = quality.raw_measures(np.full((64, 64, 3), 20, "uint8")); assert dark["illum"] < 0.3
    df = pd.DataFrame([a, b, c]); n = quality.fit_norm(df); q = quality.normalize(df, n)
    assert q.Q.between(0, 1).all()

def test_hard_negatives_and_subsplit():
    rng = np.random.default_rng(3)
    df = _df(); E = rng.normal(size=(len(df), 16)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    E[1] = E[3]                                   # image 1 (p0) and image 3 (p1) identical: must be each other's top hit if same gender
    h = pairs.hard_negatives(df, E, 3)
    assert (h.person_id_1 != h.person_id_2).all() and (h.a1 <= h.a2).all() and (h.label == 0).all()
    g = df.set_index("img_id").gender
    assert (h.img_id_1.map(g) == h.img_id_2.map(g)).all()
    assert pairs.train_subsplit(["b", "a", "c", "d"], 0.5, 1) == pairs.train_subsplit(["d", "c", "b", "a"], 0.5, 1)

import tempfile, shutil
from aafv import gen_worker
def test_worker_restart_and_retry():
    G = gap.grid(15, 50)
    pr = pd.DataFrame({"img_id_1": ["a", "c"], "img_id_2": ["b", "d"], "a1": [18, 22], "a2": [43, 24]})
    m = manifest.build_manifest(pr, G, generators=("sam",))
    tmp = tempfile.mkdtemp(); crops = tmp + "/crops"; os.makedirs(crops)
    for i in "abcd": open(f"{crops}/{i}.png", "w").write("x")
    calls = []
    def runner(age, in_dir, exp):
        calls.append(age)
        if len(calls) == 1: raise RuntimeError("boom")           # forced failure: retried later
        for f in os.listdir(in_dir):
            p = exp / "inference_results" / str(age) / f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("y")
    tot = 0
    for s in range(8):
        st = gen_worker.run_worker(m, s, crops, tmp + "/gen", tmp + "/t", runner, log=lambda *_: None); tot += st["missing_after"]
    assert tot == 0 and len(calls) > len(set(m.target_age)) - 1
    assert gen_worker.pending_by_age(m, tmp + "/gen", 0) == {}
    shutil.rmtree(tmp)

from aafv import pairvec
def test_pairvec_worked_example():
    # one pair, K_used = 2, hand-computed
    Sf = np.array([[0.6, .5, .5, .5, .5]]); Sb = np.array([[0.4, .5, .5, .5, .5]])
    Ssh = np.zeros((1, 7, 5)); Ssh[0, 0] = [0.8, .5, .5, .5, .5]; Ssh[0, 1] = [0.2, .5, .5, .5, .5]
    mf = np.array([True]); mb = np.array([True]); msh = np.zeros((1, 7), bool); msh[0, :2] = True
    x, av, K, sp, used = pairvec.branch_vector(Sf, Sb, Ssh, mf, mb, msh, (np.zeros(5), np.ones(5)))
    assert av[0] and K[0] == 2
    assert np.isclose(x[0, 0], 0.5 * 0.5 + 0.5 * 0.5)                  # base 0.5, shared 0.5
    assert np.isclose(sp[0], np.std([0.6, 0.4, 0.8, 0.2]))
    # missing bwd -> branch unavailable, zeros
    x2, av2, K2, sp2, _ = pairvec.branch_vector(Sf, Sb, Ssh, mf, np.array([False]), msh, (np.zeros(5), np.ones(5)))
    assert not av2[0] and K2[0] == 0 and (x2 == 0).all()
    # reliability: inside eps -> Ra = 1; at eps + 2*eps -> Ra = 0
    r_in = pairvec.r_target(np.array([0.5]), np.array([30.0]), np.array([30]), np.array([0.0]), np.array([0.4]), 7.0)
    assert np.isclose(r_in[0], (0.5 + 1.0 + 0.4) / 3)
    r_out = pairvec.r_target(np.array([0.5]), np.array([30.0 + 21.0]), np.array([30]), np.array([0.0]), np.array([0.4]), 7.0)
    assert np.isclose(r_out[0], (0.5 + 0.0 + 0.4) / 3)
    R = pairvec.reliability(np.array([.6]), np.array([.6]), np.zeros((1, 7)), np.zeros((1, 7)), np.zeros((1, 7), bool), np.array([True]), np.array([0.0]), 0.1)
    assert np.isclose(R[0], 0.7 * 0.6 + 0.3 * 1.0)

def test_target_arrays():
    G = gap.grid(15, 60)
    tf, tb, T, K = pairvec.target_arrays(np.array([18, 30]), np.array([43, 30]), G)
    assert tf[0] == 45 and tb[0] == 20 and K[0] == 5 and K[1] == 0 and list(T[0, :5]) == [20, 25, 30, 35, 40]

from aafv import ssim as ssim_mod
def test_ssim_matches_skimage():
    from skimage.metrics import structural_similarity as sk
    rng = np.random.default_rng(5)
    a = (rng.random((64, 64)) * 255).astype("uint8"); b = np.clip(a.astype(int) + rng.normal(0, 25, a.shape), 0, 255).astype("uint8")
    assert abs(ssim_mod.ssim_gray(a, b) - sk(a, b, data_range=255)) < 1e-9
    assert abs(ssim_mod.ssim_gray(a, a) - 1.0) < 1e-12

def test_dupids_rule():
    from aafv import dupids
    import numpy as np
    ids = ["31274", "131274", "49", "322149", "07", "333307", "5", "10005"]
    cos = {("31274", "131274"): 0.95, ("49", "322149"): 0.03, ("07", "333307"): 0.93, ("5", "10005"): 0.1}
    q, rows = dupids.quarantine_people(ids, lambda a, b: cos.get((a, b), 0.0))
    assert q == {"31274", "131274", "07", "333307"}      # '49'/'322149' (short, low cos) and '5'/'10005' stay; 5-digit suffix rule fires on 31274
