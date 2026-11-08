"""E0.2/E0.3 scoring-environment check: import each library, load its weights, run one forward pass on GPU.
Prints PASS/FAIL per component and the exact error. Needs: pip install timm lpips onnxruntime-gpu insightface scikit-image opencv-python-headless
Run:  CUDA_VISIBLE_DEVICES=<one MIG uuid> python scripts/e0_install_check.py"""
import traceback, importlib, importlib.metadata
def t(name, fn):
    try: print(f"PASS {name}: {fn()}")
    except Exception as e: print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
import torch
x = torch.rand(1, 3, 224, 224).cuda()
def timm_model(n):
    def f():
        import timm
        m = timm.create_model(n, pretrained=True, num_classes=0).cuda().eval()
        with torch.no_grad(): return tuple(m(x).shape), "timm " + timm.__version__
    return f
t("C8 EfficientNet-B0 (timm, HF weights)", timm_model("efficientnet_b0"))
t("C9 VGG16 (timm vgg16.tv_in1k, HF weights)", timm_model("vgg16.tv_in1k"))
def torchvision_vgg():
    import torchvision
    torchvision.models.vgg16(weights="IMAGENET1K_V1"); return "torchvision download worked"
t("C9 alt: torchvision VGG16 (download.pytorch.org)", torchvision_vgg)
def lp():
    import lpips
    m = lpips.LPIPS(net="alex").cuda()
    return float(m(torch.rand(1, 3, 256, 256).cuda() * 2 - 1, torch.rand(1, 3, 256, 256).cuda() * 2 - 1).item()), "lpips " + importlib.metadata.version("lpips")
t("C10 LPIPS alex", lp)
def arc():
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"]); app.prepare(ctx_id=0)
    rec = app.models["recognition"]
    import numpy as np
    out = rec.session.run(None, {rec.session.get_inputs()[0].name: np.random.rand(1, 3, 112, 112).astype("float32")})[0]
    return "session providers actually used: %s; output %s; onnxruntime %s" % (rec.session.get_providers(), out.shape, ort.__version__)
t("C7 ArcFace buffalo_l (onnxruntime)", arc)
def ssim():
    import numpy as np; from skimage.metrics import structural_similarity as s
    a = np.random.rand(256, 256); return s(a, a, data_range=1.0)
t("C11 SSIM", ssim)
def dl():
    import dlib; return "dlib " + dlib.__version__
t("C1 dlib import", dl)
