"""P6 model: one linear score learner per branch, then an 11-input MLP. Torch. See 04 section P6."""
import numpy as np, pandas as pd, torch, torch.nn as nn
COLS = ["arc", "eff", "vgg", "lpips", "ssim"]
CTX = ["g", "Q", "enhanced_pair", "R_sam", "R_diff", "avail_sam", "avail_diff", "avail_multi"]   # z = [W_d, W_s, W_f] + CTX = 11 inputs
I_RSAM, I_ASAM, I_AMULTI = CTX.index("R_sam"), CTX.index("avail_sam"), CTX.index("avail_multi")

def arrays(df):
    f = lambda p: df[[p + c for c in COLS]].to_numpy(np.float32)
    return f("d_"), f("s_"), f("f_"), df[CTX].to_numpy(np.float32)

class Fusion(nn.Module):
    def __init__(self, sizes=(32, 16), dropout=0.1):
        super().__init__()
        self.ld, self.ls, self.lf = nn.Linear(5, 1), nn.Linear(5, 1), nn.Linear(5, 1)
        h1, h2 = sizes
        self.mlp = nn.Sequential(nn.Linear(11, h1), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h2, 1))
    def forward(self, D, S, F, C):
        W = torch.cat([self.ld(D), self.ls(S), self.lf(F)], 1)
        return self.mlp(torch.cat([W, C], 1)).squeeze(1), W

def mask_sam(S, C, mask):
    """Stage A training mask: the SAM branch is hidden for the rows in `mask` (direct never masked; diffusion is always absent)."""
    S = S.clone(); C = C.clone(); S[mask] = 0; C[mask, I_RSAM] = 0; C[mask, I_ASAM] = 0; C[mask, I_AMULTI] = 0
    return S, C

def make_model(sizes=(32, 16)):
    return Fusion(tuple(sizes))

def save_model(model, path, meta):
    torch.save({"state": model.state_dict(), "meta": meta}, path)

def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False); m = make_model(ck["meta"]["sizes"]); m.load_state_dict(ck["state"]); return m.to(device).eval()

def load_model_from_state(state, device="cpu", sizes=(32, 16)):
    m = make_model(sizes); m.load_state_dict(state); return m.to(device).eval()

@torch.no_grad()
def predict(models, df, device="cpu", bs=65536):
    """Seed-averaged logit S (N,) and branch scores W (N,3) (averaged over models)."""
    D, S, F, C = [torch.from_numpy(a) for a in arrays(df)]
    outS, outW = 0, 0
    for m in models:
        s_l, w_l = [], []
        for i in range(0, len(df), bs):
            s, w = m(D[i:i+bs].to(device), S[i:i+bs].to(device), F[i:i+bs].to(device), C[i:i+bs].to(device))
            s_l.append(s.cpu()); w_l.append(w.cpu())
        outS = outS + torch.cat(s_l).numpy(); outW = outW + torch.cat(w_l).numpy()
    return outS / len(models), outW / len(models)

def train_one(tr, va, seed, lr, wd, p_mask=0.12, sizes=(32, 16), device="cpu", max_epochs=100, patience=10, bs=2048):
    """tr, va: DataFrames of pair vectors (train_fit, train_val). Early stopping on UNWEIGHTED log loss of va. Returns (model, history, n_skipped)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = make_model(sizes).to(device)
    T = [torch.from_numpy(a).to(device) for a in arrays(tr)]; y = torch.tensor(tr.label.to_numpy(np.float32), device=device); w = torch.tensor(tr.weight.to_numpy(np.float32), device=device)
    V = [torch.from_numpy(a).to(device) for a in arrays(va)]; yv = torch.tensor(va.label.to_numpy(np.float32), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd); bce = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, bad, hist, skipped = 1e9, None, 0, [], 0
    for ep in range(max_epochs):
        model.train(); perm = torch.from_numpy(rng.permutation(len(y))).to(device); tl = 0.0; nb = 0
        for i in range(0, len(y), bs):
            ix = perm[i:i+bs]
            if y[ix].sum() == 0: skipped += 1; continue
            D, S, F, C = (t[ix] for t in T)
            S, C = mask_sam(S, C, torch.rand(len(ix), device=device) < p_mask)
            s, _ = model(D, S, F, C); loss = (w[ix] * bce(s, y[ix])).mean()
            if not torch.isfinite(loss): raise RuntimeError("NaN loss, stopping")
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()); nb += 1
        model.eval()
        with torch.no_grad(): sv, _ = model(*V); vl = float(bce(sv, yv).mean())
        hist.append(dict(epoch=ep, train_loss=tl / max(nb, 1), val_logloss=vl))
        if vl < best - 1e-5: best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model.eval(), hist, skipped
