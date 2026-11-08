"""P11 baseline: vision-language model (Unsloth, Qwen2-VL-2B 4-bit + LoRA) fine-tuned to answer 'same person? Yes/No' on the ORIGINAL pair (two crops in one prompt).
Score = logit(Yes) - logit(No) at the first answer token (continuous, so it can be calibrated like every other variant). UNTESTED against real Unsloth in the sandbox: run `selftest` first.
  python scripts/p11_vlm.py subset  --p5q artifacts/p5q --out artifacts/baselines            # label-free common subset (pair ids) for all baselines
  python scripts/p11_vlm.py selftest --p5q artifacts/p5q --out artifacts/baselines            # 40 train steps + 20 scored pairs
  python scripts/p11_vlm.py train   --p5q artifacts/p5q --out artifacts/baselines [--n_train 8000 --epochs 1]
  python scripts/p11_vlm.py score   --p5q artifacts/p5q --out artifacts/baselines             # launches one worker per GPU/MIG slice
Run inside the Unsloth venv. Output: <out>/vlm/scores.npz (pair_id, logit)."""
import sys, os, re, json, argparse, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["subset", "selftest", "train", "score", "worker"])
ap.add_argument("--p5q", required=True); ap.add_argument("--out", required=True); ap.add_argument("--crops", default="/dev/shm/aafv/crops"); ap.add_argument("--model", default="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
ap.add_argument("--n_train", type=int, default=8000); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--img", type=int, default=224)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--extra", default="")   # pair table (pair_id,img_id_1,img_id_2): score it instead of the subset, output scores_<stem>.npz
a = ap.parse_args(); out = Path(a.out); vd = out / "vlm"; vd.mkdir(parents=True, exist_ok=True); sub_f = out / "subset_pair_ids.txt"
rd = lambda s, cols=None: pd.read_csv(Path(a.p5q) / f"pair_vectors_{s}.csv", dtype={"img_id_1": str, "img_id_2": str, "pair_id": str}, usecols=cols or ["pair_id", "img_id_1", "img_id_2", "label", "weight", "set_name"])
Q = "Are these two face photos of the same person? The photos may be taken years apart. Answer Yes or No."

def make_subset():
    rng = np.random.default_rng(a.seed); keep = []
    tv = rd("train_val"); keep += list(tv.sample(min(2000, len(tv)), random_state=a.seed).pair_id)
    for s in ("calibration", "test"):                                   # selection by pair-construction set only, never by model score
        d = rd(s); keep += list(d[d.set_name == "D_random"].pair_id)
        for nm, k in (("same", 1500), ("D_hard", 1500)): x = d[d.set_name == nm]; keep += list(x.sample(min(k, len(x)), random_state=a.seed).pair_id)
        x = d[~d.set_name.isin(["D_random", "same", "D_hard"])]; keep += list(x.sample(min(1500, len(x)), random_state=a.seed).pair_id)
    sub_f.write_text("\n".join(dict.fromkeys(keep))); print("subset pairs:", len(set(keep)), "->", sub_f)

def img(k): 
    from PIL import Image
    p = Path(a.crops) / f"{k}.png"; p = p if p.exists() else Path(a.crops) / f"{Path(k).stem}.png"
    return Image.open(p).convert("RGB").resize((a.img, a.img))
def msg(k1, k2, ans=None):
    u = [{"type": "image", "image": img(k1)}, {"type": "image", "image": img(k2)}, {"type": "text", "text": Q}]; m = [{"role": "user", "content": u}]
    if ans is not None: m.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
    return m

def load_model(adapter=None):
    from unsloth import FastVisionModel
    src = adapter or a.model; model, tok = FastVisionModel.from_pretrained(src, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    if not adapter:
        model = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=a.seed)
    return model, tok

def train(max_steps=None):
    import torch
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    tr = rd("train_fit"); n = a.n_train if not max_steps else 80
    tr = pd.concat([tr[tr.label == 1].sample(n // 2, random_state=a.seed, replace=len(tr[tr.label == 1]) < n // 2), tr[tr.label == 0].sample(n // 2, random_state=a.seed)]).sample(frac=1, random_state=a.seed)
    data = [{"messages": msg(r.img_id_1, r.img_id_2, "Yes" if r.label == 1 else "No")} for r in tr.itertuples()]
    model, tok = load_model(); FastVisionModel.for_training(model)
    cfg = SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_steps=5, num_train_epochs=a.epochs, max_steps=max_steps if max_steps else -1, learning_rate=a.lr, fp16=not is_bf16_supported(), bf16=is_bf16_supported(),
                    logging_steps=10, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear", seed=a.seed, output_dir=str(vd / "ckpt"), report_to="none", save_strategy="no",
                    remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True}, dataset_num_proc=4, max_seq_length=2048)
    SFTTrainer(model=model, tokenizer=tok, data_collator=UnslothVisionDataCollator(model, tok), train_dataset=data, args=cfg).train()
    model.save_pretrained(str(vd / "lora")); tok.save_pretrained(str(vd / "lora")); print("saved adapter", vd / "lora")

def score_pairs(df, tag):
    import torch
    from unsloth import FastVisionModel
    model, tok = load_model(str(vd / "lora")); FastVisionModel.for_inference(model); yes = tok.tokenizer.encode("Yes", add_special_tokens=False)[0]; no = tok.tokenizer.encode("No", add_special_tokens=False)[0]
    pid, lg = [], []
    for i, r in enumerate(df.itertuples()):
        m = msg(r.img_id_1, r.img_id_2); text = tok.apply_chat_template(m[:1], add_generation_prompt=True); ims = [m[0]["content"][0]["image"], m[0]["content"][1]["image"]]
        x = tok(ims, text, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.no_grad(): lo = model(**x).logits[0, -1].float()
        pid.append(r.pair_id); lg.append(float(lo[yes] - lo[no]))
        if i % 200 == 0: print(tag, i, len(df), flush=True)
    assert np.isfinite(lg).all(); np.savez(vd / f"scores_part_{tag}.npz", pair_id=np.array(pid), logit=np.array(lg))

def pairs_to_score():
    if a.extra: return pd.read_csv(a.extra, dtype={"img_id_1": str, "img_id_2": str, "pair_id": str}, usecols=["pair_id", "img_id_1", "img_id_2"])
    keep = set(sub_f.read_text().split()); d = pd.concat([rd(s) for s in ("train_val", "calibration", "test")]); return d[d.pair_id.isin(keep)].reset_index(drop=True)

if a.mode == "subset": make_subset()
elif a.mode == "selftest":
    train(max_steps=40); d = pd.concat([rd("train_val").head(10), rd("calibration").sample(10, random_state=1)]); score_pairs(d, "selftest"); z = np.load(vd / "scores_part_selftest.npz"); print("selftest ok", z["logit"][:5], "-> now run: train, score")
elif a.mode == "train": train()
elif a.mode == "worker": d = pairs_to_score(); d = d.iloc[a.shard::a.nshards]; d = d.head(a.limit) if a.limit else d; score_pairs(d, f"{a.shard}" if not a.extra else f"{Path(a.extra).stem}_{a.shard}")
elif a.mode == "score":
    devs = re.findall(r"(?:GPU-|MIG-)[0-9a-f-]+", subprocess.check_output(["nvidia-smi", "-L"], text=True)); mig = [x for x in devs if x.startswith("MIG")]; devs = mig or devs
    ws = devs + devs[-1:] * max(0, 8 - len(devs)); n = len(ws); ps = [subprocess.Popen([sys.executable, __file__, "worker", "--p5q", a.p5q, "--out", a.out, "--crops", a.crops, "--model", a.model, "--shard", str(i), "--nshards", str(n), "--img", str(a.img)] + (["--extra", a.extra] if a.extra else []) + (["--limit", str(a.limit)] if a.limit else []),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": d}) for i, d in enumerate(ws)]
    for p in ps: p.wait()
    pre = f"{Path(a.extra).stem}_" if a.extra else ""; parts = [np.load(vd / f"scores_part_{pre}{i}.npz") for i in range(n) if (vd / f"scores_part_{pre}{i}.npz").exists()]; assert len(parts) == n, "a worker failed"
    np.savez(vd / (f"scores_{Path(a.extra).stem}.npz" if a.extra else "scores.npz"), pair_id=np.concatenate([p["pair_id"] for p in parts]), logit=np.concatenate([p["logit"] for p in parts])); print("saved scores")
