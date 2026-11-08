"""Make the FADING code (commit 14fefbf7edbf, written for diffusers 2023) work with current diffusers (0.42).
Three text changes, all behaviour-preserving, idempotent:
 1. unet.in_channels -> unet.config.in_channels
 2. attention hook: class is now named `Attention` (was `CrossAttention`); head reshape helpers were renamed; the transformer blocks
    call attention with keyword `encoder_hidden_states=` instead of positional `context`.
Usage: python scripts/patch_fading.py FADING_DIR"""
import sys
from pathlib import Path

def sub(text, old, new, label):
    if new in text and old not in text: return text, False
    assert old in text, f"pattern not found for {label}; the FADING code differs from the one I read"
    return text.replace(old, new), True

def main(root):
    root = Path(root)
    for f in root.rglob("*.py"):
        t = f.read_text()
        if ".unet.in_channels" in t:
            f.write_text(t.replace(".unet.in_channels", ".unet.config.in_channels")); print("patched in_channels:", f)
    f = root / "FADING_util" / "ptp_utils.py"; t = f.read_text(); changed = []
    for old, new, label in [
        ("def forward(x, context=None, mask=None):",
         "def forward(x, context=None, mask=None, encoder_hidden_states=None, attention_mask=None, **_kw):\n"
         "            if encoder_hidden_states is not None:\n                context = encoder_hidden_states",
         "forward signature"),
        ("self.reshape_heads_to_batch_dim(", "self.head_to_batch_dim(", "head_to_batch"),
        ("self.reshape_batch_dim_to_heads(", "self.batch_to_head_dim(", "batch_to_head"),
        ("net_.__class__.__name__ == 'CrossAttention'", "net_.__class__.__name__ in ('CrossAttention', 'Attention')", "class name"),
    ]:
        t, c = sub(t, old, new, label); changed.append((label, c))
    f.write_text(t); print("ptp_utils.py:", changed)

if __name__ == "__main__":
    main(sys.argv[1])
