"""Replace SAM's two run-time-compiled CUDA ops with pure-PyTorch versions of the same math.
Reason (01 D6/R-02): the repo pins torch 1.6 and compiles .cu files at import; we run torch 2.13 / CUDA 13 on Blackwell.
The fused op is leaky_relu(x + bias) * scale. The upfirdn2d op is the standard upsample / pad / FIR filter / downsample.
These replace the files in place (originals saved as *.orig). Validation is a visual reconstruction test on the user's machine (E0.3 SAM)."""
import sys, shutil
from pathlib import Path

FUSED = '''import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest), negative_slope=negative_slope) * scale
'''

UPFIR = '''import torch
from torch.nn import functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    n, c, h, w = input.shape
    x = input.reshape(-1, h, w, 1)
    out = upfirdn2d_native(x, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out.reshape(n, c, out.shape[1], out.shape[2])


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    return out[:, ::down_y, ::down_x, :]
'''

def main(sam_dir):
    op = Path(sam_dir) / "models" / "stylegan2" / "op"
    assert op.is_dir(), f"not found: {op}"
    for name, text in (("fused_act.py", FUSED), ("upfirdn2d.py", UPFIR)):
        f = op / name
        if not (op / (name + ".orig")).exists():
            shutil.copy(f, op / (name + ".orig"))
        f.write_text(text)
        print("patched", f)

if __name__ == "__main__":
    main(sys.argv[1])
