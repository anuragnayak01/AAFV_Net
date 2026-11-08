"""Run the official SAM inference script under torch >= 2.6 (torch.load defaults to weights_only=True there).
The checkpoint is the official file whose sha256 is recorded in 01; loading it as a full pickle is a deliberate, recorded choice.
Usage: python scripts/run_sam_inference.py SAM_DIR <args of SAM scripts/inference.py>"""
import sys, os, functools, runpy
import torch
_orig = torch.load
torch.load = functools.partial(lambda *a, **k: _orig(*a, **{**{"weights_only": False}, **k}))
sam = os.path.abspath(sys.argv[1]); os.chdir(sam)
sys.path.insert(0, sam)
sys.argv = ["inference.py"] + sys.argv[2:]
runpy.run_path("scripts/inference.py", run_name="__main__")
