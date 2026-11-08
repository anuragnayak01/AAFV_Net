#!/usr/bin/env bash
# E0.1: run on the 8-slice machine and paste the full output back. Changes nothing.
set +e
echo "== date"; date -u
echo "== gpus"; nvidia-smi -L; nvidia-smi
echo "== cpu/ram/disk"; nproc; free -h; df -h .
echo "== python"; python3 --version; pip --version
echo "== torch"; python3 -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())" 2>&1
echo "== reachability (HTTP status; 000 = blocked)"
for u in https://pypi.org https://github.com https://huggingface.co https://drive.google.com https://download.pytorch.org https://dlib.net; do
  printf "%s " "$u"; curl -s -o /dev/null -m 10 -w "%{http_code}\n" "$u"; done
