#!/usr/bin/env bash
#
# Create the Python environment the validation scripts need: PyTorch plus the
# runtime dependencies of the ultralytics package (the ultralytics source itself
# is used from a checkout, via PYTHONPATH, so that the exact revision being
# validated against is under your control).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv venv --python 3.12 "$HERE/.venv"
uv pip install --python "$HERE/.venv/bin/python" \
  torch torchvision \
  numpy opencv-python-headless pillow pyyaml requests scipy tqdm psutil \
  py-cpuinfo polars ultralytics-thop

"$HERE/.venv/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
