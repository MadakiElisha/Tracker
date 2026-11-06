#!/usr/bin/env bash
# Set up a separate Python environment on the dev PC for compiling the VisDrone model to a Hailo HEF.
#
#   bash scripts/setup_hailo_dfc.sh ~/Downloads/hailo_dataflow_compiler-3.xx.x-py3-none-linux_x86_64.whl
#
# The DFC wheel only supports specific Python versions, so this reads the wheel's tag and builds
# venv-hailo/ with a matching interpreter (via uv, which can fetch any Python version). It does not
# touch the tracker's own venv. Afterwards:
#
#   source venv-hailo/bin/activate
#   python3 deployment/export_hailo.py
set -euo pipefail

WHEEL="${1:-}"
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
  echo "usage: bash scripts/setup_hailo_dfc.sh /path/to/hailo_dataflow_compiler-*.whl"
  echo "Download the DFC v3.x (Linux x86_64) from the Hailo Developer Zone (free account)."
  exit 1
fi
cd "$(dirname "$0")/.."

# System libraries the DFC needs (graph visualisation, Tk for its CLI tools).
sudo apt-get install -y graphviz graphviz-dev python3-tk >/dev/null

# Python version from the wheel tag, e.g. ...-cp310-cp310-linux_x86_64.whl -> 3.10.
TAG="$(basename "$WHEEL" | grep -oE 'cp3[0-9]+' | head -1 || true)"
if [[ -n "$TAG" ]]; then
  PYVER="3.${TAG#cp3}"
else
  PYVER="3.10"   # py3-none wheels: 3.10 is the version Hailo documents for DFC 3.x
fi
echo "[hailo] DFC wheel wants Python $PYVER"

if ! command -v uv >/dev/null; then
  echo "[hailo] installing uv (Python version manager) for this user"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PYVER" venv-hailo
# shellcheck disable=SC1091
source venv-hailo/bin/activate
uv pip install "$WHEEL"
# Freeze what the DFC pinned (numpy, onnx, tensorflow...) so Ultralytics cannot upgrade it underneath.
uv pip freeze | grep -viE '^(torch|torchvision)==' > venv-hailo/dfc-constraints.txt
# Same Ultralytics generation as the Pi (Hailo export + Hailo inference backend); CPU torch is enough here.
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install -c venv-hailo/dfc-constraints.txt "ultralytics>=8.4.203" onnxslim lap huggingface_hub

python3 - <<'EOF'
import hailo_sdk_client, ultralytics
print("[hailo] DFC import OK; Ultralytics", ultralytics.__version__)
EOF
echo "[hailo] ready:  source venv-hailo/bin/activate && python3 deployment/export_hailo.py"
