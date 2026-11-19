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

# The DFC's first-run check requires the system python${PYVER}-dev package (C headers). Ubuntu
# 24.04 ships only Python 3.12, so older versions come from the deadsnakes PPA.
if ! dpkg -s "python${PYVER}-dev" >/dev/null 2>&1; then
  if ! apt-cache show "python${PYVER}-dev" >/dev/null 2>&1; then
    echo "[hailo] python${PYVER}-dev is not in this Ubuntu release: adding ppa:deadsnakes/ppa"
    sudo apt-get install -y software-properties-common >/dev/null
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
  fi
  sudo apt-get install -y "python${PYVER}-dev" "python${PYVER}-venv"
fi

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
# Run the DFC's own first-use system check now, answering its "continue?" prompt, so missing
# requirements show up here rather than half-way through a compile.
echo "[hailo] DFC system check (RAM < 32 GB or no CUDA are warnings only: it compiles on the CPU)"
hailo --version < <(yes y) || { echo "[hailo] DFC system check failed: see the [Error] lines above"; exit 1; }
echo "[hailo] ready:  source venv-hailo/bin/activate && python3 deployment/export_hailo.py"
