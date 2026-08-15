#!/usr/bin/env bash
# Download the official BitNet b1.58 GGUF and optionally build microsoft/BitNet
# (bitnet.cpp) for real ternary kernels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${BITNET_MODEL_DIR:-$ROOT/models/BitNet-b1.58-2B-4T}"
HF_REPO="${BITNET_HF_REPO:-microsoft/BitNet-b1.58-2B-4T-gguf}"
BITNET_SRC="${BITNET_HOME:-$ROOT/third_party/BitNet}"
BUILD_CPP="${BITNET_BUILD_CPP:-0}"

echo "==> Project root: $ROOT"
echo "==> Model dir:    $MODEL_DIR"

python3 -m pip install -q "huggingface_hub>=0.26.0" >/dev/null

mkdir -p "$MODEL_DIR"
echo "==> Baking $HF_REPO into $MODEL_DIR ..."
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<PY
from pathlib import Path
import os
os.environ["BITNET_MODEL_DIR"] = r"$MODEL_DIR"
os.environ["BITNET_HF_REPO"] = r"$HF_REPO"
os.environ["BITNET_AUTO_DOWNLOAD"] = "True"
from autogpt.llm.providers.bitnet_bake import ensure_bitnet_gguf, write_env_model_path
env = Path(r"$ROOT") / ".env"
if not env.exists() and (Path(r"$ROOT") / ".env.template").exists():
    env.write_text((Path(r"$ROOT") / ".env.template").read_text())
gguf = ensure_bitnet_gguf(project_root=Path(r"$ROOT"))
write_env_model_path(gguf, env)
print(gguf)
PY

if [[ "$BUILD_CPP" == "1" ]]; then
  echo "==> Building microsoft/BitNet (bitnet.cpp) into $BITNET_SRC"
  mkdir -p "$(dirname "$BITNET_SRC")"
  if [[ ! -d "$BITNET_SRC/.git" ]]; then
    git clone --recursive https://github.com/microsoft/BitNet.git "$BITNET_SRC"
  fi
  (
    cd "$BITNET_SRC"
    python3 -m pip install -r requirements.txt
    python3 setup_env.py -md "$MODEL_DIR" -q i2_s
  )
  {
    echo "BITNET_HOME=$BITNET_SRC"
    echo "BITNET_BACKEND=bitnet.cpp"
  } >> "$ROOT/.env"
  echo "==> bitnet.cpp build requested; BITNET_HOME set in .env"
else
  echo
  echo "Optional: for official BitNet ternary kernels (recommended for production):"
  echo "  BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh"
  echo "  # or clone microsoft/BitNet, run setup_env.py, then set:"
  echo "  #   BITNET_HOME=/path/to/BitNet"
  echo "  #   BITNET_BACKEND=bitnet.cpp"
fi

echo
echo "Done. Next:"
echo "  python3 -m pip install -r requirements.txt"
echo "  ./run.sh"
