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

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub CLI..."
  python3 -m pip install -U "huggingface_hub[cli]"
fi

mkdir -p "$(dirname "$MODEL_DIR")"
echo "==> Downloading $HF_REPO ..."
huggingface-cli download "$HF_REPO" --local-dir "$MODEL_DIR"

GGUF=""
for candidate in \
  "$MODEL_DIR/ggml-model-i2_s.gguf" \
  "$MODEL_DIR/BitNet-b1.58-2B-4T.i2_s.gguf"; do
  if [[ -f "$candidate" ]]; then
    GGUF="$candidate"
    break
  fi
done
if [[ -z "$GGUF" ]]; then
  GGUF="$(find "$MODEL_DIR" -name '*.gguf' | head -n 1 || true)"
fi
if [[ -z "$GGUF" || ! -f "$GGUF" ]]; then
  echo "ERROR: no .gguf found under $MODEL_DIR" >&2
  exit 1
fi

echo "==> Chat GGUF: $GGUF"

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.template" "$ROOT/.env"
  echo "Created .env from template"
fi

# Upsert BITNET_MODEL_PATH in .env
python3 - <<PY
from pathlib import Path
path = Path(r"$ROOT/.env")
gguf = r"$GGUF"
text = path.read_text() if path.exists() else ""
lines = []
found = False
for line in text.splitlines():
    if line.startswith("BITNET_MODEL_PATH="):
        lines.append(f"BITNET_MODEL_PATH={gguf}")
        found = True
    else:
        lines.append(line)
if not found:
    lines.append(f"BITNET_MODEL_PATH={gguf}")
path.write_text("\n".join(lines) + "\n")
print(f"Wrote BITNET_MODEL_PATH={gguf}")
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
