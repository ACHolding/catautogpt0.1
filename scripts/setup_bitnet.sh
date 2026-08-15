#!/usr/bin/env bash
# Download the official BitNet b1.58 GGUF and build microsoft/BitNet (bitnet.cpp).
# Official i2_s GGUF cannot load in stock llama-cpp-python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${BITNET_MODEL_DIR:-$ROOT/models/BitNet-b1.58-2B-4T}"
HF_REPO="${BITNET_HF_REPO:-microsoft/BitNet-b1.58-2B-4T-gguf}"
BITNET_SRC="${BITNET_HOME:-}"
if [[ -z "$BITNET_SRC" ]]; then
  # Make breaks on '#' / ':' in the path (this USB volume uses both).
  case "$ROOT" in
    *\#*|*\:*) BITNET_SRC="$HOME/.cache/catautogpt/BitNet" ;;
    *) BITNET_SRC="$ROOT/third_party/BitNet" ;;
  esac
fi
BUILD_CPP="${BITNET_BUILD_CPP:-1}"

echo "==> Project root: $ROOT"
echo "==> Model dir:    $MODEL_DIR"
echo "==> BitNet home:  $BITNET_SRC"

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
  echo "==> Building bitnet.cpp into $BITNET_SRC (Ninja/static; patches src1_cont)…"
  export BITNET_HOME="$BITNET_SRC"
  export BITNET_AUTO_BUILD=True
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<PY
from pathlib import Path
import os
os.environ["BITNET_HOME"] = r"$BITNET_SRC"
os.environ["BITNET_AUTO_BUILD"] = "True"
from autogpt.llm.providers.bitnet_cpp import ensure_bitnet_cli, write_env_bitnet_home
gguf = Path(r"$MODEL_DIR") / "ggml-model-i2_s.gguf"
cli = ensure_bitnet_cli(gguf=gguf if gguf.is_file() else None)
write_env_bitnet_home(Path(r"$BITNET_SRC"), Path(r"$ROOT") / ".env")
print(cli)
PY
  echo "==> bitnet.cpp ready; BITNET_HOME=$BITNET_SRC"
else
  echo
  echo "Skipped bitnet.cpp build (BITNET_BUILD_CPP=0)."
  echo "Official i2_s GGUF will NOT load in llama-cpp-python."
  echo "Re-run with BITNET_BUILD_CPP=1 or set BITNET_AUTO_BUILD=True."
fi

echo
echo "Done. Next:"
echo "  python3 -m pip install -r requirements.txt"
echo "  ./run_continuous.sh"
