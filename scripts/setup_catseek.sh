#!/usr/bin/env bash
# Bake CatSeek-GPU 0.1 (DeepSeek-R1-Distill-Qwen-14B GGUF) and vibe-train it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "==> CatSeek-GPU 0.1 setup in $ROOT"
python3 -m pip install -q "huggingface_hub>=0.26.0" "llama-cpp-python>=0.3.0" >/dev/null || true

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from pathlib import Path
from autogpt.llm.providers.catseek_bake import ensure_catseek_gguf, write_env_model_path
root = Path('.').resolve()
env = root / '.env'
if not env.exists() and (root / '.env.template').exists():
    env.write_text((root / '.env.template').read_text())
gguf = ensure_catseek_gguf(project_root_path=root)
write_env_model_path(gguf, env)
print(gguf)
PY

echo "==> vibe-train"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/vibe_train_catseek.py

echo
echo "Done. Run: ./run_continuous.sh"
echo "Artifacts: auto_gpt_workspace/catseek-gpu-0.1/"
