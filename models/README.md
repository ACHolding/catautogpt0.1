# Pre-baked BitNet weights (local)

This folder holds the official **BitNet b1.58 2B** GGUF used by catautogpt.

Weights are **not** committed to git (too large). On first run Auto-GPT will
auto-download them from Hugging Face when `BITNET_AUTO_DOWNLOAD=True`
(default in `.env.template`):

```text
microsoft/BitNet-b1.58-2B-4T-gguf  →  models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

**Important:** `ggml-model-i2_s.gguf` only loads in **bitnet.cpp** (microsoft/BitNet).
Stock `llama-cpp-python` fails with `Failed to load model from file`. With
`BITNET_AUTO_BUILD=True` (default), the first run clones/builds
`third_party/BitNet` and uses its `llama-cli`.

Manual bake + build:

```bash
./scripts/setup_bitnet.sh          # downloads GGUF + builds bitnet.cpp
# or
python3 -c "from autogpt.llm.providers.bitnet_bake import ensure_bitnet_gguf; print(ensure_bitnet_gguf())"
BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh
```

After download, `.env` gets `BITNET_MODEL_PATH=...` (and `BITNET_HOME` after build).
