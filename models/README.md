# Pre-baked BitNet weights (local)

This folder holds the official **BitNet b1.58 2B** GGUF used by catautogpt.

Weights are **not** committed to git (too large). On first run Auto-GPT will
auto-download them from Hugging Face when `BITNET_AUTO_DOWNLOAD=True`
(default in `.env.template`):

```text
microsoft/BitNet-b1.58-2B-4T-gguf  →  models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

Manual bake:

```bash
./scripts/setup_bitnet.sh
# or
python3 -c "from autogpt.llm.providers.bitnet_bake import ensure_bitnet_gguf; print(ensure_bitnet_gguf())"
```

After download, `.env` gets `BITNET_MODEL_PATH=...` so later vibe-checks start cold-fast.
