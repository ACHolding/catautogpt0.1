# Pre-baked CatSeek-GPU 0.1 weights (local)

This folder holds the **DeepSeek-R1-Distill-Qwen-14B** GGUF used by catautogpt
as **CatSeek-GPU 0.1**.

Weights are **not** committed to git (too large). On first run Auto-GPT will
auto-download them from Hugging Face when `CATSEEK_AUTO_DOWNLOAD=True`
(default in `.env.template`):

```text
bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
  →  models/CatSeek-GPU-0.1-14B/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf
```

(~8–9GB Q4_K_M — loads via stock `llama-cpp-python` with Metal/CUDA offload.)

Manual bake + vibe-train (artifacts → `auto_gpt_workspace/catseek-gpu-0.1/`):

```bash
./scripts/setup_catseek.sh
# or
python3 scripts/vibe_train_catseek.py
```

After download, `.env` gets `CATSEEK_MODEL_PATH=...` (and a BitNet alias for
older scripts).
