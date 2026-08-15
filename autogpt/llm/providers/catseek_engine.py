"""CatSeek-GPU 0.1 — real local DeepSeek-R1-Distill-Qwen-14B GGUF engine.

Replaces the broken BitNet i2_s path with a standard llama.cpp GGUF that actually
generates text. Branded as ``catseek-gpu-0.1`` (DeepSeek-R1 class, ~14B Q4_K_M).

All vibe-train / trace artifacts go under ``auto_gpt_workspace/catseek-gpu-0.1/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from colorama import Fore

from autogpt.logs import logger

MODEL_ID = "catseek-gpu-0.1"
DEFAULT_N_CTX = 8192
DEFAULT_N_BATCH = 512
DEFAULT_MAX_TOKENS = 1024
DEFAULT_EMBED_DIM = 1536
DEFAULT_HF_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"
# ~8.4GB — fits Apple Silicon with Metal offload; override via CATSEEK_HF_FILE.
DEFAULT_HF_FILE = "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"

_CHAT_LOCK = threading.RLock()
_chat_llm: Any = None
_chat_path: str | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    """Always dump CatSeek artifacts into auto_gpt_workspace."""
    root = project_root() / "auto_gpt_workspace" / MODEL_ID
    root.mkdir(parents=True, exist_ok=True)
    return root


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_threads() -> int:
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 2 if cpus > 4 else cpus))


def _default_gpu_layers() -> int:
    # -1 = offload all layers to Metal/CUDA when available.
    return _env_int("CATSEEK_N_GPU_LAYERS", -1)


def model_dir() -> Path:
    override = os.getenv("CATSEEK_MODEL_DIR")
    if override and override.strip():
        return Path(override).expanduser()
    return project_root() / "models" / "CatSeek-GPU-0.1-14B"


def resolve_model_path() -> Path:
    for raw in (
        os.getenv("CATSEEK_MODEL_PATH"),
        os.getenv("BITNET_MODEL_PATH"),
        os.getenv("LLM_MODEL_PATH"),
    ):
        if not raw or not raw.strip():
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path

    root = model_dir()
    preferred = os.getenv("CATSEEK_HF_FILE") or DEFAULT_HF_FILE
    candidate = root / preferred
    if candidate.is_file():
        return candidate
    hits = sorted(root.glob("*.gguf")) if root.is_dir() else []
    if hits:
        # Prefer Q4_K_M / Q5_K_M sized files when several are present.
        for token in ("Q4_K_M", "Q5_K_M", "Q4_K_S", "Q3_K_M"):
            for h in hits:
                if token.lower() in h.name.lower():
                    return h
        return hits[0]
    raise FileNotFoundError(
        f"No CatSeek GGUF under {root}. Set CATSEEK_MODEL_PATH or run "
        f"./scripts/setup_catseek.sh / vibe-train."
    )


def ensure_model_path(*, auto_download: bool | None = None) -> Path:
    try:
        path = resolve_model_path()
        os.environ["CATSEEK_MODEL_PATH"] = str(path)
        return path
    except FileNotFoundError:
        if auto_download is None:
            auto_download = _env_bool("CATSEEK_AUTO_DOWNLOAD", True)
        if not auto_download:
            raise
        from autogpt.llm.providers.catseek_bake import ensure_catseek_gguf

        path = ensure_catseek_gguf()
        os.environ["CATSEEK_MODEL_PATH"] = str(path)
        return path


def apply_deepseek_chat_template(
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """DeepSeek-R1-Distill-Qwen / ChatML-style template."""
    parts: list[str] = []
    for message in messages:
        role = message.get("role") or "user"
        content = (message.get("content") or "").strip()
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _llama_kwargs() -> dict[str, Any]:
    n_ctx = _env_int("CATSEEK_N_CTX", DEFAULT_N_CTX)
    n_batch = _env_int("CATSEEK_N_BATCH", DEFAULT_N_BATCH)
    n_threads = _env_int("CATSEEK_N_THREADS", _default_threads())
    return {
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ubatch": _env_int("CATSEEK_N_UBATCH", min(n_batch, 512)),
        "n_threads": n_threads,
        "n_threads_batch": _env_int("CATSEEK_N_THREADS_BATCH", n_threads),
        "n_gpu_layers": _default_gpu_layers(),
        "use_mmap": _env_bool("CATSEEK_USE_MMAP", True),
        "use_mlock": _env_bool("CATSEEK_USE_MLOCK", False),
        "logits_all": False,
        "embedding": False,
        "verbose": _env_bool("CATSEEK_VERBOSE", False),
        "seed": _env_int("CATSEEK_SEED", -1),
        "chat_format": os.getenv("CATSEEK_CHAT_FORMAT", "chatml"),
    }


def get_chat_llm(force_reload: bool = False) -> Any:
    global _chat_llm, _chat_path
    model_path = str(ensure_model_path())
    if _chat_llm is not None and _chat_path == model_path and not force_reload:
        return _chat_llm

    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise SystemExit(
            "llama-cpp-python is required for CatSeek-GPU 0.1. "
            "Install with: pip install llama-cpp-python"
        ) from err

    kwargs = _llama_kwargs()
    logger.typewriter_log(
        "CatSeek-GPU: ",
        Fore.GREEN,
        f"loading {MODEL_ID} ← {model_path} "
        f"(n_ctx={kwargs['n_ctx']}, batch={kwargs['n_batch']}, "
        f"threads={kwargs['n_threads']}, gpu_layers={kwargs['n_gpu_layers']})",
    )
    try:
        _chat_llm = Llama(model_path=model_path, **kwargs)
    except Exception as first_err:
        kwargs.pop("chat_format", None)
        logger.warn(f"CatSeek reload without chat_format ({first_err})")
        _chat_llm = Llama(model_path=model_path, **kwargs)

    _chat_path = model_path
    return _chat_llm


def _trace_path() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = workspace_root() / "traces" / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_trace(record: dict[str, Any]) -> None:
    try:
        with _trace_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as err:
        logger.warn(f"CatSeek trace write skipped ({err})")


def create_chat_completion_raw(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SimpleNamespace:
    """Real llama.cpp chat completion; traces to auto_gpt_workspace."""
    cleaned = [
        {"role": m.get("role") or "user", "content": m.get("content") or ""}
        for m in messages
    ]
    max_tokens = max(1, int(max_tokens or DEFAULT_MAX_TOKENS))
    temperature = float(temperature if temperature is not None else 0.2)
    t0 = time.time()

    with _CHAT_LOCK:
        llm = get_chat_llm()
        stop = ["<|im_end|>", "<|endoftext|>"]
        gen_kwargs: dict[str, Any] = {
            "temperature": max(0.0, temperature),
            "max_tokens": max_tokens,
            "stop": stop,
        }
        if temperature <= 0:
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 1

        try:
            result = llm.create_chat_completion(
                messages=cleaned,
                **gen_kwargs,
            )
            choice = result["choices"][0]
            content = (choice.get("message") or {}).get("content") or choice.get("text") or ""
            usage = result.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        except Exception:
            # Raw completion fallback with ChatML template.
            prompt = apply_deepseek_chat_template(cleaned, add_generation_prompt=True)
            prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
            room = max(16, _env_int("CATSEEK_N_CTX", DEFAULT_N_CTX) - prompt_tokens - 8)
            result = llm(
                prompt,
                max_tokens=min(max_tokens, room),
                temperature=gen_kwargs["temperature"],
                stop=stop,
                echo=False,
            )
            content = result["choices"][0]["text"]
            usage = result.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(
                usage.get("completion_tokens") or max(1, len(content) // 4)
            )

    content = (content or "").strip()
    # Strip DeepSeek thinking blocks for agent JSON loops unless kept.
    if _env_bool("CATSEEK_STRIP_THINK", True) and "</think>" in content:
        content = content.split("</think>", 1)[-1].strip()
    if content.startswith("<think>"):
        # Unclosed think — keep a short stub rather than empty.
        content = content.replace("<think>", "").strip() or content

    _append_trace(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "latency_s": round(time.time() - t0, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "messages": cleaned,
            "content": content[:8000],
        }
    )

    # Also mirror last reply into workspace for vibe-check browsing.
    try:
        latest = workspace_root() / "latest_reply.txt"
        latest.write_text(content, encoding="utf-8")
    except Exception:
        pass

    return SimpleNamespace(
        model=MODEL_ID,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[SimpleNamespace(message={"role": "assistant", "content": content})],
    )


def create_text_completion_raw(
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SimpleNamespace:
    result = create_chat_completion_raw(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = result.choices[0].message["content"]
    return SimpleNamespace(
        model=f"{MODEL_ID}-text",
        usage=result.usage,
        choices=[SimpleNamespace(text=text)],
    )


def _hash_embedding(text: str, dims: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dims).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


def create_embedding_raw(texts: list[str]) -> SimpleNamespace:
    dims = _env_int("CATSEEK_EMBED_DIMS", DEFAULT_EMBED_DIM)
    data = [
        {"index": i, "embedding": _hash_embedding(t, dims)} for i, t in enumerate(texts)
    ]
    return SimpleNamespace(data=data, model=f"{MODEL_ID}-embed")


def warm_start() -> Path:
    path = ensure_model_path()
    os.environ["CATSEEK_MODEL_PATH"] = str(path)
    # Persist into BITNET_MODEL_PATH too so older env checks stay happy.
    os.environ.setdefault("BITNET_MODEL_PATH", str(path))
    get_chat_llm()
    logger.typewriter_log(
        "CatSeek-GPU: ",
        Fore.GREEN,
        f"{MODEL_ID} ready · model={path} · workspace={workspace_root()}",
    )
    # Manifest for vibe-check / agents.
    try:
        manifest = {
            "model_id": MODEL_ID,
            "gguf": str(path),
            "n_ctx": _env_int("CATSEEK_N_CTX", DEFAULT_N_CTX),
            "gpu_layers": _default_gpu_layers(),
            "workspace": str(workspace_root()),
            "ready_at": datetime.now(timezone.utc).isoformat(),
        }
        (workspace_root() / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return path
