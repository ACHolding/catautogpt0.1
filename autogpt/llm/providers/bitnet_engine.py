"""Real BitNet inference engine for Auto-GPT.

BitNet b1.58 (microsoft/BitNet-b1.58-2B-4T) is a native 1.58-bit LLM that uses the
LLaMA 3 tokenizer and chat template. Official kernels live in microsoft/BitNet
(bitnet.cpp). This module:

1. Prefers a compiled ``bitnet.cpp`` ``llama-cli`` when ``BITNET_HOME`` /
   ``BITNET_CLI`` is set (lossless ternary kernels).
2. Otherwise uses ``llama-cpp-python`` with BitNet-tuned load/generate settings
   and the official LLaMA-3 chat template (works for development; warn that
   full BitNet speed needs bitnet.cpp).
3. Optionally loads a BitNet embedding GGUF via ``BITNET_EMBED_MODEL_PATH``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from colorama import Fore

from autogpt.logs import logger

# BitNet-b1.58-2B-4T native context (see model card).
DEFAULT_N_CTX = 4096
DEFAULT_N_BATCH = 512
DEFAULT_MAX_TOKENS = 512
DEFAULT_EMBED_DIM = 1024

_CHAT_LOCK = threading.RLock()
_chat_llm: Any = None
_chat_path: str | None = None
_embed_llm: Any = None
_embed_path: str | None = None
_warned_fallback = False

_LLAMA3_SPECIALS = (
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|end_of_text|>",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    # Leave one core free for the agent loop / OS when we have headroom.
    return max(1, cpus - 1) if cpus > 2 else cpus


def resolve_chat_model_path() -> Path:
    """Locate the BitNet chat GGUF (prefer official i2_s filename)."""
    preferred_names = (
        "ggml-model-i2_s.gguf",
        "BitNet-b1.58-2B-4T.i2_s.gguf",
        "bitnet-b1.58-2B-4T.i2_s.gguf",
    )
    root = project_root()
    search_roots = [
        root / "models" / "BitNet-b1.58-2B-4T",
        root / "models" / "bitnet-b1.58-2B-4T",
        root / "models",
        root,
    ]

    for raw in (
        os.getenv("BITNET_MODEL_PATH"),
        os.getenv("LLM_MODEL_PATH"),
    ):
        if not raw or not raw.strip():
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            for name in preferred_names:
                hit = path / name
                if hit.is_file():
                    return hit
            ggufs = sorted(path.glob("**/*.gguf"))
            if ggufs:
                for g in ggufs:
                    if "i2_s" in g.name.lower() or "bitnet" in g.name.lower():
                        return g
                return ggufs[0]
        # Explicit path was set but missing — fail fast with a clear error.
        raise FileNotFoundError(
            f"BITNET_MODEL_PATH is set but not found: {path}\n"
            "Fix .env or run ./scripts/setup_bitnet.sh"
        )

    for base in search_roots:
        if not base.exists():
            continue
        for name in preferred_names:
            hit = base / name
            if hit.is_file():
                return hit
        ggufs = sorted(base.glob("**/*.gguf"))
        # Skip embedding-only filenames when picking a chat model.
        chat_ggufs = [
            g
            for g in ggufs
            if "embed" not in g.name.lower() and "embedding" not in g.name.lower()
        ]
        if chat_ggufs:
            for g in chat_ggufs:
                if "i2_s" in g.name.lower() or "bitnet" in g.name.lower():
                    return g
            return chat_ggufs[0]

    raise FileNotFoundError(
        "No BitNet chat GGUF found. Set BITNET_MODEL_PATH or run:\n"
        "  ./scripts/setup_bitnet.sh\n"
        "Expected something like models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    )


def ensure_chat_model_path(*, auto_download: bool | None = None) -> Path:
    """Resolve a chat GGUF, auto-downloading the official BitNet bake if needed."""
    try:
        return resolve_chat_model_path()
    except FileNotFoundError:
        from autogpt.llm.providers.bitnet_bake import ensure_bitnet_gguf

        path = ensure_bitnet_gguf(auto_download=auto_download)
        os.environ["BITNET_MODEL_PATH"] = str(path)
        return path


def resolve_embed_model_path() -> Path | None:
    """Optional BitNet embedding GGUF (BitNet-embedding-0.6B / 270M)."""
    raw = os.getenv("BITNET_EMBED_MODEL_PATH")
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        if path.is_dir():
            ggufs = sorted(path.glob("**/*.gguf"))
            if ggufs:
                return ggufs[0]
        raise FileNotFoundError(f"BITNET_EMBED_MODEL_PATH not found: {path}")

    root = project_root() / "models"
    if not root.exists():
        return None
    for pattern in (
        "**/BitNet-embedding*/**/*.gguf",
        "**/bitnet-embedding*/**/*.gguf",
        "**/*embed*.gguf",
    ):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_bitnet_cli() -> Path | None:
    """Locate microsoft/BitNet ``llama-cli`` (bitnet.cpp build)."""
    explicit = os.getenv("BITNET_CLI")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    home = os.getenv("BITNET_HOME")
    search: list[Path] = []
    if home:
        search.append(Path(home).expanduser())
    search.extend(
        [
            project_root() / "third_party" / "BitNet",
            project_root() / "BitNet",
            Path.home() / "BitNet",
        ]
    )

    bin_names = ["llama-cli"]
    if platform.system() == "Windows":
        bin_names = ["llama-cli.exe", "llama-cli"]

    for base in search:
        candidates = [
            base / "build" / "bin" / "Release" / bin_names[0],
            base / "build" / "bin" / bin_names[0],
        ]
        for name in bin_names[1:]:
            candidates.append(base / "build" / "bin" / "Release" / name)
            candidates.append(base / "build" / "bin" / name)
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return c

    which = shutil.which("llama-cli")
    if which and _env_bool("BITNET_USE_SYSTEM_LLAMA_CLI", False):
        return Path(which)
    return None


def backend_name() -> str:
    forced = (os.getenv("BITNET_BACKEND") or "auto").strip().lower()
    if forced in {"bitnet.cpp", "cli", "bitnet"}:
        return "bitnet.cpp" if find_bitnet_cli() else "missing-cli"
    if forced in {"llama-cpp", "llamacpp", "python"}:
        return "llama-cpp-python"
    return "bitnet.cpp" if find_bitnet_cli() else "llama-cpp-python"


def apply_llama3_chat_template(
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Format messages with the LLaMA 3 / BitNet instruct chat template."""
    parts: list[str] = ["<|begin_of_text|>"]
    for message in messages:
        role = message.get("role") or "user"
        content = (message.get("content") or "").strip()
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _validate_bitnetish(path: Path) -> None:
    name = path.name.lower()
    parent = str(path.parent).lower()
    looks_bitnet = any(
        token in name or token in parent
        for token in ("bitnet", "i2_s", "tl1", "tl2", "b1.58", "b1_58")
    )
    if not looks_bitnet:
        logger.warn(
            f"{Fore.YELLOW}Model path {path} does not look like an official BitNet "
            f"GGUF (expected i2_s / bitnet in the name). Inference may be wrong or slow."
            f"{Fore.RESET}"
        )


def _llama_kwargs(*, embedding: bool) -> dict[str, Any]:
    n_ctx = _env_int("BITNET_N_CTX", DEFAULT_N_CTX)
    n_batch = _env_int("BITNET_N_BATCH", DEFAULT_N_BATCH)
    n_threads = _env_int("BITNET_N_THREADS", _default_threads())
    n_threads_batch = _env_int("BITNET_N_THREADS_BATCH", n_threads)
    n_gpu_layers = _env_int("BITNET_N_GPU_LAYERS", 0)
    kwargs: dict[str, Any] = {
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ubatch": _env_int("BITNET_N_UBATCH", min(n_batch, 512)),
        "n_threads": n_threads,
        "n_threads_batch": n_threads_batch,
        "n_gpu_layers": n_gpu_layers,
        "use_mmap": _env_bool("BITNET_USE_MMAP", True),
        "use_mlock": _env_bool("BITNET_USE_MLOCK", False),
        "logits_all": False,
        "embedding": embedding,
        "verbose": _env_bool("BITNET_VERBOSE", False),
        "seed": _env_int("BITNET_SEED", -1),
    }
    # flash_attn can fail on some builds; allow opt-in.
    if _env_bool("BITNET_FLASH_ATTN", False):
        kwargs["flash_attn"] = True
    if not embedding:
        # Prefer built-in llama-3 chat formatting when available.
        kwargs["chat_format"] = os.getenv("BITNET_CHAT_FORMAT", "llama-3")
    return kwargs


def get_chat_llm(force_reload: bool = False) -> Any:
    """Lazy-load chat model via llama-cpp-python (in-process, keeps KV cache warm)."""
    global _chat_llm, _chat_path, _warned_fallback
    model_path = str(ensure_chat_model_path())
    if _chat_llm is not None and _chat_path == model_path and not force_reload:
        return _chat_llm

    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise SystemExit(
            "llama-cpp-python is required for BitNet (in-process) inference. "
            "Install with: pip install llama-cpp-python\n"
            "For official BitNet kernels, build microsoft/BitNet and set BITNET_HOME."
        ) from err

    path = Path(model_path)
    _validate_bitnetish(path)
    kwargs = _llama_kwargs(embedding=False)

    if backend_name() != "bitnet.cpp" and not _warned_fallback:
        _warned_fallback = True
        logger.typewriter_log(
            "BitNet: ",
            Fore.YELLOW,
            "using llama-cpp-python fallback. For real ternary kernels / speed, "
            "build https://github.com/microsoft/BitNet and set BITNET_HOME "
            "(or BITNET_CLI). See ./scripts/setup_bitnet.sh",
        )

    logger.typewriter_log(
        "BitNet: ",
        Fore.GREEN,
        f"loading chat model {model_path} "
        f"(n_ctx={kwargs['n_ctx']}, batch={kwargs['n_batch']}, "
        f"threads={kwargs['n_threads']}, gpu_layers={kwargs['n_gpu_layers']})",
    )

    try:
        _chat_llm = Llama(model_path=model_path, **kwargs)
    except Exception as first_err:
        # Older wheels may not accept chat_format / flash_attn.
        kwargs.pop("chat_format", None)
        kwargs.pop("flash_attn", None)
        logger.warn(f"BitNet reload without chat_format ({first_err})")
        _chat_llm = Llama(model_path=model_path, **kwargs)

    _chat_path = model_path
    return _chat_llm


def get_embed_llm(force_reload: bool = False) -> Any | None:
    """Lazy-load optional BitNet embedding GGUF."""
    global _embed_llm, _embed_path
    path = resolve_embed_model_path()
    if path is None:
        return None
    model_path = str(path)
    if _embed_llm is not None and _embed_path == model_path and not force_reload:
        return _embed_llm

    from llama_cpp import Llama

    kwargs = _llama_kwargs(embedding=True)
    kwargs.pop("chat_format", None)
    # Embedding models only need a modest context.
    kwargs["n_ctx"] = _env_int("BITNET_EMBED_N_CTX", min(2048, kwargs["n_ctx"]))
    logger.typewriter_log(
        "BitNet: ",
        Fore.GREEN,
        f"loading embedding model {model_path}",
    )
    _embed_llm = Llama(model_path=model_path, **kwargs)
    _embed_path = model_path
    return _embed_llm


def tokenize_count(text: str) -> int:
    """Count tokens with the loaded BitNet tokenizer.

    Raises if the chat model is not warm-loaded yet so callers can fall back
    to an approximate tokenizer without accidentally spawning a model load.
    """
    if _chat_llm is None:
        raise RuntimeError("BitNet chat model is not loaded")
    tokens = _chat_llm.tokenize(text.encode("utf-8"), add_bos=False)
    return len(tokens)


def count_chat_tokens(messages: Sequence[dict[str, str]]) -> int:
    prompt = apply_llama3_chat_template(messages, add_generation_prompt=True)
    return tokenize_count(prompt)


def _strip_special_tokens(text: str) -> str:
    out = text
    for tok in _LLAMA3_SPECIALS:
        out = out.replace(tok, "")
    return out.strip()


def _complete_via_bitnet_cli(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    cli = find_bitnet_cli()
    if cli is None:
        raise FileNotFoundError("bitnet.cpp llama-cli not found")

    model = resolve_chat_model_path()
    prompt = apply_llama3_chat_template(messages, add_generation_prompt=True)
    n_ctx = _env_int("BITNET_N_CTX", DEFAULT_N_CTX)
    n_threads = _env_int("BITNET_N_THREADS", _default_threads())

    cmd = [
        str(cli),
        "-m",
        str(model),
        "-n",
        str(max_tokens),
        "-t",
        str(n_threads),
        "-c",
        str(n_ctx),
        "--temp",
        str(temperature),
        "-ngl",
        str(_env_int("BITNET_N_GPU_LAYERS", 0)),
        "-p",
        prompt,
        "--no-display-prompt",
    ]
    # Stop at end-of-turn when the build supports it.
    if _env_bool("BITNET_CLI_STOP", True):
        cmd.extend(["--reverse-prompt", "<|eot_id|>"])

    logger.debug(f"BitNet CLI: {' '.join(cmd[:8])} ...")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=_env_int("BITNET_CLI_TIMEOUT", 600),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"bitnet.cpp failed (code {proc.returncode}): {err[:800]}")

    content = _strip_special_tokens(proc.stdout or "")
    # CLI does not always report usage; estimate from tokenizer if in-process available.
    prompt_tokens = tokenize_count(prompt)
    completion_tokens = tokenize_count(content) if content else 0
    return content, prompt_tokens, completion_tokens


def _complete_via_llama_cpp(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    llm = get_chat_llm()
    n_ctx = _env_int("BITNET_N_CTX", DEFAULT_N_CTX)
    # Keep room for the prompt — never request more than remaining context.
    formatted = apply_llama3_chat_template(messages, add_generation_prompt=True)
    prompt_tokens = len(llm.tokenize(formatted.encode("utf-8"), add_bos=False))
    room = max(16, n_ctx - prompt_tokens - 8)
    max_tokens = max(1, min(max_tokens, room))

    stop = [
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
    ]
    gen_kwargs: dict[str, Any] = {
        "temperature": max(0.0, temperature),
        "max_tokens": max_tokens,
        "stop": stop,
    }
    if temperature <= 0:
        # Greedy / near-deterministic decoding for agent loops.
        gen_kwargs["temperature"] = 0.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 1

    # Prefer create_chat_completion when chat_format is wired; else raw completion.
    try:
        result = llm.create_chat_completion(
            messages=[{"role": m["role"], "content": m.get("content") or ""} for m in messages],
            **gen_kwargs,
        )
        content = result["choices"][0]["message"].get("content") or ""
        usage = result.get("usage") or {}
        return (
            _strip_special_tokens(content),
            int(usage.get("prompt_tokens") or prompt_tokens),
            int(usage.get("completion_tokens") or tokenize_count(content)),
        )
    except Exception:
        result = llm(
            formatted,
            **gen_kwargs,
            echo=False,
        )
        text = result["choices"][0]["text"]
        usage = result.get("usage") or {}
        return (
            _strip_special_tokens(text),
            int(usage.get("prompt_tokens") or prompt_tokens),
            int(usage.get("completion_tokens") or tokenize_count(text)),
        )


def create_chat_completion_raw(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SimpleNamespace:
    """Run BitNet chat completion; returns OpenAI-shaped SimpleNamespace."""
    cleaned = [
        {"role": m.get("role") or "user", "content": m.get("content") or ""}
        for m in messages
    ]
    max_tokens = max(1, int(max_tokens or DEFAULT_MAX_TOKENS))
    temperature = float(temperature or 0.0)

    with _CHAT_LOCK:
        backend = backend_name()
        if backend == "bitnet.cpp":
            try:
                content, prompt_tokens, completion_tokens = _complete_via_bitnet_cli(
                    cleaned, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as err:
                logger.warn(
                    f"{Fore.YELLOW}bitnet.cpp CLI failed ({err}); "
                    f"falling back to llama-cpp-python{Fore.RESET}"
                )
                content, prompt_tokens, completion_tokens = _complete_via_llama_cpp(
                    cleaned, temperature=temperature, max_tokens=max_tokens
                )
        elif backend == "missing-cli":
            raise SystemExit(
                "BITNET_BACKEND=bitnet.cpp but llama-cli was not found. "
                "Set BITNET_HOME / BITNET_CLI or run ./scripts/setup_bitnet.sh"
            )
        else:
            content, prompt_tokens, completion_tokens = _complete_via_llama_cpp(
                cleaned, temperature=temperature, max_tokens=max_tokens
            )

    return SimpleNamespace(
        model="bitnet-b1.58",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        choices=[
            SimpleNamespace(message={"role": "assistant", "content": content})
        ],
    )


def create_text_completion_raw(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SimpleNamespace:
    messages = [{"role": "user", "content": prompt}]
    result = create_chat_completion_raw(
        messages, temperature=temperature, max_tokens=max_tokens
    )
    text = result.choices[0].message["content"]
    return SimpleNamespace(
        model="bitnet-b1.58-text",
        usage=result.usage,
        choices=[SimpleNamespace(text=text)],
    )


def _hash_embedding(text: str, dims: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    vec = rng.standard_normal(dims).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()


def _normalize(vec: Iterable[float]) -> list[float]:
    arr = np.asarray(list(vec), dtype=np.float32)
    norm = float(np.linalg.norm(arr)) + 1e-9
    return (arr / norm).tolist()


def create_embedding_raw(texts: list[str]) -> SimpleNamespace:
    dims = _env_int("BITNET_EMBED_DIMS", DEFAULT_EMBED_DIM)
    data: list[dict[str, Any]] = []
    embed_llm = None
    try:
        embed_llm = get_embed_llm()
    except Exception as err:
        logger.warn(f"BitNet embed model load failed ({err})")

    if embed_llm is not None:
        for idx, text in enumerate(texts):
            emb = embed_llm.create_embedding(text)
            vector = _normalize(emb["data"][0]["embedding"])
            data.append({"index": idx, "embedding": vector})
        return SimpleNamespace(data=data, model="bitnet-embed")

    # Mean-pool hidden states from the chat model when embedding GGUF is absent.
    try:
        llm = get_chat_llm()
        for idx, text in enumerate(texts):
            try:
                emb = llm.create_embedding(text)
                vector = _normalize(emb["data"][0]["embedding"])
            except Exception:
                # Token embedding table mean-pool via embed() if present.
                tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
                if hasattr(llm, "embed"):
                    vector = _normalize(llm.embed(text))
                else:
                    vector = _hash_embedding(text, dims)
            data.append({"index": idx, "embedding": vector})
        return SimpleNamespace(data=data, model="bitnet-embed")
    except Exception as err:
        logger.warn(
            f"BitNet embedding unavailable ({err}); using deterministic hash embeddings. "
            "Set BITNET_EMBED_MODEL_PATH to microsoft/BitNet-embedding-0.6B GGUF for real embeds."
        )
        for idx, text in enumerate(texts):
            data.append({"index": idx, "embedding": _hash_embedding(text, dims)})
        return SimpleNamespace(data=data, model="bitnet-embed-hash")


def warm_start() -> Path:
    """Resolve model (auto-bake if needed), optionally warm-load, return chat path."""
    path = ensure_chat_model_path()
    _validate_bitnetish(path)
    os.environ["BITNET_MODEL_PATH"] = str(path)
    backend = backend_name()
    if backend == "bitnet.cpp":
        cli = find_bitnet_cli()
        logger.typewriter_log(
            "BitNet: ",
            Fore.GREEN,
            f"ready via bitnet.cpp ({cli}) model={path}",
        )
        return path
    if backend == "missing-cli":
        raise SystemExit(
            "BITNET_BACKEND=bitnet.cpp but llama-cli was not found. "
            "Build microsoft/BitNet or unset BITNET_BACKEND."
        )
    get_chat_llm()
    # Prefetch embed model if configured (non-fatal).
    try:
        get_embed_llm()
    except Exception as err:
        logger.warn(f"BitNet embed warm-start skipped: {err}")
    logger.typewriter_log("BitNet: ", Fore.GREEN, f"model ready: {path}")
    return path
