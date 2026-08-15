"""Real BitNet inference engine for Auto-GPT.

BitNet b1.58 (microsoft/BitNet-b1.58-2B-4T) is a native 1.58-bit LLM that uses the
LLaMA 3 tokenizer and chat template. Official ``i2_s`` GGUFs only load in
microsoft/BitNet (bitnet.cpp) — stock llama.cpp / llama-cpp-python fails with
"Failed to load model from file".

This module:

1. Detects official BitNet i2_s GGUFs and routes them through bitnet.cpp
   ``llama-cli`` (auto-clones/builds under ``third_party/BitNet`` when
   ``BITNET_AUTO_BUILD=True``).
2. Uses ``llama-cpp-python`` only for non-i2_s / experimental GGUFs.
3. Optionally loads a BitNet embedding GGUF via ``BITNET_EMBED_MODEL_PATH``.
"""

from __future__ import annotations

import hashlib
import os
import re
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
    from autogpt.llm.providers.bitnet_cpp import find_llama_cli

    cli = find_llama_cli()
    if cli is not None:
        return cli

    which = shutil.which("llama-cli")
    if which and _env_bool("BITNET_USE_SYSTEM_LLAMA_CLI", False):
        return Path(which)
    return None


def requires_bitnet_cpp(path: Path | None = None) -> bool:
    """Official BitNet i2_s GGUFs cannot load in stock llama-cpp-python."""
    if _env_bool("BITNET_FORCE_LLAMA_CPP", False):
        return False
    from autogpt.llm.providers.bitnet_cpp import is_i2s_bitnet_gguf

    try:
        target = path or resolve_chat_model_path()
    except FileNotFoundError:
        return True
    return is_i2s_bitnet_gguf(Path(target))


def backend_name() -> str:
    forced = (os.getenv("BITNET_BACKEND") or "auto").strip().lower()
    if forced in {"bitnet.cpp", "cli", "bitnet"}:
        return "bitnet.cpp" if find_bitnet_cli() else "missing-cli"
    if forced in {"llama-cpp", "llamacpp", "python"}:
        if requires_bitnet_cpp():
            # i2_s always needs bitnet.cpp — ignore forced llama-cpp.
            return "bitnet.cpp" if find_bitnet_cli() else "missing-cli"
        return "llama-cpp-python"
    if requires_bitnet_cpp():
        return "bitnet.cpp" if find_bitnet_cli() else "missing-cli"
    return "bitnet.cpp" if find_bitnet_cli() else "llama-cpp-python"

def apply_llama3_chat_template(
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Format messages with the LLaMA 3 / BitNet instruct chat template.

    Omits ``<|begin_of_text|>`` by default — llama-completion already injects BOS
    from the model metadata (double-BOS degrades / can crash some builds).
    """
    include_bos = _env_bool("BITNET_PROMPT_BOS", False)
    parts: list[str] = ["<|begin_of_text|>"] if include_bos else []
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
    """Lazy-load chat model via llama-cpp-python (non-i2_s GGUFs only)."""
    global _chat_llm, _chat_path, _warned_fallback
    model_path = str(ensure_chat_model_path())
    path = Path(model_path)

    if requires_bitnet_cpp(path):
        raise RuntimeError(
            "Official BitNet i2_s GGUF cannot be loaded with llama-cpp-python. "
            "Use bitnet.cpp (BITNET_AUTO_BUILD / BITNET_HOME). "
            f"model={model_path}"
        )

    if _chat_llm is not None and _chat_path == model_path and not force_reload:
        return _chat_llm

    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise SystemExit(
            "llama-cpp-python is required for non-i2_s GGUF inference. "
            "Install with: pip install llama-cpp-python\n"
            "For official BitNet i2_s, build microsoft/BitNet and set BITNET_HOME."
        ) from err

    _validate_bitnetish(path)
    kwargs = _llama_kwargs(embedding=False)

    if backend_name() != "bitnet.cpp" and not _warned_fallback:
        _warned_fallback = True
        logger.typewriter_log(
            "BitNet: ",
            Fore.YELLOW,
            "using llama-cpp-python (non-i2_s GGUF). Official BitNet kernels need "
            "bitnet.cpp — see ./scripts/setup_bitnet.sh",
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
        try:
            _chat_llm = Llama(model_path=model_path, **kwargs)
        except Exception as second_err:
            raise RuntimeError(
                f"Failed to load GGUF with llama-cpp-python: {second_err}\n"
                "If this is an official BitNet i2_s model, set BITNET_AUTO_BUILD=True "
                "or BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh"
            ) from second_err

    _chat_path = model_path
    return _chat_llm


def get_embed_llm(force_reload: bool = False) -> Any | None:
    """Lazy-load optional BitNet embedding GGUF."""
    global _embed_llm, _embed_path
    path = resolve_embed_model_path()
    if path is None:
        return None
    if requires_bitnet_cpp(path):
        # Embedding i2_s also needs bitnet.cpp; skip in-process load.
        logger.warn(
            "BitNet embed GGUF looks like i2_s — skipping llama-cpp load; "
            "using hash embeddings unless a llama.cpp-compatible embed GGUF is set."
        )
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
    try:
        _embed_llm = Llama(model_path=model_path, **kwargs)
    except Exception as err:
        logger.warn(f"BitNet embed model load failed ({err})")
        return None
    _embed_path = model_path
    return _embed_llm


def _estimate_tokens(text: str) -> int:
    # LLaMA-ish heuristic when bitnet.cpp CLI has no in-process tokenizer.
    return max(1, (len(text) + 3) // 4)


def tokenize_count(text: str) -> int:
    """Count tokens with the loaded BitNet tokenizer.

    Falls back to a char heuristic when only bitnet.cpp CLI is available
    (official i2_s path never loads llama-cpp-python).
    """
    if _chat_llm is None:
        return _estimate_tokens(text)
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
    import tempfile

    from autogpt.llm.providers.bitnet_cpp import (
        ensure_cached_gguf,
        path_unsafe_for_make,
    )

    cli = find_bitnet_cli()
    if cli is None:
        raise FileNotFoundError("bitnet.cpp llama-cli/llama-completion not found")

    # Newer BitNet builds split interactive chat (llama-cli) from one-shot
    # completion (llama-completion). Prefer the latter for agent loops.
    if cli.name.startswith("llama-cli"):
        sibling = cli.with_name(cli.name.replace("llama-cli", "llama-completion", 1))
        if sibling.is_file() and os.access(sibling, os.X_OK):
            cli = sibling

    model = resolve_chat_model_path()
    # USB paths with '#'/':' + mmap are a common segfault source on macOS.
    if path_unsafe_for_make(model) or _env_bool("BITNET_CACHE_GGUF", True):
        try:
            model = ensure_cached_gguf(model)
        except Exception as err:
            logger.warn(f"BitNet GGUF cache copy skipped ({err})")

    prompt = apply_llama3_chat_template(messages, add_generation_prompt=True)
    # Conservative defaults: Apple Silicon + Metal/BLAS + large ubatch often SIGSEGV
    # (exit -11) on i2_s. Prefer CPU + small batches.
    n_ctx = _env_int("BITNET_N_CTX", 2048)
    n_threads = min(_env_int("BITNET_N_THREADS", min(8, _default_threads())), 8)
    n_batch = _env_int("BITNET_N_BATCH", 1)
    n_ubatch = _env_int("BITNET_N_UBATCH", 1)

    def _run(extra: list[str], *, use_jinja: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(prompt)
            prompt_file = fh.name
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
            "-b",
            str(n_batch),
            "-ub",
            str(n_ubatch),
            "--temp",
            str(max(0.0, temperature)),
            "-ngl",
            "0",
            "--device",
            "none",
            "--fit",
            "off",
            "--no-mmap",
            "-no-cnv",
            "--override-kv",
            "tokenizer.ggml.pre=str:llama-bpe",
            "-f",
            prompt_file,
            "--no-display-prompt",
            *extra,
        ]
        if use_jinja:
            cmd.append("--jinja")
        env = os.environ.copy()
        env["GGML_METAL"] = "0"
        env["LLAMA_ARG_N_GPU_LAYERS"] = "0"
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=_env_int("BITNET_CLI_TIMEOUT", 600),
                env=env,
            )
        finally:
            try:
                Path(prompt_file).unlink(missing_ok=True)
            except Exception:
                pass

    logger.debug(f"BitNet CLI: {cli.name} model={model} thr={n_threads} ctx={n_ctx}")
    # Default: no jinja (custom LLaMA-3 prompt already applied).
    use_jinja = _env_bool("BITNET_CLI_JINJA", False)
    proc = _run([], use_jinja=use_jinja)

    # Exit -11 = SIGSEGV. Retry once with even safer settings.
    if proc.returncode in (-11, 139, 245) or (
        proc.returncode != 0 and "Segmentation" in ((proc.stderr or "") + (proc.stdout or ""))
    ):
        logger.warn(
            f"{Fore.YELLOW}bitnet.cpp crashed (code {proc.returncode}); "
            f"retrying CPU/gemv-safe settings{Fore.RESET}"
        )
        n_ctx = min(n_ctx, 1024)
        n_threads = min(n_threads, 4)
        n_batch = 1
        n_ubatch = 1
        proc = _run([], use_jinja=False)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"bitnet.cpp failed (code {proc.returncode}): {err[:800]}")

    # llama-completion prints sampler logs to stdout; keep generation lines only.
    raw = (proc.stdout or "").strip()
    content_lines = [
        ln
        for ln in raw.splitlines()
        if ln.strip()
        and not re.match(r"^\d+\.\d+", ln.strip())
        and not ln.strip().startswith("common_perf_print")
        and "sampler " not in ln
        and "system_info:" not in ln
        and "generate:" not in ln
        and not ln.lstrip().startswith("repeat_")
        and not ln.lstrip().startswith("dry_")
        and not ln.lstrip().startswith("top_")
        and not ln.lstrip().startswith("mirostat")
        and "llama_completion:" not in ln
        and "print_info:" not in ln
    ]
    content = _strip_special_tokens("\n".join(content_lines) if content_lines else raw)
    # Drop pure-punctuation garbage runs from broken tokenizer/kernels.
    if content and set(content) <= set("@Gg \n"):
        logger.warn(
            "BitNet returned placeholder garbage (@/G). Known issue with some "
            "bitnet.cpp builds / i2_s kernels; try rebuilding BitNet or smaller -b 1."
        )
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
    path = ensure_chat_model_path()
    must_use_cpp = requires_bitnet_cpp(path)

    with _CHAT_LOCK:
        backend = backend_name()
        if backend == "missing-cli" or (must_use_cpp and find_bitnet_cli() is None):
            from autogpt.llm.providers.bitnet_cpp import ensure_bitnet_cli

            ensure_bitnet_cli(gguf=path)
            backend = backend_name()

        if backend == "bitnet.cpp" or must_use_cpp:
            content, prompt_tokens, completion_tokens = _complete_via_bitnet_cli(
                cleaned, temperature=temperature, max_tokens=max_tokens
            )
        elif backend == "missing-cli":
            raise SystemExit(
                "BITNET_BACKEND=bitnet.cpp but llama-cli was not found. "
                "Set BITNET_HOME / BITNET_CLI or run BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh"
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

    # Mean-pool from in-process chat model only when llama-cpp can load it.
    if not requires_bitnet_cpp():
        try:
            llm = get_chat_llm()
            for idx, text in enumerate(texts):
                try:
                    emb = llm.create_embedding(text)
                    vector = _normalize(emb["data"][0]["embedding"])
                except Exception:
                    if hasattr(llm, "embed"):
                        vector = _normalize(llm.embed(text))
                    else:
                        vector = _hash_embedding(text, dims)
                data.append({"index": idx, "embedding": vector})
            return SimpleNamespace(data=data, model="bitnet-embed")
        except Exception as err:
            logger.warn(f"BitNet chat-model embedding unavailable ({err})")

    logger.warn(
        "BitNet embedding unavailable for i2_s / missing embed GGUF; "
        "using deterministic hash embeddings. "
        "Set BITNET_EMBED_MODEL_PATH to a llama.cpp-compatible embed GGUF for real embeds."
    )
    for idx, text in enumerate(texts):
        data.append({"index": idx, "embedding": _hash_embedding(text, dims)})
    return SimpleNamespace(data=data, model="bitnet-embed-hash")


def warm_start() -> Path:
    """Resolve model (auto-bake if needed), ensure bitnet.cpp for i2_s, return path."""
    path = ensure_chat_model_path()
    _validate_bitnetish(path)
    os.environ["BITNET_MODEL_PATH"] = str(path)

    if requires_bitnet_cpp(path):
        from autogpt.llm.providers.bitnet_cpp import ensure_bitnet_cli

        cli = ensure_bitnet_cli(gguf=path)
        logger.typewriter_log(
            "BitNet: ",
            Fore.GREEN,
            f"ready via bitnet.cpp ({cli}) model={path}",
        )
        return path

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
