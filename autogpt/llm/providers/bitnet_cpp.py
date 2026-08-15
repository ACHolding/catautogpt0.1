"""Ensure microsoft/BitNet (bitnet.cpp) is available for official i2_s GGUFs.

Stock llama.cpp / llama-cpp-python cannot load BitNet ``ggml-model-i2_s.gguf``
(custom tensor types). Inference must use the BitNet fork's ``llama-cli``.

Build trees are placed under ``~/.cache/catautogpt/BitNet`` when the project
path contains ``#`` or ``:`` (Make treats those as special — common on this USB volume).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from colorama import Fore

from autogpt.logs import logger

BITNET_REPO = "https://github.com/microsoft/BitNet.git"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def path_unsafe_for_make(path: Path | str) -> bool:
    """Make interprets '#' as comments; ':' is also unsafe in targets."""
    return any(ch in str(path) for ch in "#:")


def ensure_cached_gguf(gguf: Path) -> Path:
    """Copy GGUF off USB / odd paths into ``~/.cache`` for stable mmap-free loads.

    Paths containing ``#`` or ``:`` (this project's USB volume) are associated with
    bitnet.cpp SIGSEGVs during Metal/mmap init.
    """
    gguf = Path(gguf).expanduser().resolve()
    if not gguf.is_file():
        raise FileNotFoundError(gguf)
    if not path_unsafe_for_make(gguf) and not _env_bool("BITNET_FORCE_CACHE_GGUF", False):
        return gguf

    cache_dir = Path.home() / ".cache" / "catautogpt" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / gguf.name
    if dest.is_file() and dest.stat().st_size == gguf.stat().st_size:
        return dest

    logger.typewriter_log(
        "BitNet: ",
        Fore.YELLOW,
        f"caching GGUF to {dest} (stable path for inference)…",
    )
    tmp = dest.with_suffix(dest.suffix + ".partial")
    shutil.copy2(gguf, tmp)
    tmp.replace(dest)
    return dest


def default_bitnet_home() -> Path:
    home = os.getenv("BITNET_HOME")
    if home and home.strip():
        return Path(home).expanduser()
    root = project_root()
    if path_unsafe_for_make(root):
        return Path.home() / ".cache" / "catautogpt" / "BitNet"
    return root / "third_party" / "BitNet"


def is_i2s_bitnet_gguf(path: Path) -> bool:
    """True when this GGUF needs bitnet.cpp (not stock llama.cpp)."""
    blob = f"{path.name} {path.parent.name}".lower().replace("-", "_")
    raw = f"{path.name} {path.parent.name}".lower()
    markers = (
        "i2_s",
        "i2s",
        "bitnet_b1.58",
        "bitnet_b1_58",
        "b1.58_2b",
        "b1_58_2b",
        "bitnet-b1.58",
    )
    return any(m in blob or m in raw for m in markers)


def find_llama_cli(home: Path | None = None) -> Path | None:
    """Locate built llama-cli / llama-completion under BITNET_HOME."""
    explicit = os.getenv("BITNET_CLI")
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path

    bases: list[Path] = []
    if home:
        bases.append(home)
    bases.append(default_bitnet_home())
    bases.append(Path.home() / ".cache" / "catautogpt" / "BitNet")
    bases.append(project_root() / "third_party" / "BitNet")
    bases.append(project_root() / "BitNet")
    bases.append(Path.home() / "BitNet")

    # Prefer llama-completion for non-interactive -p prompts (newer BitNet builds).
    names = [
        "llama-completion",
        "llama-cli",
        "llama-completion.exe",
        "llama-cli.exe",
    ]
    for base in bases:
        for name in names:
            for candidate in (
                base / "build" / "bin" / name,
                base / "build" / "bin" / "Release" / name,
            ):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
    return None


def _model_dir_for_setup(gguf: Path) -> Path:
    """setup_env.py expects the directory containing the GGUF."""
    return gguf.parent if gguf.is_file() else gguf


def _patch_src1_cont(home: Path) -> None:
    """Upstream BitNet uses ``src1_cont`` outside its declaring #ifdef — fix locally."""
    path = home / "3rdparty" / "llama.cpp" / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "const bool src1_cont = ggml_is_contiguous(src1);\n        const void * src1_wdata"
    if marker in text:
        return
    old = (
        "    if (src0->type == GGML_TYPE_I2_S && ggml_n_dims(src0) == 2) {\n"
        "        const void * src1_wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;\n"
        "        const size_t i2s_row_size = ggml_row_size(vec_dot_type, ne10);\n"
        "        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? i2s_row_size : nb11;"
    )
    new = (
        "    if (src0->type == GGML_TYPE_I2_S && ggml_n_dims(src0) == 2) {\n"
        "        const bool src1_cont = ggml_is_contiguous(src1);\n"
        "        const void * src1_wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;\n"
        "        const size_t i2s_row_size = ggml_row_size(vec_dot_type, ne10);\n"
        "        const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? i2s_row_size : nb11;"
    )
    if old not in text:
        logger.warn(f"Could not patch src1_cont in {path}; build may fail on this BitNet revision.")
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    logger.typewriter_log("BitNet: ", Fore.YELLOW, "patched ggml-cpu.c src1_cont scope bug")


def _patch_bitnet_relu_sqr(home: Path) -> None:
    """Recent BitNet llama.cpp uses SILU for b1.58 FFN; correct activation is ReLU²."""
    path = home / "3rdparty" / "llama.cpp" / "src" / "models" / "bitnet.cpp"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);" in text:
        return
    needle = (
        "        cur = build_ffn(cur,\n"
        "                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,\n"
        "                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,\n"
        "                NULL,                      NULL, NULL,\n"
        "                NULL,\n"
        "                LLM_FFN_SILU, LLM_FFN_PAR, il);"
    )
    if needle not in text:
        return
    path.write_text(
        text.replace(needle, needle.replace("LLM_FFN_SILU", "LLM_FFN_RELU_SQR"), 1),
        encoding="utf-8",
    )
    logger.typewriter_log(
        "BitNet: ", Fore.YELLOW, "patched bitnet.cpp FFN SILU → ReLU²"
    )


def _ninja_path() -> str | None:
    for candidate in ("/opt/homebrew/bin/ninja", shutil.which("ninja")):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _cmake_build_fallback(home: Path, env: dict[str, str]) -> None:
    """Build bitnet.cpp with Ninja + static libs (shared dylib link fails for i2_s)."""
    _patch_src1_cont(home)
    _patch_bitnet_relu_sqr(home)
    build = home / "build"
    if build.exists():
        shutil.rmtree(build)

    ninja = _ninja_path()
    cmake_cmd = [
        "cmake",
        "-B",
        str(build),
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBITNET_ARM_TL1=OFF",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ]
    if ninja:
        cmake_cmd[1:1] = ["-G", "Ninja"]
        cmake_cmd.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
    else:
        make_prog = env.get("CMAKE_MAKE_PROGRAM") or _cmake_make_program()
        if make_prog:
            # Avoid x86_64 /usr/local/bin/gmake on Apple Silicon (EBADARCH -86).
            cmake_cmd.append(f"-DCMAKE_MAKE_PROGRAM={make_prog}")

    subprocess.run(cmake_cmd, cwd=str(home), check=True, env=env)
    jobs = str(max(1, (os.cpu_count() or 4) - 1))
    subprocess.run(
        ["cmake", "--build", str(build), "--config", "Release", "-j", jobs],
        cwd=str(home),
        check=True,
        env=env,
    )


def _run_setup_env(home: Path, model_dir: Path) -> None:
    """Build bitnet.cpp. Prefer our cmake path — setup_env.py often breaks on macOS."""
    logger.typewriter_log(
        "BitNet: ",
        Fore.YELLOW,
        f"building bitnet.cpp under {home} (several minutes)…",
    )
    env = os.environ.copy()
    # Keep PATH free of x86_64 /usr/local/bin/gmake when possible.
    path_parts = [p for p in env.get("PATH", "").split(":") if p != "/usr/local/bin"]
    if "/usr/bin" not in path_parts:
        path_parts.insert(0, "/usr/bin")
    env["PATH"] = ":".join(path_parts)
    make_prog = _cmake_make_program()
    if make_prog:
        env["CMAKE_MAKE_PROGRAM"] = make_prog
        env["MAKE"] = make_prog

    # Official setup_env.py: skip if GGUF already present (avoids CLT python/pip hell).
    # Always use the cmake/Ninja recipe that works on Apple Silicon.
    try:
        _cmake_build_fallback(home, env)
    except Exception as err:
        raise RuntimeError(
            f"bitnet.cpp build failed ({err}). "
            f"Need cmake + clang (+ ninja recommended). Build outside paths with '#'/':':\n"
            f"  export BITNET_HOME=\"$HOME/.cache/catautogpt/BitNet\"\n"
            f"  ./scripts/setup_bitnet.sh\n"
            f"Model dir was: {model_dir}"
        ) from err


def write_env_bitnet_home(home: Path, env_file: Path | None = None) -> None:
    """Persist BITNET_HOME / BITNET_BACKEND into .env."""
    env_path = env_file or (project_root() / ".env")
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    updates = {
        "BITNET_HOME": str(home),
        "BITNET_BACKEND": "bitnet.cpp",
    }
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = (
            line.split("=", 1)[0].strip()
            if "=" in line and not line.strip().startswith("#")
            else ""
        )
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def ensure_bitnet_cli(
    *,
    gguf: Path | None = None,
    auto_build: bool | None = None,
) -> Path:
    """Return path to bitnet.cpp ``llama-cli``, cloning/building if allowed."""
    existing = find_llama_cli()
    if existing is not None:
        home = default_bitnet_home()
        if (home / "build").exists():
            os.environ.setdefault("BITNET_HOME", str(home))
        os.environ.setdefault("BITNET_CLI", str(existing))
        return existing

    if auto_build is None:
        auto_build = _env_bool("BITNET_AUTO_BUILD", True)
    if not auto_build:
        raise SystemExit(
            "Official BitNet i2_s GGUF requires bitnet.cpp (stock llama.cpp cannot load it).\n"
            "Build once:\n"
            "  BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh\n"
            "Or set BITNET_HOME to a built microsoft/BitNet tree, or BITNET_AUTO_BUILD=True.\n"
            "Note: if this repo lives under a path with '#' or ':', set\n"
            "  BITNET_HOME=$HOME/.cache/catautogpt/BitNet"
        )

    if shutil.which("cmake") is None:
        raise SystemExit(
            "cmake is required to build bitnet.cpp. Install cmake, then re-run.\n"
            "  brew install cmake   # macOS\n"
            "Or: BITNET_BUILD_CPP=1 ./scripts/setup_bitnet.sh"
        )

    home = default_bitnet_home()
    if path_unsafe_for_make(home):
        raise SystemExit(
            f"BITNET_HOME={home} contains '#' or ':' which breaks Make/CMake.\n"
            f"Use: export BITNET_HOME=\"$HOME/.cache/catautogpt/BitNet\""
        )

    _clone_bitnet(home)

    if gguf is None:
        from autogpt.llm.providers.bitnet_engine import resolve_chat_model_path

        gguf = resolve_chat_model_path()
    model_dir = _model_dir_for_setup(Path(gguf))
    _run_setup_env(home, model_dir)

    cli = find_llama_cli(home)
    if cli is None:
        raise SystemExit(
            f"bitnet.cpp build finished but llama-cli was not found under {home}/build/bin.\n"
            f"Check the build log, then set BITNET_CLI explicitly."
        )

    os.environ["BITNET_HOME"] = str(home)
    os.environ["BITNET_CLI"] = str(cli)
    os.environ["BITNET_BACKEND"] = "bitnet.cpp"
    try:
        write_env_bitnet_home(home)
    except Exception:
        pass

    logger.typewriter_log(
        "BitNet: ",
        Fore.GREEN,
        f"bitnet.cpp ready: {cli}",
    )
    return cli
