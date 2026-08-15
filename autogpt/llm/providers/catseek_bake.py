"""Download DeepSeek-R1-Distill-Qwen-14B GGUF for CatSeek-GPU 0.1."""

from __future__ import annotations

import os
from pathlib import Path

from colorama import Fore

from autogpt.logs import logger

DEFAULT_HF_REPO = "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF"
DEFAULT_HF_FILE = "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "CatSeek-GPU-0.1-14B"


def write_env_model_path(gguf: Path, env_file: Path) -> None:
    line = f"CATSEEK_MODEL_PATH={gguf}"
    bitnet_line = f"BITNET_MODEL_PATH={gguf}"
    if not env_file.exists():
        env_file.write_text(line + "\n" + bitnet_line + "\n", encoding="utf-8")
        return
    lines = env_file.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen_c = seen_b = False
    for raw in lines:
        if raw.startswith("CATSEEK_MODEL_PATH="):
            out.append(line)
            seen_c = True
        elif raw.startswith("BITNET_MODEL_PATH="):
            out.append(bitnet_line)
            seen_b = True
        else:
            out.append(raw)
    if not seen_c:
        out.append(line)
    if not seen_b:
        out.append(bitnet_line)
    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")


def ensure_catseek_gguf(
    *,
    project_root_path: Path | None = None,
    auto_download: bool | None = None,
    repo_id: str | None = None,
    filename: str | None = None,
) -> Path:
    root = project_root_path or project_root()
    model_dir = Path(os.getenv("CATSEEK_MODEL_DIR") or default_model_dir(root))
    model_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or os.getenv("CATSEEK_HF_FILE") or DEFAULT_HF_FILE
    existing = model_dir / fname
    if existing.is_file() and existing.stat().st_size > 1_000_000:
        return existing
    # Any gguf already present.
    hits = sorted(model_dir.glob("*.gguf"))
    if hits:
        return hits[0]

    if auto_download is None:
        auto_download = os.getenv("CATSEEK_AUTO_DOWNLOAD", "True").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not auto_download:
        raise FileNotFoundError(
            f"No CatSeek GGUF under {model_dir} and CATSEEK_AUTO_DOWNLOAD is disabled."
        )

    repo = repo_id or os.getenv("CATSEEK_HF_REPO") or DEFAULT_HF_REPO
    logger.typewriter_log(
        "CatSeek-GPU: ",
        Fore.YELLOW,
        f"downloading {repo}/{fname} → {model_dir} (~8–9GB, one-time)…",
    )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise SystemExit(
            "huggingface_hub required to bake CatSeek GGUF. "
            "pip install huggingface_hub"
        ) from err

    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=fname,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    )
    if not path.is_file():
        raise FileNotFoundError(f"Download finished but {path} missing")

    env = root / ".env"
    try:
        if not env.exists() and (root / ".env.template").exists():
            env.write_text((root / ".env.template").read_text(encoding="utf-8"))
        write_env_model_path(path, env)
    except Exception:
        pass
    logger.typewriter_log("CatSeek-GPU: ", Fore.GREEN, f"baked {path}")
    return path
