"""Pre-bake / auto-fetch the official BitNet b1.58 GGUF for local inference."""

from __future__ import annotations

import os
from pathlib import Path

from colorama import Fore

from autogpt.logs import logger

DEFAULT_HF_REPO = "microsoft/BitNet-b1.58-2B-4T-gguf"
DEFAULT_GGUF_NAME = "ggml-model-i2_s.gguf"
# Alternate filenames some mirrors / older uploads use.
ALTERNATE_GGUF_NAMES = (
    "ggml-model-i2_s.gguf",
    "BitNet-b1.58-2B-4T.i2_s.gguf",
    "bitnet-b1.58-2B-4T-gguf.gguf",
)


def default_model_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[3]
    return root / "models" / "BitNet-b1.58-2B-4T"


def find_local_gguf(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    for name in ALTERNATE_GGUF_NAMES:
        hit = model_dir / name
        if hit.is_file() and hit.stat().st_size > 1_000_000:
            return hit
    ggufs = sorted(model_dir.glob("**/*.gguf"))
    for g in ggufs:
        if g.stat().st_size > 1_000_000:
            return g
    return None


def ensure_bitnet_gguf(
    *,
    project_root: Path | None = None,
    repo_id: str | None = None,
    auto_download: bool | None = None,
) -> Path:
    """Return path to a local BitNet GGUF, downloading the official bake if needed.

    Controlled by:
      BITNET_AUTO_DOWNLOAD=True|False (default True)
      BITNET_HF_REPO (default microsoft/BitNet-b1.58-2B-4T-gguf)
      BITNET_MODEL_DIR (optional override for download destination)
    """
    root = project_root or Path(__file__).resolve().parents[3]
    model_dir = Path(
        os.getenv("BITNET_MODEL_DIR") or default_model_dir(root)
    ).expanduser()
    existing = find_local_gguf(model_dir)
    if existing:
        return existing

    if auto_download is None:
        auto_download = os.getenv("BITNET_AUTO_DOWNLOAD", "True").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if not auto_download:
        raise FileNotFoundError(
            f"No BitNet GGUF under {model_dir} and BITNET_AUTO_DOWNLOAD is disabled."
        )

    repo = repo_id or os.getenv("BITNET_HF_REPO") or DEFAULT_HF_REPO
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as err:
        raise SystemExit(
            "huggingface_hub is required to auto-download BitNet. "
            "Install with: pip install huggingface_hub\n"
            "Or run: ./scripts/setup_bitnet.sh"
        ) from err

    logger.typewriter_log(
        "BitNet: ",
        Fore.CYAN,
        f"pre-baking model from {repo} → {model_dir} (first run only)…",
    )

    # Prefer known i2_s name; otherwise pick the first .gguf in the repo.
    try:
        remote_files = list_repo_files(repo)
    except Exception:
        remote_files = []

    filename = DEFAULT_GGUF_NAME
    for candidate in ALTERNATE_GGUF_NAMES:
        if candidate in remote_files:
            filename = candidate
            break
    else:
        remote_ggufs = [f for f in remote_files if f.endswith(".gguf")]
        if remote_ggufs:
            # Prefer i2_s if any
            i2 = [f for f in remote_ggufs if "i2_s" in f.lower()]
            filename = i2[0] if i2 else remote_ggufs[0]

    try:
        local_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(model_dir),
        )
    except Exception as err:
        raise FileNotFoundError(
            f"Failed to download BitNet GGUF from {repo}: {err}\n"
            "Check network access, or place a .gguf under "
            f"{model_dir} and set BITNET_MODEL_PATH."
        ) from err

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(f"Download finished but file missing: {path}")

    logger.typewriter_log("BitNet: ", Fore.GREEN, f"baked model ready at {path}")
    return path


def write_env_model_path(gguf: Path, env_file: Path) -> None:
    """Upsert BITNET_MODEL_PATH in a .env file."""
    line = f"BITNET_MODEL_PATH={gguf}"
    if env_file.exists():
        text = env_file.read_text()
        lines = []
        found = False
        for raw in text.splitlines():
            if raw.startswith("BITNET_MODEL_PATH="):
                lines.append(line)
                found = True
            else:
                lines.append(raw)
        if not found:
            lines.append(line)
        env_file.write_text("\n".join(lines) + "\n")
    else:
        template = env_file.parent / ".env.template"
        if template.exists():
            env_file.write_text(template.read_text())
            write_env_model_path(gguf, env_file)
        else:
            env_file.write_text(line + "\n")
