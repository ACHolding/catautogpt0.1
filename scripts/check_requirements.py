#!/usr/bin/env python3
"""Check that packages listed in requirements.txt are installed."""

from __future__ import annotations

import re
import sys
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _requirement_name(req: str) -> str | None:
    """Extract the distribution name from a requirement line."""
    if req.startswith(("-e ", "--", "git+", "http://", "https://", "file:")):
        return None
    if "@" in req:
        # e.g. "en-core-web-sm @ https://..."
        req = req.split("@", 1)[0].strip()
    req = re.split(r"[\[<>=!~]", req, maxsplit=1)[0].strip()
    return req or None


def _is_local_package(name: str) -> bool:
    """True if the project ships this package as a local directory."""
    candidate = ROOT / name.replace("-", "_")
    alt = ROOT / name.replace("_", "-")
    return (candidate.is_dir() and (candidate / "__init__.py").exists()) or (
        alt.is_dir() and (alt / "__init__.py").exists()
    )


def _is_installed(req_line: str) -> bool:
    name = _requirement_name(req_line)
    if name is None:
        return True
    if _is_local_package(name):
        return True
    try:
        metadata.version(name)
        return True
    except PackageNotFoundError:
        alt = name.replace("-", "_") if "-" in name else name.replace("_", "-")
        try:
            metadata.version(alt)
            return True
        except PackageNotFoundError:
            return False


def main() -> None:
    requirements_file = sys.argv[1]
    # By default only enforce runtime deps (same cut as the Docker image).
    check_all = "--all" in sys.argv[2:]
    with open(requirements_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    required_packages = []
    for line in lines:
        if (
            not check_all
            and "Items below this point will not be included in the Docker Image"
            in line
        ):
            break
        required_packages.append(line.strip().split("#")[0].strip())

    missing_packages = []
    for required_package in required_packages:
        if not required_package:
            continue
        if not _is_installed(required_package):
            missing_packages.append(required_package)

    if missing_packages:
        print("Missing packages:")
        print(", ".join(missing_packages))
        sys.exit(1)

    print("All packages are installed.")


if __name__ == "__main__":
    main()
