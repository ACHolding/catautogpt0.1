import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

if "pytest" in sys.argv or "pytest" in sys.modules or os.getenv("CI"):
    print("Setting random seed to 42")
    random.seed(42)

# Always load .env from the project root (parent of the autogpt package),
# not from whatever directory the user happened to launch from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_FILE, verbose=False, override=True)
# Also allow a cwd .env to override for local experiments.
load_dotenv(verbose=False, override=True)

del load_dotenv
