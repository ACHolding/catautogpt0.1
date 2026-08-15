#!/usr/bin/env bash
# Keep Auto-GPT running in continuous mode.
# Restarts automatically if the process exits (crash, limit, etc.).
# Stop with Ctrl+C.
#
# Exit code 2 = configuration error (e.g. missing CATSEEK_MODEL_PATH):
# the supervisor will NOT restart in that case.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

find_python_command() {
    if command -v python >/dev/null 2>&1; then
        echo "python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        echo "Python not found. Please install Python." >&2
        exit 1
    fi
}

PYTHON_CMD="$(find_python_command)"

if ! "$PYTHON_CMD" -c "import sys; sys.exit(0 if sys.version_info >= (3, 14) else 1)"; then
    echo "Python 3.14 or higher is required to run Auto GPT." >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    echo "Missing $SCRIPT_DIR/.env"
    echo "Create one from the template and bake CatSeek-GPU 0.1:"
    echo "  cp .env.template .env"
    echo "  ./scripts/setup_catseek.sh"
    exit 2
fi

# shellcheck disable=SC1091
set -a
# Load .env for the supervisor pre-check (python also loads it).
# shellcheck source=/dev/null
source "$SCRIPT_DIR/.env" 2>/dev/null || true
set +a

if [[ -z "${CATSEEK_MODEL_PATH:-}${BITNET_MODEL_PATH:-}${LLM_MODEL_PATH:-}" ]]; then
    # Allow auto-discovery of models/*.gguf inside the project.
    if ! compgen -G "$SCRIPT_DIR/models/**/*.gguf" >/dev/null \
        && ! compgen -G "$SCRIPT_DIR/models/*.gguf" >/dev/null; then
        echo "No CatSeek GGUF model configured."
        echo "Bake DeepSeek-R1-Distill-Qwen-14B with:"
        echo "  ./scripts/setup_catseek.sh"
        echo "  # or set CATSEEK_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
        exit 2
    fi
fi

# Lightweight deps check (same as run.sh, but non-fatal).
if ! "$PYTHON_CMD" scripts/check_requirements.py requirements.txt >/dev/null 2>&1; then
    echo "Installing missing packages..."
    "$PYTHON_CMD" -m pip install -r requirements.txt || {
        echo "Warning: some packages failed to install; continuing anyway."
    }
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Restart delay after an unexpected exit (seconds).
RESTART_DELAY="${CATAUTOGPT_RESTART_DELAY:-5}"

# Extra continuous flags:
#   -c / --continuous  : no per-command user authorization loop
#   -y / --skip-reprompt: skip startup AI settings prompts when possible
#   --skip-news        : don't block on bulletin
CONTINUOUS_ARGS=(-c -y --skip-news)

# One-shot for help; don't enter the supervisor loop.
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            exec "$PYTHON_CMD" -m autogpt --help
            ;;
    esac
done

echo "Starting Auto-GPT continuous supervisor in ${SCRIPT_DIR}"
echo "LLM: CatSeek-GPU 0.1 (DeepSeek-R1-Distill-Qwen-14B)"
echo "Workspace: ${SCRIPT_DIR}/auto_gpt_workspace"
echo "Press Ctrl+C to stop."
echo

trap 'echo; echo "Stopping continuous supervisor."; exit 0' INT TERM

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching: $PYTHON_CMD -m autogpt ${CONTINUOUS_ARGS[*]} $*"
    set +e
    "$PYTHON_CMD" -m autogpt "${CONTINUOUS_ARGS[@]}" "$@"
    exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-GPT exited cleanly (code 0)."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-GPT exited with code ${exit_code}."
    fi

    # 2 = configuration error (missing CatSeek model, etc.) — do not restart-loop.
    if [[ $exit_code -eq 2 ]]; then
        echo "Configuration error. Fix CatSeek runtime (CATSEEK_MODEL_PATH / llama-cpp-python), then re-run."
        echo "  ./scripts/setup_catseek.sh"
        exit 2
    fi

    echo "Restarting in ${RESTART_DELAY}s... (Ctrl+C to quit)"
    sleep "$RESTART_DELAY"
done
