#!/usr/bin/env bash

# Always run from the directory containing this script (project root),
# so invoking via an absolute path from ~ still works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

function find_python_command() {
    if command -v python &> /dev/null
    then
        echo "python"
    elif command -v python3 &> /dev/null
    then
        echo "python3"
    else
        echo "Python not found. Please install Python."
        exit 1
    fi
}

PYTHON_CMD=$(find_python_command)

if $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    CHECK_OUT=$($PYTHON_CMD scripts/check_requirements.py requirements.txt 2>&1) || {
        echo "$CHECK_OUT"
        # Install only the packages reported missing (second line of checker output).
        MISSING=$(printf '%s\n' "$CHECK_OUT" | sed -n '2p')
        if [ -n "$MISSING" ]; then
            echo "Installing missing packages..."
            OLD_IFS=$IFS
            IFS=','
            # shellcheck disable=SC2086
            set -- $MISSING
            IFS=$OLD_IFS
            PKGS=()
            for pkg in "$@"; do
                # trim whitespace without xargs
                pkg="${pkg#"${pkg%%[![:space:]]*}"}"
                pkg="${pkg%"${pkg##*[![:space:]]}"}"
                [ -n "$pkg" ] && PKGS+=("$pkg")
            done
            if [ ${#PKGS[@]} -gt 0 ]; then
                $PYTHON_CMD -m pip install "${PKGS[@]}" || {
                    echo "Warning: some packages failed to install; continuing anyway."
                }
            fi
        fi
    }
    # Ensure the project root is importable when not installed as a package.
    export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
    $PYTHON_CMD -m autogpt "$@"
    # Only pause when running interactively in a terminal.
    if [ -t 0 ] && [ -t 1 ]; then
        read -p "Press any key to continue..."
    fi
else
    echo "Python 3.10 or higher is required to run Auto GPT."
    exit 1
fi
