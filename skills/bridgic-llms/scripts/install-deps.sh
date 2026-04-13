#!/bin/bash
# install-deps.sh — Install bridgic-llms skill dependencies.
#
# 1. Checks uv availability.
# 2. Ensures a uv project is initialized (pyproject.toml exists).
# 3. Checks each required package and installs missing ones via uv add.
#
# By default installs bridgic-llms-openai (the most common provider).
# Pass a provider name to install a different one:
#   install-deps.sh [PROJECT_DIR] [PROVIDER]
#
# Supported providers: openai (default), openai-like, vllm
#
# Exit codes:
#   0  All dependencies installed
#   1  uv not installed
#   2  uv init failed
#   3  uv add failed

set -euo pipefail

PROJECT_DIR="${1:-.}"
PROVIDER="${2:-openai}"
cd "$PROJECT_DIR"

# Map provider name to package
case "$PROVIDER" in
    openai)
        LLM_PACKAGE="bridgic-llms-openai"
        ;;
    openai-like)
        LLM_PACKAGE="bridgic-llms-openai-like"
        ;;
    vllm)
        LLM_PACKAGE="bridgic-llms-vllm"
        ;;
    *)
        echo "ERROR: Unknown provider '$PROVIDER'. Supported: openai, openai-like, vllm"
        exit 1
        ;;
esac

# Required packages
PACKAGES=(
    "$LLM_PACKAGE"
    "python-dotenv"
)

# ──────────────────────────────────────────────
# 1. Check uv
# ──────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing ..."
    case "$(uname -s)" in
        CYGWIN*|MINGW*|MSYS*|Windows_NT*)
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" \
                || { echo "Error: uv installation failed on Windows."; exit 1; }
            ;;
        *)
            curl -LsSf https://astral.sh/uv/install.sh | sh \
                || { echo "Error: uv installation failed."; exit 1; }
            ;;
    esac
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "Error: uv was installed but not found on PATH."
        exit 1
    fi
    echo "uv installed successfully."
fi

echo "uv: $(uv --version 2>&1)"

# ──────────────────────────────────────────────
# 2. Initialize uv project if needed
# ──────────────────────────────────────────────
if [ ! -f pyproject.toml ]; then
    echo "No pyproject.toml found — running uv init --bare ..."
    uv init --bare || { echo "Error: uv init failed."; exit 2; }
    echo "Created pyproject.toml"
else
    echo "pyproject.toml already exists, skipping init"
fi

# ──────────────────────────────────────────────
# 3. Check and install missing packages
# ──────────────────────────────────────────────

# Helper: check if a package is already in pyproject.toml dependencies
is_installed() {
    local pkg="$1"
    grep -qiE "^[[:space:]]*\"?${pkg}\"?[[:space:]]*[>=<~!,\"]" pyproject.toml 2>/dev/null
}

MISSING=()

for pkg in "${PACKAGES[@]}"; do
    if is_installed "$pkg"; then
        echo "✓ $pkg already installed"
    else
        MISSING+=("$pkg")
        echo "✗ $pkg not found — will install"
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "Installing: ${MISSING[*]} ..."
    uv add "${MISSING[@]}" || { echo "Error: uv add failed for: ${MISSING[*]}"; exit 3; }
fi

echo ""
echo "=== DEPS_READY (bridgic-llms, provider: $PROVIDER) ==="
