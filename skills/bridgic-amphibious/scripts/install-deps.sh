#!/bin/bash
# install-deps.sh — Install bridgic-amphibious skill dependencies.
#
# 1. Checks uv availability.
# 2. Ensures a uv project is initialized (pyproject.toml exists).
# 3. Checks each required package and installs missing ones via uv add.
#
# Usage:
#   install-deps.sh [PROJECT_DIR]   (defaults to current directory)
#
# Exit codes:
#   0  All dependencies installed
#   1  uv not installed
#   2  uv init failed
#   3  uv add failed

set -euo pipefail

PROJECT_DIR="${1:-.}"
cd "$PROJECT_DIR"

PRIVATE_INDEX="http://8.130.156.165:3141/btsk/test/+simple"

# Required packages for bridgic-amphibious skill
PACKAGES=(
    "bridgic-core"
    "bridgic-amphibious"
    "bridgic-llms-openai"
    "python-dotenv"
)

# Packages that need the private index
PRIVATE_PACKAGES=(
    "bridgic-amphibious"
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
    # Match the package name in dependencies (handles both quoted and unquoted forms)
    grep -qiE "^[[:space:]]*\"?${pkg}\"?[[:space:]]*[>=<~!,\"]" pyproject.toml 2>/dev/null
}

# Helper: check if a package needs the private index
needs_private_index() {
    local pkg="$1"
    for pp in "${PRIVATE_PACKAGES[@]}"; do
        if [ "$pkg" = "$pp" ]; then
            return 0
        fi
    done
    return 1
}

MISSING=()
MISSING_PRIVATE=()

for pkg in "${PACKAGES[@]}"; do
    if is_installed "$pkg"; then
        echo "✓ $pkg already installed"
    else
        if needs_private_index "$pkg"; then
            MISSING_PRIVATE+=("$pkg")
        else
            MISSING+=("$pkg")
        fi
        echo "✗ $pkg not found — will install"
    fi
done

# Install standard packages
if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "Installing: ${MISSING[*]} ..."
    uv add "${MISSING[@]}" || { echo "Error: uv add failed for: ${MISSING[*]}"; exit 3; }
fi

# Install private-index packages
if [ ${#MISSING_PRIVATE[@]} -gt 0 ]; then
    echo ""
    echo "Installing (private index): ${MISSING_PRIVATE[*]} ..."
    uv add "${MISSING_PRIVATE[@]}" --extra-index-url "$PRIVATE_INDEX" || {
        echo "Error: uv add failed for: ${MISSING_PRIVATE[*]}"
        exit 3
    }
fi

echo ""
echo "=== DEPS_READY (bridgic-amphibious) ==="
