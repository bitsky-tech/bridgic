#!/bin/bash
# install-deps.sh — Install bridgic-llms skill dependencies.
#
# 1. Checks uv availability.
# 2. Ensures a uv project is initialized (pyproject.toml exists).
# 3. Optionally reads a developer-local deps manifest (BRIDGIC_DEPS_MANIFEST)
#    to route specific packages through a private dev index. When unset, all
#    dependencies resolve from public PyPI (production default).
# 4. Installs missing packages via uv add.
# 5. Runs uv sync to finalize the project environment so the caller's venv
#    matches pyproject.toml on exit (no manual sync step required).
#
# By default installs bridgic-llms-openai (the most common provider).
# Pass a provider name to install a different one:
#   install-deps.sh [PROJECT_DIR] [PROVIDER]
#
# Supported providers: openai (default), openai-like, vllm
#
# Environment:
#   BRIDGIC_DEPS_MANIFEST   Optional. Path to a developer-local manifest file
#                           describing per-package source routing. Lives
#                           outside this repo so internal index URLs are never
#                           committed. Relative paths are resolved against the
#                           caller's working directory.
#
#                           Manifest format (one directive per line):
#
#                             # comments allowed (lines starting with #)
#                             index https://your-dev-index.example.com/simple/
#                             bridgic-core             dev
#                             bridgic-llms-openai      dev
#                             bridgic-llms-openai-like dev
#                             python-dotenv            default
#
#                           Directives:
#                             - `index <url>`     declares the dev index URL
#                                                 (required if any package is
#                                                 marked `dev`)
#                             - `<pkg> dev`       inject pkg into
#                                                 [tool.uv.sources] and route
#                                                 via the dev index
#                             - `<pkg> default`   resolve from public PyPI
#                                                 (also the implicit default
#                                                 for any package not listed)
#
#                           Note: transitive bridgic-* deps must also be
#                           declared in the manifest if you want them routed
#                           through the dev index — otherwise they resolve
#                           from public PyPI and may drift from the dev
#                           release of their parent package.
#
#                           When BRIDGIC_DEPS_MANIFEST is unset, the manifest
#                           step is skipped entirely and every package
#                           resolves from public PyPI.
#
# Exit codes:
#   0  All dependencies installed and synced
#   1  uv not installed (or unknown provider)
#   2  uv init failed
#   3  uv add failed
#   4  uv sync failed
#   5  manifest file not found or malformed
#
# Output markers:
#   On success: "=== DEPS_READY (...) ==="
#   On failure: "=== DEPS_FAILED reason=<label> exit=<N> ==="

set -euo pipefail

PROJECT_DIR="${1:-.}"
PROVIDER="${2:-openai}"
MANIFEST_FILE="${BRIDGIC_DEPS_MANIFEST:-}"

# Resolve manifest path against caller's PWD before we cd into PROJECT_DIR,
# so users can pass a relative path naturally.
if [ -n "$MANIFEST_FILE" ] && [ "${MANIFEST_FILE:0:1}" != "/" ]; then
    MANIFEST_FILE="$PWD/$MANIFEST_FILE"
fi

cd "$PROJECT_DIR"

DEV_INDEX_NAME="bridgic-repo"
INJECTION_BEGIN_MARKER="# BEGIN bridgic-deps-injection"
INJECTION_END_MARKER="# END bridgic-deps-injection"

# Map provider name to its main package. Transitive bridgic-* deps are now
# resolved automatically by uv (and may be routed via the manifest if the
# user wants them coherent with the parent package).
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
        echo "ERROR: Unknown provider '$PROVIDER'. Supported: openai, openai-like, vllm" >&2
        echo ""
        echo "=== DEPS_FAILED reason=unknown_provider exit=1 ==="
        exit 1
        ;;
esac

# Required packages (only the main LLM package is added explicitly;
# bridgic transitive deps are resolved automatically by uv)
PACKAGES=(
    "$LLM_PACKAGE"
    "python-dotenv"
)

# Shared log file capturing stdout+stderr of each uv invocation. The trap
# guarantees cleanup even on early exit.
LOG_FILE="$(mktemp -t bridgic-deps.XXXXXX)"
trap 'rm -f "$LOG_FILE"' EXIT

# ──────────────────────────────────────────────
# Failure helper — emits structured marker and exits.
# ──────────────────────────────────────────────
fail() {
    local reason="$1"
    local code="$2"
    echo ""
    echo "=== DEPS_FAILED reason=${reason} exit=${code} ==="
    exit "$code"
}

# ──────────────────────────────────────────────
# uv runner — captures full output, prints it, and on failure leaves the
# captured output visible to the caller before emitting the failure marker.
# Usage: run_uv <fail_label> <fail_exit_code> <cmd> [args...]
# ──────────────────────────────────────────────
run_uv() {
    local label="$1"
    local exit_code="$2"
    shift 2
    if ! "$@" > "$LOG_FILE" 2>&1; then
        cat "$LOG_FILE"
        fail "$label" "$exit_code"
    fi
    cat "$LOG_FILE"
}

# ──────────────────────────────────────────────
# 0. Parse manifest (if BRIDGIC_DEPS_MANIFEST is set)
# ──────────────────────────────────────────────
DEV_INDEX_URL=""
DEV_PACKAGES=()

if [ -n "$MANIFEST_FILE" ]; then
    if [ ! -f "$MANIFEST_FILE" ]; then
        echo "Error: BRIDGIC_DEPS_MANIFEST is set but file not found: $MANIFEST_FILE" >&2
        fail "manifest_not_found" 5
    fi
    echo "Reading deps manifest: $MANIFEST_FILE"

    manifest_lineno=0
    while IFS= read -r raw_line || [ -n "${raw_line:-}" ]; do
        manifest_lineno=$((manifest_lineno + 1))
        # Strip inline comments (# and anything after).
        line="${raw_line%%#*}"
        # Tokenize: read trims whitespace and handles tabs/spaces uniformly.
        f1=""
        f2=""
        read -r f1 f2 _ <<< "$line" || true
        # Skip blank or comment-only lines.
        [ -z "$f1" ] && continue

        if [ "$f1" = "index" ]; then
            if [ -z "$f2" ]; then
                echo "Error: manifest line $manifest_lineno: 'index' directive missing URL" >&2
                fail "manifest_malformed" 5
            fi
            DEV_INDEX_URL="$f2"
        else
            case "$f2" in
                dev)
                    DEV_PACKAGES+=("$f1")
                    ;;
                default|"")
                    : # no-op; default routing (public PyPI)
                    ;;
                *)
                    echo "Error: manifest line $manifest_lineno: unknown source '$f2' for package '$f1' (expected: dev or default)" >&2
                    fail "manifest_malformed" 5
                    ;;
            esac
        fi
    done < "$MANIFEST_FILE"

    if [ ${#DEV_PACKAGES[@]} -gt 0 ] && [ -z "$DEV_INDEX_URL" ]; then
        echo "Error: manifest declares 'dev' packages but no 'index <url>' line" >&2
        fail "manifest_missing_index" 5
    fi

    if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
        echo "Manifest dev packages (${#DEV_PACKAGES[@]}): ${DEV_PACKAGES[*]}"
    else
        echo "Manifest declares no dev packages — all dependencies will resolve from public PyPI"
    fi
fi

# ──────────────────────────────────────────────
# 1. Check uv
# ──────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing ..."
    case "$(uname -s)" in
        CYGWIN*|MINGW*|MSYS*|Windows_NT*)
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" \
                || { echo "Error: uv installation failed on Windows." >&2; fail "uv_install_failed" 1; }
            ;;
        *)
            curl -LsSf https://astral.sh/uv/install.sh | sh \
                || { echo "Error: uv installation failed." >&2; fail "uv_install_failed" 1; }
            ;;
    esac
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "Error: uv was installed but not found on PATH." >&2
        fail "uv_not_on_path" 1
    fi
    echo "uv installed successfully."
fi

echo "uv: $(uv --version 2>&1)"

# ──────────────────────────────────────────────
# 2. Initialize uv project if needed
# ──────────────────────────────────────────────
if [ ! -f pyproject.toml ]; then
    echo "No pyproject.toml found — running uv init --bare ..."
    run_uv "uv_init_failed" 2 uv init --bare
    echo "Created pyproject.toml"
else
    echo "pyproject.toml already exists, skipping init"
fi

# ──────────────────────────────────────────────
# 3. Inject dev index sources from manifest (re-entrant via markers)
# ──────────────────────────────────────────────
if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
    # Remove any previous bridgic-deps injection block so manifest changes
    # take effect on re-run without manual cleanup.
    if grep -qF "$INJECTION_BEGIN_MARKER" pyproject.toml 2>/dev/null; then
        echo "Replacing previous bridgic-deps injection block in pyproject.toml"
        awk -v begin="$INJECTION_BEGIN_MARKER" -v end="$INJECTION_END_MARKER" '
            index($0, begin) { skip=1; next }
            skip && index($0, end) { skip=0; next }
            !skip
        ' pyproject.toml > pyproject.toml.bridgic.tmp \
            && mv pyproject.toml.bridgic.tmp pyproject.toml
    fi

    echo "Injecting dev index for ${#DEV_PACKAGES[@]} package(s) into pyproject.toml"
    {
        echo ""
        echo "$INJECTION_BEGIN_MARKER (auto-generated by install-deps.sh, do not edit by hand)"
        echo "[[tool.uv.index]]"
        echo "name = \"${DEV_INDEX_NAME}\""
        echo "url = \"${DEV_INDEX_URL}\""
        echo "explicit = true"
        echo ""
        echo "[tool.uv.sources]"
        for pkg in "${DEV_PACKAGES[@]}"; do
            echo "${pkg} = { index = \"${DEV_INDEX_NAME}\" }"
        done
        echo "$INJECTION_END_MARKER"
    } >> pyproject.toml
    echo "Dev index injected for: ${DEV_PACKAGES[*]}"
fi

# ──────────────────────────────────────────────
# 4. Check and install missing packages
# ──────────────────────────────────────────────

# Helper: check if a package is already in [project.dependencies].
# Must match only quoted dependency strings like "pkg>=1.0" — NOT
# [tool.uv.sources] entries like `pkg = { index = "..." }`, which would
# otherwise cause false positives and silently skip package installation.
is_installed() {
    local pkg="$1"
    grep -qiE "^[[:space:]]*\"${pkg}[[:space:]]*[>=<~!\"]" pyproject.toml 2>/dev/null
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
    # uv's default prerelease mode (if-necessary-or-explicit) handles mixed
    # routing correctly: dev-only packages resolve to their dev release
    # because no stable match exists, while packages routed to public PyPI
    # keep picking stable versions. No --prerelease flag is needed.
    run_uv "uv_add_failed" 3 uv add "${MISSING[@]}"
fi

# ──────────────────────────────────────────────
# 5. Sync project environment
# ──────────────────────────────────────────────
echo ""
echo "Syncing project environment ..."
run_uv "uv_sync_failed" 4 uv sync

echo ""
if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
    echo "=== DEPS_READY (bridgic-llms, provider: $PROVIDER, dev packages: ${DEV_PACKAGES[*]}) ==="
else
    echo "=== DEPS_READY (bridgic-llms, provider: $PROVIDER) ==="
fi
