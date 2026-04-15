#!/bin/bash
# install-deps.sh — Install bridgic-amphibious skill dependencies.
#
# Reads the per-skill INI config (deps.ini, sibling of this script) which
# declares every required package, its source ("default" for public PyPI,
# or the literal name of a private index — e.g. "btsk-repo"), and an
# optional pinned version constraint. When BRIDGIC_DEV_INDEX is set,
# packages with a non-default source are routed through that URL via
# [[tool.uv.index]] + [tool.uv.sources] in pyproject.toml, using the source
# field as the index name. Only one private index per config is supported.
#
# Steps:
#   1. Parse deps.ini (pure-bash + awk, no Python/jq/yq dependency).
#   2. Check uv availability (auto-install if missing).
#   3. Run `uv init --bare` if no pyproject.toml yet.
#   4. Inject dev index block into pyproject.toml (re-entrant via markers).
#   5. `uv add` any missing packages and re-pin packages whose config
#      declares an explicit version.
#   6. `uv sync` to finalize the project environment.
#
# Config file: ./deps.ini (sibling of this script)
#
# Environment:
#   BRIDGIC_DEV_INDEX   URL of the private package index. Required when
#                       deps.ini declares any package whose source is not
#                       "default". When unset, presence of such packages
#                       is a fatal error.
#
# Usage:
#   install-deps.sh [PROJECT_DIR]   (defaults to current directory)
#
# Exit codes:
#   0  All dependencies installed and synced
#   1  uv not installed
#   2  uv init failed
#   3  uv add failed
#   4  uv sync failed
#   5  deps.ini missing or malformed
#   6  deps.ini declares private-index packages but BRIDGIC_DEV_INDEX is unset
#
# Output markers:
#   On success: "=== DEPS_READY (...) ==="
#   On failure: "=== DEPS_FAILED reason=<label> exit=<N> ==="

set -euo pipefail

# Resolve script directory before cd'ing into the project so deps.ini is
# found regardless of the caller's PWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/deps.ini"

PROJECT_DIR="${1:-.}"
DEV_INDEX="${BRIDGIC_DEV_INDEX:-}"

cd "$PROJECT_DIR"

INJECTION_BEGIN_MARKER="# BEGIN bridgic-deps-injection"
INJECTION_END_MARKER="# END bridgic-deps-injection"
# Populated from deps.ini during parsing — the literal name of the private
# index that non-default packages route through. Empty means all packages
# resolve from public PyPI.
DEV_INDEX_NAME=""

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
# 0. Parse deps.ini
# ──────────────────────────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    fail "config_not_found" 5
fi

echo "Reading deps config: $CONFIG_FILE"

# Parse INI via awk → emit one TSV line per package: name<TAB>source<TAB>version
# Awk does the heavy lifting (sections, key=value, comments, quotes); bash
# just consumes the TSV stream into parallel arrays. Awk exits 2 on any
# malformed line (unknown key, key outside any section, garbage line).
PARSED_TSV="$(awk '
function trim(s) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); return s }
function emit() {
    if (cur_section != "") {
        if (cur_source == "") cur_source = "default"
        print cur_section "\t" cur_source "\t" cur_version
    }
}
BEGIN { cur_section = ""; cur_source = ""; cur_version = "" }
/^[[:space:]]*[#;]/ { next }                          # full-line comments
/^[[:space:]]*$/    { next }                          # blank lines
/^[[:space:]]*\[.*\][[:space:]]*$/ {                  # [section] header
    emit()
    line = $0
    sub(/^[[:space:]]*\[/, "", line)
    sub(/\][[:space:]]*$/, "", line)
    cur_section = trim(line)
    cur_source = ""
    cur_version = ""
    next
}
/=/ {                                                 # key = value
    if (cur_section == "") {
        print "ini_error: key/value outside any section: " $0 > "/dev/stderr"
        exit 2
    }
    eq_idx = index($0, "=")
    key = trim(substr($0, 1, eq_idx - 1))
    val = trim(substr($0, eq_idx + 1))
    sub(/^"/, "", val); sub(/"$/, "", val)
    if (key == "source")       cur_source  = val
    else if (key == "version") cur_version = val
    else {
        print "ini_error: unknown key \"" key "\" in section [" cur_section "]" > "/dev/stderr"
        exit 2
    }
    next
}
{
    print "ini_error: malformed line: " $0 > "/dev/stderr"
    exit 2
}
END { emit() }
' "$CONFIG_FILE")" || fail "config_malformed" 5

# Read parallel arrays from TSV (bash 3.2 compatible — no associative arrays).
PKG_NAMES=()
PKG_SOURCES=()
PKG_VERSIONS=()
DEV_PACKAGES=()

while IFS=$'\t' read -r pkg_name pkg_source pkg_version; do
    [ -z "${pkg_name:-}" ] && continue
    if [ "$pkg_source" != "default" ]; then
        # Any non-default value is the literal name of a private index
        # (e.g. "btsk-repo"). The URL still comes from BRIDGIC_DEV_INDEX.
        # Only one private index per config — reject mixed names.
        if [ -n "$DEV_INDEX_NAME" ] && [ "$DEV_INDEX_NAME" != "$pkg_source" ]; then
            echo "Error: package '$pkg_name' declares index '$pkg_source', but" >&2
            echo "       '$DEV_INDEX_NAME' was already declared by another package." >&2
            echo "       Only one private index per config is supported." >&2
            fail "config_malformed" 5
        fi
        DEV_INDEX_NAME="$pkg_source"
        DEV_PACKAGES+=("$pkg_name")
    fi
    PKG_NAMES+=("$pkg_name")
    PKG_SOURCES+=("$pkg_source")
    PKG_VERSIONS+=("$pkg_version")
done <<< "$PARSED_TSV"

if [ ${#PKG_NAMES[@]} -eq 0 ]; then
    echo "Error: deps.ini declares no packages" >&2
    fail "config_empty" 5
fi

echo "Config declares ${#PKG_NAMES[@]} package(s)"

if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
    if [ -z "$DEV_INDEX" ]; then
        echo "Error: deps.ini declares packages on private index '$DEV_INDEX_NAME'" >&2
        echo "       but BRIDGIC_DEV_INDEX is not set." >&2
        echo "       affected packages: ${DEV_PACKAGES[*]}" >&2
        fail "dev_index_missing" 6
    fi
    echo "Private index '$DEV_INDEX_NAME' (URL from BRIDGIC_DEV_INDEX) routes: ${DEV_PACKAGES[*]}"
else
    echo "All packages resolve from public PyPI"
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
# 3. Inject dev index sources from config (re-entrant via markers)
# ──────────────────────────────────────────────
if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
    # Remove any previous bridgic-deps injection block so config changes
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
        echo "url = \"${DEV_INDEX}\""
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
# 4. Install / re-pin packages from config
# ──────────────────────────────────────────────

# Helper: check if a package is already in [project.dependencies].
# Must match only quoted dependency strings like "pkg>=1.0" — NOT
# [tool.uv.sources] entries like `pkg = { index = "..." }`, which would
# otherwise cause false positives and silently skip package installation.
is_installed() {
    local pkg="$1"
    grep -qiE "^[[:space:]]*\"${pkg}[[:space:]]*[>=<~!\"]" pyproject.toml 2>/dev/null
}

# Build the install spec list:
#   - packages with a `version =` in the config are always re-added (so the
#     constraint in pyproject.toml is forced to match the config on re-run)
#   - packages without a version are added only if they aren't already in
#     pyproject.toml (existing constraints are left untouched)
TO_INSTALL=()
for i in "${!PKG_NAMES[@]}"; do
    name="${PKG_NAMES[$i]}"
    version="${PKG_VERSIONS[$i]}"
    if [ -n "$version" ]; then
        TO_INSTALL+=("${name}${version}")
        echo "→ ${name}${version} (re-pinning from config)"
    elif is_installed "$name"; then
        echo "✓ ${name} already present (no version pinned in config)"
    else
        TO_INSTALL+=("$name")
        echo "→ ${name} (missing, will install latest)"
    fi
done

if [ ${#TO_INSTALL[@]} -gt 0 ]; then
    echo ""
    echo "Running: uv add ${TO_INSTALL[*]}"
    run_uv "uv_add_failed" 3 uv add "${TO_INSTALL[@]}"
fi

# ──────────────────────────────────────────────
# 5. Sync project environment
# ──────────────────────────────────────────────
echo ""
echo "Syncing project environment ..."
run_uv "uv_sync_failed" 4 uv sync

echo ""
if [ ${#DEV_PACKAGES[@]} -gt 0 ]; then
    echo "=== DEPS_READY (bridgic-amphibious, dev packages: ${DEV_PACKAGES[*]}) ==="
else
    echo "=== DEPS_READY (bridgic-amphibious) ==="
fi
