#!/bin/bash

# Usage:
#     bash ./scripts/publish_packages.sh [repo_name]
# Return:
#     Space seperated packages name failed to be published.

repo=${1:-"btsk"}

echo "Publishing to repository [$repo]."
echo ""

if [ -z "$UV_PUBLISH_USERNAME" ]; then
    read -p "Input your username: " UV_PUBLISH_USERNAME
    export UV_PUBLISH_USERNAME
fi

if [ -z "$UV_PUBLISH_PASSWORD" ]; then
    read -sp "Input your password: " UV_PUBLISH_PASSWORD
    export UV_PUBLISH_PASSWORD
fi

echo ""
echo "Now publishing all of your packages..."
echo ""


# Define main package which is the last one to publish.
main_package=$(basename "$PWD")

# Define priority packages, which will be firstly published in order.
priority_packages=("bridgic-core" "bridgic-asl")

failures_file=$(mktemp)
successes_file=$(mktemp)

###############################################################################
# Utility functions
###############################################################################

# Function to print colorful text
# Arguments:
#   $1: color
#   $2: text to print
#   $3: optional icon/prefix
colorful_print() {
    local color="$1"
    local text="$2"
    local icon="${3:-}"

    local color_code=""
    local reset="\033[0m"

    case "$color" in
        red)         color_code="\033[31m" ;;
        green)       color_code="\033[32m" ;;
        yellow)      color_code="\033[33m" ;;
        white)       color_code="\033[37m" ;;
        bold_green)  color_code="\033[1;32m" ;;
        bold_yellow) color_code="\033[1;33m" ;;
        bold_white)  color_code="\033[1;37m" ;;
        *)           color_code="" ;;
    esac

    if [ -n "$icon" ]; then
        printf "${color_code}%s${reset} %s\n" "$icon" "$text"
    else
        printf "${color_code}%s${reset}\n" "$text"
    fi
}

# Function to check if a value is in an array
# Arguments:
#   $1: target value to search for
#   $@: array elements to search in
# Returns:
#   0 if found, 1 if not found
contains() {
    local target="$1"
    shift
    local item
    for item in "$@"; do
        [ "$item" = "$target" ] && return 0
    done
    return 1
}

# Function to publish a single package
# Arguments:
#   $1: required package directory (empty str for main package)
#   $2: required package name
#   $3: optional message suffix
publish_package() {
    local pkg_dir="$1"
    local pkg_name="$2"
    local msg_suffix="${3:-}"

    # Determine if this is the main package based on whether pkg_dir is empty
    local is_main_package=false
    if [ -z "$pkg_dir" ]; then
        is_main_package=true
    fi

    local pyproject_file
    if [ "$is_main_package" = true ]; then
        pyproject_file="pyproject.toml"
        echo "==> Found package: $pkg_name"
    else
        pyproject_file="$pkg_dir/pyproject.toml"
        echo "==> Found subpackage: $pkg_name"
    fi

    local version
    version=$(uv run python -c "import tomli; print(tomli.load(open('$pyproject_file', 'rb'))['project']['version'])")

    if ! python scripts/version_check.py --version "$version" --repo "$repo" --package "$pkg_name"; then
        echo "==> Skipping $pkg_name (version $version incompatible with $repo)"
        if [ "$is_main_package" = true ]; then
            echo "$pkg_name" >> "$failures_file"
        else
            echo "$pkg_dir" >> "$failures_file"
        fi
        return 1
    fi

    local publish_msg="==> Publishing"
    if [ "$is_main_package" = true ]; then
        publish_msg="$publish_msg package [$pkg_name]"
    else
        publish_msg="$publish_msg subpackage [$pkg_name]"
    fi
    if [ -n "$msg_suffix" ]; then
        publish_msg="$publish_msg ($msg_suffix)"
    fi
    publish_msg="$publish_msg..."
    echo "$publish_msg"

    if [ "$is_main_package" = true ]; then
        if ! make publish repo="$repo"; then
            echo "$pkg_name" >> "$failures_file"
            echo "Failed to publish: $pkg_name"
            return 1
        fi
    else
        if ! make -C "$pkg_dir" publish repo="$repo"; then
            echo "$pkg_dir" >> "$failures_file"
            echo "Failed to publish: $pkg_dir"
            return 1
        fi
    fi

    if [ "$is_main_package" = true ]; then
        echo "Successfully published: $pkg_name"
        echo "$pkg_name" >> "$successes_file"
    else
        echo "Successfully published: $pkg_dir"
        echo "$pkg_name" >> "$successes_file"
    fi
    return 0
}

# Collect all subpackages
declare -a subpackage_list=()
while IFS= read -r dir; do
    if [ -f "$dir/Makefile" ] && [ -f "$dir/pyproject.toml" ]; then
        subpackage_list+=("$dir")
    fi
done < <(find ./packages -maxdepth 4 -type d -name "bridgic-*" | sort)

###############################################################################
# Step 1: Publish priority packages first in order
###############################################################################
for priority_pkg in "${priority_packages[@]}"; do
    for dir in "${subpackage_list[@]}"; do
        pkg_name=$(basename "$dir")
        if [ "$pkg_name" = "$priority_pkg" ]; then
            publish_package "$dir" "$pkg_name" "priority: $pkg_name"
            echo ""
            break
        fi
    done
done

###############################################################################
# Step 2: Publish remaining packages (skip priority packages)
###############################################################################
for dir in "${subpackage_list[@]}"; do
    pkg_name=$(basename "$dir")
    if ! contains "$pkg_name" "${priority_packages[@]}"; then
        publish_package "$dir" "$pkg_name"
        echo ""
    fi
done
echo ""

###############################################################################
# Step 3: Publish main package
###############################################################################
publish_package "" "$main_package"
echo ""

###############################################################################
# Step 4: Show the publish results
###############################################################################
echo ""
echo "==============================================================================="
colorful_print "bold_white" " 📦 PUBLISH SUMMARY"
echo "==============================================================================="
echo ""

# Count successes and failures
success_count=0
failure_count=0

if [ -s "$successes_file" ]; then
    success_count=$(wc -l < "$successes_file" | xargs)
fi

if [ -s "$failures_file" ]; then
    failure_count=$(wc -l < "$failures_file" | xargs)
fi

# Show successful packages
if [ -s "$successes_file" ]; then
    colorful_print "green" "✓ Successfully Published ($success_count):"
    while IFS= read -r pkg; do
        colorful_print "green" "$pkg" "✓"
    done < "$successes_file"
    echo ""
fi

# Show failed packages
if [ -s "$failures_file" ]; then
    colorful_print "red" "✗ Failed to Publish ($failure_count):"
    while IFS= read -r pkg; do
        colorful_print "red" "$pkg" "✗"
    done < "$failures_file"
    echo ""
fi

# Print summary line
echo "==============================================================================="
colorful_print "bold_yellow" " 🔨 Completed with: $success_count succeeded, $failure_count failed"
echo "==============================================================================="
echo ""

# Clean up temporary files
rm -f "$failures_file" "$successes_file"
