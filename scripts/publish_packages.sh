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

main_package=$(basename "$PWD")
failures_file=$(mktemp)

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
    else
        echo "Successfully published: $pkg_dir"
    fi
    return 0
}

# Collect all subpackages
declare -a subpackages=()
while IFS= read -r -d '' dir; do
    if [ -f "$dir/Makefile" ] && [ -f "$dir/pyproject.toml" ]; then
        subpackages+=("$dir")
    fi
done < <(find . -maxdepth 4 -type d -name "bridgic-*" -print0)

# Sort packages to ensure bridgic-core is published first
IFS=$'\n' sorted_packages=($(printf '%s\n' "${subpackages[@]}" | sort))
unset IFS

# Publish bridgic-core first if it exists
for dir in "${sorted_packages[@]}"; do
    sub_package=$(basename "$dir")
    if [ "$sub_package" = "bridgic-core" ]; then
        publish_package "$dir" "$sub_package" "priority: core package"
        echo ""
        break
    fi
done

# Wait a moment for the package to be available in the repository
sleep 2

# Publish remaining packages
for dir in "${sorted_packages[@]}"; do
    sub_package=$(basename "$dir")
    if [ "$sub_package" = "bridgic-core" ]; then
        continue  # Already published
    fi
    publish_package "$dir" "$sub_package"
    echo ""
done
echo ""

# Publish main package
publish_package "" "$main_package"
echo ""

if [ -s "$failures_file" ]; then
    printf "==> The following packages failed to be published:\n"
    printf "\033[31m"
    while IFS= read -r pkg; do
        printf "  %s\n" "$pkg"
    done < "$failures_file"
    printf "\033[0m\n"
    rm -f "$failures_file"
    exit 1
else
    echo "==> All packages published successfully!"
    rm -f "$failures_file"
    exit 0
fi
