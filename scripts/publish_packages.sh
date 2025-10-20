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

while IFS= read -r -d '' dir; do
    if [ -f "$dir/Makefile" ] && [ -f "$dir/pyproject.toml" ]; then
        sub_package=$(basename "$dir")
        echo "==> Found Bridgic subpackage: $sub_package"

        version=$(python -c "import tomllib; print(tomllib.load(open('$dir/pyproject.toml', 'rb'))['project']['version'])")

        if python scripts/version_check.py --version "$version" --repo "$repo" --package "$sub_package"; then
            echo "==> Publishing subpackage [$sub_package]..."
            if ! make -C "$dir" publish repo="$repo"; then
                echo "$dir" >> "$failures_file"
                echo "Failed to publish: $dir"
            else
                echo "Successfully published: $dir"
            fi
        else
            echo "==> Skipping $sub_package (version $version incompatible with $repo)"
            echo "$dir" >> "$failures_file"
        fi
        echo ""
    fi
done < <(find . -maxdepth 4 -type d -name "bridgic-*" -print0)
echo ""

version=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
if python scripts/version_check.py --version "$version" --repo "$repo" --package "$main_package"; then
    echo "==> Publishing package [$main_package]..."
    if ! make publish repo="$repo"; then
        echo "$main_package" >> "$failures_file"
        echo "Failed to publish: $main_package"
    else
        echo "Successfully published: $main_package"
    fi
else
    echo "==> Skipping $main_package (version $version incompatible with $repo)"
    echo "$main_package" >> "$failures_file"
fi
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
