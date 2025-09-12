#!/bin/bash

# Usage:
#     bash ./scripts/publish_packages.sh [repo_name]
# Return:
#     Space seperated packages name failed to be published.

set -e

repo=${1:-"btsk"}

echo "Publishing to repository [$repo]."

main_package=$(basename "$PWD")
failures=()

find . -maxdepth 4 -type d -name "bridgic-*" | while read dir; do
    if [ -f "$dir/Makefile" ] && [ -f "$dir/pyproject.toml" ]; then
        sub_package=$(basename "$dir")
        echo "==> Found Bridgic subpackage: $sub_package"
        echo "==> Publishing subpackage [$sub_package]..."
        if ! make -C "$dir" publish repo="$repo"; then
            failures+=("$dir")
        else
            echo "Successfully published: $dir"
        fi
    fi
done

echo "==> Publishing package [$main_package]..."
if ! make publish repo="$repo"; then
    failures+=("$main_package")
else
    echo "Successfully published: $main_package"
fi

if [ ${#failures[@]} -ne 0 ]; then
    printf "==> The following packages failed to be published:\n"
    printf "\033[31m"
    for pkg in "${failures[@]}"; do
        printf "  [%s]\n" "$pkg"
    done
    printf "\033[0m\n"
    exit 1
else
    echo "==> All packages published successfully!"
    exit 0
fi
