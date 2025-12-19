#!/bin/bash
# Set publish credentials (username and password) for uv publish.
# If credentials are already set in environment variables, they will be reused.
# Otherwise, prompts the user to input them and exports them.

if [ -z "$UV_PUBLISH_USERNAME" ]; then
    read -p "Input your username: " UV_PUBLISH_USERNAME
    export UV_PUBLISH_USERNAME
fi

if [ -z "$UV_PUBLISH_PASSWORD" ]; then
    read -sp "Input your password: " UV_PUBLISH_PASSWORD
    export UV_PUBLISH_PASSWORD
fi

