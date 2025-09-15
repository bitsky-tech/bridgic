.PHONY: install serve build check help version clean

# Configurable parameters
HOST ?= 127.0.0.1
PORT ?= 8000

# Command wrapper
MKDOCS = uv run mkdocs

# Common MkDocs commands

# Install dependencies
install:
	uv sync

# Start development server
serve:
	$(MKDOCS) serve -a $(HOST):$(PORT)

# Build static site
build:
	$(MKDOCS) build

# Validate configuration
check:
	$(MKDOCS) build --strict

# Show help
help:
	$(MKDOCS) --help

# Show version
version:
	$(MKDOCS) --version

# Clean build artifacts
clean:
	rm -rf site/ .cache/
