.PHONY: install serve build check help version clean

# Configurable parameters
HOST ?= 127.0.0.1
PORT ?= 8000

MKDOCS = uv run mkdocs

# Build static site
build-doc:
	@uv sync
	$(MKDOCS) build --strict

# Start development server
serve-doc:
	@uv sync
	$(MKDOCS) serve -a $(HOST):$(PORT)

# Clean build artifacts
clean:
	rm -rf site/ .cache/
