.PHONY: install serve build check help version clean

# Configurable parameters
HOST ?= 127.0.0.1
PORT ?= 8000

MKDOCS = uv run mkdocs

# Generate API reference documentation
gen-ref-pages:
	@uv run python scripts/gen_ref_pages_safe.py

# Build static site
build-doc:
	@uv sync
	@make gen-ref-pages
	$(MKDOCS) build

# Start development server
serve-doc:
	@uv sync
	$(MKDOCS) serve -a $(HOST):$(PORT)

# Clean build artifacts
clean:
	rm -rf site/ .cache/
