.PHONY: install serve build check help version clean

# Configurable parameters
HOST ?= 127.0.0.1
PORT ?= 8000

MKDOCS = uv run mkdocs

# Generate mkdocs.yml file from API reference documentation
gen-mkdocs-yml:
	@uv run python scripts/gen_mkdocs_yml.py

# Build static site
build-doc:
	@uv sync
	@make gen-mkdocs-yml
	$(MKDOCS) build

# Start development server
serve-doc:
	@uv sync
	@make gen-mkdocs-yml
	$(MKDOCS) serve -a $(HOST):$(PORT)

# Clean build artifacts
clean:
	rm -rf site/ .cache/
