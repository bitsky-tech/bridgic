.PHONY: install serve build check help version clean

# Configurable parameters
HOST ?= 127.0.0.1
PORT ?= 8000

MKDOCS = uv run mkdocs
MIKE = uv run mike

# Generate mkdocs.yml file from API reference documentation
gen-mkdocs-yml:
	@uv run python scripts/gen_mkdocs_yml.py

# Build static site
build-doc:
	@uv sync
	@make gen-mkdocs-yml
	$(MKDOCS) build

require-version:
	@if [ -z "$(version)" ]; then \
		echo "ERROR: version is required. Usage: make build-doc-version version=1.2.3"; \
		exit 1; \
	fi

# build a specific version of the docs and point the latest alias to it
build-doc-version: require-version
	@make gen-mkdocs-yml
	$(MIKE) deploy --push --update-aliases $(version) latest
	$(MIKE) set-default --push latest

# Delete a deployed docs version and point the latest alias to the next highest version
delete-doc-version: require-version
	@make gen-mkdocs-yml
	$(MIKE) delete --push $(version)

# Start development server
serve-doc:
	@uv sync
	@make gen-mkdocs-yml
	$(MIKE) serve -a $(HOST):$(PORT)

# Clean build artifacts
clean:
	rm -rf site/ .cache/
