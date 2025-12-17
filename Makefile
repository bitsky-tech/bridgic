SHELL := /bin/bash
ROOT_DIR := $(shell git rev-parse --show-toplevel)
VERSION_CHECK := $(ROOT_DIR)/scripts/version_check.py

.PHONY: init-dev test-all build build-all publish publish-all

package_name := $(notdir $(CURDIR))
repo ?= btsk

init-dev:
	@test -d .git || (echo "Not a git repo"; exit 1)
	@git config --local core.hooksPath .githooks
	@echo "\n==> Successfully installed git hooks."
	@echo "\n==> Preparing virtual environment for project."
	@if [ -d .venv ]; then echo ".venv already exists, removing..."; rm -rf .venv; fi
	@uv venv --python=python3.9 .venv && echo ".venv created."
	@echo "\n==> Installing development dependencies for the project..."
	@source .venv/bin/activate && uv sync --group dev --group publish
	@echo "\n==> Initializing virtual environment for subpackages..."
	@source .venv/bin/activate && \
	find ./packages -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo "==> Found Bridgic subpackage: $$dir"; \
			$(MAKE) -C "$$dir" venv-collect; \
		fi \
	done

test-all:
	@find ./packages -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo ""; \
			echo "==> Testing subpackage [$$dir]..."; \
			$(MAKE) -C "$$dir" test; \
		fi \
	done
	@cd ./tests && uv run -- pytest -v

build:
	@mkdir -p dist
	@rm -rf dist/*
	@uv build

build-all:
	@find ./packages -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo "==> Building subpackage [$$dir]..."; \
			$(MAKE) -C "$$dir" build; \
		fi \
	done
	@echo "==> Building package [${package_name}]..."
	${MAKE} build

publish:
	@version=$$(uv run python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])") && \
    uv run python $(VERSION_CHECK) --version "$$version" --repo "$(repo)" --package "$(package_name)" && \
	$(MAKE) _publish_$(repo)

_publish_btsk:
	@uv publish --index btsk-repo --config-file $(ROOT_DIR)/uv.toml

_publish_testpypi:
	@uv publish --index test-pypi --config-file $(ROOT_DIR)/uv.toml

_publish_pypi:
	@uv publish --config-file $(ROOT_DIR)/uv.toml

publish-all:
	@bash ./scripts/publish_packages.sh $(repo)
