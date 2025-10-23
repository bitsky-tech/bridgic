.PHONY: venv-init test build publish

package_name := $(notdir $(CURDIR))
repo ?= btsk

ROOT_DIR := $(shell git rev-parse --show-toplevel)
VERSION_CHECK := $(ROOT_DIR)/scripts/version_check.py

venv-init:
	@echo "\n==> Initializing virtual environment for [${package_name}]..."
	@if [ -d ${CURDIR}/.venv ]; then echo ".venv already exists, removing..."; rm -rf ${CURDIR}/.venv; fi
	@uv venv --python=python3.9 ${CURDIR}/.venv && echo ".venv created."
	@echo "\n==> Installing dependencies for [${package_name}]..."
	@source ${CURDIR}/.venv/bin/activate && uv pip install -e .

venv-collect:
	@echo "\n==> Installing dependencies for [${package_name}]..."
	@uv pip install -e .

test:
	@uv run -- pytest

build:
	@mkdir -p dist
	@rm -rf dist/*
	@uv build

publish:
	@version=$$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])") && \
	python $(VERSION_CHECK) --version "$$version" --repo "$(repo)" --package "$(package_name)" && \
	$(MAKE) _publish_$(repo)

_publish_btsk:
	@uv publish --index btsk-repo --config-file $(ROOT_DIR)/uv.toml

_publish_testpypi:
	@uv publish --index test-pypi --config-file $(ROOT_DIR)/uv.toml

_publish_pypi:
	@uv publish --config-file $(ROOT_DIR)/uv.toml