.PHONY: venv-init test build publish

package_name := $(notdir $(CURDIR))
repo ?= btsk

ROOT_DIR := $(shell git rev-parse --show-toplevel)
VERSION_CHECK := $(ROOT_DIR)/scripts/version_check.py
SET_CREDENTIALS := $(ROOT_DIR)/scripts/set_publish_credentials.sh

venv-collect:
	@echo "\n==> Installing dependencies for [${package_name}]..."
	@uv pip install -e .

test:
	@uv run -- pytest

build:
	@mkdir -p dist
	@rm -rf dist/*
	@package_name=$$(uv run python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['name'])") && \
	uv build --package "$$package_name" --out-dir dist

publish:
	@source $(SET_CREDENTIALS) && \
	version=$$(uv run python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])") && \
	uv run python $(VERSION_CHECK) --version "$$version" --repo "$(repo)" --package "$(package_name)" && \
	$(MAKE) _publish_$(repo)

_publish_btsk:
	@uv publish dist/* --index btsk-repo --config-file $(ROOT_DIR)/uv.toml

_publish_testpypi:
	@uv publish dist/* --index test-pypi --config-file $(ROOT_DIR)/uv.toml

_publish_pypi:
	@uv publish dist/* --config-file $(ROOT_DIR)/uv.toml

