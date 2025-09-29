.PHONY: venv-init test build publish

package_name := $(notdir $(CURDIR))
repo ?= btsk

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
ifeq ($(repo), btsk)
	@uv publish --index btsk-repo
else ifeq ($(repo), pypi)
	$(error "Now we don't support publishing to pypi!"); exit 1
else
	$(error "Unknown repository: [${repo}]"); exit 1
endif