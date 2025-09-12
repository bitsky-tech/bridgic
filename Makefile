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
	@echo "\n==> Initializing virtual environment for subpackages..."
	@source .venv/bin/activate && \
	find . -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo "==> Found Bridgic subpackage: $$dir"; \
			$(MAKE) -C "$$dir" venv-init; \
		fi \
	done

test-all:
	@find . -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo "==> Testing subpackage [$$dir]..."; \
			$(MAKE) -C "$$dir" test; \
		fi \
	done

build:
	@mkdir -p dist
	@rm -rf dist/*
	@uv build

build-all:
	@find . -maxdepth 4 -type d -name "bridgic-*" | while read dir; do \
		if [ -f "$$dir/Makefile" ] && [ -f "$$dir/pyproject.toml" ]; then \
			echo "==> Building subpackage [$$dir]..."; \
			$(MAKE) -C "$$dir" build; \
		fi \
	done
	@echo "==> Building package [${package_name}]..."
	${MAKE} build

publish:
ifeq ($(repo), btsk)
	@uv publish --index btsk-repo
else ifeq ($(repo), pypi)
	$(error "Now we don't support publishing to pypi!"); exit 1
else
	$(error "Unknown repository: [${repo}]"); exit 1
endif

publish-all:
	@bash ./scripts/publish_packages.sh $(repo)