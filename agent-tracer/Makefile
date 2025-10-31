.PHONY: help install install-dev test lint format clean build publish

help:
	@echo "Available commands:"
	@echo "  install       - Install the package"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black"
	@echo "  clean         - Remove build artifacts"
	@echo "  build         - Build package"
	@echo "  publish       - Publish to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

test:
	pytest tests/ -v --cov=agent_tracer --cov-report=html --cov-report=term

lint:
	ruff check src/agent_tracer
	mypy src/agent_tracer

format:
	black src/agent_tracer tests examples
	ruff check --fix src/agent_tracer

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

examples:
	@echo "Running basic example..."
	python examples/basic_usage.py

