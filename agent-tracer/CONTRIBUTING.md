# Contributing to Agent Tracer

Thank you for your interest in contributing to Agent Tracer! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agent-tracer.git
   cd agent-tracer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev,all]"
   ```

## Code Standards

### Style Guide

- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Maximum line length: 120 characters
- Use Black for code formatting
- Use Ruff for linting

### Code Formatting

```bash
# Format code with Black
black src/agent_tracer

# Run linter
ruff check src/agent_tracer

# Type checking
mypy src/agent_tracer
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent_tracer --cov-report=html
```

## Adding a New Tracer

To add support for a new tracing backend:

1. **Create a new tracer file** in `src/agent_tracer/tracers/`
   ```python
   # src/agent_tracer/tracers/my_tracer.py
   from agent_tracer.base import BaseTracer
   
   class MyTracer(BaseTracer):
       def __init__(self, ...):
           # Implementation
           pass
       
       # Implement all abstract methods
   ```

2. **Update service.py** to initialize your tracer
   ```python
   def _initialize_my_tracer(self, trace_context: TraceContext) -> None:
       try:
           from agent_tracer.tracers.my_tracer import MyTracer
           
           trace_context.tracers["my_tracer"] = MyTracer(...)
       except ImportError:
           logger.debug("MyTracer not available")
       except Exception as e:
           logger.debug(f"Error initializing MyTracer: {e}")
   ```

3. **Update pyproject.toml** with optional dependencies
   ```toml
   [project.optional-dependencies]
   my-tracer = [
       "my-tracer-package>=1.0.0",
   ]
   ```

4. **Add tests** in `tests/tracers/test_my_tracer.py`

5. **Update documentation** in README.md and add examples

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

**Examples:**
```
feat(tracers): add support for LangWatch tracer

Implement LangWatch tracer with full span support and
LangChain callback integration.

Closes #123
```

```
fix(service): handle None trace context in get_callbacks

Check for None trace context before accessing tracers
to prevent AttributeError.

Fixes #456
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feat/my-new-feature
   ```

2. **Make your changes** following the code standards

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Run tests and linting**
   ```bash
   pytest
   black src/agent_tracer
   ruff check src/agent_tracer
   mypy src/agent_tracer
   ```

6. **Commit your changes** with clear commit messages

7. **Push to your fork**
   ```bash
   git push origin feat/my-new-feature
   ```

8. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

### PR Review Process

- At least one maintainer approval required
- All tests must pass
- Code coverage should not decrease
- Documentation must be updated

## Project Structure

```
agent-tracer/
├── src/
│   └── agent_tracer/
│       ├── __init__.py         # Public API
│       ├── base.py             # BaseTracer abstract class
│       ├── schema.py           # Data schemas
│       ├── service.py          # TracingService
│       ├── utils.py            # Utility functions
│       └── tracers/            # Tracer implementations
│           ├── __init__.py
│           ├── langfuse.py
│           ├── langwatch.py
│           ├── opik.py
│           └── console.py
├── tests/                      # Test files
├── examples/                   # Example scripts
├── docs/                       # Documentation
└── pyproject.toml             # Project configuration
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Update examples for new features

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Join our community chat [link]

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

