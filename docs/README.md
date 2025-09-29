## Bridgic Docs 📚

A lightweight documentation site powered by MkDocs Material and mkdocstrings. It hosts the project guides and API reference for Bridgic. ✨

### Prerequisites ✅

- Python 3.9+
- uv (Python package manager) — see `https://github.com/astral-sh/uv`

All Python dependencies are defined in `pyproject.toml` and installed via `uv sync`.

### Quick Start 🚀

```bash
# 1) Install dependencies
make install

# 2) Start the dev server (default: 127.0.0.1:8000)
make serve

# Or customize host/port
make serve HOST=0.0.0.0 PORT=8001
```

Open `http://127.0.0.1:8000` in your browser. Live reload is enabled. 🔁

### Common Commands 🛠️

```bash
# Build the static site into ./site
make build

# Strict build to validate configuration and references
make check

# Show MkDocs help / version
make help
make version

# Clean build artifacts and cache
make clean
```

Under the hood, these targets invoke MkDocs via `uv run mkdocs` so you don't need to manually activate a virtual environment. 🧰

### API Documentation Generation 🤖

The project includes an automated API documentation generation system:

```bash
# Generate API reference documentation
uv run python scripts/gen_ref_pages_safe.py

# Configuration file for customization
scripts/doc_config.yaml
```

#### Features:
- **Safe Configuration Updates**: Preserves all MkDocs settings while updating only the API Reference section
- **Automatic Module Discovery**: Recursively scans Python packages and generates documentation
- **Configurable Exclusions**: Skip test files, private modules, and other unwanted content
- **Multi-Package Support**: Generate docs for multiple packages in one run
- **mkdocstrings Integration**: Seamless integration with mkdocstrings for rich API documentation

#### Configuration:
Edit `scripts/doc_config.yaml` to customize:
- Package paths and descriptions
- Exclusion patterns for files and directories
- Documentation generation options
- mkdocstrings rendering settings

#### Debugging & Troubleshooting 🐛

The documentation generator includes comprehensive debugging capabilities:

**Enable Debug Output:**
```bash
# Method 1: Edit doc_config.yaml
# Set verbose: true and logging.level: "DEBUG"

# Method 2: Temporarily enable verbose logging
# Edit scripts/gen_ref_pages_safe.py line 18:
# logging.basicConfig(level=logging.DEBUG, ...)
```

**Debug Output Features:**
- **Verbose Mode**: Detailed file processing information
- **Statistics Report**: Shows generated/skipped/error file counts
- **Error Details**: Full error messages and stack traces when verbose is enabled
- **Path Resolution**: Debug info about working directory changes
- **Navigation Structure**: Detailed MkDocs config update process

**Generated Debug Files:**
- `docs/reference/SUMMARY.md` - Complete navigation structure for verification
- Console output with processing statistics
- Error file list with detailed failure reasons

**Common Debug Scenarios:**
```bash
# Check what files are being processed
uv run python scripts/gen_ref_pages_safe.py | grep "Generated documentation"

# View skipped files (when verbose=true)
uv run python scripts/gen_ref_pages_safe.py | grep "Skipped file"

# See full error details for failed files
uv run python scripts/gen_ref_pages_safe.py | grep -A 5 "Error files:"

# Debug navigation structure
cat docs/reference/SUMMARY.md
```

**Configuration Debug Options:**
```yaml
# In scripts/doc_config.yaml
generation_options:
  verbose: true                    # Enable detailed logging
  skip_empty_modules: true         # Skip empty Python files
  show_source: true               # Show source links in docs
  
logging:
  level: "DEBUG"                  # DEBUG, INFO, WARNING, ERROR
```

### Configuration ⚙️

- Main config: `mkdocs.yml`
- Theme: Material (`mkdocs-material`)
- API reference: `mkdocstrings[python]` with `griffe-fieldz`
- Source paths for API docs point to `../bridgic-core`
- API generation config: `scripts/doc_config.yaml`

You can tweak navigation, theme options, and docstring rendering in `mkdocs.yml`. Use the generation script configuration for automated API documentation settings.

### Project Layout 🗂️

```
docs/
  ├─ docs/               # Markdown sources (guides, API index, etc.)
  │   ├─ reference/      # Generated API documentation
  │   ├─ home/           # Getting started guides
  │   └─ about/          # Project information
  ├─ scripts/            # Documentation generation tools
  │   ├─ gen_ref_pages_safe.py  # Safe API doc generator
  │   └─ doc_config.yaml        # Generation configuration
  ├─ site/               # Built static site (generated)
  ├─ mkdocs.yml          # MkDocs configuration
  ├─ pyproject.toml      # Docs dependencies
  └─ Makefile            # Helpful shortcuts (serve/build/check/...)
```

### Tips & Troubleshooting 🧭

**General Issues:**
- Port already in use? Change it with `make serve PORT=8001`.
- Stale content or layout? Run `make clean && make build`.
- API pages missing members? Ensure the `../bridgic-core` code is present and importable.
- Need to regenerate API docs? Run `uv run python scripts/gen_ref_pages_safe.py`.
- Configuration issues? Check `scripts/doc_config.yaml` for proper package paths.

**Documentation Generation Issues:**
- **Script fails with "Config file 'mkdocs.yml' does not exist"**: Ensure you're running from the correct directory or use `cd docs && uv run python scripts/gen_ref_pages_safe.py`.
- **Empty or missing documentation pages**: Enable verbose mode and check the "Error files" section in the output.
- **Navigation not updating**: Verify the script completed successfully and check the generated `SUMMARY.md` file.
- **Module import errors**: Ensure the Python packages are properly installed and importable from the script's working directory.

**Debug Commands:**
```bash
# Full debug output
cd docs && uv run python scripts/gen_ref_pages_safe.py 2>&1 | tee debug.log

# Check generated file count
ls -la docs/reference/bridgic-core/bridgic/core/**/*.md | wc -l

# Verify navigation structure
head -20 docs/reference/SUMMARY.md
```

### Deployment 🌐

TODO: Add CICD

### License 📄

This documentation is distributed under the same license as the repository. See the root `LICENSE` file.


