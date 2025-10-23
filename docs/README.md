# Bridgic Documentation 📚

A modern, collaborative documentation workflow powered by MkDocs with automated API reference generation.

## Prerequisites 🚀

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager


## Development Workflow 🛠

### For general document writer

1. Start development server by execute `make serve-doc`.
2. Edit markdown files in docs/ and view the real-time changes at http://127.0.0.1:8000. 

### For API developer

1. Add docstrings to your Python code.
2. Run the generation script `make gen-mkdocs-yml`.
3. Start development server to preview by execute `make serve-doc`.

### For online deployer

1. Build production-ready site static files by execute `make build-doc`.
2. Start development server to preview by execute `make serve-doc`.

## Project Structure 📁

```
docs/
├── docs/                   # Content directory
│   ├── reference/          # Auto-generated API docs (don't edit manually)
│   └── ...
├── scripts/                # Documentation tools
│   ├── mkdocs_template.yml # Template for mkdocs.yml (edit this for config changes)
│   ├── gen_mkdocs_yml.py   # API documentation generator
│   └── doc_config.yaml     # Generation configuration
├── mkdocs.yml              # Generated MkDocs config (auto-generated)
├── site/                   # Built static site (generated)
├── pyproject.toml          # Python dependencies
└── Makefile                # Development commands
```

## Configuration ⚙

### Template System

The documentation uses a template-based system to avoid merge conflicts:

- **Edit**: `scripts/mkdocs_template.yml` for static configuration
- **Generated**: `mkdocs.yml` is auto-generated (don't edit manually)
- **Benefit**: Team members can edit configuration without conflicts

### API Documentation

The system automatically generates API documentation from:
- Numpy style Python docstrings
- `__all__` exports in `__init__.py` files
- Module structure and imports

## Deployment 🚀

### Static Site Hosting

```bash
# Build for production
make build-doc

# Deploy site/ directory to your hosting service
# Examples: GitHub Pages, Netlify, Vercel, etc.
```

### Avoiding Conflicts

- **Don't edit** `mkdocs.yml` directly
- **Do edit** `scripts/mkdocs_template.yml` for configuration changes
- **Always run** `make gen-mkdocs-yml` after template changes
