# Bridgic Documentation 📚

A modern, collaborative documentation system powered by MkDocs Material with automated API reference generation. Built for developer teams who need reliable, conflict-free documentation workflows.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager


## 🛠️ Development Workflow

### For Content Writers

```bash
# 1. Start development server
make serve-doc

# 2. Edit markdown files in docs/
# 3. View changes at http://127.0.0.1:8000
```

### For Developers Adding API Documentation

```bash
# 1. Add docstrings to your Python code
# 2. Run the generation script
make gen-mkdocs-yml

# 3. Start development server to preview
make serve-doc
```

### Build Static Site

```bash
# Build production-ready static site
make build-doc
```

Output will be in the `site/` directory, ready for deployment.

## 📁 Project Structure

```
docs/
├── docs/                    # Content directory
│   ├── reference/          # Auto-generated API docs (don't edit manually)
    ...
├── scripts/                # Documentation tools
│   ├── mkdocs_template.yml # Template for mkdocs.yml (edit this for config changes)
│   ├── gen_mkdocs_yml.py   # API documentation generator
│   └── doc_config.yaml     # Generation configuration
├── mkdocs.yml              # Generated MkDocs config (auto-generated)
├── site/                   # Built static site (generated)
├── pyproject.toml          # Python dependencies
└── Makefile               # Development commands
```

## ⚙️ Configuration

### Template System

The documentation uses a template-based system to avoid merge conflicts:

- **Edit**: `scripts/mkdocs_template.yml` for static configuration
- **Generated**: `mkdocs.yml` is auto-generated (don't edit manually)
- **Benefit**: Team members can edit configuration without conflicts

### API Documentation

The system automatically generates API documentation from:
- Python docstrings (NumPy style recommended)
- `__all__` exports in `__init__.py` files
- Module structure and imports

## 🚀 Deployment

### Static Site Hosting

```bash
# Build for production
make build-doc

# Deploy site/ directory to your hosting service
# Examples: GitHub Pages, Netlify, Vercel, etc.
```

## 🤝 Contributing

### For Documentation Writers

1. Edit markdown files in `docs/`
2. Test with `make serve-doc`
3. Submit pull request

### For Developers

1. Add docstrings to Python code
2. Run `make gen-mkdocs-yml` to update API docs
3. Test with `make serve-doc`
4. Submit pull request

### Avoiding Conflicts

- **Don't edit** `mkdocs.yml` directly
- **Do edit** `scripts/mkdocs_template.yml` for configuration changes
- **Always run** `make gen-mkdocs-yml` after template changes

## 📄 License

This documentation is distributed under the same license as the repository. See the root `LICENSE` file.

