# Bridgic Skills

Self-contained capability modules that Claude (or any skill-aware agent) loads on demand. Each skill bundles a `SKILL.md` (what the skill does and when to use it), supporting references, and a dependency-bootstrap script — so a caller can install everything the skill needs into their own uv-managed project with a single command.

## Available skills

| Skill | Purpose |
|-------|---------|
| [`bridgic-amphibious`](./bridgic-amphibious) | Dual-mode agent framework — combines LLM-driven (`on_agent`) and deterministic (`on_workflow`) execution with automatic fallback and human-in-the-loop support. |
| [`bridgic-llms`](./bridgic-llms) | LLM provider initialization — `bridgic-llms-openai`, `bridgic-llms-openai-like`, `bridgic-llms-vllm`, exposed through a unified `BaseLlm` + protocol interface. |

Each skill's own `SKILL.md` is the authoritative source for its API, quickstart, and usage boundaries.

## Directory layout

Every skill follows the same structure:

```
skills/<skill-name>/
├── SKILL.md                 # purpose, quickstart, when-to-use (read by Claude)
├── references/              # extra docs, examples, templates
└── scripts/
    ├── install-deps.sh      # dependency bootstrap (shared grammar across skills)
    ├── deps.ini             # single-variant skill
    └── deps.<variant>.ini   # one file per variant (e.g. LLM provider)
```

`install-deps.sh` and the `deps*.ini` format are identical across skills, so once you understand one you understand all of them.

## Running a skill's installer

```bash
# single-variant skill (bridgic-amphibious)
bash skills/<skill>/scripts/install-deps.sh <project-dir>

# multi-variant skill (bridgic-llms — default variant is "openai")
bash skills/<skill>/scripts/install-deps.sh <project-dir> <variant>
```

The script will: auto-install `uv` if missing → `uv init --bare` if `<project-dir>` has no `pyproject.toml` → `uv add` any missing packages and re-pin those with an explicit `version` → `uv sync`.

Outcome markers (grep-friendly):

- Success: `=== DEPS_READY (...) ===`
- Failure: `=== DEPS_FAILED reason=<label> exit=<N> ===`

When the script exits 0 the target project is fully initialized — no follow-up `uv add` / `uv sync` required.

## `deps.ini` format

Per-variant dependency manifest. Standard INI:

- `[section]` — section name **is** the pip package name
- `key = value` inside a section — only `source` and `version` are recognized
- Comments: full-line `#` or `;`; blank lines OK
- Values may be quoted with `"..."` (quotes are stripped)
- Unknown key, key outside a section, or malformed line → fatal (exit 5)

### Fields

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `source` | no | `"default"` | Must be `"default"` (public PyPI) for anything shipped to `main`. |
| `version` | no | (none) | PEP 440 specifier, e.g. `">=1.0,<2.0"` or `"==0.5.2"`. If set, the package is **re-pinned to this constraint on every run**; if unset, install latest only when missing. |

### Example

```ini
# Public PyPI, latest
[bridgic-core]
source = "default"

# Public PyPI, pinned
[python-dotenv]
source = "default"
version = ">=1.0,<2.0"
```

## CI gate

The `Skill Install-Deps Check` workflow runs on every PR targeting `main` and, for each `deps*.ini` found under `skills/*/scripts/`:

1. **Policy guard** — rejects any section whose `source` is not `"default"`, listing every offender with a clear message.
2. **Live bootstrap** — runs `install-deps.sh` in a fresh `mktemp -d` project dir and asserts `=== DEPS_READY` reaches stdout. This validates INI syntax, package name correctness, and version resolvability against public PyPI.

The matrix is discovered dynamically by scanning `skills/*/scripts/deps*.ini` — adding a new skill or variant needs **no** workflow change.

## Exit codes

| Code | Reason |
|------|--------|
| 1 | `uv` not installed or not on PATH after install attempt |
| 2 | `uv init` failed |
| 3 | `uv add` failed |
| 4 | `uv sync` failed |
| 5 | `deps.ini` missing, malformed, or empty |

## Not supported

- `extras` (e.g. `pkg[openai]`) — workaround: depend on the extras-bearing meta-package directly
- markers / environment conditions
