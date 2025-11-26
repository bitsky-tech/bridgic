"""
MkDocs hook to patch mkdocs-static-i18n so that .ipynb files keep behaving as
documentation pages after mkdocs-jupyter wraps them.
"""

from __future__ import annotations

import types
from pathlib import Path

from mkdocs.plugins import get_plugin_logger

from mkdocs_static_i18n import suffix

log = get_plugin_logger("hooks.i18n_ipynb_fix")

_original_create_i18n_file = getattr(suffix, "_original_create_i18n_file", suffix.create_i18n_file)


def _mark_as_documentation_page(i18n_file):
    """Force the given file to behave like a documentation page."""
    if not hasattr(i18n_file, "_i18n_force_doc"):
        i18n_file.is_documentation_page = types.MethodType(lambda self: True, i18n_file)
        i18n_file._i18n_force_doc = True
        log.debug(f"Forced '{i18n_file.src_uri}' to be handled as a documentation page.")


def create_i18n_file_with_ipynb_support(*args, **kwargs):
    """Wrapper around mkdocs-static-i18n.suffix.create_i18n_file."""
    i18n_file = _original_create_i18n_file(*args, **kwargs)

    if Path(i18n_file.src_uri).suffix.lower() == ".ipynb":
        _mark_as_documentation_page(i18n_file)

    return i18n_file


def on_config(config):
    """Patch mkdocs-static-i18n before the plugins start running."""
    if suffix.create_i18n_file is not create_i18n_file_with_ipynb_support:
        suffix._original_create_i18n_file = _original_create_i18n_file
        suffix.create_i18n_file = create_i18n_file_with_ipynb_support
        log.info("Applied mkdocs-static-i18n .ipynb handling fix.")
    return config

