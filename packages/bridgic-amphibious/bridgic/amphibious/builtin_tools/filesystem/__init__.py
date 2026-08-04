"""Filesystem built-in tools.

These mirror Claude-Code's ``Read`` / ``Write`` / ``Edit`` / ``Glob`` /
``Grep`` capabilities, adapted for ``AmphibiousAutoma``.

The three file-mutating tools (``read_file``, ``write_file``, ``edit_file``)
share a per-agent read-tracker stored on the running agent so that
``write_file`` and ``edit_file`` can refuse to act on a file that has not
been read first or has changed since the last read.

``glob`` and ``grep`` are stateless — they do not touch the agent.
"""

from .read_file import read_file, read_file_tool
from .write_file import write_file, write_file_tool
from .edit_file import edit_file, edit_file_tool
from .glob import glob, glob_tool
from .grep import grep, grep_tool

__all__ = [
    "read_file",
    "read_file_tool",
    "write_file",
    "write_file_tool",
    "edit_file",
    "edit_file_tool",
    "glob",
    "glob_tool",
    "grep",
    "grep_tool",
]
