"""Helper functions for Opik tracing integration."""

import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)


def resolve_child_span_project_name(
    parent_project_name: Optional[str] = None,
    child_project_name: Optional[str] = None,
    show_warning: bool = True,
) -> Optional[str]:
    """
    Resolve the project name for a child span.

    If both parent and child project names are provided and they differ,
    the child project name takes precedence. A warning may be logged
    if show_warning is True.

    Parameters
    ----------
    parent_project_name : Optional[str]
        The project name of the parent span/trace.
    child_project_name : Optional[str]
        The project name for the child span.
    show_warning : bool
        Whether to show a warning if project names differ. Default is True.

    Returns
    -------
    Optional[str]
        The resolved project name for the child span.
    """
    if child_project_name is not None:
        if (
            show_warning
            and parent_project_name is not None
            and parent_project_name != child_project_name
        ):
            LOGGER.warning(
                f"Child span project name '{child_project_name}' differs from "
                f"parent project name '{parent_project_name}'. Using child project name."
            )
        return child_project_name

    return parent_project_name
