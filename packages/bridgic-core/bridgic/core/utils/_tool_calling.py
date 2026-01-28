from typing import Any, List
import uuid


def generate_tool_id() -> str:
    """
    Generate a unique tool ID and make sure its length is no more than 30 characters.
    """
    return f"tool_{uuid.uuid4().hex[:25]}"

def stringify_tool_result(tool_result: Any, verbose: bool = True) -> str:
    """
    Convert a tool result to a string representation for logging purposes.

    Parameters
    ----------
    tool_result : Any
        The tool result to stringify. Can be any type, but if it's an MCP CallToolResult with
        a content attribute and verbose=True, it will be specially formatted.
    verbose : bool, default=True
        If True, provides detailed formated content extraction.
        If False, returns a simple truncated string representation.

    Returns
    -------
    str
        A string representation of the tool result, suitable for logging.
    """
    if not hasattr(tool_result, "content") or not verbose:
        return str(tool_result)[:1000] + "..." if len(str(tool_result)) > 1000 else str(tool_result)

    try:
        from mcp import types
        _mcp_installed = True
    except ImportError:
        _mcp_installed = False

    if not _mcp_installed:
        return str(tool_result)

    from mcp.types import CallToolResult

    if not isinstance(tool_result, CallToolResult):
        return str(tool_result)

    content = tool_result.content
    if not isinstance(content, List):
        return str(tool_result)

    content_str = ""
    for idx, item in enumerate(content):
        if idx > 0:
            content_str += "\n"
        match item.type:
            case "text":
                content_str += f"[{idx}]-[{item.type}]:\n\n" + item.text
            case "image":
                content_str += f"[{idx}]-[{item.type}]-{item.mimeType}:\n\n" + item.data
            case "audio":
                content_str += f"[{idx}]-[{item.type}]-{item.mimeType}:\n\n" + item.data
            case _:
                content_str += str(item)

    return content_str
