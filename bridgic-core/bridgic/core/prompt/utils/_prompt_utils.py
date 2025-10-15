from bridgic.core.model.types import Message, Role
from bridgic.core.agentic.types._chat_message import ChatMessage
from bridgic.core.model.types import ToolCallBlock

def transform_chat_message_to_llm_message(message: ChatMessage) -> Message:
    """
    Transform a `ChatMessage` to a `Message` expected by the LLM.
    """
    role = message["role"]
    extras = {}
    if role == "system":
        name = message.get("name", None)
        if name:
            extras["name"] = name
        return Message.from_text(message["content"], Role.SYSTEM, extras)
    elif role == "user":
        name = message.get("name", None)
        if name:
            extras["name"] = name
        return Message.from_text(message["content"], Role.USER, extras)
    elif role == "assistant":
        name = message.get("name", None)
        if name:
            extras["name"] = name
        tool_calls = message.get("tool_calls", None)
        if tool_calls:
            tool_call_blocks = [ToolCallBlock(
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                arguments=tool_call["function"]["arguments"],
            ) for tool_call in tool_calls]
        else:
            tool_call_blocks = []
        return Message.from_tool_call(
            text=message.get("content", None),
            tool_calls=tool_call_blocks,
        )
    elif role == "tool":
        # tool_call_id is a required field
        return Message.from_tool_result(
            tool_id=message["tool_call_id"],
            content=message["content"],
        )
    else:
        raise ValueError(f"Invalid role: `{role}` in message: `{message}`.")
