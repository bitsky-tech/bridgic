from bridgic.core.intelligence.base_llm import Message, Role
from bridgic.core.prompt.chat_message import ChatMessage

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
        # TODO: handle tool_calls; choose between content and tool_calls
        return Message.from_text(message["content"], Role.AI, extras)
    elif role == "tool":
        # tool_call_id is required
        extras["tool_call_id"] = message["tool_call_id"]
        return Message.from_text(message["content"], Role.TOOL, extras)
    else:
        raise ValueError(f"Invalid role: `{role}` in message: `{message}`.")
