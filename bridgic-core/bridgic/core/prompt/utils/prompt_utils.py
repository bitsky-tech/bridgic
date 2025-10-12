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
        tool_calls = message.get("tool_calls", None)
        if tool_calls:
            # TODO: from_tool_call need to update to support multiple tool calls & content
            msg = Message.from_tool_call(
                tool_id=tool_calls[0]["id"],
                tool_name=tool_calls[0]["function"]["name"],
                arguments=tool_calls[0]["function"]["arguments"],
            )
        else:
            msg = Message.from_text(message["content"], Role.AI, extras)
        return msg
    elif role == "tool":
        # tool_call_id is a required field
        return Message.from_tool_result(
            tool_id=message["tool_call_id"],
            content=message["content"],
        )
    else:
        raise ValueError(f"Invalid role: `{role}` in message: `{message}`.")
