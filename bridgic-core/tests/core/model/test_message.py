from bridgic.core.model.types import Message
from bridgic.core.model.types import Role

def test_message_from_text():
    message = Message(role=Role.USER)
    message.content = "Hello, how are you?"
    assert message == Message.from_text(text="Hello, how are you?", role="user")

def test_message_from_tool_result():
    message = Message.from_tool_result(
        tool_id="call_weather_123",
        content="The weather in Tokyo is 22°C and sunny."
    )
    assert message.blocks[0].id == "call_weather_123"
    assert message.blocks[0].content == "The weather in Tokyo is 22°C and sunny."
