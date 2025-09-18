from bridgic.core.intelligence.base_llm import Message
from bridgic.core.intelligence.base_llm import Role

def test_message_from_text():
    message = Message(role=Role.USER)
    message.content = "Hello, how are you?"
    assert message == Message.from_text(text="Hello, how are you?", role="user")