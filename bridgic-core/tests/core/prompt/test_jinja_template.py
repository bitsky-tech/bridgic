import pytest

from typing import List

from bridgic.core.prompt._ejinja_template import EjinjaPromptTemplate
from bridgic.core.model.types import Message

@pytest.mark.asyncio
async def test_jinja_format_message():
    template = EjinjaPromptTemplate(template_str="""
{% if name -%}Hello, {{ name }}! {% else -%}Hello, guest! {% endif -%}
{% if fruits -%}
I will introduce to you some fruits and where they come from:
{% for fruit in fruits -%}
- {{ fruit.name }}: {{ fruit.origin | default("Unknown") }}
{% endfor %}
{% else -%}
Which kinds of fruits do you like?
{% endif %}
""")
    message: Message = template.format_message(
        role="user",
        name="John",
        fruits=[
            {"name": "Apple", "origin": "China"},
            {"name": "Kiwi"},
        ],
    )
    assert message.role.value == "user"
    assert ("- Apple: China" in message.blocks[0].text)
    assert ("- Kiwi: Unknown" in message.blocks[0].text)

@pytest.mark.asyncio
async def test_jinja_format_messages():
    template = EjinjaPromptTemplate(template_str="""
{% msg role="system" %}You are a helpful assistant.{% endmsg %}
{% msg role="user" %}{{ query }}{% endmsg %}
""")
    messages: List[Message] = template.format_messages(query="Hi! How are you?")
    assert messages[0].role.value == "system"
    assert messages[0].blocks[0].text == "You are a helpful assistant."
    assert messages[1].role.value == "user"
    assert messages[1].blocks[0].text == "Hi! How are you?"
