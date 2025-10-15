import pytest

from bridgic.core.prompt.fstring_template import FstringPromptTemplate
from bridgic.core.model.base_llm import Message
from bridgic.core.types.error import PromptRenderError

TEMPLATE_STR = "Hello, {name}! Here I will introduce a project named {project} to you."

@pytest.mark.asyncio
async def test_fstring_format_message():
    template = FstringPromptTemplate(template_str=TEMPLATE_STR)
    assert template._find_variables() == ["name", "project"]

    message = template.format_message(role="user", name="John", project="Bridgic")
    assert message == Message.from_text(
        role="user",
        text=TEMPLATE_STR.format(name="John", project="Bridgic"),
    )

@pytest.mark.asyncio
async def test_fstring_format_message_missing_vars():
    template = FstringPromptTemplate(template_str=TEMPLATE_STR)
    with pytest.raises(PromptRenderError):
        template.format_message(role="user", name="John")