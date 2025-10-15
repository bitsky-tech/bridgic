import re

from typing import List, Union
from jinja2 import Environment, Template, nodes
from jinja2.ext import Extension

from bridgic.core.types._error import PromptRenderError, PromptSyntaxError
from bridgic.core.model.types import Message, Role, ContentBlock, TextBlock
from bridgic.core.prompt._base_template import BasePromptTemplate
from bridgic.core.utils._cache import MemoryCache

SUPPORTED_TYPES = Role.get_all_roles()
CONTENT_BLOCK_REGEX = re.compile(r"(<content_block>\{.*?\}<\/content_block>)|([^<](?:(?!<content_block>)[\s\S])*)")

def _chat_message_from_text(role: str, content: str) -> Message:
    content_blocks: list[ContentBlock] = []

    # Find all content block matches
    matches = CONTENT_BLOCK_REGEX.finditer(content)
    for match in matches:
        if match.group(1):
            # content block match
            content_block_json_str = (
                match.group(1).strip().removeprefix("<content_block>").removesuffix("</content_block>")
            )
            content_blocks.append(ContentBlock.model_validate_json(content_block_json_str))
        elif match.group(2):
            # plain-text match
            text = match.group(2).strip()
            if text:
                content_blocks.append(TextBlock(text=text))

    # If no content blocks were found, treat entire content as text
    if not content_blocks:
        content_blocks.append(TextBlock(text=content))

    final_content = content_blocks
    return Message(role=role, blocks=final_content)

class MsgExtension(Extension):
    """
    `msg` can be used to render prompt text as structured Message objects.

    Example:
        ```
        {% msg role="system" %}
        You are a helpful assistant.
        {% endmsg %}
        ```
    """

    tags = {"msg"}

    def parse(self, parser):
        # We get the line number of the first token for error reporting
        lineno = next(parser.stream).lineno

        # Gather tokens up to the next block_end ('%}')
        gathered = []
        while parser.stream.current.type != "block_end":
            gathered.append(next(parser.stream))

        # If all has gone well, we will have one triplet of tokens:
        #   (type='name, value='role'),
        #   (type='assign', value='='),
        #   (type='string', value='user'),
        # Anything else is a parse error
        error_msg = f"Invalid syntax for chat attribute, got '{gathered}', expected role=\"value\""
        try:
            attr_name, attr_assign, attr_value = gathered  # pylint: disable=unbalanced-tuple-unpacking
        except ValueError:
            raise PromptSyntaxError(error_msg, lineno) from None

        # Validate tag attributes
        if attr_name.value != "role" or attr_assign.value != "=":
            raise PromptSyntaxError(error_msg, lineno)

        if attr_value.value not in SUPPORTED_TYPES:
            types = ", ".join(SUPPORTED_TYPES)
            msg = f"Unknown role type '{attr_value.value}', use one of ({types})"
            raise PromptSyntaxError(msg, lineno)

        # Pass the role name to the CallBlock node
        args: list[nodes.Expr] = [nodes.Const(attr_value.value)]

        # Message body
        body = parser.parse_statements(("name:endmsg",), drop_needle=True)

        # Build messages list
        return nodes.CallBlock(self.call_method("_store_chat_messages", args), [], [], body).set_lineno(lineno)

    def _store_chat_messages(self, role, caller):
        """
        Helper callback.
        """
        cm = _chat_message_from_text(role=role, content=caller())
        return cm.model_dump_json(exclude_none=True) + "\n"

env = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
)
env.add_extension(MsgExtension)

class EjinjaPromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses extended Jinja syntax to render the prompt.
    """

    _env_template: Template
    _render_cache: MemoryCache

    def __init__(self, template_str: str):
        super().__init__(template_str=template_str)
        self._env_template = env.from_string(template_str)
        self._render_cache = MemoryCache()

    def format_message(self, role: Union[Role, str] = None, **kwargs) -> Message:
        if isinstance(role, str):
            role = Role(role)

        rendered = self._env_template.render(**kwargs)
        match_list = re.findall(r"{%\s*msg\s*role=\"(.*?)\"\s*%}(.*?){%\s*endmsg\s*%}", rendered)
        if len(match_list) > 1:
            raise PromptSyntaxError(
                f"It is required to just have one {{% msg %}} block in the template, "
                f"but got {len(match_list)}"
            )
        elif len(match_list) == 1:
            if role is not None:
                raise PromptRenderError(
                    f"If you want to render a single message, the role has to be only specified in the template "
                    f"and not be passed as an argument to the \"format_message\" method in {type(self).__name__}"
                )
            role, content = match_list[0][0], match_list[0][1]
        else:
            if role is None:
                raise PromptRenderError(
                    f"If you want to render a template without {{% msg %}} blocks, the role has to be specified "
                    f"as an argument to the \"format_message\" method in {type(self).__name__}"
                )
            role, content = role, rendered
        return Message.from_text(text=content, role=role)

    def format_messages(self, **kwargs) -> List[Message]:
        rendered = self._render_cache.get(kwargs)
        if not rendered:
            rendered = self._env_template.render(kwargs)
            self._render_cache.set(kwargs, rendered)

        messages: List[Message] = []
        for line in rendered.strip().split("\n"):
            try:
                messages.append(Message.model_validate_json(line))
            except Exception:
                raise PromptRenderError(
                    f"It is required to wrap each content in a {{% msg %}} block when calling the "
                    f"\"format_messages\" method of {type(self).__name__}, but got: {line}"
                )

        if not messages and rendered.strip():
            messages.append(_chat_message_from_text(role="user", content=rendered))
        return messages