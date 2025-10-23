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
    Extended Jinja2-based prompt template with custom message blocks.
    
    This template implementation extends the standard Jinja2 syntax with custom
    `{% msg %}` blocks to create structured Message objects. It supports both
    single message and multiple message rendering with variable substitution
    and content block parsing.
    
    Attributes
    ----------
    _env_template : Template
        The compiled Jinja2 template object.
    _render_cache : MemoryCache
        Cache for rendered template results to improve performance.
    
    Methods
    -------
    format_message(role, **kwargs)
        Format a single message from the template.
    format_messages(**kwargs)
        Format multiple messages from the template.
    
    Notes
    -----
    This template supports two rendering modes:
    
    1. **Single Message Mode**: Use `format_message()` to render one message.    
    2. **Multiple Messages Mode**: Use `format_messages()` to render multiple messages.
    
    Examples
    --------
    Single message with role in template:
    >>> template = EjinjaPromptTemplate('''
    ... {% msg role="system" %}
    ... You are a helpful assistant. User name: {{ name }}
    ... {% endmsg %}
    ... ''')
    >>> message = template.format_message(name="Alice")
    
    Single message with role as parameter:
    >>> template = EjinjaPromptTemplate("Hello {{ name }}, how are you?")
    >>> message = template.format_message(role="user", name="Bob")
    
    Multiple messages:
    >>> template = EjinjaPromptTemplate('''
    ... {% msg role="system" %}You are helpful{% endmsg %}
    ... {% msg role="user" %}Hello {{ name }}{% endmsg %}
    ... ''')
    >>> messages = template.format_messages(name="Charlie")
    """

    _env_template: Template
    _render_cache: MemoryCache

    def __init__(self, template_str: str):
        """
        Initialize the EjinjaPromptTemplate.
        
        Parameters
        ----------
        template_str : str
            The Jinja2 template string with optional `{% msg %}` blocks.
        """
        super().__init__(template_str=template_str)
        self._env_template = env.from_string(template_str)
        self._render_cache = MemoryCache()

    def format_message(self, role: Union[Role, str] = None, **kwargs) -> Message:
        """
        Format a single message from the template.
        
        Parameters
        ----------
        role : Union[Role, str], optional
            The role of the message. If the template contains a `{% msg %}` block,
            this parameter should be None as the role will be extracted from
            the template. If no `{% msg %}` block exists, this parameter is required.
        **kwargs
            Additional keyword arguments to be substituted into the template.
            
        Returns
        -------
        Message
            A formatted message object with the specified role and content.
            
        Raises
        ------
        PromptSyntaxError
            If the template contains more than one `{% msg %}` block.
        PromptRenderError
            If role parameter conflicts with template-defined role, or if
            no role is specified when template has no `{% msg %}` block.
        """
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
        """
        Format multiple messages from the template.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments to be substituted into the template.
            
        Returns
        -------
        List[Message]
            A list of formatted message objects. Each line of the rendered
            template should be a valid JSON representation of a Message object.
            If no valid messages are found but content exists, a default user
            message is created.
            
        Raises
        ------
        PromptRenderError
            If any line in the rendered template is not a valid JSON
            representation of a Message object.
            
        Notes
        -----
        This method uses caching to improve performance for repeated calls
        with the same parameters. The rendered template is cached based on
        the provided keyword arguments.
        """
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