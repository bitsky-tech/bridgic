import re

from typing import List, Union

from bridgic.core.types.llm_basic import Message, Role
from bridgic.core.types.error import PromptRenderError
from bridgic.core.prompt.base_template import BasePromptTemplate
from bridgic.core.utils.collection import unique_list_in_order

class FstringPromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses f-string to render the prompt.
    """

    def format_message(self, role: Union[Role, str], **kwargs) -> Message:
        if isinstance(role, str):
            role = Role(role)

        all_vars = self._find_variables()
        missing_vars = set(all_vars) - set(kwargs.keys())
        if missing_vars:
            raise PromptRenderError(f"Missing variables that are required to render the prompt template: {', '.join(missing_vars)}")

        rendered = self.template_str.format(**kwargs)
        return Message(role=role, content=rendered)

    def _find_variables(self) -> List[str]:
        var_list = re.findall(r'{([^}]+)}', self.template_str)
        var_list = [var.strip() for var in var_list]
        return unique_list_in_order(var_list)
