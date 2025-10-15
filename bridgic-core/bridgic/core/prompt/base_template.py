from typing import List, Union
from pydantic import BaseModel, Field

from bridgic.core.model.base_llm import Message, Role

class BasePromptTemplate(BaseModel):
    """
    Abstract class for prompt templates.
    """

    template_str: str
    """
    The template string.
    """

    def format_message(self, role: Union[Role, str] = Role.USER, **kwargs) -> Message:
        raise NotImplementedError(f"format_message is not implemented in class {self.__class__.__name__}")

    def format_messages(self, **kwargs) -> List[Message]:
        raise NotImplementedError(f"format_messages is not implemented in class {self.__class__.__name__}")