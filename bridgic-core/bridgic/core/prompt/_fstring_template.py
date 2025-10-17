import re

from typing import List, Union

from bridgic.core.model.types import Message, Role
from bridgic.core.types._error import PromptRenderError
from bridgic.core.prompt._base_template import BasePromptTemplate
from bridgic.core.utils._collection import unique_list_in_order

class FstringPromptTemplate(BasePromptTemplate):
    """    
    This template implementation uses Python's f-string syntax (braces `{}`).
    
    Methods
    -------
    format_message(role, **kwargs)
        Format a single message from the template.
    
    Notes
    -----
    This template supports single message rendering via `format_message()`.
    The template uses Python's built-in `str.format()` method for variable
    substitution, which provides basic formatting capabilities.
    
    Examples
    --------
    Basic usage:
    >>> template = FstringPromptTemplate("Hello {name}, you are {age} years old.")
    >>> message = template.format_message(role="user", name="Alice", age=25)
    
    With context:
    >>> template = FstringPromptTemplate('''
    ... Context: {context}
    ... Question: {question}
    ... Please provide a helpful answer.
    ... ''')
    >>> message = template.format_message(
    ...     role="system", 
    ...     context="Python programming", 
    ...     question="What is a decorator?"
    ... )
    
    Multiple variables:
    >>> template = FstringPromptTemplate("{greeting} {name}! Today is {date}.")
    >>> message = template.format_message(
    ...     role="assistant",
    ...     greeting="Good morning",
    ...     name="Bob", 
    ...     date="Monday"
    ... )
    """

    def format_message(self, role: Union[Role, str], **kwargs) -> Message:
        """
        Format a single message from the template.
        
        Parameters
        ----------
        role : Union[Role, str]
            The role of the message (e.g., 'user', 'assistant', 'system').
            Required parameter for this template implementation.
        **kwargs
            Keyword arguments containing values for all variables referenced
            in the template string. All variables must be provided.
            
        Returns
        -------
        Message
            A formatted message object with the specified role and rendered content.
            
        Raises
        ------
        PromptRenderError
            If any variables referenced in the template are missing from
            the provided keyword arguments.
        """
        if isinstance(role, str):
            role = Role(role)

        all_vars = self._find_variables()
        missing_vars = set(all_vars) - set(kwargs.keys())
        if missing_vars:
            raise PromptRenderError(f"Missing variables that are required to render the prompt template: {', '.join(missing_vars)}")

        rendered = self.template_str.format(**kwargs)
        return Message.from_text(text=rendered, role=role)

    def _find_variables(self) -> List[str]:
        """
        Extract variable names from the template string.
        
        Returns
        -------
        List[str]
            A list of unique variable names found in the template string,
            in the order they first appear. Variable names are extracted
            from curly brace syntax `{variable_name}`.
        """
        var_list = re.findall(r'{([^}]+)}', self.template_str)
        var_list = [var.strip() for var in var_list]
        return unique_list_in_order(var_list)
