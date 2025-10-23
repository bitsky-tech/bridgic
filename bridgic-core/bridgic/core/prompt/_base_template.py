from typing import List, Union
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message, Role

class BasePromptTemplate(BaseModel):
    """
    Abstract base class for prompt templates.
    
    This class provides a common interface for messages from template strings with variable substitutions.    
    
    Attributes
    ----------
    template_str : str
        The template string containing placeholders for variable substitution.
        The specific placeholder syntax depends on the concrete implementation
        (e.g., f-string, Jinja2, etc.).
    
    Methods
    -------
    format_message(role, **kwargs)
        Format a single message from the template.
    format_messages(**kwargs)
        Format multiple messages from the template.
    
    Notes
    -----
    This is an abstract base class that must be subclassed to provide
    concrete implementations. Subclasses should implement the `format_message`
    and `format_messages` methods according to their specific template
    formatting requirements.
    
    Examples
    --------
    >>> class MyTemplate(BasePromptTemplate):
    ...     def format_message(self, role=Role.USER, **kwargs):
    ...         # Implementation here
    ...         pass
    ...     
    ...     def format_messages(self, **kwargs):
    ...         # Implementation here
    ...         pass
    >>> 
    >>> template = MyTemplate(template_str="Hello {name}!")
    >>> message = template.format_message(name="World")
    """

    template_str: str

    def format_message(self, role: Union[Role, str] = Role.USER, **kwargs) -> Message:
        """
        Format a single message from the template.
        
        Parameters
        ----------
        role : Union[Role, str], default=Role.USER
            The role of the message (e.g., 'user', 'assistant', 'system').
        **kwargs
            Additional keyword arguments to be substituted into the template.
            
        Returns
        -------
        Message
            A formatted message object.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError(f"format_message is not implemented in class {self.__class__.__name__}")

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
            A list of formatted message objects.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError(f"format_messages is not implemented in class {self.__class__.__name__}")