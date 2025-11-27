from typing import List, Union
from pydantic import BaseModel, Field

from bridgic.core.model.types import Message, Role

class BasePromptTemplate(BaseModel):
    """
    Abstract base class for prompt templates.
    """

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