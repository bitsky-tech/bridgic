from enum import Enum
from typing import Annotated, Union, Literal, List, Dict, Any, Optional, Generator, AsyncGenerator
from pydantic import BaseModel, Field

class TextBlock(BaseModel):
    """
    Encapsulates plain text data that is passed to or received from language models.

    Attributes
    ----------
    block_type : Literal["text"]
        The type identifier for this content block.
    text : str
        The actual text content.
    """
    block_type: Literal["text"] = Field(default="text")
    text: str

class ToolCallBlock(BaseModel):
    """
    Encapsulates tool invocation data that is received from language models.

    Attributes
    ----------
    block_type : Literal["tool_call"]
        The type identifier for this content block.
    id : str
        Unique identifier for the tool call instance.
    name : str
        Name of the tool to be called.
    arguments : Dict[str, Any]
        Parameters to be passed to the tool function.
    """
    block_type: Literal["tool_call"] = Field(default="tool_call")
    id: str = Field(..., description="The ID of the tool call.")
    name: str = Field(..., description="The name of the tool call.")
    arguments: Dict[str, Any] = Field(..., description="The arguments of the tool call.")

class ToolResultBlock(BaseModel):
    """
    Encapsulates the results returned from tool executions.

    Attributes
    ----------
    block_type : Literal["tool_result"]
        The type identifier for this content block.
    id : str
        Unique identifier matching the corresponding tool call.
    content : str
        The result content returned from the tool execution.
    """
    block_type: Literal["tool_result"] = Field(default="tool_result")
    id: str = Field(..., description="The ID of the tool call.")
    content: str = Field(..., description="The result content of the tool call.")

# TODO : Vision modal support.
# class ImageBlock(BaseModel):
#     """
#     A representation of image data that pass to/from the LLM.
#     """
#     block_type: Literal["image"] = Field(default="image")
#     data: bytes = None
#     image_url: str = None
#     image_path: str = None
#     image_mimetype: str = None

#     def to_base64(self) -> str:
#         pass

# TODO : Audio modal support.
# class AudioBlock(BaseModel):
#     """
#     A representation of audio data that pass to/from the LLM.
#     """
#     block_type: Literal["audio"] = Field(default="audio")
#     data: bytes = None
#     audio_url: str = None
#     audio_path: str = None
#     audio_mimetype: str = None

#     def to_base64(self) -> str:
#         pass

# TODO : Document modal support.
# class DocumentBlock(BaseModel):
#     """
#     A representation of document data that pass to the LLM.
#     """
#     block_type: Literal["document"] = Field(default="document")
#     data: bytes = None
#     document_url: str = None
#     document_path: str = None
#     document_title: str = None
#     document_mimetype: str = None

#     def to_base64(self) -> str:
#         pass

ContentBlock = Annotated[
    Union[
        TextBlock,
        ToolCallBlock,
        ToolResultBlock,
        # ImageBlock,
        # AudioBlock,
        # DocumentBlock,
    ],
    Field(discriminator="block_type"),
]
