from enum import Enum
from typing import Annotated, Union, Literal, List, Dict, Any, Optional, Generator, AsyncGenerator
from pydantic import BaseModel, Field

class TextBlock(BaseModel):
    """
    A representation of text data that pass to/from the LLM.
    """
    block_type: Literal["text"] = Field(default="text")
    text: str

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
        # ImageBlock,
        # AudioBlock,
        # DocumentBlock,
    ],
    Field(discriminator="block_type"),
]
