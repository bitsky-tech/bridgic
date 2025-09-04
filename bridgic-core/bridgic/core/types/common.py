from typing_extensions import TypeAlias
from enum import Enum
from pathlib import Path
from typing import Union

ZeroToOne: TypeAlias = float

class LLMOutputFormat(Enum):
    FreeText = 1
    Json = 2

PromptTemplate: TypeAlias = Union[str, Path]
