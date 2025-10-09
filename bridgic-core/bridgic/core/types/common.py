from typing_extensions import TypeAlias
from enum import Enum
from pathlib import Path
from typing import Union

ZeroToOne: TypeAlias = float

class AutomaType(Enum):
    Graph = 1
    Concurrent = 2
    Sequential = 3
    ReAct = 4
