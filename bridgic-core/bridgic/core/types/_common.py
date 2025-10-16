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

class ArgsMappingRule(Enum):
    """
    Definitions of arguments mapping rules:
    - AS_IS: The arguments mapping is as is from the return values of its predecessor workers, preserving the order of 
    the dependencies when the current worker is added. The types of arguments are preserved as is (no unpacking or 
    merging is performed).
    - UNPACK: The returned value of the predecessor worker is unpacked and passed as arguments to the current worker. 
    Only valid when the current worker has exactly one dependency and the returned value of the predecessor worker 
    is a list/tuple or dict.
    - MERGE: The returned value(s) of the predecessor worker(s) are merged as a single tuple to be the only argument 
    of the current worker.
    - SUPPRESSED: The returned value(s) of the predecessor worker(s) will NOT be passed to the current worker directly. 
    """
    AS_IS = "as_is"
    UNPACK = "unpack"
    MERGE = "merge"
    SUPPRESSED = "suppressed"