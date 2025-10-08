from typing_extensions import TypeAlias
from enum import Enum
from pathlib import Path
from typing import Union

ZeroToOne: TypeAlias = float

class AutomaType(Enum):
    Fragment = 0
    Graph = 1
    Concurrent = 2
    Sequential = 3
    ReAct = 4

class ArgsMappingRule(Enum):
    """
    Definitions of arguments mapping rules:
    - AS_IS: The arguments for a worker are passed as is from the return values of its predecessors, preserving the order of the dependencies list as specified when the current worker is added. All types of return values, including list/tuple, dict, and single value, are all passed in as is (no unpacking or merging is performed).
    - UNPACK: The return value of the predecessor worker is unpacked and passed as arguments to the current worker. Only valid when the current worker has exactly one dependency and the return value of the predecessor worker is a list/tuple or dict.
    - MERGE: The return values of the predecessor workers are merged and passed as arguments to the current worker. Only valid when the current worker has multiple (at least two) dependencies.
    - SUPPRESSED: The return values of the predecessor worker(s) are NOT passed to the current worker. The current worker has to access the outbuf mechanism to get the output data of its predecessor workers.

    Please refer to test/bridgic/automa/test_automa_args_mapping.py for more details.
    """
    AS_IS = "as_is"
    UNPACK = "unpack"
    MERGE = "merge"
    SUPPRESSED = "suppressed"
