from typing_extensions import TypeAlias
from enum import Enum

ZeroToOne: TypeAlias = float

class AutomaType(Enum):
    Graph = 1
    Concurrent = 2
    Sequential = 3
    ReAct = 4

class ArgsMappingRule(Enum):
    """
    Enumeration of Arguments Mapping rules for worker parameter passing.

    ArgsMappingRule defines how the return values from predecessor workers are mapped 
    to the parameters of the current worker. This controls the data flow between workers 
    in an automa execution graph.

    Attributes
    ----------
    AS_IS: Enum
        Preserves the exact order and types of return values from predecessor workers.
        No unpacking or merging is performed.
    UNPACK: Enum
        Unpacks the return value from the predecessor worker and passes as individual 
        arguments. Only valid when the current worker has exactly one dependency and 
        the return value is a list/tuple or dict.
    MERGE: Enum
        Merges all return values from predecessor workers into a single tuple as the 
        only argument of the current worker.
    SUPPRESSED: Enum
        Suppresses all return values from predecessor workers. No arguments are passed 
        to the current worker from its dependencies.

    Examples
    --------
    ```python
    class MyAutoma(GraphAutoma):
        @worker(is_start=True)
        def worker_0(self, user_input: int) -> int:
            return user_input + 1
        
        @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.AS_IS)
        def worker_1(self, worker_0_output: int) -> int:
            # Receives the exact return value from worker_0
            return worker_0_output + 1
        
        @worker(dependencies=["worker_0"], args_mapping_rule=ArgsMappingRule.UNPACK)
        def worker_2(self, user_input: int, result: int) -> int:
            # Unpacks the return value from worker_0 (assuming it returns a tuple)
            return user_input + result
        
        @worker(dependencies=["worker_0", "worker_1"], args_mapping_rule=ArgsMappingRule.MERGE)
        def worker_3(self, all_results: tuple) -> int:
            # Receives all results as a single tuple
            return sum(all_results)
        
        @worker(dependencies=["worker_3"], args_mapping_rule=ArgsMappingRule.SUPPRESSED)
        def worker_4(self, custom_input: int = 10) -> int:
            # Ignores return value from worker_3, uses custom input
            return custom_input + 1
    ```

    Note
    ----
    1. AS_IS is the default mapping rule when not specified
    2. UNPACK requires exactly one dependency and a list/tuple/dict return value
    3. MERGE combines all predecessor outputs into a single tuple argument
    4. SUPPRESSED allows workers to ignore dependency outputs completely
    """
    AS_IS = "as_is"
    UNPACK = "unpack"
    MERGE = "merge"
    SUPPRESSED = "suppressed"