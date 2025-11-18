from typing import Dict, List, Callable, _ProtocolMeta
from collections import defaultdict, deque

from bridgic.core.automa.worker._worker_decorator import packup_worker_decorator_rumtime_args, get_worker_decorator_default_paramap
from bridgic.core.automa.worker._worker_callback import WorkerCallbackBuilder
from bridgic.core.types._common import AutomaType
from bridgic.core.types._error import AutomaDeclarationError, AutomaCompilationError

class GraphMeta(_ProtocolMeta):
    """
    This metaclass is used to:
    - Correctly handle worker registration during the class-level definition of `GraphAutoma` subclass.
    - Maintain static edge relationships (dependencies) across the entire graph structure, while verifying 
    that the entire static graph structure satisfies the DAG constraint.
    """

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        # Inherit the graph structure from the parent classes and maintain the related data structures.
        registered_worker_funcs: Dict[str, Callable] = {}
        worker_static_forwards: Dict[str, List[str]] = {}
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(
                    cls, 
                    cls.AUTOMA_TYPE, 
                    worker_kwargs
                )
                default_paramap = get_worker_decorator_default_paramap(AutomaType.Graph)
                func = attr_value

                setattr(func, "__is_worker__", True)
                setattr(func, "__worker_key__", complete_args.get("key", default_paramap["key"]))
                setattr(func, "__dependencies__", complete_args.get("dependencies", default_paramap["dependencies"]))
                setattr(func, "__is_start__", complete_args.get("is_start", default_paramap["is_start"]))
                setattr(func, "__is_output__", complete_args.get("is_output", default_paramap["is_output"]))
                setattr(func, "__args_mapping_rule__", complete_args.get("args_mapping_rule", default_paramap["args_mapping_rule"]))

                callback_builders = complete_args.get("callback_builders", default_paramap["callback_builders"])
                if not callback_builders:
                    callback_builders = []
                else:
                    for cb in callback_builders:
                        if not isinstance(cb, WorkerCallbackBuilder):
                            raise TypeError(
                                f"Expected WorkerCallbackBuilder for callback_builders, got {type(cb)}. "
                                f"Use WorkerCallbackBuilder(callback_type, init_kwargs) instead of callback instances."
                            )
                setattr(func, "__callback_builders__", callback_builders)
        
        for attr_name, attr_value in dct.items():
            # Attributes with __is_worker__ will be registered as workers.
            if hasattr(attr_value, "__is_worker__"):
                worker_key = getattr(attr_value, "__worker_key__", None) or attr_name
                dependencies = list(set(attr_value.__dependencies__))

                # Update the registered workers for current class.
                if worker_key not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_key] = attr_value
                else:
                    raise AutomaDeclarationError(
                        f"Duplicate worker keys are not allowed: "
                        f"worker={worker_key}"
                    )

                # Update the table of static forwards.
                for trigger in dependencies:
                    if trigger not in worker_static_forwards.keys():
                        worker_static_forwards[trigger] = []
                    worker_static_forwards[trigger].append(worker_key)

        # Validate if the DAG constraint is met.
        # TODO: this is indeed a chance to detect risks. Add more checks here or remove totally!
        mcls.validate_dag_constraints(worker_static_forwards)

        setattr(cls, "_registered_worker_funcs", registered_worker_funcs)
        return cls
    
    @classmethod
    def validate_dag_constraints(mcls, forward_dict: Dict[str, List[str]]):
        """
        Use Kahn's algorithm to check if the input graph described by the forward_dict satisfies
        the DAG constraints. If the graph doesn't meet the DAG constraints, AutomaDeclarationError will be raised. 

        More about [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
        could be read from the link.

        Parameters
        ----------
        forward_dict : Dict[str, List[str]]
            A dictionary that describes the graph structure. The keys are the nodes, and the values 
            are the lists of nodes that are directly reachable from the keys.

        Raises
        ------
        AutomaDeclarationError
            If the graph doesn't meet the DAG constraints.
        """
        # 1. Initialize the in-degree.
        in_degree = defaultdict(int)
        for current, target_list in forward_dict.items():
            for target in target_list:
                in_degree[target] += 1

        # 2. Create a queue of workers with in-degree 0.
        queue = deque([node for node in forward_dict.keys() if in_degree[node] == 0])

        # 3. Continuously pop workers from the queue and update the in-degree of their targets.
        while queue:
            node = queue.popleft()
            for target in forward_dict.get(node, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        # 4. If the in-degree were all 0, then the graph meets the DAG constraints.
        if not all([in_degree[node] == 0 for node in in_degree.keys()]):
            nodes_in_cycle = [node for node in forward_dict.keys() if in_degree[node] != 0]
            raise AutomaCompilationError(
                f"the graph automa does not meet the DAG constraints, because the "
                f"following workers are in cycle: {nodes_in_cycle}"
            )