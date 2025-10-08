from typing import Dict, List, Callable, Type, Union, Any, ClassVar, _ProtocolMeta
from bridgic.core.types.error import *
from bridgic.core.utils.graph_tools import validate_dag_constraints
from bridgic.core.types.common import AutomaType
from bridgic.core.automa.worker_decorator import (
    packup_worker_decor_runtime_args,
    get_worker_decor_default_params,
)

class GraphMeta(_ProtocolMeta):
    """
    This metaclass is used to:
    1. Correctly handle graph-fragment inheritance, ensuring that subgraph structures
    defined by base classes and those defined by subclass coexist on the same plane, 
    allowing workers within the subgraph to reference each other.
    2. Maintain static edge relationships (dependencies) across the entire graph structure, 
    while verifying that the entire static graph structure satisfies the DAG constraint.
    """

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)
        automa_type = getattr(cls, 'AUTOMA_TYPE')
        inherit_level = mcls.decide_inherit_level(cls)

        # Maintain the necessary data structures for declaring the static graph.
        registered_worker_funcs: Dict[str, Callable] = {}
        worker_static_forwards: Dict[str, List[str]] = {}

        # Decorate all worker methods.
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                # According to automa_type, validate input arguments and decorate worker methods.
                complete_args = packup_worker_decor_runtime_args(cls, automa_type, worker_kwargs)
                default_params = get_worker_decor_default_params(AutomaType.Graph)
                func = attr_value
                setattr(func, "__is_worker__", True)
                setattr(func, "__worker_key__", complete_args.get("key", default_params["key"]))
                setattr(func, "__dependencies__", complete_args.get("dependencies", default_params["dependencies"]))
                setattr(func, "__is_start__", complete_args.get("is_start", default_params["is_start"]))
                setattr(func, "__is_output__", complete_args.get("is_output", default_params["is_output"]))
                setattr(func, "__args_mapping_rule__", complete_args.get("args_mapping_rule", default_params["args_mapping_rule"]))

        # Inherit the graph structure from the parent classes.
        for base in bases:
            for worker_key, worker_func in getattr(base, "_registered_worker_funcs", {}).items():
                if worker_key not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_key] = worker_func
                else:
                    raise AutomaDeclarationError(
                        f"worker is defined in multiple base classes: "
                        f"base={base}, worker={worker_key}"
                    )

            for current, forward_list in getattr(base, "_worker_static_forwards", {}).items():
                if current not in worker_static_forwards.keys():
                    worker_static_forwards[current] = []
                worker_static_forwards[current].extend(forward_list)

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
        validate_dag_constraints(worker_static_forwards)

        setattr(cls, "_registered_worker_funcs", registered_worker_funcs)
        setattr(cls, "_worker_static_forwards", worker_static_forwards)
        return cls

    @classmethod
    def decide_inherit_level(mcls, cls: Type) -> int:
        # Avoid direct import to prevent ImportError during class initialization.
        rev_mro = list(reversed(cls.mro()))
        graph_automa_cls = None
        for c in rev_mro:
            if getattr(c, "__name__", None) == "GraphAutoma":
                graph_automa_cls = c
                break
        if graph_automa_cls is not None:
            graph_automa_idx = rev_mro.index(graph_automa_cls)
            cls_idx = rev_mro.index(cls) - graph_automa_idx
        else:
            cls_idx = -1
        return cls_idx