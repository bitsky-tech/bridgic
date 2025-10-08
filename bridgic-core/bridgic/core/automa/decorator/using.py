from typing import Callable, Type, Tuple, Dict, List
from bridgic.core.automa.graph_fragment import GraphFragment
from bridgic.core.types.error import AutomaDeclarationError

def using(*fragments: Type[GraphFragment]) -> Callable:
    """
    Decorator for merging and flattening multiple GraphFragment classes into a GraphAutoma.

    Parameters
    ----------
    *fragments : Type[GraphFragment]
        The GraphFragment classes to be merged.

    Returns
    -------
    Callable
        The decorator function.
        
    Examples
    --------
    ```python
    @using(SearchFinancialFragment, SearchMedicalFragment)
    class SynthesizeAutoma(GraphAutoma):
        @worker(
            dependencies=["search_financial_data", "search_medical_data"],
            args_mapping_rule=ArgsMappingRule.AS_IS,
            is_output=True
        )
        def synthesize(self, financial_data: list[str], medical_data: list[str]) -> str:
            pass
    ```
    """
    def decorator(cls: Type) -> Type:
        for fragment in fragments:
            if not issubclass(fragment, GraphFragment):
                raise TypeError(
                    f"All arguments to @using must be GraphFragment subclasses, "
                    f"but got {fragment} instead."
                )

        registered_worker_funcs: Dict[str, Callable] = getattr(cls, "_registered_worker_funcs")
        worker_static_forwards: Dict[str, List[str]] = getattr(cls, "_worker_static_forwards")

        for fragment in fragments:
            for worker_key, worker_func in getattr(fragment, "_registered_worker_funcs").items():
                if worker_key not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_key] = worker_func
                else:
                    raise AutomaDeclarationError(
                        f"worker is defined in multiple fragments: "
                        f"fragment={fragment}, worker={worker_key}"
                    )

            for current, forward_list in getattr(fragment, "_worker_static_forwards").items():
                if current not in worker_static_forwards.keys():
                    worker_static_forwards[current] = []
                worker_static_forwards[current].extend(forward_list)
            
        return cls
    
    return decorator
