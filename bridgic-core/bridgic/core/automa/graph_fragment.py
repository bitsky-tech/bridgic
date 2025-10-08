from typing import ClassVar
from bridgic.core.types.common import AutomaType
from bridgic.core.automa.graph_meta import GraphMeta

class GraphFragment(metaclass=GraphMeta):
    """
    `GraphFragment` is used to define a subgraph of a complete runnable `GraphAutoma`. 
    It allows graph definitions to be developed in a multi-module fashion, with each 
    module focusing solely on the graph structure and internal logic it cares about. 
    Users can then reuse these modules as needed.

    To maximize the reusability of `GraphFragments`, the graph structure declared by a 
    `GraphFragment` only defines the process and order of node execution. Workers 
    defined within it are not allowed to specify the `is_output` option. This allows for 
    flexible composition and inheritance of different `GraphFragments`.

    Examples
    --------
    The following example shows how to use `GraphFragment` to define two basic fragments
    for searching and compose them into a complete executable `GraphAutoma` that supports
    answering questions about both financial and medical topics.

    ```python
    class SearchFinancialFragment(GraphFragment):
        @worker(is_start=True)
        def search_financial_data(self, query: str) -> list[str]:
            pass

    class SearchMedicalFragment(GraphFragment):
        @worker(is_start=True)
        def search_medical_data(self, query: str) -> list[str]:
            pass

    @using(SearchFinancialFragment, SearchMedicalFragment)
    class SynthesizeAutoma(GraphAutoma):
        @worker(
            dependencies=["search_financial_data", "search_medical_data"],
            args_mapping_rule=ArgsMappingRule.AS_IS,
            is_output=True
        )
        def synthesize(self, financial_data: list[str], medical_data: list[str]) -> str:
            pass

    automa_obj = SynthesizeAutoma(name="Financial and Medical Assistant")
    answer = await automa_obj.arun(query="recent changes in the medical industry")
    ```
    """
    # Automa type identifier
    AUTOMA_TYPE: ClassVar[AutomaType] = AutomaType.Fragment
