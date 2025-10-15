from typing import Optional, Dict, Any, Callable
from typing_extensions import override
from types import MethodType
from bridgic.core.model.types import Tool
from bridgic.core.automa.worker import Worker, CallableWorker
from bridgic.core.agentic.tool_specs._base_tool_spec import ToolSpec
from bridgic.core.utils._json_schema import create_func_params_json_schema
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func
from bridgic.core.agentic.tool_specs._automa_tool_spec import get_tool_description_from

class FunctionToolSpec(ToolSpec):
    _func: Callable
    """The python function to be used as a tool"""

    def __init__(
        self,
        func: Callable,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters
        )
        self._func = func

    @classmethod
    def from_raw(
        cls,
        func: Callable,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
    ) -> "FunctionToolSpec":
        """
        Create a FunctionToolSpec from a python function. By default, the tool name, description and parameters' json schema will be extracted from the function's docstring and the parameters' type and description. However, these values can be customized by passing in the corresponding arguments.

        Parameters
        ----------
        func : Callable
            The python function to create a FunctionToolSpec from.
        tool_name : Optional[str]
            The name of the tool. If not provided, the function name will be used.
        tool_description : Optional[str]
            The description of the tool. If not provided, the function docstring will be used.
        tool_parameters : Optional[Dict[str, Any]]
            The JSON schema of the tool's parameters. If not provided, the JSON schema will be constructed properly from the parameters' annotations, the function's signature and/or docstring.

        Returns
        -------
        FunctionToolSpec
            A new `FunctionToolSpec` object.
        """
        if isinstance(func, MethodType):
            raise ValueError(f"`func` is not allowed to be a bound method: {func}.")

        if not tool_name:
            tool_name = func.__name__
        
        if not tool_description:
            tool_description = get_tool_description_from(func, tool_name)

        if not tool_parameters:
            tool_parameters = create_func_params_json_schema(func)
            # TODO: whether to remove the `title` field of the params_schema?
        
        return cls(
            func=func,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters
        )

    @override
    def to_tool(self) -> Tool:
        """
        Transform this FunctionToolSpec to a `Tool` object used by LLM.

        Returns
        -------
        Tool
            A `Tool` object that can be used by LLM.
        """
        return Tool(
            name=self._tool_name,
            description=self._tool_description,
            parameters=self._tool_parameters
        )

    @override
    def create_worker(self) -> Worker:
        """
        Create a Worker from the information included in this FunctionToolSpec.

        Returns
        -------
        Worker
            A new `Worker` object that can be added to an Automa to execute the tool.
        """
        # TODO: some initialization arguments may be needed in future, e.g., `bound_needed`.
        return CallableWorker(self._func)

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["func"] = self._func.__module__ + "." + self._func.__qualname__
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._func = load_qualified_class_or_func(state_dict["func"])