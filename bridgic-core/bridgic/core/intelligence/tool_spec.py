from typing import Optional, Union, Dict, List, Any, Type, Callable, Annotated, get_origin
from typing_extensions import override
from abc import abstractmethod
from types import MethodType
from docstring_parser import parse as parse_docstring # type: ignore
import inspect
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.types.serialization import Serializable
from bridgic.core.intelligence.protocol import Tool
from bridgic.core.automa.automa import Automa
from bridgic.core.automa.worker import Worker, CallableWorker
from bridgic.core.utils.json_schema import create_func_params_json_schema
from bridgic.core.utils.inspect_tools import load_qualified_class_or_func

class ToolSpec(Serializable):
    """
    ToolSpec is an abstract class that represents a tool specification that describes all necessary information about a tool used by the LLM. 

    ToolSpec and its subclasses are responsible for providing four categories of interfaces:
    1. Transformations to LLM Tool: `to_tool`.
    2. Worker Creation: `create_worker`.
    3. Serialization and Deserialization.
    4. ToolSpec initialization from raw resources: `from_raw`.
    """
    _tool_id: Optional[Union[str, int]]
    """The unique ID of the tool, used to uniquely identify a tool across the entire system. This tool can be of various types."""

    def __init__(self):
        self._tool_id = None

    ###############################################################
    ######## Part One of interfaces: Transformations to Tool ######
    ###############################################################

    @abstractmethod
    def to_tool(self) -> Tool:
        """
        Transform this ToolSpec to a `Tool` object used by LLM.

        Returns
        -------
        Tool
            A `Tool` object that can be used by LLM.
        """
        ...

    ###############################################################
    ######## Part Two of interfaces: Worker Creation ##############
    ###############################################################

    @abstractmethod
    def create_worker(self) -> Worker:
        """
        Create a Worker from the information included in this ToolSpec.

        Returns
        -------
        Worker
            A new `Worker` object that can be added to an Automa to execute the tool.
        """
        ...

    ###############################################################
    ######## Part Three of interfaces: 
    ######## Serialization and Deserialization ####################
    ###############################################################

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        if self._tool_id:
            state_dict["tool_id"] = self._tool_id
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        self._tool_id = state_dict.get("tool_id")

    ###############################################################
    ######## Part Four of interfaces: 
    ######## ToolSpec initialization from raw resources ###########
    ######## `from(...)`: See subclasses for details ##############
    ###############################################################

class FunctionToolSpec(ToolSpec):
    _func: Callable
    _tool_name: Optional[str]
    """The name of the tool to be called"""
    _tool_description: Optional[str]
    """A description of what the tool does, used by the model to choose when and how to call the tool."""
    _tool_parameters: Optional[Dict[str, Any]]
    """The JSON schema of the tool's parameters"""

    def __init__(
        self,
        func: Callable,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._func = func
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._tool_parameters = tool_parameters

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
            docstring = parse_docstring(func.__doc__)
            tool_description = docstring.description
            if tool_description:
                tool_description = tool_description.strip()
        if not tool_description:
            # No description provided, use the function signature as the description.
            fn_sig = inspect.signature(func)
            filtered_params = []
            ignore_params: List[str] = ["self", "cls"]
            for param_name, param in fn_sig.parameters.items():
                if param_name in ignore_params:
                    continue

                # Resolve the original type of the parameter.
                param_type = param.annotation
                if get_origin(param_type) is Annotated:
                    param_type = param_type.__origin__

                # Note: Remove the default of the parameter.
                filtered_params.append(param.replace(
                    annotation=param_type,
                    default=inspect.Parameter.empty
                ))

            fn_sig = fn_sig.replace(parameters=filtered_params)
            tool_description = f"{tool_name}{fn_sig}\n"

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
        if self._tool_name:
            state_dict["tool_name"] = self._tool_name
        if self._tool_description:
            state_dict["tool_description"] = self._tool_description
        if self._tool_parameters:
            state_dict["tool_parameters"] = self._tool_parameters
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._func = load_qualified_class_or_func(state_dict["func"])
        self._tool_name = state_dict.get("tool_name")
        self._tool_description = state_dict.get("tool_description")
        self._tool_parameters = state_dict.get("tool_parameters")

class AutomaToolSpec(ToolSpec):
    _automa_cls: Type[Automa]
    _tool_name: Optional[str]
    """The name of the tool to be called"""
    _tool_description: Optional[str]
    """A description of what the tool does, used by the model to choose when and how to call the tool."""
    _tool_parameters: Optional[Dict[str, Any]]
    """The JSON schema of the tool's parameters"""
    # initialization arguments
    _automa_name: Optional[str]
    _automa_thread_pool: Optional[ThreadPoolExecutor]
    _init_kwargs: Dict[str, Any]

    def __init__(
        self,
        automa_cls: Type[Automa],
        tool_name: Optional[str],
        tool_description: Optional[str],
        tool_parameters: Optional[Dict[str, Any]],
        *,
        automa_name: Optional[str] = None,
        automa_thread_pool: Optional[ThreadPoolExecutor] = None,
        **init_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self._automa_cls = automa_cls
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._tool_parameters = tool_parameters
        self._automa_name = automa_name
        self._automa_thread_pool = automa_thread_pool
        self._init_kwargs = init_kwargs

    @classmethod
    def from_raw(
        cls,
        automa_cls: Type[Automa],
        tool_name: Optional[str],
        tool_description: Optional[str],
        tool_parameters: Optional[Dict[str, Any]],
        *,
        automa_name: Optional[str] = None,
        automa_thread_pool: Optional[ThreadPoolExecutor] = None,
        **init_kwargs: Dict[str, Any],
    ) -> "AutomaToolSpec":
        """
        Create an AutomaToolSpec from an Automa class.
        """
        # TODO: how to automatically extract tool metadata from the automa class?
        return cls(
            automa_cls=automa_cls,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters,
            automa_name=automa_name,
            automa_thread_pool=automa_thread_pool,
            **init_kwargs
        )

    @override
    def to_tool(self) -> Tool:
        """
        Transform this AutomaToolSpec to a `Tool` object used by LLM.

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
        Create a Worker from the information included in this AutomaToolSpec.

        Returns
        -------
        Worker
            A new `Worker` object that can be added to an Automa to execute the tool.
        """
        return self._automa_cls(
            name=self._automa_name,
            thread_pool=self._automa_thread_pool,
            **self._init_kwargs
        )

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["automa_cls"] = self._automa_cls.__module__ + "." + self._automa_cls.__qualname__
        if self._tool_name:
            state_dict["tool_name"] = self._tool_name
        if self._tool_description:
            state_dict["tool_description"] = self._tool_description
        if self._tool_parameters:
            state_dict["tool_parameters"] = self._tool_parameters
        if self._automa_name:
            state_dict["automa_name"] = self._automa_name
        if self._automa_thread_pool:
            state_dict["automa_thread_pool"] = self._automa_thread_pool
        if self._init_kwargs:
            state_dict["init_kwargs"] = self._init_kwargs
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._automa_cls = load_qualified_class_or_func(state_dict["automa_cls"])
        self._tool_name = state_dict.get("tool_name")
        self._tool_description = state_dict.get("tool_description")
        self._tool_parameters = state_dict.get("tool_parameters")
        self._automa_name = state_dict.get("automa_name")
        self._automa_thread_pool = state_dict.get("automa_thread_pool")
        self._init_kwargs = state_dict.get("init_kwargs") or {}
