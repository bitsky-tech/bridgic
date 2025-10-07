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

# TODO:
# 1， 生成tool； 2，创建worker； 3，序列化和反序列化（可定制）
# 4，从原材料生成spec对象； 
class ToolSpec(Serializable):
    _tool_id: Optional[Union[str, int]]

    def __init__(self):
        self._tool_id = None

    ###############################################################
    ######## Part One of interfaces: Transformations to Tool ######
    ###############################################################

    @abstractmethod
    def to_tool(self) -> Tool:
        ...

    ###############################################################
    ######## Part Two of interfaces: Worker Creation ##############
    ###############################################################

    @abstractmethod
    def create_worker(self) -> Worker:
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
    ######## ToolSpec initialization from raw materials ###########
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
        return Tool(
            name=self._tool_name,
            description=self._tool_description,
            parameters=self._tool_parameters
        )

    @override
    def create_worker(self) -> Worker:
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
        return Tool(
            name=self._tool_name,
            description=self._tool_description,
            parameters=self._tool_parameters
        )

    @override
    def create_worker(self) -> Worker:
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
