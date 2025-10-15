import inspect
from typing import Optional, Union, Dict, List, Any, Type, Callable, Annotated, get_origin
from typing_extensions import override
from abc import abstractmethod
from types import MethodType
from docstring_parser import parse as parse_docstring # type: ignore

from bridgic.core.model.types import Tool
from bridgic.core.automa import Automa
from bridgic.core.automa.worker import Worker
from bridgic.core.agentic.tool_specs._base_tool_spec import ToolSpec
from bridgic.core.utils._json_schema import create_func_params_json_schema
from bridgic.core.utils._inspect_tools import load_qualified_class_or_func

def as_tool(spec_func: Callable) -> Callable:
    """
    A decorator that transforms a class to a tool that may be used by LLM.

    Parameters
    ----------
    spec_func : Callable
        The function used to declare the tool spec. Note that this function is not intended to be called directly.
    """
    def decorator(cls):
        if not isinstance(spec_func, Callable):
            raise ValueError(f"A function argument is expected, but got {type(spec_func)}.")
        if isinstance(spec_func, MethodType):
            raise ValueError(f"`spec_func` is not allowed to be a bound method: {spec_func}.")
        cls.spec_func = spec_func
        return cls
    return decorator

def get_tool_description_from(
    spec_func: Callable,
    tool_name: Optional[str] = None
) -> str:
    """
    Get the tool description from the spec function.
    """
    docstring = parse_docstring(spec_func.__doc__)
    tool_description = docstring.description
    if tool_description:
        tool_description = tool_description.strip()

    if not tool_description:
        # No description provided, use the function signature as the description.
        fn_sig = inspect.signature(spec_func)
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
        if not tool_name:
            tool_name = spec_func.__name__
        tool_description = f"{tool_name}{fn_sig}\n"
    return tool_description

class AutomaToolSpec(ToolSpec):
    _automa_cls: Type[Automa]
    """The Automa class to be used as a tool"""
    _automa_init_kwargs: Dict[str, Any]
    """The initialization arguments for the Automa"""

    def __init__(
        self,
        automa_cls: Type[Automa],
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        **automa_init_kwargs: Dict[str, Any],
    ):
        super().__init__(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters
        )
        self._automa_cls = automa_cls
        self._automa_init_kwargs = automa_init_kwargs

    @classmethod
    def from_raw(
        cls,
        automa_cls: Type[Automa],
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        tool_parameters: Optional[Dict[str, Any]] = None,
        **automa_init_kwargs: Dict[str, Any],
    ) -> "AutomaToolSpec":
        """
        Create an AutomaToolSpec from an Automa class.
        """

        def check_spec_func(automa_cls):
            if hasattr(automa_cls, "spec_func") and isinstance(automa_cls.spec_func, Callable):
                return
            raise ValueError(f"The Automa class {automa_cls} must be decorated with `@as_tool` in order to be used as a tool.")

        if (not tool_name) or (not tool_description) or (not tool_parameters):
            check_spec_func(automa_cls)

        if not tool_name:
            tool_name = automa_cls.spec_func.__name__
        
        if not tool_description:
            tool_description = get_tool_description_from(automa_cls.spec_func, tool_name)

        if not tool_parameters:
            tool_parameters = create_func_params_json_schema(automa_cls.spec_func)
            # TODO: whether to remove the `title` field of the params_schema?

        return cls(
            automa_cls=automa_cls,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_parameters=tool_parameters,
            **automa_init_kwargs
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
        return self._automa_cls(**self._automa_init_kwargs)

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["automa_cls"] = self._automa_cls.__module__ + "." + self._automa_cls.__qualname__
        if self._automa_init_kwargs:
            state_dict["automa_init_kwargs"] = self._automa_init_kwargs
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self._automa_cls = load_qualified_class_or_func(state_dict["automa_cls"])
        self._automa_init_kwargs = state_dict.get("automa_init_kwargs") or {}
