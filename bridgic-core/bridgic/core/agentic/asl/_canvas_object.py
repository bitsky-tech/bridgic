import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from pydantic_core import PydanticUndefinedType
from pydantic.fields import FieldInfo
from typing import List, Any, Union, Callable, Tuple, Type, Optional, Dict

from bridgic.core.automa.worker import Worker
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.args import ArgsMappingRule, Distribute
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.utils._inspect_tools import override_func_signature, set_method_signature

graph_stack: ContextVar[List["_Canvas"]] = ContextVar("graph_stack", default=[])


class Data:
    def __init__(self, **kwargs: Any) -> None:
        data = {}
        for key, value in kwargs.items():
            data[key] = {
                "type": Any,
                "default": value,
            }
        self.data = data

    def __rmul__(self, other: Union[Callable, Worker]):
        if isinstance(other, Worker):
            worker_name = other.__class__.__name__
            override_func = other.arun if other._is_arun_overridden() else other.run
            override_func_signature(worker_name, override_func, self.data)
        elif isinstance(other, Callable):
            func_name = getattr(other, "__name__", repr(other))
            override_func_signature(func_name, other, self.data)

        return other


@dataclass
class Settings:
    key: str = None
    is_start: bool = None
    is_output: bool = None
    dependencies: List[str] = None
    args_mapping_rule: ArgsMappingRule = None
    destroy_timely: bool = None

    def __post_init__(self):
        if not self.key:
            self.key = ""
        if not self.is_start:
            self.is_start = False
        if not self.is_output:
            self.is_output = False
        if not self.dependencies:
            self.dependencies = []
        if not self.args_mapping_rule:
            self.args_mapping_rule = ArgsMappingRule.AS_IS
        if not self.destroy_timely:
            self.destroy_timely = False

    def update(self, other: "Settings") -> None:
        if other.key != self.key:
            self.key = other.key
        if other.is_start != self.is_start:
            self.is_start = other.is_start
        if other.is_output != self.is_output:
            self.is_output = other.is_output
        if other.dependencies != self.dependencies:  # detect duplicate error
            if len(other.dependencies) != len(set(other.dependencies)):
                raise ValueError(f"Duplicate dependency: {other.dependencies}.")
            self.dependencies = other.dependencies
        if other.args_mapping_rule != self.args_mapping_rule:
            self.args_mapping_rule = other.args_mapping_rule

    def __rmul__(self, other: Union[Callable, Worker]):
        """
        "*" operator is used to set the settings of a obj.
        """
        setattr(other, "__settings__", self)
        return other


class _CanvasObject:
    def __init__(self, key: str, worker_material: Optional[Union[Worker, Callable]]) -> None:
        self.worker_material = worker_material
        self.parent_canvas = None
        self.left_canvas_obj = None
        self.right_canvas_obj = None
        self.is_lambda = False

        # settings initialization
        self.settings = Settings(
            key=key,
            is_start=False,
            is_output=False,
            dependencies=[],
            args_mapping_rule=ArgsMappingRule.AS_IS
        )

    def update_settings(self, settings: Settings) -> None:
        self.settings.update(settings)
        return self

    def __rshift__(self, other: Union["_CanvasObject", Tuple["_CanvasObject"]]) -> None:
        """
        ">>" operator is used to set the current worker as a dependency of the other worker.
        """
        def check_duplicate_dependency(current_canvas_obj: _CanvasObject, left_canvas_objs: List[str]) -> None:
            for dependency in left_canvas_objs:
                if dependency in current_canvas_obj.settings.dependencies:
                    raise ValueError(f"Duplicate dependency: {dependency}.")

        current_canvas_obj = self
        left_canvas_objs = [self.settings.key]
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            left_canvas_objs.append(current_canvas_obj.settings.key)
        left_canvas_objs.reverse()  # Keep the order consistent with the declaration.

        check_duplicate_dependency(other, left_canvas_objs)
        other.settings.dependencies.extend(left_canvas_objs)

        current_canvas_obj = other
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            check_duplicate_dependency(current_canvas_obj, left_canvas_objs)
            current_canvas_obj.settings.dependencies.extend(left_canvas_objs)
        return other

    def __or__(self, other: "_CanvasObject") -> None:
        """
        "|" operator.
        """
        raise NotImplementedError("`|` operator is not supported for AgentCanvasObject.")

    def __and__(self, other: "_CanvasObject") -> None:
        """
        "&" operator is used to group multiple workers.
        """
        self.right_canvas_obj = other
        other.left_canvas_obj = self
        return other

    def __pos__(self) -> None:
        """
        "+" operator is used to set the worker as a start worker.
        """
        self.settings.is_start = True
        
        current_canvas_obj = self
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj.settings.is_start = True
            current_canvas_obj = current_canvas_obj.left_canvas_obj
        current_canvas_obj.settings.is_start = True  # set the last one
        
        return self

    def __invert__(self) -> None:
        """
        "~" operator is used to set the worker as an output worker.
        """
        self.settings.is_output = True

        current_canvas_obj = self
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj.settings.is_output = True
            current_canvas_obj = current_canvas_obj.left_canvas_obj
        current_canvas_obj.settings.is_output = True  # set the last one

        return self

    def __str__(self) -> str:
        return (
            f"CanvasObject("
            f"key={self.settings.key}, "
            f"worker_material={self.worker_material}, "
            f"settings={self.settings}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class _Canvas(_CanvasObject):
    """
    The input parameters of a graph were recorded.
    """
    def __init__(self, automa_type: str, params: Dict[str, Any]) -> None:
        super().__init__(None, None)
        self.automa_type = automa_type
        self.params = params

        self.elements: Dict[str, Union["_Element", "_Canvas"]] = {}

    def register(self, key: str, value: Union["_Element", "_Canvas"]):
        if isinstance(value, _Element) and value.is_lambda and self.automa_type == "graph":
            raise ValueError("Lambda dynamic logic must be written under a `concurrent`, `sequential`, not `graph`.")
            
        value.parent_canvas = self
        self.elements[key] = value

    def make_automa(self, running_options: RunningOptions = None):
        if self.automa_type == "graph":
            self.worker_material = GraphAutoma(
                name=self.settings.key, 
                running_options=running_options
            )
        elif self.automa_type == "concurrent":
            self.worker_material = ConcurrentAutoma(
                name=self.settings.key, 
                running_options=running_options
            )
        else:
            raise ValueError(f"Invalid automa type: {self.automa_type}")

        if self.params:
            set_method_signature(self.worker_material.arun, self.params)

    def is_top_level(self) -> bool:
        return self.parent_canvas is self

    def __str__(self) -> str:
        return (
            f"_Canvas("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas.settings.key if self.parent_canvas else None}, "
            f"elements={self.elements}, "
            f"settings={self.settings}"
        )


class _Element(_CanvasObject):
    def __init__(self, key: str, worker_material: Union[Worker, Callable]) -> None:
        super().__init__(key, worker_material)
        self.parent_canvas = None

    def __str__(self) -> str:
        return (
            f"_Element("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas.settings.key if self.parent_canvas else None}, "
            f"settings={self.settings})"
        )


class _GraphContextManager:
    def __init__(self, automa_type: str):
        self.automa_type = automa_type
        self.params = {}

    def __call__(self, **kwargs: Dict[str, "ASLField"]) -> _Canvas:
        params = {}
        for key, value in kwargs.items():
            if not isinstance(value, ASLField):
                raise ValueError(f"Invalid field type: {type(value)}.")
            default_value = Distribute(
                value.default 
                if not isinstance(value.default, PydanticUndefinedType) 
                else inspect._empty
            ) if value.distribute else value.default
            params[key] = {
                "type": value.type,
                "default": default_value
            }
        self.params = params
        return self

    def __enter__(self) -> _Canvas:
        ctx = _Canvas(automa_type=self.automa_type, params=self.params)

        stack = list(graph_stack.get())
        stack.append(ctx)
        graph_stack.set(stack)
        return ctx

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        stack = list(graph_stack.get())
        stack.pop()
        graph_stack.set(stack)
        

# Create module-level instances (not global variables, but singleton objects)
graph = _GraphContextManager(automa_type="graph")
concurrent = _GraphContextManager(automa_type="concurrent")


class ASLField(FieldInfo):
    """
    A custom Field class that extends Pydantic's FieldInfo with support for storing default type information.
    
    The `default_type` parameter is stored as metadata and does not automatically generate default values.
    You must explicitly provide a `default` value if you want a default.
    """
    
    def __init__(
        self,
        type: Type[Any] = Any,
        *,
        default: Any = ...,
        distribute: bool = False,
        **kwargs: Any
    ):
        """
        Initialize ASLField with optional default_type metadata.
        
        Parameters
        ----------
        default : Any
            Explicit default value. Must be provided if you want a default value.
        type : Type[Any]
            Type information stored as metadata. Does not automatically generate default values.
        distribute : bool
            Whether to distribute the data to multiple workers.
        **kwargs : Any
            Other Field parameters (description, ge, le, etc.)
        """
        super().__init__(default=default, **kwargs)
        self.type = type
        self.distribute = distribute
