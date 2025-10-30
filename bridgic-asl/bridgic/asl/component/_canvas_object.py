import inspect
from dataclasses import dataclass
from contextvars import ContextVar
from typing import List, Any, Union, Callable, Tuple

from bridgic.core.automa.worker import Worker
from bridgic.core.automa import Automa, GraphAutoma
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.agentic import ConcurrentAutoma

graph_stack = ContextVar("graph_stack", default=[])


class TrackingNamespace(dict):
    def __setitem__(self, key: str, value: Any):
        # get the settings from the value
        settings = getattr(value, "__settings__", None)

        # only track the special values
        if not key.startswith('_'):
            if isinstance(value, Worker) or isinstance(value, Callable):
                if key != 'arun' and key != 'run':
                    value: _Element = _Element(key, value)
                    if settings:
                        value.update_settings(settings)
        super().__setitem__(key, value)

        # get current activate `with` context
        stack = graph_stack.get()
        if not stack:  
            return
        current_graph = stack[-1]

        # register the nested canvas
        if isinstance(value, _Canvas):
            if value not in stack:
                return

            if not value.settings.key:
                value.settings.key = key

            if settings:
                value.update_settings(settings)

            if len(stack) >= 2 and stack[-1] is value:
                parent = stack[-2]
                parent.register(value.settings.key, value)
            return

        # register the normal object
        current_graph.register(key, value)


def make_automa(automa_type: str) -> Automa:
    if automa_type == "graph":
        return GraphAutoma()
    elif automa_type == "concurrent":
        return ConcurrentAutoma()
    else:
        raise ValueError(f"Invalid automa type: {automa_type}")


class Data:
    def __init__(self, **kwargs: Any) -> None:
        self.data = kwargs

    def __rmul__(self, other: Union[Callable, Worker, "_Canvas"]):
        """
        "*" operator is used to set the data dependencies of a obj.
        """
        pass


@dataclass
class Settings:
    key: str = None
    is_start: bool = None
    is_output: bool = None
    dependencies: List[str] = None
    args_mapping_rule: ArgsMappingRule = None

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
    def __init__(self, key: str, worker_material: Union[Worker, Callable]) -> None:
        self.worker_material = worker_material
        self.parent_canvas = None
        self.left_canvas_obj = None
        self.right_canvas_obj = None

        # settings initialization
        self.settings = Settings(
            key=key,
            is_start=False,
            is_output=False,
            dependencies=[],
            args_mapping_rule=ArgsMappingRule.AS_IS
        )

        self.parameters = {}

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
    def __init__(self, automa_type: str) -> None:
        worker_material = make_automa(automa_type)
        super().__init__(None, worker_material)
        self.elements = {}

    def __enter__(self) -> "_Canvas":
        stack = list(graph_stack.get())
        stack.append(self)
        graph_stack.set(stack)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        stack = list(graph_stack.get())
        stack.pop()
        graph_stack.set(stack)

    def register(self, key, value):
        value.parent_canvas = self.settings.key
        self.elements[key] = value
        if isinstance(value, _Canvas):
            value.settings.key = key

    def __str__(self) -> str:
        return (
            f"_Canvas("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas}, "
            f"elements={self.elements}, "
            f"settings={self.settings}"
        )

def graph() -> _Canvas:
    ctx = _Canvas(automa_type="graph")
    return ctx

def concurrent() -> _Canvas:
    ctx = _Canvas(automa_type="concurrent")
    return ctx


class _Element(_CanvasObject):
    def __init__(self, key: str, worker_material: Union[Worker, Callable]) -> None:
        super().__init__(key, worker_material)
        self.parent_canvas = None

    def __str__(self) -> str:
        return (
            f"_Element("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas}, "
            f"settings={self.settings})"
        )