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
        # only track the special values
        if not key.startswith('_'):
            if isinstance(value, Worker) or isinstance(value, Callable):
                if key != 'arun' and key != 'run':
                    value = _Element(key, value)
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

            if not value.name:
                value.name = key

            if len(stack) >= 2 and stack[-1] is value:
                parent = stack[-2]
                parent.register(value.name, value)
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
    key: str
    is_start: bool
    is_output: bool
    dependencies: List[str]
    args_mapping_rule: ArgsMappingRule

    def update(self, other: "Settings") -> None:
        if other.key != self.key:
            self.key = other.key
        if other.is_start != self.is_start:
            self.is_start = other.is_start
        if other.is_output != self.is_output:
            self.is_output = other.is_output
        if other.dependencies != self.dependencies:  # detect duplicate error
            other_dependencies_keys = [dependency_obj.name for dependency_obj in other.dependencies]
            if len(other_dependencies_keys) != len(set(other_dependencies_keys)):
                raise ValueError(f"Duplicate dependency: {other_dependencies_keys}.")
            self.dependencies = other_dependencies_keys
        if other.args_mapping_rule != self.args_mapping_rule:
            self.args_mapping_rule = other.args_mapping_rule

    def __rmul__(self, other: Union[Callable, Worker, "_Canvas"]):
        """
        "*" operator is used to set the settings of a obj.
        """
        pass


class _CanvasObject:
    def __init__(self, name: str, worker_material: Union[Worker, Callable]) -> None:
        self.name = name
        self.worker_material = worker_material
        self.parent_canvas = None
        self.left_canvas_obj = None
        self.right_canvas_obj = None

        # settings initialization
        # TODO: 配合 Setting 右结合 * 运算符更新一个 setting
        self.settings = Settings(
            key=self.name,
            is_start=False,
            is_output=False,
            dependencies=[],
            args_mapping_rule=ArgsMappingRule.AS_IS
        )

        self.parameters = {}

    def __rshift__(self, other: Union["_CanvasObject", Tuple["_CanvasObject"]]) -> None:
        """
        ">>" operator is used to set the current worker as a dependency of the other worker.
        """
        current_canvas_obj = self
        left_canvas_objs = [self.name]
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            left_canvas_objs.append(current_canvas_obj.name)
        left_canvas_objs.reverse()  # Keep the order consistent with the declaration.

        other.settings.dependencies.extend(left_canvas_objs)
        current_canvas_obj = other
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            current_canvas_obj_dependencies = set(current_canvas_obj.settings.dependencies)
            for dependency_name in left_canvas_objs:
                if dependency_name in current_canvas_obj_dependencies:
                    raise ValueError(f"Duplicate dependency: {dependency_name}.")
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
            f"name={self.name}, "
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

    def register(self, name, value):
        value.parent_canvas = self.name
        self.elements[name] = value
        if isinstance(value, _Canvas):
            value.settings.key = name

    def __str__(self) -> str:
        return (
            f"_Canvas("
            f"name={self.name}, "
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
    def __init__(self, name: str, worker_material: Union[Worker, Callable]) -> None:
        super().__init__(name, worker_material)
        self.parent_canvas = None

    def __str__(self) -> str:
        return (
            f"_Element("
            f"name={self.settings.key}, "
            f"parent_canvas={self.parent_canvas}, "
            f"settings={self.settings})"
        )