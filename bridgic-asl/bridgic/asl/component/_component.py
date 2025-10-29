import uuid
from dataclasses import dataclass
from turtle import left
from typing import get_type_hints, Union, Callable, List, Any, Dict, Tuple

from bridgic.core.automa import Automa, GraphAutoma
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.automa.args import ArgsMappingRule
from bridgic.core.automa.worker import Worker


class Component(GraphAutoma):
    def __init__(self, name: str, original_cls: type):
        super().__init__(name=name)
        self._original_cls = original_cls
        self._collect_canvases(original_cls)
        self._build_graph()

    def _build_graph(self) -> None:
        # build the graph
        for canvas in self._canvases:
            automa = canvas.worker_material
            elements = canvas.elements
            self._inner_build_logic_flow(automa, elements)

    def _inner_build_data_flow(self, automa: GraphAutoma, elements: List["_Element"]) -> None:
        pass
            
    def _inner_build_logic_flow(self, automa: GraphAutoma, elements: List["_Element"]) -> None:
        for element in elements:
            key = element.settings.key
            is_start = element.settings.is_start
            is_output = element.settings.is_output
            dependencies = self._get_dependencies(element)
            args_mapping_rule = element.settings.args_mapping_rule
            
            if isinstance(element.worker_material, Worker):
                automa.add_worker(
                    key=key,
                    worker=element.worker_material,
                    is_start=is_start,
                    is_output=is_output,
                    dependencies=dependencies,
                    args_mapping_rule=args_mapping_rule,
                )
            elif isinstance(element.worker_material, Callable):
                automa.add_func_as_worker(
                    key=key,
                    func=element.worker_material,
                    is_start=is_start,
                    is_output=is_output,
                    dependencies=dependencies,
                    args_mapping_rule=args_mapping_rule,
                )

    def _collect_canvases(self, original_cls: type):
        # collect all canvases and elements
        root = None
        canvases_dict: Dict[str, _Canvas] = {}
        for name, value in original_cls.__dict__.items():
            if isinstance(value, _Canvas):
                value.settings.key = name
                if not value.parent_canvas:  # find the root canvas
                    if not root:
                        root = value
                    else:
                        raise ValueError("Multiple root canvases are not allowed.")
                canvases_dict[value.obj_id] = value
        
        canvases_obj_dict = {}
        for name, value in original_cls.__dict__.items():
            if isinstance(value, _CanvasObject):
                value.settings.key = name
                parent_key = value.parent_canvas
                if parent_key:
                    canvases_dict[parent_key].elements.append(value)
                canvases_obj_dict[value.obj_id] = value
        self._canvases_obj_dict = canvases_obj_dict

        # bottom up order traversal to get the canvases
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 1:
                return canvases
            
            canvas_map = {canvas.obj_id: canvas for canvas in canvases}
            all_parents = {canvas.parent_canvas for canvas in canvases if canvas.parent_canvas}
            leaves = [canvas for canvas in canvases if canvas.obj_id not in all_parents]
            return leaves + [element for canvas in leaves for element in bottom_up_order_traversal([canvas_map[canvas.parent_canvas]])]

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        bottom_up_order[-1].worker_material = self  # the root graph is self
        self._canvases = bottom_up_order

    def _get_dependencies(self, element: "_Element") -> List[str]:
        dependency_elements = element.settings.dependencies
        dependencies_keys = []
        for dependency_element_id in dependency_elements:
            dependency_element = self._canvases_obj_dict[dependency_element_id]
            dependencies_keys.append(dependency_element.settings.key)
        return dependencies_keys

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return await super().arun(*args, **kwargs)
        

def component(original_cls: type):
    """ 
    Decorator for defining a component. 
    
    Parameters 
    ---------- 
    cls : type 
        The class to be decorated. 
    """
    annotations = get_type_hints(original_cls)
    defaults = {
        name: getattr(original_cls, name)
        for name in dir(original_cls)
        if not name.startswith("__")
    }
    
    class SpecificComponent(Component):
        def __init__(self, **kwargs):
            for name, annotation in annotations.items():
                if name in kwargs:
                    value = kwargs[name]
                elif name in defaults:
                    value = defaults[name]
                else:
                    raise ValueError(
                        f"Parameter {name} is missing, please provide a value for it or add a default value. "
                        f"Annotations: {annotations}, Defaults: {defaults}, Kwargs: {kwargs}"
                    )
                setattr(self, name, value)
            
            super().__init__(name=original_cls.__name__, original_cls=original_cls)

    # copy the attributes of the original class
    for name, value in original_cls.__dict__.items():
        if not name.startswith('__') and not callable(value):
            setattr(SpecificComponent, name, value)
        if callable(value) and name != '__init__':
            setattr(SpecificComponent, name, value)
    
    # set the class name
    SpecificComponent.__name__ = original_cls.__name__
    SpecificComponent.__qualname__ = original_cls.__qualname__
    return SpecificComponent


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
            other_dependencies_keys = [dependency_obj.obj_id for dependency_obj in other.dependencies]
            if len(other_dependencies_keys) != len(set(other_dependencies_keys)):
                raise ValueError(f"Duplicate dependency: {other_dependencies_keys}.")
            self.dependencies = other_dependencies_keys
        if other.args_mapping_rule != self.args_mapping_rule:
            self.args_mapping_rule = other.args_mapping_rule


class _CanvasObject:
    def __init__(self, worker_material: Union[Worker, Callable]) -> None:
        self.obj_id = str(id(worker_material)) + "-" + str(uuid.uuid4().hex[:8])
        self.worker_material = worker_material
        self.parent_canvas = None
        self.left_canvas_obj = None
        self.right_canvas_obj = None

        # settings initialization
        self.settings = Settings(
            key=worker_material.__class__.__name__ if isinstance(worker_material, Worker) else worker_material.__name__,
            is_start=False,
            is_output=False,
            dependencies=[],
            args_mapping_rule=ArgsMappingRule.AS_IS
        )

        self.parameters = {}

    def __mul__(self, other: Union[Data, Settings]) -> None:
        if isinstance(other, Settings):
            self.settings.update(other)

        if isinstance(other, Data):
            pass

    def __rshift__(self, other: Union["_CanvasObject", Tuple["_CanvasObject"]]) -> None:
        """
        ">>" operator is used to set the current worker as a dependency of the other worker.
        """
        current_canvas_obj = self
        left_canvas_objs = [self.obj_id]
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            left_canvas_objs.append(current_canvas_obj.obj_id)
        left_canvas_objs.reverse()  # Keep the order consistent with the declaration.

        other.settings.dependencies.extend(left_canvas_objs)
        current_canvas_obj = other
        while current_canvas_obj.left_canvas_obj:
            current_canvas_obj = current_canvas_obj.left_canvas_obj
            current_canvas_obj_dependencies = set(current_canvas_obj.settings.dependencies)
            for dependency_obj_id in left_canvas_objs:
                if dependency_obj_id in current_canvas_obj_dependencies:
                    raise ValueError(f"Duplicate dependency: {dependency_obj_id}.")
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
            f"obj_id={self.obj_id}, "
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
        super().__init__(worker_material)
        self.elements = []

    def __enter__(self) -> "_Canvas":
        self.runtime_id = self.obj_id
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.runtime_id = None

    def __rmatmul__(self, other: Union[Worker, Callable]) -> _CanvasObject:
        if self.runtime_id is None:
            raise ValueError(f"graph is not initialized or has been exited. Please use the within statement `with graph(...) as g:` etc.")
            
        canvas_object = _Element(self.runtime_id, other)
        return canvas_object
    
    def __matmul__(self, other: "_Canvas") -> "_Canvas":
        self.parent_canvas = other.obj_id
        return self

    def __str__(self) -> str:
        return (
            f"_Canvas("
            f"obj_id={self.obj_id}, "
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
    def __init__(self, parent_id: str, worker_material: Union[Worker, Callable]) -> None:
        super().__init__(worker_material)
        self.parent_canvas = parent_id

    def __str__(self) -> str:
        return (
            f"_Element("
            f"obj_id={self.obj_id}, "
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas}, "
            f"settings={self.settings})"
        )
