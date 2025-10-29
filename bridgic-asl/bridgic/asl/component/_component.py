from typing import Callable, List, Any, Dict, Tuple

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.asl.component._canvas_object import (
    _Canvas,
    _Element,
    TrackingNamespace
)


class ComponentMeta(GraphMeta):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()


class Component(GraphAutoma, metaclass=ComponentMeta):
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._collect_canvases()
        self._build_graph()

    def _build_graph(self) -> None:
        # build the graph
        for canvas in self._canvases:
            automa = canvas.worker_material
            elements = canvas.elements
            self._inner_build_logic_flow(automa, elements)

    def _inner_build_data_flow(self, automa: GraphAutoma, elements: List["_Element"]) -> None:
        pass
            
    def _inner_build_logic_flow(self, automa: GraphAutoma, elements: Dict[str, "_Element"]) -> None:
        for _, element in elements.items():
            key = element.settings.key
            is_start = element.settings.is_start
            is_output = element.settings.is_output
            dependencies = element.settings.dependencies
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

    def _collect_canvases(self):
        # collect all canvases and elements
        root = None
        canvases_dict: Dict[str, _Canvas] = {}
        # 访问类属性，需要使用 type(self).__dict__ 而不是 self.__dict__
        cls = type(self)
        for name, value in cls.__dict__.items():  # find the root canvas
            if isinstance(value, _Canvas):
                if not value.parent_canvas:
                    if not root:
                        root = value
                    else:
                        raise ValueError("Multiple root canvases are not allowed.")
                canvases_dict[value.name] = value

        # bottom up order traversal to get the canvases
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 1:
                return canvases
            
            canvas_map = {canvas.name: canvas for canvas in canvases}
            all_parents = {canvas.parent_canvas for canvas in canvases if canvas.parent_canvas}
            leaves = [canvas for canvas in canvases if canvas.name not in all_parents]
            return leaves + [element for canvas in leaves for element in bottom_up_order_traversal([canvas_map[canvas.parent_canvas]])]

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        bottom_up_order[-1].worker_material = self  # the root graph is self
        self._canvases = bottom_up_order

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return await super().arun(*args, **kwargs)
        
