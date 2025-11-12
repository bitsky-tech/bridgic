import copy
import inspect
import functools
from typing import Callable, List, Any, Dict, Tuple, Union

from bridgic.core.automa import GraphAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import ArgsMappingRule, System
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.asl._canvas_object import _Canvas, _Element, _CanvasObject, graph_stack


class TrackingNamespace(dict):
    def __setitem__(self, key: str, value: Any):
        # get the settings from the value and clear the settings of the value
        settings = copy.deepcopy(getattr(value, "__settings__", None))
        if settings:
            setattr(value, "__settings__", None)

        # track the _Element values
        if not key.startswith('_') and (isinstance(value, Worker) or isinstance(value, Callable)):
            if key != 'arun' and key != 'run':
                # create the _Element instance
                element: _Element = _Element(key, value)

                # update the settings of the element
                if settings:
                    element.update_settings(settings)

                # judge if the value is a lambda function
                if isinstance(value, Callable) and getattr(value.__code__, "co_name", None) == '<lambda>':
                    element.is_lambda = True

                # cover the value with the element
                value = element
        
        # record the key-value pair in the tracking namespace
        super().__setitem__(key, value)

        # register the _CanvasObject to the current canvas
        if isinstance(value, _CanvasObject):
            stack = graph_stack.get()
            
            # if is a nested canvas, register to the parent canvas.
            if isinstance(value, _Canvas):
                # set the key of the nested canvas
                if not value.settings.key:
                    value.settings.key = key

                # update the settings of the nested canvas
                if settings:
                    value.update_settings(settings)
                
                # if the stack is only one, the current canvas is the root canvas.
                if len(stack) == 1:
                    return
                
                # get the parent canvas
                parent_canvas: _Canvas = stack[-2]

                # register to the parent canvas, the parent canvas is the root canvas.
                parent_canvas.register(value.settings.key, value)
            elif isinstance(value, _Element):
                if len(stack) == 0:
                    raise ValueError("All workers must be written under one graph.")
                current_canvas: _Canvas = stack[-1]
                current_canvas.register(value.settings.key, value)


class ASLAutomaMeta(GraphMeta):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        return cls


class ASLAutoma(GraphAutoma, metaclass=ASLAutomaMeta):
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._dynamic_workers = {}
        self._collect_canvases()
        self._build_graph()

    def _build_graph(self) -> None:
        # build the graph
        for canvas in self._canvases:
            automa = canvas.worker_material
            elements = canvas.elements
            self._inner_build_logic_flow(automa, elements)
            
    def _inner_build_logic_flow(self, automa: GraphAutoma, elements: Dict[str, "_Element"]) -> None:
        exit_keys = []
        for _, element in elements.items():
            if element.settings.key in exit_keys:
                continue

            key = element.settings.key
            worker_material = element.worker_material
            is_start = element.settings.is_start
            is_output = element.settings.is_output
            dependencies = element.settings.dependencies
            args_mapping_rule = element.settings.args_mapping_rule

            if element.is_lambda:
                bound_element_start = functools.partial(
                    dynamic_lambda_worker_start,
                    __dynamic_lambda_worker__=element,
                    __dynamic_workers__=self._dynamic_workers,
                )
                worker_material = bound_element_start

            new_dependencies = []
            for dependency in dependencies:
                if dependency in self._dynamic_workers.keys():
                    new_dependencies.extend(self._dynamic_workers[dependency])
                else:
                    new_dependencies.append(dependency)
            
            if isinstance(automa, ConcurrentAutoma):
                build_concurrent(
                    automa=automa,
                    key=key,
                    worker_material=worker_material,
                )
            elif isinstance(automa, GraphAutoma):
                build_graph(
                    automa=automa,
                    key=key,
                    worker_material=worker_material,
                    is_start=is_start,
                    is_output=is_output,
                    dependencies=new_dependencies,
                    args_mapping_rule=args_mapping_rule,
                )
            else:
                raise ValueError(f"Invalid automa type: {type(automa)}.")

            exit_keys.append(key)

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
                canvases_dict[value.settings.key] = value

        # bottom up order traversal to get the canvases
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 1:
                return canvases
            
            canvas_map = {canvas.settings.key: canvas for canvas in canvases}
            all_parents = {canvas.parent_canvas.settings.key for canvas in canvases if canvas.parent_canvas}
            leaves = [canvas for canvas in canvases if canvas.settings.key not in all_parents]
            return leaves + [
                        element 
                        for canvas in leaves 
                        for element in bottom_up_order_traversal(
                            [canvas_map[canvas.parent_canvas.settings.key]]
                        )
                    ]

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        bottom_up_order[-1].worker_material = self  # the root graph is self
        self._canvases = bottom_up_order

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return await super().arun(*args, **kwargs)


def build_concurrent(
    automa: ConcurrentAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
) -> None:
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
        )
    elif isinstance(worker_material, Callable):
        automa.add_func_as_worker(
            key=key,
            func=worker_material,
        )

def build_graph(
    automa: GraphAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
    is_start: bool,
    is_output: bool,
    dependencies: List[str],
    args_mapping_rule: ArgsMappingRule,
) -> None:
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
            is_start=is_start,
            is_output=is_output,
            dependencies=dependencies,
            args_mapping_rule=args_mapping_rule,
        )
    elif isinstance(worker_material, Callable):
        automa.add_func_as_worker(
            key=key,
            func=worker_material,
            is_start=is_start,
            is_output=is_output,
            dependencies=dependencies,
            args_mapping_rule=args_mapping_rule,
        )

async def dynamic_lambda_worker_start(
    *args,
    __dynamic_lambda_worker__: _Element,
    __dynamic_workers__: Dict[str, _Element],
    **kwargs,
) -> Any:
    print(args, kwargs)
    automa = __dynamic_lambda_worker__.parent_canvas.worker_material
    key = __dynamic_lambda_worker__.settings.key  
    lambda_func = __dynamic_lambda_worker__.worker_material
    is_output = __dynamic_lambda_worker__.settings.is_output
    args_mapping_rule = __dynamic_lambda_worker__.settings.args_mapping_rule

    # run the lambda function to get the dynamic workers
    dynamic_worker_materials = lambda_func(*args, **kwargs)
    print(dynamic_worker_materials)

    # collect the dynamic workers and build the graph or concurrent
    dynamic_workers_keys = []
    for worker_material in dynamic_worker_materials:
        settings = copy.deepcopy(getattr(worker_material, "__settings__", None))
        print(settings)
        if settings:
            setattr(worker_material, "__settings__", None)
        dynamic_workers_key = settings.key
        dynamic_workers_is_output = settings.is_output
        dynamic_workers_args_mapping_rule = settings.args_mapping_rule
        dynamic_workers_keys.append(dynamic_workers_key)

        if isinstance(automa, GraphAutoma):
            build_graph(
                automa=automa,
                key=dynamic_workers_key,
                worker_material=worker_material,
                is_start=False,
                is_output=is_output or dynamic_workers_is_output,
                dependencies=[key],
                args_mapping_rule = ArgsMappingRule.UNPACK,
            )
        elif isinstance(automa, ConcurrentAutoma):
            pass
    __dynamic_workers__[key] = dynamic_workers_keys
    
    bound_element_end = functools.partial(
        dynamic_lambda_worker_end,
        __dynamic_workers_keys__=dynamic_workers_keys,
    )
    automa.add_func_as_worker(
        key=f"__del_{key}__",
        func=bound_element_end,
        is_start=False,
        is_output=False,
        dependencies=dynamic_workers_keys,
    )
    return args[0]


async def dynamic_lambda_worker_end(
    *args, 
    __dynamic_workers_keys__: List[str],
    __automa__ = System("automa"),
    **kwargs,
) -> Any:
    for dynamic_worker_key in __dynamic_workers_keys__:
        __automa__.remove_worker(dynamic_worker_key)