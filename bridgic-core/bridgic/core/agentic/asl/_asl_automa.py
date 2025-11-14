import copy
import inspect
import functools
from re import L
from typing import Callable, List, Any, Dict, Tuple, Union, Optional

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
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
                # if the stack is empty, indicates that the element is not under any canvas.
                if len(stack) == 0:
                    raise ValueError("All workers must be written under one graph.")

                # get the current canvas
                current_canvas: _Canvas = stack[-1]

                # register to the current canvas
                current_canvas.register(value.settings.key, value)


class ASLAutomaMeta(GraphMeta):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        canvases = cls._collect_canvases(namespace)
        setattr(cls, "_canvases", canvases)
        return cls

    @classmethod
    def _collect_canvases(mcls, dct: Dict[str, Any]) -> List[_Canvas]:
        # collect all canvases and elements
        root = None
        canvases_dict: Dict[str, _Canvas] = {}

        for _, value in dct.items():
            if isinstance(value, _Canvas):

                # find the root canvas and set its parent canvas to itself
                if not value.parent_canvas:
                    if not root:
                        root = value
                        value.parent_canvas = value
                    else:
                        raise ValueError("Multiple root canvases are not allowed.")

                # record the canvas to the canvases dictionary
                canvases_dict[value.settings.key] = value

        # bottom up order traversal to get the canvases
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 1:
                return canvases
            
            canvas_map = {canvas.settings.key: canvas for canvas in canvases}
            all_parents = {canvas.parent_canvas.settings.key for canvas in canvases if canvas.parent_canvas}
            leaves = [canvas for canvas in canvases if canvas.settings.key not in all_parents]
            result = leaves + [
                        element 
                        for canvas in leaves 
                        for element in bottom_up_order_traversal(
                            [canvas_map[canvas.parent_canvas.settings.key]]
                        )
                    ]
            return result

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        return bottom_up_order


class ASLAutoma(GraphAutoma, metaclass=ASLAutomaMeta):
    # The canvases of the automa.
    _canvases: List[_Canvas] = []

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self._dynamic_workers = {}
        self._build_graph()

    def _build_graph(self) -> None:
        for canvas in self._canvases:
            static_elements = {key: value for key, value in canvas.elements.items() if not value.is_lambda}
            dynamic_elements = {key: value for key, value in canvas.elements.items() if value.is_lambda}            
            self._inner_build_graph(canvas, static_elements, dynamic_elements)
            
    def _inner_build_graph(
        self, 
        canvas: _Canvas, 
        static_elements: Dict[str, "_Element"],
        dynamic_elements: Dict[str, "_Element"]
    ) -> None:
        automa = None
        current_canvas_key = canvas.settings.key

        # if the element is already built, skip it, `exit_keys_set` is to deal with such situations:
        #   ...
        #   with graph as g:
        #       a = worker1
        #       b = worker2
        #       c = worker3
        # 
        #       arrangement1 = +a >> b
        #       arrangement2 = ~c
        # 
        #       arrangement1 >> arrangement2
        #   ...
        #   In the end, the `arrangement1` is still `b`, but by this time, `b` has already been built.
        exit_keys_set = set()


        ###############################
        # build the dynamic logic flow
        ###############################
        running_options_callback = []
        for _, element in dynamic_elements.items():
            if element.settings.key in exit_keys_set:
                continue
            
            # if the canvas is top level, should use `RunningOptions` to add callback.
            if canvas.is_top_level():
                running_options_callback.append(
                    WorkerCallbackBuilder(
                        DynamicLambdaWorkerCallback, 
                        init_kwargs={"__dynamic_lambda_worker__": element},
                        is_shared=False
                    )
                )

            # else should delegate parent automa add callback during using `add_worker`.
            else:
                parent_key = canvas.parent_canvas.settings.key
                if parent_key not in self._dynamic_workers:
                    self._dynamic_workers[parent_key] = {}
                if current_canvas_key not in self._dynamic_workers[parent_key]:
                    self._dynamic_workers[parent_key][current_canvas_key] = []
                self._dynamic_workers[parent_key][current_canvas_key].append(element)
            
            # add the key to the exit keys set
            exit_keys_set.add(element.settings.key)

        # make the automa
        canvas.make_automa(running_options=RunningOptions(callback_builders=running_options_callback))
        automa = canvas.worker_material if not canvas.is_top_level() else self

        
        ###############################
        # build the static logic flow
        ###############################
        for _, element in static_elements.items():
            if element.settings.key in exit_keys_set:
                continue

            key = element.settings.key
            worker_material = element.worker_material
            is_start = element.settings.is_start
            is_output = element.settings.is_output
            dependencies = element.settings.dependencies
            args_mapping_rule = element.settings.args_mapping_rule

            # prepare the callback builders
            callback_builders = []
            # if current element delegated dynamic workers to be added in current canvas
            if current_canvas_key in self._dynamic_workers and key in self._dynamic_workers[current_canvas_key]:
                delegated_dynamic_workers = self._dynamic_workers[current_canvas_key][key]
                for delegated_dynamic_element in delegated_dynamic_workers:
                    callback_builders.append(WorkerCallbackBuilder(
                        DynamicLambdaWorkerCallback,
                        init_kwargs={"__dynamic_lambda_worker__": delegated_dynamic_element},
                        is_shared=False
                    ))

            if isinstance(automa, ConcurrentAutoma):
                build_concurrent(
                    automa=automa,
                    key=key,
                    worker_material=worker_material,
                    callback_builders=callback_builders
                )
            elif isinstance(automa, GraphAutoma):
                build_graph(
                    automa=automa,
                    key=key,
                    worker_material=worker_material,
                    is_start=is_start,
                    is_output=is_output,
                    dependencies=dependencies,
                    args_mapping_rule=args_mapping_rule,
                    callback_builders=callback_builders
                )
            else:
                raise ValueError(f"Invalid automa type: {type(automa)}.")

            # add the key to the exit keys set
            exit_keys_set.add(key)

    def _inner_build_static_graph(self):
        pass

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return await super().arun(*args, **kwargs)


def build_concurrent(
    automa: ConcurrentAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> None:
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
            callback_builders=callback_builders,
        )
    elif isinstance(worker_material, Callable):
        automa.add_func_as_worker(
            key=key,
            func=worker_material,
            callback_builders=callback_builders,
        )

def build_graph(
    automa: GraphAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
    is_start: bool,
    is_output: bool,
    dependencies: List[str],
    args_mapping_rule: ArgsMappingRule,
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> None:
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
            is_start=is_start,
            is_output=is_output,
            dependencies=dependencies,
            args_mapping_rule=args_mapping_rule,
            callback_builders=callback_builders
        )
    elif isinstance(worker_material, Callable):
        automa.add_func_as_worker(
            key=key,
            func=worker_material,
            is_start=is_start,
            is_output=is_output,
            dependencies=dependencies,
            args_mapping_rule=args_mapping_rule,
            callback_builders=callback_builders
        )


class DynamicLambdaWorkerCallback(WorkerCallback):
    def __init__(self, __dynamic_lambda_worker__: _Element):
        self.__dynamic_lambda_worker__ = __dynamic_lambda_worker__

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        # if the worker is not the top-level worker, skip it
        if not is_top_level:
            return

        

        if is_top_level:
            specific_automa = parent
        else:
            specific_automa = parent._get_worker_instance(key)

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        pass


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