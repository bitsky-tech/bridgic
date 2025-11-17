import copy
import uuid
from typing import Callable, List, Any, Dict, Tuple, Union

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from bridgic.core.automa.args import ArgsMappingRule, System, Distribute
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.asl._canvas_object import _Canvas, _Element, _CanvasObject, graph_stack, Settings, Data, KeyUnDifined
from bridgic.core.utils._inspect_tools import get_func_signature_data, set_method_signature, override_func_signature


class TrackingNamespace(dict):
    def __setitem__(self, key: str, value: Any):
        # if the value is a _Element, It indicates that the corresponding Worker object or 
        # callable has already been declared. Declaring it again is likely to be a fragment.
        # So skip the registration process.
        if isinstance(value, _Element):
            super().__setitem__(key, value)
            return

        # get the settings and data from the value and clear the settings and data of the value
        settings = copy.deepcopy(getattr(value, "__settings__", None))
        if settings:
            setattr(value, "__settings__", None)
        data = copy.deepcopy(getattr(value, "__data__", None))
        if data:
            setattr(value, "__data__", None)

        # track the _Element values
        if not key.startswith('_') and (isinstance(value, Worker) or isinstance(value, Callable)):
            if key != 'arun' and key != 'run':
                # create the _Element instance
                element: _Element = _Element(key, value)

                # update the settings of the element
                if settings:
                    element.update_settings(settings)

                # update the data of the element
                if data:
                    element.update_data(data)

                # judge if the value is a lambda function
                if isinstance(value, Callable) and getattr(value.__code__, "co_name", None) == '<lambda>':
                    element.is_lambda = True

                # cover the value with the element
                value = element

        # register the _CanvasObject to the current canvas
        if isinstance(value, _CanvasObject):
            stack = list[_Canvas](graph_stack.get())
            
            # if is a nested canvas, register to the parent canvas.
            if isinstance(value, _Canvas):
                # if the canvas is already in the tracking namespace, raise error.
                if super().__contains__(key):
                    raise ValueError(f"Duplicate key: {key}, every key must be unique.")

                # set the key of the nested canvas
                if isinstance(value.settings.key, KeyUnDifined):
                    value.settings.key = key

                # update the settings of the nested canvas
                if settings:
                    value.update_settings(settings)

                # update the data of the nested canvas
                if data:
                    value.update_data(data)

                # register current canvas to the tracking namespace
                super().__setitem__(key, {})
                current_canvas_namespace = super().__getitem__(key)
                current_canvas_namespace["__self__"] = value
                
                # if the stack is only one, the current canvas is the root canvas, skip the registration.
                if len(stack) > 1:
                    # get the parent canvas
                    parent_canvas: _Canvas = stack[-2]

                    # register to the parent canvas
                    parent_canvas.register(key, value)

                    # register to the tracking namespace
                    parent_canvas_namespace = super().__getitem__(parent_canvas.settings.key)
                    parent_canvas_namespace[value.settings.key] = value

            elif isinstance(value, _Element):
                # if the stack is empty, indicates that the element is not under any canvas.
                if len(stack) == 0:
                    raise ValueError("All workers must be written under one graph.")

                # get the current canvas
                parent_canvas: _Canvas = stack[-1]

                # register to the current canvas
                parent_canvas.register(value.settings.key, value)

                # register to the tracking namespace
                parent_canvas_namespace = super().__getitem__(parent_canvas.settings.key)
                parent_canvas_namespace[value.settings.key] = value
        else:
            # record the normal key-value pair in the tracking namespace
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        # get the current canvas
        stack = list[_Canvas](graph_stack.get())

        # if the stack is empty, indicates that the element may be a normal object.
        if not stack:
            return super().__getitem__(key)

        # get the current canvas and its namespace
        current_canvas: _Canvas = stack[-1]
        current_canvas_key = current_canvas.settings.key
        current_canvas_namespace = super().__getitem__(current_canvas_key)
        if key == current_canvas_key:
            raise ValueError(f"Invalid canvas: {key}, cannot use the canvas itself.")

        # if the key is in the current canvas namespace, return the element.
        if key in current_canvas_namespace:
            res = current_canvas_namespace[key]
            return res

        return super().__getitem__(key)


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
            if isinstance(value, Dict) and "__self__" in value:
                value = value["__self__"]

                # find the root canvas and set its parent canvas to itself
                if not value.parent_canvas:
                    value.parent_canvas = value
                    if not root:
                        root = value
                    else:
                        raise ValueError("Multiple root canvases are not allowed.")

                # record the canvas to the canvases dictionary
                canvases_dict[value.settings.key] = value

        # bottom up order traversal to get the canvases
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 0:
                return []
            if len(canvases) == 1:
                return canvases
            
            result: List[_Canvas] = []
            remaining = {canvas.settings.key: canvas for canvas in canvases}
            
            while remaining:
                remaining_keys = set(remaining.keys())
                all_parents = {
                    canvas.parent_canvas.settings.key 
                    for canvas in remaining.values() 
                    if canvas.parent_canvas 
                    and canvas.parent_canvas.settings.key in remaining_keys
                    and canvas.parent_canvas.settings.key != canvas.settings.key
                }
                leaves = [
                    canvas for canvas in remaining.values() 
                    if canvas.settings.key not in all_parents
                ]
                
                if not leaves:
                    root_nodes = [
                        canvas for canvas in remaining.values()
                        if canvas.parent_canvas 
                        and canvas.parent_canvas.settings.key == canvas.settings.key
                    ]
                    if root_nodes:
                        result.extend(root_nodes)
                        break
                    else:
                        raise ValueError("Circular dependency detected in canvas hierarchy")
                
                result.extend(leaves)
                
                for leaf in leaves:
                    del remaining[leaf.settings.key]
            
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
    ) -> GraphAutoma:
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
                        AsTopLevelDynamicCallback, 
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
        if canvas.is_top_level():
            automa = self
            params_data = get_func_signature_data(canvas.worker_material.arun)
            set_method_signature(self.arun, params_data)
        else:
            automa = canvas.worker_material

        
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
                        AsWorkerDynamicCallback,
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

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return await super().arun(*args, **kwargs)


def build_concurrent(
    automa: ConcurrentAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> None:
    # TODO: need to add callback builders of ConcurrentAutoma
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
            # callback_builders=callback_builders,
        )
    elif isinstance(worker_material, Callable):
        automa.add_func_as_worker(
            key=key,
            func=worker_material,
            # callback_builders=callback_builders,
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


class DynamicCallback(WorkerCallback):
    def __init__(self, __dynamic_lambda_worker__: _Element):
        self.__dynamic_lambda_worker__ = __dynamic_lambda_worker__
        self.__dynamic_worker_keys__ = []

    def get_dynamic_worker_settings(self, worker_material: Union[Worker, Callable]) -> Settings:
        settings = getattr(worker_material, "__settings__", Settings())
        return settings

    def get_dynamic_worker_data(self, worker_material: Union[Worker, Callable]) -> Data:
        data = getattr(worker_material, "__data__", Data())
        return data

    def _generate_dynamic_worker_key(self, automa_name: str) -> str:
        return f"{automa_name}-dynamic-worker-{uuid.uuid4().hex[:8]}"

    def _update_dynamic_worker_params(self, worker_material: Union[Worker, Callable], data: Data) -> None:
        if isinstance(worker_material, Worker):
            worker_name = worker_material.__class__.__name__
            override_func = worker_material.arun if worker_material._is_arun_overridden() else worker_material.run
            override_func_signature(worker_name, override_func, data.data)
        elif isinstance(worker_material, Callable):
            func_name = getattr(worker_material, "__name__", repr(worker_material))
            override_func_signature(func_name, worker_material, data.data)


    def build_dynamic_workers(
        self, 
        lambda_func: Callable,
        in_args: Tuple[Any, ...],
        in_kwargs: Dict[str, Any],
        automa: GraphAutoma,
    ) -> None:
        args = [
            item 
            if not isinstance(item, Distribute) 
            else item.data 
            for item in in_args
        ]
        kwargs = {
            key: item 
            if not isinstance(item, Distribute) 
            else item.data 
            for key, item in in_kwargs.items()
        }
        dynamic_worker_materials = lambda_func(*args, **kwargs)
        for worker_material in dynamic_worker_materials:
            # get the settings and data of the dynamic worker
            dynamic_worker_settings = self.get_dynamic_worker_settings(worker_material)
            dynamic_worker_data = self.get_dynamic_worker_data(worker_material)
            dynamic_worker_key = (
                dynamic_worker_settings.key 
                if dynamic_worker_settings.key 
                else self._generate_dynamic_worker_key(automa.name)
            )
            self._update_dynamic_worker_params(worker_material, dynamic_worker_data)

            # build the dynamic worker
            # TODO: need to add callback builders to the dynamic worker
            build_concurrent(
                automa=automa,
                key=dynamic_worker_key,
                worker_material=worker_material,
            )
            self.__dynamic_worker_keys__.append(dynamic_worker_key)


class AsTopLevelDynamicCallback(DynamicCallback):
    def __init__(self, __dynamic_lambda_worker__: _Element):
        super().__init__(__dynamic_lambda_worker__)

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

        # get the specific automa to add dynamic workers
        specific_automa = parent

        # build the dynamic workers
        self.build_dynamic_workers(
            lambda_func=self.__dynamic_lambda_worker__.worker_material,
            in_args=arguments["args"],
            in_kwargs=arguments["kwargs"],
            automa=specific_automa
        )

    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        if not is_top_level:
            return

        # get the specific automa to remove dynamic workers
        specific_automa = parent

        # remove the dynamic workers from the specific automa
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)
        

class AsWorkerDynamicCallback(DynamicCallback):
    def __init__(self, __dynamic_lambda_worker__: _Element):
        super().__init__(__dynamic_lambda_worker__)

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        # get the specific automa to add dynamic workers
        specific_automa = parent._get_worker_instance(key).get_decorated_worker()

        # build the dynamic workers
        self.build_dynamic_workers(
            lambda_func=self.__dynamic_lambda_worker__.worker_material,
            in_args=arguments["args"],
            in_kwargs=arguments["kwargs"],
            automa=specific_automa
        )
            
    async def on_worker_end(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
        result: Any = None,
    ) -> None:
        # get the specific automa to remove dynamic workers
        specific_automa = parent._get_worker_instance(key).get_decorated_worker()

        # remove the dynamic workers from the specific automa
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)
