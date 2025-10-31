import copy
import inspect
import functools
from typing import Callable, List, Any, Dict, Tuple, Union

from bridgic.core.automa import GraphAutoma
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.automa.worker import Worker
from bridgic.core.automa.args import ArgsMappingRule, System
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.core.agentic.asl._canvas_object import (
    _Canvas,
    _Element,
    TrackingNamespace,
    set_method_signature,
)


class ASLAutomaMeta(GraphMeta):
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        # collect the input parameters
        annotations = namespace.get("__annotations__", {}) or {}
        input_params = {}
        for param_name, param_type in annotations.items():
            default_value = namespace.get(param_name, None)
            input_params[param_name] = {
                "type": param_type,
                "default": default_value,
            }

        # dynamically set the signature of the arun method
        arun_func = getattr(cls, "arun", None)
        input_params["self"] = {
            "type": inspect._empty,
            "default": inspect._empty,
        }
        set_method_signature(arun_func, input_params)

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
            automa_type = canvas.automa_type
            automa = canvas.worker_material
            elements = canvas.elements
            self._inner_build_logic_flow(automa_type, automa, elements)
            
    def _inner_build_logic_flow(self, automa_type: str, automa: GraphAutoma, elements: Dict[str, "_Element"]) -> None:
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
            
            if automa_type == "concurrent":
                build_concurrent(
                    automa=automa,
                    key=key,
                    worker_material=worker_material,
                )
            elif automa_type == "graph":
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
                raise ValueError(f"Invalid automa type: {automa_type}")

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