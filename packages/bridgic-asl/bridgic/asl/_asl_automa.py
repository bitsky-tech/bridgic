import copy
import uuid
import enum
import warnings
import inspect
from typing import Callable, List, Any, Dict, Tuple, Union, Optional
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from bridgic.core.automa.args._args_binding import ArgsMappingRule, ResultDispatchingRule, InOrder, override_func_signature, set_method_signature, safely_map_args
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.asl._error import ASLCompilationError
from bridgic.core.utils._inspect_tools import get_param_names_all_kinds
from bridgic.asl._canvas_object import _Canvas, _Element, _CanvasObject, _Fragment, graph_stack, Settings, Data, KeyUnDifined, _GraphContextManager


class TrackingNamespace(dict):
    """
    A custom dictionary that tracks and manages canvas objects and elements during class definition.
    
    This namespace is used as the class preparation namespace in the metaclass to automatically
    register canvases and elements into their respective parent canvases. It handles
    the extraction of settings and data from worker materials and creates appropriate _Element
    or _Canvas instances.
    """
    def __init__(self):
        super().__init__()
        self.canvas_definition_start = False

    def __setitem__(self, key: str, value: Any):
        """
        Register a key-value pair in the tracking namespace.
        
        This method handles the registration of canvases and elements. It extracts
        settings and data from worker materials, creates _Element instances for workers and
        callables, and registers them to their parent canvases. 
        
        Parameters
        ----------
        key : str
            The key to register in the namespace.
        value : Any
            The value to register. Can be a Worker, Callable, _Canvas, _Element, or any other object.
            
        Raises
        ------
        ASLCompilationError
            - If a duplicate canvas key is detected.
            - If a worker is declared outside of any canvas.

        Notes
        -----
        - If a value is already an _Element, it skips re-registration to handle fragment declarations.
        - If a value is a _Canvas and has already been declared, it also indicates the corresponding
          canvas object is a fragment.
        """
        # Before the class body is truly executed, the class body definition of the
        # parent class will be executed first, so we skip them for now.
        if not self.canvas_definition_start:
            super().__setitem__(key, value)
            if key == '__qualname__' and value != "ASLAutoma":
                self.canvas_definition_start = True
            return

        # If the value is a _CanvasObject before the object is registered, it indicates that
        # the corresponding canvas object is a fragment.
        # TODO: Need Fragment class to handle the fragment logic.
        if isinstance(value, _CanvasObject):
            stack = list[_Canvas](graph_stack.get())
            parent_canvas: _Canvas = stack[-1]
            parent_canvas_namespace = super().__getitem__(parent_canvas.key)
            parent_canvas_fragment_namespace = parent_canvas_namespace.get('__fragment__')
            if parent_canvas_namespace.get(key) or parent_canvas_fragment_namespace.get(key):
                raise ASLCompilationError(
                    f"Duplicate key: {key} under graph: {parent_canvas.key} of a fragment or a registered worker."
                )
            
            value = _Fragment(key, value)
            parent_canvas_fragment_namespace[key] = value

        # Get the settings and data from the value and clear the settings and data of the value.
        settings = copy.deepcopy(getattr(value, "__settings__", None))
        if settings:
            setattr(value, "__settings__", None)
        data = copy.deepcopy(getattr(value, "__data__", None))
        if data:
            setattr(value, "__data__", None)

        # Track the _Element values.
        if (
            (isinstance(value, Worker) or isinstance(value, Callable) or isinstance(value, ASLAutoma)) and 
            not isinstance(value, _GraphContextManager)
        ):
            # Create the _Element instance.
            element: _Element = _Element(key, value)

            # Check if the value is a lambda function.
            if isinstance(value, Callable) and getattr(value.__code__, "co_name", None) == '<lambda>':
                element.is_lambda = True
                element.cached_param_names = get_param_names_all_kinds(value)

            # Replace the value with the element.
            value = element
        
        # Track the _Canvas values.
        if isinstance(value, _GraphContextManager):
            stack = list[_Canvas](graph_stack.get())
            current_canvas: _Canvas = stack[-1]
            value = current_canvas

        # Register the _CanvasObject to the current canvas.
        if isinstance(value, _CanvasObject):
            # Update the key, settings and data of the canvas object.
            if isinstance(value.key, KeyUnDifined):
                value.key = key
            if settings:
                value.update_settings(settings)
            if data:
                value.update_data(data)
            
            # Get the parent canvas and register to the parent canvas.
            stack = list[_Canvas](graph_stack.get())
            parent_canvas: Optional[_Canvas] = (
                None if len(stack) == 0 
                else stack[-1] if isinstance(value, _Element) 
                else None if len(stack) == 1 else stack[-2]
            )
            parent_key = parent_canvas.key if parent_canvas else None

            if parent_key and parent_key == key:
                raise ASLCompilationError(f"Invalid worker key: {key}, cannot use the canvas itself as a worker.")
            if isinstance(parent_key, KeyUnDifined):
                raise ASLCompilationError(f"The parent of worker {key} has no name! Please declare the parent key with `with graph as <name>:`.")

            if isinstance(value, _Element):
                if parent_canvas:
                    parent_canvas_namespace = super().__getitem__(parent_canvas.key)
                    parent_canvas_fragment_namespace = parent_canvas_namespace.get('__fragment__')
                    if parent_canvas_namespace.get(value.key) or parent_canvas_fragment_namespace.get(value.key):
                        raise ASLCompilationError(f"Duplicate key: {value.key} under graph: {parent_canvas.key} of a fragment or a registered worker.")
                    parent_canvas_namespace[value.key] = value
                    parent_canvas.register(key, value)
                else:
                    # Check if element is declared outside any canvas
                    if isinstance(value, _Element):
                        raise ASLCompilationError("All workers must be written under one graph.")
            elif isinstance(value, _Canvas):
                if parent_canvas:
                    parent_canvas_namespace = super().__getitem__(parent_canvas.key)
                    parent_canvas_fragment_namespace = parent_canvas_namespace.get('__fragment__')
                    if parent_canvas_namespace.get(value.key) or parent_canvas_fragment_namespace.get(value.key):
                        raise ASLCompilationError(f"Duplicate key: {value.key} under graph: {parent_canvas.key} of a fragment or a registered worker.")
                    parent_canvas_namespace[value.key] = value
                    parent_canvas.register(key, value)
                super().__setitem__(key, {})
                current_canvas_namespace = super().__getitem__(key)
                current_canvas_namespace["__self__"] = value
                current_canvas_namespace["__fragment__"] = {}
        else:
            # record the normal key-value pair in the tracking namespace
            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """
        Retrieve an item from the tracking namespace.
        
        This method first checks if the key exists in the current canvas's namespace. If not,
        it falls back to the parent namespace. This allows for proper scoping of elements
        within nested canvases.
        
        Parameters
        ----------
        key : str
            The key to retrieve from the namespace.
            
        Returns
        -------
        Dict[str, Any]
            The value associated with the key, typically an _Element or _Canvas instance.
            
        Raises
        ------
        ASLCompilationError
            If attempting to access the canvas itself using its own key.
        """
        # get the current canvas
        stack = list[_Canvas](graph_stack.get())

        # If the stack is empty, it indicates that the element may be a normal object.
        if not stack:
            return super().__getitem__(key)

        # Get the current canvas and its namespace.
        current_canvas: _Canvas = stack[-1]
        current_canvas_key = current_canvas.key
        current_canvas_namespace = super().__getitem__(current_canvas_key)
        current_canvas_fragment_namespace = current_canvas_namespace.get('__fragment__')

        if key == current_canvas_key:
            raise ASLCompilationError(f"Invalid worker key: {key}, cannot use the canvas itself as a worker.")

        # If the key is in the current canvas namespace, return the element.
        if key in current_canvas_namespace:
            return current_canvas_namespace[key]
        if key in current_canvas_fragment_namespace:
            return current_canvas_fragment_namespace[key]

        return super().__getitem__(key)


class ASLAutomaMeta(GraphMeta):
    """
    Metaclass for ASLAutoma that collects and organizes canvases during class definition.
    
    This metaclass uses TrackingNamespace to intercept class attribute assignments and
    automatically collect all canvas definitions. It then organizes them in bottom-up order
    to ensure proper initialization sequence.
    """
    @classmethod
    def __prepare__(mcls, name, bases):
        return TrackingNamespace()

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        top_canvas = cls._collect_canvases(namespace)
        setattr(cls, "_top_canvas", top_canvas)
        return cls

    @classmethod
    def _collect_canvases(mcls, dct: Dict[str, Any]) -> _Canvas:
        """
        Collect all canvases from the class namespace and organize them in bottom-up order.
        
        This method identifies all canvas objects in the namespace, finds the root canvas,
        and organizes all canvases in a bottom-up traversal order. This ensures that nested
        canvases are processed before their parent canvases during graph construction.
        
        Parameters
        ----------
        dct : Dict[str, Any]
            The class namespace dictionary containing all attributes.
            
        Returns
        -------
        List[_Canvas]
            A list of canvases ordered from leaves to root (bottom-up order).
            
        Raises
        ------
        ASLCompilationError
            - If multiple root canvases are detected.
            - If a circular dependency exists in the canvas hierarchy.
            
        Raises
        ------
        ASLCompilationError
            - If multiple root canvases are detected.
            - If a circular dependency exists
        """
        # Collect all canvases and elements.
        root = None
        canvases_dict: Dict[str, _Canvas] = {}

        for _, value in dct.items():
            if isinstance(value, Dict) and "__self__" in value:
                value = value["__self__"]

                # Find the root canvas and set its parent canvas to itself.
                if not value.parent_canvas:
                    value.parent_canvas = value
                    if not root:
                        root = value
                    else:
                        raise ASLCompilationError("Multiple root graph are not allowed.")

                # Record the canvas to the canvases dictionary.
                canvases_dict[value.key] = value

        # Bottom-up order traversal to get the canvases.
        def bottom_up_order_traversal(canvases: List[_Canvas]) -> List[_Canvas]:
            if len(canvases) == 0:
                return []
            if len(canvases) == 1:
                return canvases
            
            result: List[_Canvas] = []
            remaining = {canvas.key: canvas for canvas in canvases}
            
            while remaining:
                remaining_keys = set(remaining.keys())
                all_parents = {
                    canvas.parent_canvas.key 
                    for canvas in remaining.values() 
                    if canvas.parent_canvas 
                    and canvas.parent_canvas.key in remaining_keys
                    and canvas.parent_canvas.key != canvas.key
                }
                leaves = [
                    canvas for canvas in remaining.values() 
                    if canvas.key not in all_parents
                ]
                
                if not leaves:
                    root_nodes = [
                        canvas for canvas in remaining.values()
                        if canvas.parent_canvas 
                        and canvas.parent_canvas.key == canvas.key
                    ]
                    if root_nodes:
                        result.extend(root_nodes)
                        break
                    else:
                        raise ASLCompilationError("Circular dependency detected in canvas hierarchy")
                
                result.extend(leaves)
                
                for leaf in leaves:
                    del remaining[leaf.key]
            
            return result

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        return bottom_up_order


class ASLAutoma(GraphAutoma, metaclass=ASLAutomaMeta):
    """
    An automa that builds agent structures from ASL (Agent Structured Language) definitions.
    
    This class extends `GraphAutoma` and uses a declarative syntax to define workflows. It
    automatically builds the graph structure from definitions during initialization, handling 
    both static and dynamic worker registration.

    Examples
    --------
    >>> from bridgic.core.agentic.asl import graph, ASLAutoma
    >>> 
    >>> def add_one(x: int):
    ...     return x + 1
    >>> 
    >>> def add_two(x: int):
    ...     return x + 2
    >>> 
    >>> class MyGraph(ASLAutoma):
    ...     with graph as g:
    ...         a = add_one
    ...         b = add_two
    ...         +a >> b  # a is the start worker, b depends on a and is the output worker.
    >>> 
    >>> graph = MyGraph()
    >>> result = await graph.arun(x=1)  # result: 4 (1+1+2)
    """
    # The canvases of the automa (stored in bottom-up order).
    _top_canvas: _Canvas = None

    def __init__(
        self, 
        name: str = None, 
        thread_pool: Optional[ThreadPoolExecutor] = None, 
        running_options: Optional[RunningOptions] = None
    ):
        """
        Initialize the ASLAutoma instance.
        
        Parameters
        ----------
        name : str, optional
            The name of the automa. If None, a default name will be assigned.
        thread_pool : ThreadPoolExecutor, optional
            The thread pool for parallel running of I/O-bound tasks. If None, a default thread pool will be used.
        running_options : RunningOptions, optional
            The running options for the automa. If None, a default running options will be used.
        """
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)
        self._dynamic_workers = {}
        if not self._top_canvas:
            self.automa = GraphAutoma(name=name, thread_pool=thread_pool, running_options=running_options)
        else:
            top_canvas = self._top_canvas[-1]
            self.automa: GraphAutoma = self._build_graph(top_canvas)

    def _build_graph(self, canvas: _Canvas) -> GraphAutoma:
        """
        Build the graph structure from all canvases.
        
        This method iterates through all canvases in bottom-up order and builds the graph
        structure for each canvas. It separates static and dynamic elements and delegates
        the actual building to _inner_build_graph.
        """
        static_elements = {
            key: value 
            for key, value in canvas.elements.items() 
            if (isinstance(value, _Element) and not value.is_lambda) or isinstance(value, _Canvas)
        }
        dynamic_elements = {
            key: value 
            for key, value in canvas.elements.items() 
            if (isinstance(value, _Element) and value.is_lambda)
        }
        return self._inner_build_graph(canvas, static_elements, dynamic_elements)
            
    def _inner_build_graph(
        self, 
        canvas: _Canvas, 
        static_elements: Dict[str, "_Element"],
        dynamic_elements: Dict[str, "_Element"]
    ) -> GraphAutoma:
        """
        Build the graph structure for a specific canvas.
        
        This method handles the construction of both dynamic and static logic flows. For dynamic
        elements (lambda functions), it sets up callbacks that will add workers at runtime and remove 
        them when the execution completes. For static elements, it immediately adds them to the 
        graph with their dependencies and settings.
        
        Parameters
        ----------
        canvas : _Canvas
            The canvas to build the graph for.
        static_elements : Dict[str, "_Element"]
            Dictionary of static elements (non-lambda workers) to add to the graph.
        dynamic_elements : Dict[str, "_Element"]
            Dictionary of dynamic elements (lambda functions) that will generate workers at runtime.
        """
        automa = None
        current_canvas_key = canvas.key

        ###############################
        # build the dynamic logic flow
        ###############################
        running_options_callback = []
        for _, element in dynamic_elements.items():
            worker_material = element.worker_material
            params_names = element.cached_param_names

            # If the canvas is top level, use `RunningOptions` to add callback.
            if canvas.is_top_level():
                running_options_callback.append(
                    WorkerCallbackBuilder(
                        AsTopLevelDynamicCallback, 
                        init_kwargs={"__dynamic_lambda_func__": worker_material, "__param_names__": simplify_param_names(params_names)}
                    )
                )

            # Otherwise, delegate parent automa to add callback during building graph.
            else:
                parent_key = canvas.parent_canvas.key
                if parent_key not in self._dynamic_workers:
                    self._dynamic_workers[parent_key] = {}
                if current_canvas_key not in self._dynamic_workers[parent_key]:
                    self._dynamic_workers[parent_key][current_canvas_key] = []
                self._dynamic_workers[parent_key][current_canvas_key].append(element)
            
        # Make the automa.
        canvas.make_automa(running_options=RunningOptions(callback_builders=running_options_callback))
        automa = canvas.worker_material
        if canvas.is_top_level():
            params_data = canvas.worker_material.get_input_param_names()
            set_method_signature(self.arun, params_data)

        
        ###############################
        # build the static logic flow
        ###############################
        for _, element in static_elements.items():
            key = element.key
            worker_material = element.worker_material
            is_start = element.is_start
            is_output = element.is_output
            dependencies = [item.key for item in element.dependencies]
            args_mapping_rule = element.args_mapping_rule
            result_dispatching_rule = element.result_dispatching_rule

            # If the object is an instance of an object instance, it must be ensured that each time
            # an instance of the current ASLAutoma is created, it is an independent one of this object
            # instance. Here, the object has these forms:
            #   1. Canvas:
            #     a. graph (exactly is GraphAutoma) or concurrent (exactly is ConcurrentAutoma) etc.
            #   2. Element:
            #     a. Callable
            #     b. ASLAutoma
            #     c. GraphAutoma etc.
            #     d. Worker
            if isinstance(element, _Canvas):
                worker_material = self._build_graph(element)
            elif isinstance(element, _Element):
                if isinstance(worker_material, ASLAutoma):
                    asl_automa_class = type(worker_material)
                    worker_material = asl_automa_class(
                        name=getattr(worker_material, "name", None), 
                        thread_pool=getattr(worker_material, "thread_pool", None), 
                        running_options=getattr(worker_material, "running_options", None)
                    )
                elif isinstance(worker_material, GraphAutoma):
                    graph_automa_class = type(worker_material)
                    worker_material = graph_automa_class(
                        name=getattr(worker_material, "name", None), 
                        thread_pool=getattr(worker_material, "thread_pool", None), 
                        running_options=getattr(worker_material, "running_options", None)
                    )
                elif isinstance(worker_material, Worker):
                    worker_material = _copy_worker_safely(worker_material)
                elif isinstance(worker_material, Callable):
                    pass
            else:
                raise ValueError(f"Invalid worker material type: {type(worker_material)}.")

            # Prepare the callback builders.
            # If current element delegated dynamic workers to be added in current canvas.
            callback_builders = []
            if current_canvas_key in self._dynamic_workers and key in self._dynamic_workers[current_canvas_key]:
                delegated_dynamic_workers = self._dynamic_workers[current_canvas_key][key]
                for delegated_dynamic_element in delegated_dynamic_workers:
                    delegated_dynamic_func = delegated_dynamic_element.worker_material
                    delegated_dynamic_params_names = delegated_dynamic_element.cached_param_names
                    callback_builders.append(WorkerCallbackBuilder(
                        AsWorkerDynamicCallback,
                        init_kwargs={"__dynamic_lambda_func__": delegated_dynamic_func, "__param_names__": simplify_param_names(delegated_dynamic_params_names)}
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
                    result_dispatching_rule=result_dispatching_rule,
                    callback_builders=callback_builders
                )
            else:
                raise ValueError(f"Invalid automa type: {type(automa)}.")
        return automa

    def dump_to_dict(self) -> Dict[str, Any]:
        """
        Dump the ASLAutoma instance to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing the serialized state of the ASLAutoma instance.
        """
        state_dict = super().dump_to_dict()
        state_dict["automa"] = self.automa.dump_to_dict()
        return state_dict

    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the ASLAutoma instance from a dictionary.
        
        Parameters
        ----------
        state_dict : Dict[str, Any]
            A dictionary containing the serialized state of the ASLAutoma instance.
        """
        super().load_from_dict(state_dict)
        self.automa = state_dict["automa"]

    async def arun(
        self,
        *args: Tuple[Any, ...],
        feedback_data = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Run the automa asynchronously.
        
        Parameters
        ----------
        *args : Tuple[Any, ...]
            Positional arguments to pass to the automa.
        feedback_data : Any, optional
            Feedback data for the execution (default: None).
        **kwargs : Dict[str, Any]
            Keyword arguments to pass to the automa.
            
        Returns
        -------
        Any
            The result of the automa execution.
        """
        if not self.automa:
            return 

        res = await self.automa.arun(*args, feedback_data=feedback_data, **kwargs)
        return res

    def __str__(self) -> str:
        return f"ASLAutoma(automa={self.automa})"

    def __repr__(self) -> str:
        return self.__str__()


def simplify_param_names(param_names: Dict[enum.IntEnum, List[Tuple[str, Any]]]) -> Dict[str, List[Tuple[str, Any]]]:
    """
    Simplify the parameter names dictionary for serialization.
    
    This function converts the parameter names dictionary from using enum.IntEnum keys
    to string keys for serialization in CallbackBuilder and DynamicCallback.
    
    Parameters
    ----------
    param_names : Dict[enum.IntEnum, List[Tuple[str, Any]]]
        The parameter names dictionary with enum keys.
        
    Returns
    -------
    Dict[str, List[Tuple[str, Any]]]
        The simplified parameter names dictionary with string keys.
    """
    res = {}
    for kind, param_list in param_names.items():
        # Convert enum.IntEnum to its string name (e.g., "POSITIONAL_OR_KEYWORD")
        kind_name = kind.name if hasattr(kind, 'name') else str(kind)
        res[kind_name] = param_list
    return res


def recover_param_names_dict(simplified_param_names_dict: Dict[str, List[Tuple[str, Any]]]) -> Dict[enum.IntEnum, List[Tuple[str, Any]]]:
    """
    Recover the parameter names dictionary from simplified format to original format.
    
    This function converts the parameter names dictionary from using string keys back
    to enum.IntEnum keys after deserialization.
    
    Parameters
    ----------
    simplified_param_names_dict : Dict[str, List[Tuple[str, Any]]]
        The simplified parameter names dictionary with string keys.
        
    Returns
    -------
    Dict[enum.IntEnum, List[Tuple[str, Any]]]
        The recovered parameter names dictionary with enum keys.
    """
    kind_map = {
        "POSITIONAL_ONLY": inspect.Parameter.POSITIONAL_ONLY,
        "POSITIONAL_OR_KEYWORD": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "VAR_POSITIONAL": inspect.Parameter.VAR_POSITIONAL,
        "KEYWORD_ONLY": inspect.Parameter.KEYWORD_ONLY,
        "VAR_KEYWORD": inspect.Parameter.VAR_KEYWORD,
    }
    
    res = {}
    for kind_name, param_list in simplified_param_names_dict.items():
        if kind_name in kind_map:
            res[kind_map[kind_name]] = param_list
    return res


def build_concurrent(
    automa: ConcurrentAutoma,
    key: str,
    worker_material: Union[Worker, Callable],
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> None:
    """
    Add a worker to a ConcurrentAutoma instance.
    
    This helper function adds either a Worker instance or a callable function to a
    ConcurrentAutoma with the specified key. The callback builders are currently not
    supported for ConcurrentAutoma.
    
    Parameters
    ----------
    automa : ConcurrentAutoma
        The ConcurrentAutoma instance to add the worker to.
    key : str
        The unique identifier for the worker.
    worker_material : Union[Worker, Callable]
        Either a Worker instance or a callable function to add as a worker.
    callback_builders : List[WorkerCallbackBuilder], optional
        List of callback builders (currently not used for ConcurrentAutoma).
        
    Notes
    -----
    TODO: Support for callback builders in ConcurrentAutoma is planned for future implementation.
    """
    # TODO: Need to add callback builders of ConcurrentAutoma.
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
    result_dispatching_rule: ResultDispatchingRule,
    callback_builders: List[WorkerCallbackBuilder] = [],
) -> None:
    """
    Add a worker to a GraphAutoma instance.
    
    This helper function adds either a Worker instance or a callable function to a
    GraphAutoma with the specified configuration including start/output flags, dependencies,
    argument mapping rules, and callback builders.
    
    Parameters
    ----------
    automa : GraphAutoma
        The GraphAutoma instance to add the worker to.
    key : str
        The unique identifier for the worker.
    worker_material : Union[Worker, Callable]
        Either a Worker instance or a callable function to add as a worker.
    is_start : bool
        Whether this worker is a start worker in the graph.
    is_output : bool
        Whether this worker is an output worker in the graph.
    dependencies : List[str]
        List of worker keys that this worker depends on.
    args_mapping_rule : ArgsMappingRule
        The rule for mapping arguments to this worker.
    result_dispatching_rule : ResultDispatchingRule
        The rule for dispatching results to this worker.
    callback_builders : List[WorkerCallbackBuilder], optional
        List of callback builders to attach to this worker.
    """
    if isinstance(worker_material, Worker):
        automa.add_worker(
            key=key,
            worker=worker_material,
            is_start=is_start,
            is_output=is_output,
            dependencies=dependencies,
            args_mapping_rule=args_mapping_rule,
            result_dispatching_rule=result_dispatching_rule,
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
            result_dispatching_rule=result_dispatching_rule,
            callback_builders=callback_builders
        )


class DynamicCallback(WorkerCallback):
    """
    Base callback class for handling dynamic worker creation from lambda functions.
    
    This callback is responsible for executing lambda functions at runtime to generate
    dynamic workers and adding them to the automa. It tracks the generated worker
    keys for later cleanup.
    
    Attributes
    ----------
    __dynamic_lambda_func__ : Callable
        The lambda function that generates dynamic workers.
    __param_names__ : Dict[str, List[Tuple[str, Any]]]
        The parameter names dictionary in simplified format (string keys).
    __dynamic_worker_keys__ : List[str]
        List of keys for dynamically created workers, used for cleanup.
    """
    def __init__(self, __dynamic_lambda_func__: Callable, __param_names__: Dict[str, List[Tuple[str, Any]]]):
        """
        Initialize the dynamic callback.
        
        Parameters
        ----------
        __dynamic_lambda_func__ : Callable
            The lambda function that will generate dynamic workers.
        __param_names__ : Dict[str, List[Tuple[str, Any]]]
            The parameter names dictionary in simplified format (string keys).
        """
        super().__init__()
        self.__dynamic_lambda_func__ = __dynamic_lambda_func__
        self.__param_names__ = __param_names__
        self.__dynamic_worker_keys__ = []


    def get_dynamic_worker_settings(self, worker_material: Union[Worker, Callable]) -> Settings:
        """
        Extract settings from a worker material.
        
        Parameters
        ----------
        worker_material : Union[Worker, Callable]
            The worker material to extract settings from.
            
        Returns
        -------
        Settings
            The settings object, or a new default Settings instance if none exists.
        """
        settings = getattr(worker_material, "__settings__", Settings())
        return settings

    def get_dynamic_worker_data(self, worker_material: Union[Worker, Callable]) -> Data:
        """
        Extract data configuration from a worker material.
        
        Parameters
        ----------
        worker_material : Union[Worker, Callable]
            The worker material to extract data from.
            
        Returns
        -------
        Data
            The data object, or a new default Data instance if none exists.
        """
        data = getattr(worker_material, "__data__", Data())
        return data

    def _generate_dynamic_worker_key(self, automa_name: str) -> str:
        """
        Generate a unique key for a dynamic worker.
        
        Parameters
        ----------
        automa_name : str
            The name of the automa to include in the key.
            
        Returns
        -------
        str
            A unique key in the format "{automa_name}-dynamic-worker-{uuid}".
        """
        return f"{automa_name}-dynamic-worker-{uuid.uuid4().hex[:8]}"

    def _update_dynamic_worker_params(self, worker_material: Union[Worker, Callable], data: Data) -> None:
        """
        Update the function signature of a worker material based on data configuration.
        
        This method overrides the function signature of the worker material to match the
        data configuration, allowing dynamic parameter injection.
        
        Parameters
        ----------
        worker_material : Union[Worker, Callable]
            The worker material whose signature should be updated.
        data : Data
            The data configuration containing parameter type and default value information.
        """
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
        simplified_param_names_dict: Dict[str, List[Tuple[str, Any]]],
        in_args: Tuple[Any, ...],
        in_kwargs: Dict[str, Any],
        automa: GraphAutoma,
    ) -> None:
        """
        Execute a lambda function to generate dynamic workers and add them to the automa.
        
        This method processes the input arguments (extracting dispatching objects), executes
        the lambda function, and then adds each returned worker material to the automa
        as a dynamic worker. The generated worker keys are tracked for cleanup.
        
        Parameters
        ----------
        lambda_func : Callable
            The lambda function that generates dynamic workers when called.
        in_args : Tuple[Any, ...]
            Positional arguments to pass to the lambda function.
        in_kwargs : Dict[str, Any]
            Keyword arguments to pass to the lambda function.
        automa : GraphAutoma
            The automa instance to add the dynamic workers to.
            
        Notes
        -----
        Dispatching objects in the arguments are automatically unwrapped before passing
        to the lambda function.
        """
        args = [
            item 
            if not isinstance(item, InOrder) 
            else item.data 
            for item in in_args
        ]
        kwargs = {
            key: item 
            if not isinstance(item, InOrder) 
            else item.data 
            for key, item in in_kwargs.items()
        }
        rx_param_names_dict = recover_param_names_dict(simplified_param_names_dict)
        rx_args, rx_kwargs = safely_map_args(args, kwargs, rx_param_names_dict)
        dynamic_worker_materials = lambda_func(*rx_args, **rx_kwargs)
        for worker_material in dynamic_worker_materials:
            # Get the settings and data of the dynamic worker.
            dynamic_worker_settings = self.get_dynamic_worker_settings(worker_material)
            dynamic_worker_data = self.get_dynamic_worker_data(worker_material)
            dynamic_worker_key = (
                dynamic_worker_settings.key 
                if dynamic_worker_settings.key 
                else self._generate_dynamic_worker_key(automa.name)
            )
            self._update_dynamic_worker_params(worker_material, dynamic_worker_data)

            # Build the dynamic worker.
            # TODO: Need to add callback builders to the dynamic worker.
            build_concurrent(
                automa=automa,
                key=dynamic_worker_key,
                worker_material=worker_material,
            )
            self.__dynamic_worker_keys__.append(dynamic_worker_key)

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        """
        Dump the DynamicCallback instance to a dictionary.
        """
        state_dict = super().dump_to_dict()
        state_dict["__dynamic_lambda_func__"] = self.__dynamic_lambda_func__
        state_dict["__param_names__"] = self.__param_names__
        state_dict["__dynamic_worker_keys__"] = self.__dynamic_worker_keys__
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the DynamicCallback instance from a dictionary.
        
        Parameters
        ----------
        state_dict : Dict[str, Any]
            A dictionary containing the serialized state of the DynamicCallback instance.
            
        Notes
        -----
        The lambda function is deserialized based on how it was serialized:
        1. If serialized by name, load it using load_qualified_class_or_func
        2. If serialized by pickle, load it using pickle.loads
        
        Note: For bound methods that were serialized by name, they will need to be
        rebound to the appropriate instance after deserialization if needed.
        """
        super().load_from_dict(state_dict)
        self.__dynamic_lambda_func__ = state_dict["__dynamic_lambda_func__"]
        self.__dynamic_worker_keys__ = state_dict["__dynamic_worker_keys__"]
        self.__param_names__ = state_dict["__param_names__"]


class AsTopLevelDynamicCallback(DynamicCallback):
    """
    Callback for handling dynamic workers at the top level of the automa.
    
    This callback is attached to the top-level automa and generates dynamic workers
    when the automa starts executing. It removes the dynamic workers when execution
    completes.
    """
    def __init__(self, __dynamic_lambda_func__: Callable, __param_names__: Dict[str, List[Tuple[str, Any]]]):
        super().__init__(__dynamic_lambda_func__, __param_names__)

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        """
        Called when a top-level worker starts execution.
        
        This method generates dynamic workers by executing the lambda function with the
        provided arguments and adds them to the parent automa.
        
        Parameters
        ----------
        key : str
            The key of the worker that started (unused for top-level callbacks).
        is_top_level : bool, optional
            Whether this is a top-level worker. Only processes if True.
        parent : GraphAutoma, optional
            The parent automa to add dynamic workers to.
        arguments : Dict[str, Any], optional
            Dictionary containing 'args' and 'kwargs' keys with the execution arguments.
        """
        # If the worker is not the top-level worker, skip it.
        if not is_top_level:
            return

        # Get the specific automa to add dynamic workers.
        specific_automa = parent

        # Build the dynamic workers.
        self.build_dynamic_workers(
            lambda_func=self.__dynamic_lambda_func__,
            simplified_param_names_dict=self.__param_names__,
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
        """
        Called when a top-level worker ends execution.
        
        This method removes all dynamically created workers from the parent automa
        to clean up resources.
        
        Parameters
        ----------
        key : str
            The key of the worker that ended (unused for top-level callbacks).
        is_top_level : bool, optional
            Whether this is a top-level worker. Only processes if True.
        parent : GraphAutoma, optional
            The parent automa to remove dynamic workers from.
        arguments : Dict[str, Any], optional
            The execution arguments (unused in cleanup).
        result : Any, optional
            The result of the worker execution (unused in cleanup).
        """
        if not is_top_level:
            return

        # Get the specific automa to remove dynamic workers.
        specific_automa = parent

        # Remove the dynamic workers from the specific automa.
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)
        

class AsWorkerDynamicCallback(DynamicCallback):
    """
    Callback for handling dynamic workers within a specific worker's automa.
    
    This callback is attached to a worker and generates dynamic workers within that
    worker's decorated automa (e.g., a nested GraphAutoma). It removes the dynamic
    workers when the worker completes execution.
    """
    def __init__(self, __dynamic_lambda_func__: Callable, __param_names__: Dict[str, List[Tuple[str, Any]]]):
        super().__init__(__dynamic_lambda_func__, __param_names__)

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        """
        Called when a worker starts execution.
        
        This method retrieves the worker's decorated automa and generates dynamic
        workers by executing the lambda function with the provided arguments.
        
        Parameters
        ----------
        key : str
            The key of the worker that started.
        is_top_level : bool, optional
            Whether this is a top-level worker (unused for worker callbacks).
        parent : GraphAutoma, optional
            The parent automa containing the worker.
        arguments : Dict[str, Any], optional
            Dictionary containing 'args' and 'kwargs' keys with the execution arguments.
        """
        # Get the specific automa to add dynamic workers.
        specific_automa = parent._get_worker_instance(key).get_decorated_worker()

        # Build the dynamic workers.
        self.build_dynamic_workers(
            lambda_func=self.__dynamic_lambda_func__,
            simplified_param_names_dict=self.__param_names__,
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
        """
        Called when a worker ends execution.
        
        This method retrieves the worker's decorated automa and removes all dynamically
        created workers to clean up resources.
        
        Parameters
        ----------
        key : str
            The key of the worker that ended.
        is_top_level : bool, optional
            Whether this is a top-level worker (unused for worker callbacks).
        parent : GraphAutoma, optional
            The parent automa containing the worker.
        arguments : Dict[str, Any], optional
            The execution arguments (unused in cleanup).
        result : Any, optional
            The result of the worker execution (unused in cleanup).
        """
        # Get the specific automa to remove dynamic workers.
        specific_automa = parent._get_worker_instance(key).get_decorated_worker()

        # Remove the dynamic workers from the specific automa.
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)


def _copy_worker_safely(worker: Worker) -> Worker:
    """
    Safely copy a Worker instance, handling unserializable objects.
    
    First attempts to use copy.deepcopy() for deep copying. If it encounters
    unserializable objects (e.g., thread locks, file handles, etc.), it falls
    back to an intelligent copying strategy: creates a new instance and deep copies
    serializable attributes as much as possible, while using shallow copy
    (shared reference) for unserializable attributes and issuing a warning.
    
    Parameters
    ----------
    worker : Worker
        The Worker instance to copy
        
    Returns
    -------
    Worker
        The copied Worker instance
        
    Warns
    -----
    UserWarning
        When encountering unserializable objects, a warning is issued indicating
        that certain attributes may be shared
    """
    try:
        # First attempt deep copy.
        return copy.deepcopy(worker)
    except (TypeError, ValueError, AttributeError) as e:
        # If deep copy fails (usually due to unserializable objects), use intelligent copying strategy.
        worker_class = type(worker)
        
        # Create new instance (without calling __init__ to avoid side effects).
        new_worker = worker_class.__new__(worker_class)
        
        # Get all attributes from the original instance.
        original_dict = worker.__dict__.copy()
        unserializable_attrs = []
        
        # Attributes that need special handling: these should not be deep copied,
        # or need to be reset after copying.
        # __parent usually points to an Automa instance and will be set correctly later, so skip it here.
        skip_attrs = {'_Worker__parent'}  # Use name-mangled attribute name
        
        # Try to deep copy each attribute individually.
        for attr_name, attr_value in original_dict.items():
            # Skip special attributes.
            if attr_name in skip_attrs:
                # For __parent, set it to the new instance itself (Worker's default behavior).
                if attr_name == '_Worker__parent':
                    setattr(new_worker, attr_name, new_worker)
                continue
            
            try:
                # Attempt to deep copy this attribute.
                setattr(new_worker, attr_name, copy.deepcopy(attr_value))
            except (TypeError, ValueError, AttributeError):
                # If this attribute cannot be deep copied, use shallow copy (shared reference).
                setattr(new_worker, attr_name, attr_value)
                # Only record user-defined attributes, not internal attributes.
                if not attr_name.startswith('_Worker__') and not attr_name.startswith('__'):
                    unserializable_attrs.append(attr_name)
        
        # If there are unserializable user-defined attributes, issue a warning.
        if unserializable_attrs:
            warnings.warn(
                f"Worker {worker_class.__name__} contains unserializable attributes: {unserializable_attrs}. "
                f"These attributes will be shared between the original and copied instances. "
                f"This may lead to unexpected behavior if these attributes are modified. "
                f"Consider avoiding unserializable objects (e.g., locks, file handles, database connections) "
                f"in Worker instances, or implement custom __deepcopy__ methods for them.",
                UserWarning,
                stacklevel=2
            )
        
        return new_worker
