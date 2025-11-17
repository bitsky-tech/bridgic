import copy
import uuid
from typing import Callable, List, Any, Dict, Tuple, Union

from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.worker import Worker, WorkerCallback, WorkerCallbackBuilder
from bridgic.core.automa.args import ArgsMappingRule, Distribute
from bridgic.core.automa._graph_automa import GraphMeta
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.agentic.asl._canvas_object import _Canvas, _Element, _CanvasObject, graph_stack, Settings, Data, KeyUnDifined
from bridgic.core.utils._inspect_tools import get_func_signature_data, set_method_signature, override_func_signature
from bridgic.core.types._error import ASLCompilationError


class TrackingNamespace(dict):
    """
    A custom dictionary that tracks and manages canvas objects and elements during class definition.
    
    This namespace is used as the class preparation namespace in the metaclass to automatically
    register canvases and elements into their respective parent canvases. It handles
    the extraction of settings and data from worker materials and creates appropriate _Element
    or _Canvas instances.
    """
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

        Notes:
        - If a value is already an _Element, it skips re-registration to handle fragment declarations.
        - If a value is a _Canvas and has already been declared, It also indicates the corresponding
        """
        # if the value is a _Element, it indicates that the corresponding Worker object or 
        # callable has already been declared. Declaring it again is likely to be a fragment.
        # So skip the registration process.
        if isinstance(value, _Element):
            super().__setitem__(key, value)
            return
        
        # if the value is a _Canvas and has already been declared, It also indicates the corresponding
        # canvas has already been registered. So skip the registration process. Declaring it again is 
        # likely to be a fragment. So skip the registration process.
        if isinstance(value, _Canvas) and super().__contains__(key):
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
                    raise ASLCompilationError(f"Duplicate key: {key}, every key of canvas must be unique.")

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
                    raise ASLCompilationError("All workers must be written under one graph.")

                # get the current canvas
                parent_canvas: _Canvas = stack[-1]

                # register to the current canvas
                if isinstance(value.settings.key, KeyUnDifined):
                    value.settings.key = key
                parent_canvas.register(value.settings.key, value)

                # register to the tracking namespace
                parent_canvas_namespace = super().__getitem__(parent_canvas.settings.key)
                parent_canvas_namespace[value.settings.key] = value
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

        # if the stack is empty, indicates that the element may be a normal object.
        if not stack:
            return super().__getitem__(key)

        # get the current canvas and its namespace
        current_canvas: _Canvas = stack[-1]
        current_canvas_key = current_canvas.settings.key
        current_canvas_namespace = super().__getitem__(current_canvas_key)
        if key == current_canvas_key:
            raise ASLCompilationError(f"Invalid canvas: {key}, cannot use the canvas itself.")

        # if the key is in the current canvas namespace, return the element.
        if key in current_canvas_namespace:
            res = current_canvas_namespace[key]
            return res

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
        canvases = cls._collect_canvases(namespace)
        setattr(cls, "_canvases", canvases)
        return cls

    @classmethod
    def _collect_canvases(mcls, dct: Dict[str, Any]) -> List[_Canvas]:
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
            - If a circular dependency exists
        """
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
                        raise ASLCompilationError("Multiple root canvases are not allowed.")

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
                        raise ASLCompilationError("Circular dependency detected in canvas hierarchy")
                
                result.extend(leaves)
                
                for leaf in leaves:
                    del remaining[leaf.settings.key]
            
            return result

        canvases_list = list(canvases_dict.values())
        bottom_up_order = bottom_up_order_traversal(canvases_list)
        return bottom_up_order


class ASLAutoma(GraphAutoma, metaclass=ASLAutomaMeta):
    """
    An automaton that builds graph structures from ASL (Agent Structured Language) definitions.
    
    This class extends GraphAutoma and uses a declarative syntax to define workflows. It
    automatically builds the graph structure from graph definitions during initialization,
    handling both static and dynamic worker registration.
    
    Attributes
    ----------
    _canvases : List[_Canvas]
        The list of canvases that define the graph structure, ordered bottom-up.
    _dynamic_workers : Dict[str, Dict[str, List[_Element]]]
        A nested dictionary tracking dynamic workers delegated to parent canvases.
        Structure: {parent_key: {canvas_key: [elements]}}
    """
    # The canvases of the automa.
    _canvases: List[_Canvas] = []

    def __init__(self, name: str = None):
        """
        Initialize the ASLAutoma instance.
        
        Parameters
        ----------
        name : str, optional
            The name of the automaton. If None, a default name will be assigned.
        """
        super().__init__(name=name)
        self._dynamic_workers = {}
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Build the graph structure from all canvases.
        
        This method iterates through all canvases in bottom-up order and builds the graph
        structure for each canvas. It separates static and dynamic elements and delegates
        the actual building to _inner_build_graph.
        """
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
        current_canvas_key = canvas.settings.key


        ###############################
        # build the dynamic logic flow
        ###############################
        running_options_callback = []
        for _, element in dynamic_elements.items():
            # if the canvas is top level, should use `RunningOptions` to add callback.
            if canvas.is_top_level():
                running_options_callback.append(
                    WorkerCallbackBuilder(
                        AsTopLevelDynamicCallback, 
                        init_kwargs={"__dynamic_lambda_worker__": element},
                        is_shared=False
                    )
                )

            # else should delegate parent automa add callback during building graph.
            else:
                parent_key = canvas.parent_canvas.settings.key
                if parent_key not in self._dynamic_workers:
                    self._dynamic_workers[parent_key] = {}
                if current_canvas_key not in self._dynamic_workers[parent_key]:
                    self._dynamic_workers[parent_key][current_canvas_key] = []
                self._dynamic_workers[parent_key][current_canvas_key].append(element)
            
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

    async def arun(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        """
        Execute the automaton asynchronously.
        
        This method runs the graph workflow with the provided arguments. The signature
        of this method is dynamically set based on the root canvas's parameter definitions.
        
        Parameters
        ----------
        *args : Tuple[Any, ...]
            Positional arguments to pass to the automaton.
        **kwargs : Dict[str, Any]
            Keyword arguments to pass to the automaton.
            
        Returns
        -------
        Any
            The result of executing the automaton workflow.
        """
        return await super().arun(*args, **kwargs)


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
        Whether this worker is a start node in the graph.
    is_output : bool
        Whether this worker is an output node in the graph.
    dependencies : List[str]
        List of worker keys that this worker depends on.
    args_mapping_rule : ArgsMappingRule
        The rule for mapping arguments to this worker.
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
    """
    Base callback class for handling dynamic worker creation from lambda functions.
    
    This callback is responsible for executing lambda functions at runtime to generate
    dynamic workers and adding them to the automaton. It tracks the generated worker
    keys for later cleanup.
    
    Attributes
    ----------
    __dynamic_lambda_worker__ : _Element
        The element containing the lambda function that generates dynamic workers.
    __dynamic_worker_keys__ : List[str]
        List of keys for dynamically created workers, used for cleanup.
    """
    def __init__(self, __dynamic_lambda_worker__: _Element):
        """
        Initialize the dynamic callback.
        
        Parameters
        ----------
        __dynamic_lambda_worker__ : _Element
            The element containing the lambda function that will generate dynamic workers.
        """
        self.__dynamic_lambda_worker__ = __dynamic_lambda_worker__
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
            The name of the automaton to include in the key.
            
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
        in_args: Tuple[Any, ...],
        in_kwargs: Dict[str, Any],
        automa: GraphAutoma,
    ) -> None:
        """
        Execute a lambda function to generate dynamic workers and add them to the automaton.
        
        This method processes the input arguments (extracting Distribute objects), executes
        the lambda function, and then adds each returned worker material to the automaton
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
            The automaton instance to add the dynamic workers to.
            
        Notes
        -----
        Distribute objects in the arguments are automatically unwrapped before passing
        to the lambda function.
        """
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
    """
    Callback for handling dynamic workers at the top level of the automaton.
    
    This callback is attached to the top-level automaton and generates dynamic workers
    when the automaton starts executing. It removes the dynamic workers when execution
    completes.
    """
    def __init__(self, __dynamic_lambda_worker__: _Element):
        super().__init__(__dynamic_lambda_worker__)

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
        provided arguments and adds them to the parent automaton.
        
        Parameters
        ----------
        key : str
            The key of the worker that started (unused for top-level callbacks).
        is_top_level : bool, optional
            Whether this is a top-level worker. Only processes if True.
        parent : GraphAutoma, optional
            The parent automaton to add dynamic workers to.
        arguments : Dict[str, Any], optional
            Dictionary containing 'args' and 'kwargs' keys with the execution arguments.
        """
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
        """
        Called when a top-level worker ends execution.
        
        This method removes all dynamically created workers from the parent automaton
        to clean up resources.
        
        Parameters
        ----------
        key : str
            The key of the worker that ended (unused for top-level callbacks).
        is_top_level : bool, optional
            Whether this is a top-level worker. Only processes if True.
        parent : GraphAutoma, optional
            The parent automaton to remove dynamic workers from.
        arguments : Dict[str, Any], optional
            The execution arguments (unused in cleanup).
        result : Any, optional
            The result of the worker execution (unused in cleanup).
        """
        if not is_top_level:
            return

        # get the specific automa to remove dynamic workers
        specific_automa = parent

        # remove the dynamic workers from the specific automa
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)
        

class AsWorkerDynamicCallback(DynamicCallback):
    """
    Callback for handling dynamic workers within a specific worker's automaton.
    
    This callback is attached to a worker and generates dynamic workers within that
    worker's decorated automaton (e.g., a nested GraphAutoma). It removes the dynamic
    workers when the worker completes execution.
    """
    def __init__(self, __dynamic_lambda_worker__: _Element):
        super().__init__(__dynamic_lambda_worker__)

    async def on_worker_start(
        self,
        key: str,
        is_top_level: bool = False,
        parent: GraphAutoma = None,
        arguments: Dict[str, Any] = None,
    ) -> None:
        """
        Called when a worker starts execution.
        
        This method retrieves the worker's decorated automaton and generates dynamic
        workers by executing the lambda function with the provided arguments.
        
        Parameters
        ----------
        key : str
            The key of the worker that started.
        is_top_level : bool, optional
            Whether this is a top-level worker (unused for worker callbacks).
        parent : GraphAutoma, optional
            The parent automaton containing the worker.
        arguments : Dict[str, Any], optional
            Dictionary containing 'args' and 'kwargs' keys with the execution arguments.
        """
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
        """
        Called when a worker ends execution.
        
        This method retrieves the worker's decorated automaton and removes all dynamically
        created workers to clean up resources.
        
        Parameters
        ----------
        key : str
            The key of the worker that ended.
        is_top_level : bool, optional
            Whether this is a top-level worker (unused for worker callbacks).
        parent : GraphAutoma, optional
            The parent automaton containing the worker.
        arguments : Dict[str, Any], optional
            The execution arguments (unused in cleanup).
        result : Any, optional
            The result of the worker execution (unused in cleanup).
        """
        # get the specific automa to remove dynamic workers
        specific_automa = parent._get_worker_instance(key).get_decorated_worker()

        # remove the dynamic workers from the specific automa
        for dynamic_worker_key in self.__dynamic_worker_keys__:
            specific_automa.remove_worker(dynamic_worker_key)
