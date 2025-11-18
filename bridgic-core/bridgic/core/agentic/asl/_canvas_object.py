import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from pydantic_core import PydanticUndefinedType
from pydantic.fields import FieldInfo
from typing import List, Any, Union, Callable, Tuple, Type, Optional, Dict

from bridgic.core.automa.worker import Worker
from bridgic.core.automa import GraphAutoma, RunningOptions
from bridgic.core.automa.args import ArgsMappingRule, Distribute
from bridgic.core.agentic import ConcurrentAutoma
from bridgic.core.utils._inspect_tools import override_func_signature, set_method_signature
from bridgic.core.types._error import ASLCompilationError

graph_stack: ContextVar[List["_Canvas"]] = ContextVar("graph_stack", default=[])


class Data:
    """
    Container for parameter data configuration.
    
    This class stores type and default value information for function parameters,
    allowing dynamic modification of function signatures at declaration time. It can be
    attached to workers or callables using the left-multiplication operator (*).
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Data container with parameter configurations.
        
        Parameters
        ----------
        **kwargs : Any
            Keyword arguments where each key is a parameter name and each value
            is the default value for that parameter. The type is set to Any by default.
        """
        data = {}
        for key, value in kwargs.items():
            data[key] = {
                "type": Any,
                "default": value,
            }
        self.data = data

    def __rmul__(self, other: Union[Callable, Worker]):
        """
        Attach this Data configuration to a worker or callable using right-multiplication.
        
        This allows syntax like `other * Data(param1=value1)` to attach parameter
        configuration to a worker or callable function.
        
        Parameters
        ----------
        other : Union[Callable, Worker]
            The worker or callable to attach the data configuration to.
            
        Returns
        -------
        Union[Callable, Worker]
            The same object with the `__data__` attribute set.
        """
        setattr(other, "__data__", self)
        return other


class KeyUnDifined:
    """
    Sentinel class used to indicate that a key has not been defined yet.
    
    This is used as a placeholder value in Settings to distinguish between
    an explicitly set None value and an uninitialized key.
    """
    ...


@dataclass
class Settings:
    """
    Configuration settings for canvas objects (workers, elements, and canvases).
    
    This dataclass stores metadata about how a canvas object should be configured
    in the graph, including its key, start/output status, dependencies, and argument
    mapping rules.
    
    Attributes
    ----------
    key : str
        The unique identifier for the canvas object. Defaults to KeyUnDifined if not set.
    is_start : bool
        Whether this object is a start node in the graph. Defaults to False.
    is_output : bool
        Whether this object is an output node in the graph. Defaults to False.
    dependencies : List[str]
        List of keys of other objects that this object depends on. Defaults to empty list.
    args_mapping_rule : ArgsMappingRule
        The rule for mapping arguments to this object. Defaults to ArgsMappingRule.AS_IS.
    """
    key: str = None
    is_start: bool = None
    is_output: bool = None
    dependencies: List[str] = None
    args_mapping_rule: ArgsMappingRule = None

    def __post_init__(self):
        """
        Initialize default values for Settings attributes after dataclass initialization.
        
        This method is automatically called by the dataclass decorator and ensures that
        all attributes have appropriate default values if they were not explicitly set.
        """
        if not self.key:
            self.key = KeyUnDifined()
        if not self.is_start:
            self.is_start = False
        if not self.is_output:
            self.is_output = False
        if not self.dependencies:
            self.dependencies = []
        if not self.args_mapping_rule:
            self.args_mapping_rule = ArgsMappingRule.AS_IS

    def update(self, other: "Settings") -> None:
        """
        Update this Settings instance with values from another Settings instance.
        
        Only non-default values from the other Settings are applied. This allows
        for incremental configuration updates.
        
        Parameters
        ----------
        other : Settings
            The Settings instance to copy values from.
            
        Raises
        ------
        ASLCompilationError
            If duplicate dependencies are detected in the other Settings.
        """
        if other.key != self.key and not isinstance(other.key, KeyUnDifined):
            self.key = other.key
        if other.is_start != self.is_start:
            self.is_start = other.is_start
        if other.is_output != self.is_output:
            self.is_output = other.is_output
        if other.dependencies != self.dependencies:  # detect duplicate error
            if len(other.dependencies) != len(set(other.dependencies)):
                raise ASLCompilationError(f"Duplicate dependency: {other.dependencies}.")
            self.dependencies = other.dependencies
        if other.args_mapping_rule != self.args_mapping_rule:
            self.args_mapping_rule = other.args_mapping_rule

    def __rmul__(self, other: Union[Callable, Worker]):
        """
        Attach this Settings configuration to a worker or callable using left-multiplication.
        
        This allows syntax like `other * Settings(key="worker1", is_start=True)` to
        attach configuration settings to a worker or callable function.
        
        Parameters
        ----------
        other : Union[Callable, Worker]
            The worker or callable to attach the settings to.
            
        Returns
        -------
        Union[Callable, Worker]
            The same object with the `__settings__` attribute set.
        """
        setattr(other, "__settings__", self)
        return other


class _CanvasObject:
    """
    Base class for all objects that can be placed on a canvas.
    
    This class provides the foundation for both _Element and _Canvas
    objects. It manages settings, parent-child relationships, and provides operator overloads
    for declarative workflow definition (e.g., `+`, `~`, `>>`, `&`).
    
    Attributes
    ----------
    worker_material : Optional[Union[Worker, Callable]]
        The underlying worker or callable that this object wraps.
    parent_canvas : Optional[_Canvas]
        The parent canvas that contains this object. self if this is the root canvas.
    left_canvas_obj : Optional[_CanvasObject]
        The left neighbor in a grouped sequence (used with `&` operator).
    right_canvas_obj : Optional[_CanvasObject]
        The right neighbor in a grouped sequence (used with `&` operator).
    is_lambda : bool
        Whether this object represents a lambda function (for dynamic workers).
    settings : Settings
        Configuration settings for this canvas object.
    """
    def __init__(self, key: str, worker_material: Optional[Union[Worker, Callable]]) -> None:
        """
        Initialize a canvas object.
        
        Parameters
        ----------
        key : str
            The unique identifier for this canvas object.
        worker_material : Optional[Union[Worker, Callable]]
            The worker or callable to wrap. None for canvas objects.
        """
        self.worker_material = worker_material
        self.parent_canvas = None
        self.left_canvas_obj = None
        self.right_canvas_obj = None
        self.is_lambda = False

        # settings initialization
        self.settings = Settings(
            key=key,
            is_start=False,
            is_output=False,
            dependencies=[],
            args_mapping_rule=ArgsMappingRule.AS_IS
        )

    def update_settings(self, settings: Settings) -> None:
        """
        Update the settings of this canvas object.
        
        Parameters
        ----------
        settings : Settings
            The settings to merge into this object's settings.
            
        Returns
        -------
        _CanvasObject
            Returns self for method chaining.
        """
        self.settings.update(settings)
        return self

    def update_data(self, data: Data) -> None:
        """
        Update the function signature of the worker material based on data configuration.
        
        This method overrides the function signature to match the data configuration,
        allowing dynamic parameter injection.
        
        Parameters
        ----------
        data : Data
            The data configuration containing parameter type and default value information.
        """
        if isinstance(self.worker_material, Worker):
            worker_name = self.worker_material.__class__.__name__
            override_func = self.worker_material.arun if self.worker_material._is_arun_overridden() else self.worker_material.run
            override_func_signature(worker_name, override_func, data.data)
        elif isinstance(self.worker_material, Callable):
            func_name = getattr(self.worker_material, "__name__", repr(self.worker_material))
            override_func_signature(func_name, self.worker_material, data.data)

    def __rshift__(self, other: Union["_CanvasObject", Tuple["_CanvasObject"]]) -> None:
        """
        Right-shift operator (>>) sets the current object as a dependency of the other object.
        
        This operator creates a dependency relationship where the left-hand object must
        complete before the right-hand object can execute. If the left-hand object is part
        of a group (connected with `&`), all objects in the group become dependencies.
        
        Parameters
        ----------
        other : Union[_CanvasObject, Tuple[_CanvasObject]]
            The canvas object(s) that will depend on this object.
            
        Returns
        -------
        _CanvasObject
            Returns the right-hand object for method chaining.
            
        Raises
        ------
        ValueError
            If duplicate dependencies are detected.
        """
        def check_duplicate_dependency(current_canvas_obj: _CanvasObject, left_canvas_objs: List[str]) -> None:
            for dependency in left_canvas_objs:
                if dependency in current_canvas_obj.settings.dependencies:
                    raise ASLCompilationError(f"Duplicate dependency: {dependency}.")

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
        Bitwise AND operator (&) groups multiple canvas objects together.
        
        This operator creates a linked list of canvas objects, allowing them to be
        treated as a single unit when used with other operators like `>>` or `+`.
        
        Parameters
        ----------
        other : _CanvasObject
            The canvas object to group with this object.
            
        Returns
        -------
        _CanvasObject
            Returns the right-hand object for method chaining.
        """
        self.right_canvas_obj = other
        other.left_canvas_obj = self
        return other

    def __pos__(self) -> None:
        """
        Unary plus operator (+) marks this object and its group as start nodes.
        
        When applied to a grouped sequence (connected with `&`), all objects in the
        group are marked as start nodes, meaning they can begin execution without
        waiting for dependencies.
        
        Returns
        -------
        _CanvasObject
            Returns self for method chaining.
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
        Unary invert operator (~) marks this object and its group as output nodes.
        
        When applied to a grouped sequence (connected with `&`), all objects in the
        group are marked as output nodes, meaning their results will be collected
        as outputs of the graph.
        
        Returns
        -------
        _CanvasObject
            Returns self for method chaining.
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

    def __hash__(self) -> int:
        return hash(self.settings.key)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _CanvasObject):
            return self.settings.key == other.settings.key
        return False


class _Canvas(_CanvasObject):
    """
    Represents a canvas (graph or concurrent automaton) that can contain elements.
    
    A canvas is a container for workers and nested canvases. It records input parameters
    and manages the registration of elements. When the graph is built, the canvas creates
    the appropriate automaton instance at appropriate time (GraphAutoma or ConcurrentAutoma).
    
    Attributes
    ----------
    automa_type : str
        The type of automaton to create: "graph" or "concurrent".
    params : Dict[str, Any]
        Dictionary of parameter definitions for the automaton's arun method.
    elements : Dict[str, Union[_Element, _Canvas]]
        Dictionary of elements (workers and nested canvases) registered to this canvas.
    """
    def __init__(self, automa_type: str, params: Dict[str, Any]) -> None:
        """
        Initialize a canvas object.
        
        Parameters
        ----------
        automa_type : str
            The type of automaton: "graph" or "concurrent".
        params : Dict[str, Any]
            Dictionary of parameter definitions with keys as parameter names and values
            containing type and default information.
        """
        super().__init__(None, None)
        self.automa_type = automa_type
        self.params = params

        self.elements: Dict[str, Union["_Element", "_Canvas"]] = {}

    def register(self, key: str, value: Union["_Element", "_Canvas"]):
        """
        Register an element (worker or nested canvas) to this canvas.
        
        Parameters
        ----------
        key : str
            The unique key for the element within this canvas.
        value : Union[_Element, _Canvas]
            The element or nested canvas to register.
            
        Raises
        ------
        ASLCompilationError
            If attempting to register a lambda element to a graph canvas (lambdas are
            only allowed in concurrent or sequential canvases).
        """
        if isinstance(value, _Element) and value.is_lambda and self.automa_type == "graph":
            raise ASLCompilationError("Lambda dynamic logic must be written under a `concurrent`, `sequential`, not `graph`.")
            
        value.parent_canvas = self
        self.elements[key] = value

    def make_automa(self, running_options: RunningOptions = None):
        """
        Create the automaton instance for this canvas.
        
        This method instantiates either a GraphAutoma or ConcurrentAutoma based on
        the `automa_type`, and sets up the method signature based on the `params`.
        
        Parameters
        ----------
        running_options : RunningOptions, optional
            Options for running the automaton, including callbacks.
            
        Raises
        ------
        ASLCompilationError
            If the automa_type is not "graph" or "concurrent".
        """
        if self.automa_type == "graph":
            self.worker_material = GraphAutoma(
                name=self.settings.key, 
                running_options=running_options
            )
        elif self.automa_type == "concurrent":
            self.worker_material = ConcurrentAutoma(
                name=self.settings.key, 
                running_options=running_options
            )
        else:
            raise ASLCompilationError(f"Invalid automa type: {self.automa_type}")

        if self.params:
            set_method_signature(self.worker_material.arun, self.params)

    def is_top_level(self) -> bool:
        """
        Check if this canvas is the top-level (root) canvas.
        
        A canvas is top-level if its parent_canvas is itself, which is set during
        the collection phase in the `ASLAutoma` metaclass.
        
        Returns
        -------
        bool
            True if this is the top-level canvas, False otherwise.
        """
        return self.parent_canvas is self

    def __str__(self) -> str:
        return (
            f"_Canvas("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas.settings.key if self.parent_canvas else None}, "
            f"elements={self.elements}, "
            f"settings={self.settings}"
        )


class _Element(_CanvasObject):
    """
    Represents a worker element on a canvas.
    
    An element wraps a Worker instance or callable function and provides canvas-specific
    metadata and operations. Elements are registered to canvases and participate in the
    graph workflow definition.
    """
    def __init__(self, key: str, worker_material: Union[Worker, Callable]) -> None:
        """
        Initialize an element with a worker or callable.
        
        Parameters
        ----------
        key : str
            The unique identifier for this element.
        worker_material : Union[Worker, Callable]
            The worker or callable function that this element wraps.
        """
        super().__init__(key, worker_material)
        self.parent_canvas = None

    def __str__(self) -> str:
        return (
            f"_Element("
            f"key={self.settings.key}, "
            f"parent_canvas={self.parent_canvas.settings.key if self.parent_canvas else None}, "
            f"settings={self.settings})"
        )


class _GraphContextManager:
    """
    Context manager for defining graph or concurrent automaton structures.
    
    This class provides a context manager interface for declaratively defining
    automaton structures. It manages the graph stack and parameter definitions.
    Instances are created at module level as `graph` and `concurrent`.
    """
    def __init__(self, automa_type: str):
        """
        Initialize the graph context manager.
        
        Parameters
        ----------
        automa_type : str
            The type of automaton: "graph" or "concurrent".
        """
        self.automa_type = automa_type
        self.params = {}

    def __call__(self, **kwargs: Dict[str, "ASLField"]) -> _Canvas:
        """
        Define parameters for the graph using ASLField instances.
        
        This method is called when the context manager is invoked with keyword arguments,
        each representing a parameter of the automaton's arun method.
        
        Parameters
        ----------
        **kwargs : Dict[str, ASLField]
            Keyword arguments where each key is a parameter name and each value is
            an ASLField instance defining the parameter's type and default value.
            
        Returns
        -------
        _GraphContextManager
            Returns self to enable use as a context manager.
            
        Raises
        ------
        ASLCompilationError
            If any value is not an ASLField instance.
        """
        params = {}
        for key, value in kwargs.items():
            if not isinstance(value, ASLField):
                raise ASLCompilationError(f"Invalid field type: {type(value)}.")
            default_value = Distribute(
                value.default
                if not isinstance(value.default, PydanticUndefinedType) 
                else inspect._empty
            ) if value.distribute else (
                value.default 
                if not isinstance(value.default, PydanticUndefinedType) 
                else inspect._empty
            )
            params[key] = {
                "type": value.type,
                "default": default_value
            }
        self.params = params
        return self

    def __enter__(self) -> _Canvas:
        """
        Enter the context and create a new canvas.
        
        This method creates a new _Canvas instance and pushes it onto the graph stack,
        making it the current active canvas for element registration.
        
        Returns
        -------
        _Canvas
            The newly created canvas instance.
        """
        ctx = _Canvas(automa_type=self.automa_type, params=self.params)

        stack = list(graph_stack.get())
        stack.append(ctx)
        graph_stack.set(stack)
        return ctx

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exit the context and pop the canvas from the graph stack.
        
        Parameters
        ----------
        exc_type : Any
            Exception type if an exception occurred (standard context manager protocol).
        exc_val : Any
            Exception value if an exception occurred (standard context manager protocol).
        exc_tb : Any
            Exception traceback if an exception occurred (standard context manager protocol).
        """
        stack = list(graph_stack.get())
        stack.pop()
        graph_stack.set(stack)
        

# Create module-level instances (not global variables, but singleton objects)
graph = _GraphContextManager(automa_type="graph")
concurrent = _GraphContextManager(automa_type="concurrent")


class ASLField(FieldInfo):
    """
    A custom Field class that extends Pydantic's FieldInfo with support for storing default type information.
    
    The `default_type` parameter is stored as metadata and does not automatically generate default values.
    You must explicitly provide a `default` value if you want a default.
    """
    
    def __init__(
        self,
        type: Type[Any] = Any,
        *,
        default: Any = ...,
        distribute: bool = False,
        **kwargs: Any
    ):
        """
        Initialize ASLField with optional default_type metadata.
        
        Parameters
        ----------
        default : Any
            Explicit default value. Must be provided if you want a default value.
        type : Type[Any]
            Type information stored as metadata. Does not automatically generate default values.
        distribute : bool
            Whether to distribute the data to multiple workers.
        **kwargs : Any
            Other Field parameters (description, ge, le, etc.)
        """
        super().__init__(default=default, **kwargs)
        self.type = type
        self.distribute = distribute
