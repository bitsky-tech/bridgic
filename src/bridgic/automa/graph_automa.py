import asyncio
import inspect
from inspect import Parameter
import json
import uuid
from abc import ABCMeta
from typing import Any, List, Dict, Set, Mapping, Callable, Tuple, Optional, Literal, Union
from types import MethodType
from collections import defaultdict, deque
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import override

from bridgic.utils.console import printer, colored
from bridgic.automa.worker import Worker
from bridgic.types.error import *
from bridgic.types.mixin import WithKeyMixin, AdaptableMixin, LandableMixin
from bridgic.utils.inspect_tools import get_param_names_by_kind
from bridgic.automa import Automa
from bridgic.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType, ArgsMappingRule
from bridgic.automa.worker.callable_worker import CallableWorker

class _GraphAdaptedWorker(WithKeyMixin, AdaptableMixin, LandableMixin, Worker):
    """
    A decorated worker used for GraphAutoma orchestration and scheduling. Not intended for external use.
    Follows the `Decorator` design pattern:
    - https://web.archive.org/web/20031204182047/http://patterndigest.com/patterns/Decorator.html
    - https://en.wikipedia.org/wiki/Decorator_pattern
    New Behavior Added: In addition to the original Worker functionality, maintains configuration and state variables related to dynamic scheduling and graph topology.
    """
    def __init__(
        self,
        *,
        key: str,
        worker: Worker,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        **kwargs,
    ):
        super().__init__(
            key=key,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
            **kwargs,
        )
        self._decorated_worker = worker

    #
    # Delegate all the properties and methods of _GraphAdaptedWorker to the decorated worker.
    # TODO: Maybe 'Worker' should be a Protocol.
    #

    @override
    async def process_async(self, *args, **kwargs) -> Any:
        return await self._decorated_worker.process_async(*args, **kwargs)

    @property
    def return_type(self) -> type:
        return self._decorated_worker.return_type

    @property
    def parent(self) -> "Automa":
        return self._decorated_worker.parent

    @parent.setter
    def parent(self, value: "Automa"):
        self._decorated_worker.parent = value

    @property
    def output_buffer(self) -> Any:
        return self._decorated_worker.output_buffer
    
    @output_buffer.setter
    def output_buffer(self, value: Any):
        self._decorated_worker.output_buffer = value

    @property
    def local_space(self) -> Dict[str, Any]:
        return self._decorated_worker.local_space
    
    @local_space.setter
    def local_space(self, value: Dict[str, Any]):
        self._decorated_worker.local_space = value

    @property
    def func(self):
        """
        Readonly property for getting the original/real function of the worker.
        """
        if isinstance(self._decorated_worker, CallableWorker):
            return self._decorated_worker.callable
        return self._decorated_worker.process_async

    @override
    def __str__(self) -> str:
        # TODO: need some refactoring
        return str(self._decorated_worker)

@dataclass
class _RunnningTask:
    """
    States of the current running task.
    The instances of this class do not need to be serialized.
    """
    worker_key: str
    task: asyncio.Task

class _KickoffInfo(BaseModel):
    worker_key: str
    # Worker key or the container "__automa__"
    last_kickoff: str
    # Whether the kickoff is triggered by ferry_to() initiated by developers.
    from_ferry: bool = False
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}

class _WorkerDynamicState(BaseModel):
    # Dynamically record the dependency workers keys of each worker.
    # Will be reset to the topology edges/dependencies of the worker
    # once the task is finished or the topology is changed.
    dependency_triggers: Set[str]

class _AddWorkerDeferredTask(BaseModel):
    task_type: Literal["add_worker"] = Field(default="add_worker")
    key: str
    worker_obj: Worker # Note: Not a pydantic model!!
    dependencies: List[str] = []
    is_start: bool = False
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS

    model_config = ConfigDict(arbitrary_types_allowed=True)

class _RemoveWorkerDeferredTask(BaseModel):
    task_type: Literal["remove_worker"] = Field(default="remove_worker")
    key: str

class _SetOutputWorkerDeferredTask(BaseModel):
    output_worker_key: str

class _FerryDeferredTask(BaseModel):
    ferry_to_worker_key: str
    kickoff_worker_key: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class GraphAutomaMeta(ABCMeta):
    def __new__(mcls, name, bases, dct):
        """
        This metaclass is used to:
            - Correctly handle inheritence of pre-defined workers during GraphAutoma class inheritence, 
            particularly for multiple inheritence scenarios, enabling easy extension of GraphAutoma
            - Maintain the static tables of trigger-workers and forward-workers for each worker
            - Verify that all of the pre-defined worker dependencies satisfy the DAG constraints
        """
        cls = super().__new__(mcls, name, bases, dct)

        # Inherit the graph structure from the parent classes and maintain the related data structures.
        registered_worker_funcs: Dict[str, Callable] = {}
        worker_static_forwards: Dict[str, List[str]] = {}
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(WorkerDecoratorType.GraphAutomaMethod, worker_kwargs)
                func = attr_value
                setattr(func, "__is_worker__", True)
                setattr(func, "__worker_key__", complete_args["key"])
                setattr(func, "__dependencies__", complete_args["dependencies"])
                setattr(func, "__is_start__", complete_args["is_start"])
                setattr(func, "__args_mapping_rule__", complete_args["args_mapping_rule"])
        
        for base in bases:
            for worker_key, worker_func in getattr(base, "_registered_worker_funcs", {}).items():
                if worker_key not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_key] = worker_func
                else:
                    raise AutomaDeclarationError(
                        f"worker is defined in multiple base classes: "
                        f"base={base}, worker={worker_key}"
                    )

            for current, forward_list in getattr(base, "_worker_static_forwards", {}).items():
                if current not in worker_static_forwards.keys():
                    worker_static_forwards[current] = []
                worker_static_forwards[current].extend(forward_list)

        for attr_name, attr_value in dct.items():
            # Attributes with __is_worker__ will be registered as workers.
            if hasattr(attr_value, "__is_worker__"):
                worker_key = getattr(attr_value, "__worker_key__", None) or attr_name
                dependencies = list(set(attr_value.__dependencies__))

                # Update the registered workers for current class.
                if worker_key not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_key] = attr_value
                else:
                    raise AutomaDeclarationError(
                        f"Duplicate worker keys are not allowed: "
                        f"worker={worker_key}"
                    )

                # Update the table of static forwards.
                for trigger in dependencies:
                    if trigger not in worker_static_forwards.keys():
                        worker_static_forwards[trigger] = []
                    worker_static_forwards[trigger].append(worker_key)

        # Validate if the DAG constraint is met.
        # TODO: this is indeed a chance to detect risks. Add more checks here or remove totally!
        mcls.validate_dag_constraints(worker_static_forwards)

        setattr(cls, "_registered_worker_funcs", registered_worker_funcs)
        return cls

    @classmethod
    def validate_dag_constraints(mcls, forward_dict: Dict[str, List[str]]):
        """
        Use Kahn's algorithm to check if the input graph described by the forward_dict satisfies
        the DAG constraints. If the graph doesn't meet the DAG constraints, AutomaDeclarationError will be raised. 

        More about [Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
        could be read from the link.

        Parameters
        ----------
        forward_dict : Dict[str, List[str]]
            A dictionary that describes the graph structure. The keys are the nodes, and the values 
            are the lists of nodes that are directly reachable from the keys.

        Raises
        ------
        AutomaDeclarationError
            If the graph doesn't meet the DAG constraints.
        """
        # 1. Initialize the in-degree.
        in_degree = defaultdict(int)
        for current, target_list in forward_dict.items():
            for target in target_list:
                in_degree[target] += 1

        # 2. Create a queue of workers with in-degree 0.
        queue = deque([node for node in forward_dict.keys() if in_degree[node] == 0])

        # 3. Continuously pop workers from the queue and update the in-degree of their targets.
        while queue:
            node = queue.popleft()
            for target in forward_dict.get(node, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        # 4. If the in-degree were all 0, then the graph meets the DAG constraints.
        if not all([in_degree[node] == 0 for node in in_degree.keys()]):
            nodes_in_cycle = [node for node in forward_dict.keys() if in_degree[node] != 0]
            raise AutomaCompilationError(
                f"the graph automa does not meet the DAG constraints, because the "
                f"following workers are in cycle: {nodes_in_cycle}"
            )

class GraphAutoma(Automa, metaclass=GraphAutomaMeta):
    """
    DDG (Dynamic Directed Graph) implementation of Automa.
    The topology of a DDG are allowed to be changed during runtime, by developers.
    """

    # The initial topology defined by @worker functions.
    _registered_worker_funcs: Dict[str, Callable] = {}

    # [IMPORTANT] 
    # The whole states of the Automa are divided into two main parts:
    #
    # [Part One: Topology-Related Runtime States] The runtime states related to topology changes.
    # -- Nodes: workers (A Worker can be another Automa): {_workers}
    # -- Edges: Dependencies between workers: {dependencies in _workers, _worker_forwards}
    # -- Worker Dynamics: the dynamic in-degrees of each workers. {_workers_dynamic_states}
    # -- Configurations: Start workers, output worker, args_mapping_rule, etc. {_output_worker_key, is_start and args_mapping_rule in _workers}
    # -- Deferred Tasks: {_topology_change_deferred_tasks, _set_output_worker_deferred_task}
    #
    # [Part Two: Task-Related Runtime States] The runtime states related to task execution.
    # -- Current kickoff workers list. {_current_kickoff_workers}
    # -- Running & Ferry Tasks list. {_running_tasks, _ferry_deferred_tasks}
    # -- Output buffer and local space for each worker. {in _workers}
    # -- Other runtime states...

    # Note: Just declarations, NOT instances!!!
    # So do not initialize them here!!!
    _workers: Dict[str, _GraphAdaptedWorker]
    _worker_forwards: Dict[str, List[str]]
    _output_worker_key: Optional[str]

    _current_kickoff_workers: List[_KickoffInfo]
    _workers_dynamic_states: Dict[str, _WorkerDynamicState]
    # The whole running process of the DDG is divided into two main phases:
    # 1. [Initialization Phase] The first phase (when _running_started is False): the initial topology of DDG was constructed.
    # 2. [Running Phase] The second phase (when _running_started is True): the DDG is running, and the workers are executed in a dynamic step-by-step manner (DS loop).
    # Once _running_started is set to True, it will always be True.
    _running_started: bool

    # The following need not to be serialized.
    _running_tasks: List[_RunnningTask]
    # TODO: The following deferred task structures need to be thread-safe.
    # TODO: Need to be refactored when parallelization features are added.
    _topology_change_deferred_tasks: List[Union[_AddWorkerDeferredTask, _RemoveWorkerDeferredTask]]
    _set_output_worker_deferred_task: _SetOutputWorkerDeferredTask
    _ferry_deferred_tasks: List[_FerryDeferredTask]

    def __init__(
        self,
        name: Optional[str] = None,
        output_worker_key: Optional[str] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        name : Optional[str] (default = None)
            The name of the automa.
        output_worker_key : Optional[str] (default = None)
            The key of the output worker whose output will be returned by the automa.
        state_dict : Optional[Dict[str, Any]] (default = None)
            A dictionary for initializing the automa's runtime state. This parameter is intended for internal framework use only, specifically for deserialization, and should not be used by developers.
        """
        super().__init__(name=name or f"{type(self).__name__}-{uuid.uuid4().hex[:8]}")
        self._workers = {} #TODO: __getattribute__() refactoring
        self._running_started = False

        # Start to initialize the states of the Automa.
        # The whole states of the Automa can be from two sources:
        if state_dict:
            # 1. One source: from the state_dict serialized previously.
            self._init_from_state_dict(state_dict)
        else:
            # 2. Another source: from the class definition.
            self._normal_init(output_worker_key)

    def _normal_init(
            self,
            output_worker_key: Optional[str] = None,
    ):
        ###############################################################################
        # Initialization of [Part One: Topology-Related Runtime States] #### Strat ####
        ###############################################################################

        cls = type(self)

        # _workers, _worker_forwards and _workers_dynamic_states will be initialized incrementally by add_worker()...
        self._worker_forwards = {}
        self._workers_dynamic_states = {}

        # The _registered_worker_funcs data are from @worker decorators.
        for worker_key, worker_func in cls._registered_worker_funcs.items():
            # The decorator based mechanism (i.e. @worker) is based on the add_worker() interface.
            # Parameters check and other implementation details can be unified.
            self.add_func_as_worker(
                key=worker_key,
                func=worker_func,
                dependencies=worker_func.__dependencies__,
                is_start=worker_func.__is_start__,
                args_mapping_rule=worker_func.__args_mapping_rule__,
            )

        self._output_worker_key = output_worker_key

        ###############################################################################
        # Initialization of [Part One: Topology-Related Runtime States] ##### End #####
        ###############################################################################

        ###############################################################################
        # Initialization of [Part Two: Task-Related Runtime States] ###### Strat ######
        ###############################################################################

        # -- Current kickoff workers list.
        # The key list of the workers that are ready to be immediately executed in the next DS (Dynamic Step). It will be lazily initialized in _compile_graph_and_detect_risks().
        self._current_kickoff_workers = []
        # -- Running Task list.
        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        self._topology_change_deferred_tasks = []
        self._set_output_worker_deferred_task = None
        self._ferry_deferred_tasks = []

        ###############################################################################
        # Initialization of [Part Two: Task-Related Runtime States] ####### End #######
        ###############################################################################

    def _init_from_state_dict(
            self,
            state_dict: Dict[str, Any],
    ):
        ...
        # TODO: deserialize from the state_dict

    def _add_worker_incrementally(
        self,
        key: str,
        worker_obj: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        Incrementally add a worker into the automa.
        This method is one of the very basic primitives of DDG for dynamic topology changes.
        For internal use only.
        """
        if key in self._workers:
            raise AutomaRuntimeError(
                f"duplicate workers with the same key '{key}' are not allowed to be added!"
            )
        new_worker_obj = _GraphAdaptedWorker(
            key=key,
            worker=worker_obj,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )

        # Register the worker_obj.
        new_worker_obj.parent = self
        self._workers[new_worker_obj.key] = new_worker_obj
        # Incrementally update the dynamic states of added workers.
        self._workers_dynamic_states[key] = _WorkerDynamicState(
            dependency_triggers=set(dependencies)
        )
        # Incrementally update the forwards table.
        # Note: Here we are not able to guarantee the existence of the trigger workers in self._workers of the final dynamic graph, so validation is needed in appropriate time later.
        # TODO: check the comment above.
        for trigger in dependencies:
            if trigger not in self._worker_forwards:
                self._worker_forwards[trigger] = []
            self._worker_forwards[trigger].append(key)

    def _remove_worker_incrementally(
        self,
        key: str
    ) -> None:
        """
        Incrementally add a worker into the automa.
        This method is one of the very basic primitives of DDG for dynamic topology changes.
        For internal use only.
        """
        if key not in self._workers:
            raise AutomaRuntimeError(
                f"fail to remove worker '{key}' that does not exist!"
            )

        worker_to_remove = self._workers[key]
        # Remove the worker.
        del self._workers[key]
        # Incrementally update the dynamic states of removed workers.
        del self._workers_dynamic_states[key]
        if key in self._worker_forwards:
            # Update the dependencies of the successor workers, if needed.
            for successor in self._worker_forwards[key]:
                self._workers[successor].dependencies.remove(key)
                # Note this detail here: use discard() instead of remove() to avoid KeyError.
                # This case occurs when a worker call remove_worker() to remove its predecessor worker.
                self._workers_dynamic_states[successor].dependency_triggers.discard(key)
            # Incrementally update the forwards table.
            del self._worker_forwards[key]
        # Note: Here we are not able to guarantee the existence of the trigger workers in self._workers of the final dynamic graph, so validation is needed in appropriate time later.
        # TODO: check the comment above.
        for trigger in worker_to_remove.dependencies:
            self._worker_forwards[trigger].remove(key)
        # clear the output worker key if needed.
        if key == self._output_worker_key:
            self._output_worker_key = None

    def add_worker(
        self,
        key: str,
        worker_obj: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        This method is used to add a worker dynamically into the automa.

        If this method is called during the [Initialization Phase], the worker will be added immediately. If this method is called during the [Running Phase], the worker will be added as a deferred task which will be executed in the next DS.

        In DDG, dependencies can only be added together with a worker. However, you can add a worker without any dependencies.

        Note: A worker that is added during the [Running Phase] is not allowed to be a start worker.

        Parameters
        ----------
        key : str
            The key of the worker.
        worker_obj : Worker
            The worker instance to be registered.
        dependencies : List[str]
            A list of worker keys that the worker depends on.
        is_start : bool
            Whether the worker is a start worker.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        """

        def _basic_worker_params_check(key: str, worker_obj: Worker):
            if not isinstance(worker_obj, Worker):
                raise TypeError(
                    f"worker_obj to be registered must be a Worker, "
                    f"but got {type(worker_obj)} for worker '{key}'"
                )

            if not asyncio.iscoroutinefunction(worker_obj.process_async):
                raise WorkerSignatureError(
                    f"process_async of Worker must be an async method, "
                    f"but got {type(worker_obj.process_async)} for worker '{key}'"
                )
            
            if not isinstance(dependencies, list):
                raise TypeError(
                    f"dependencies must be a list, "
                    f"but got {type(dependencies)} for worker '{key}'"
                )
            if not all([isinstance(d, str) for d in dependencies]):
                raise ValueError(
                    f"dependencies must be a List of str, "
                    f"but got {dependencies} for worker {key}"
                )
            
            if args_mapping_rule not in ArgsMappingRule:
                raise ValueError(
                    f"args_mapping_rule must be one of the following: {[e for e in ArgsMappingRule]}, "
                    f"but got {args_mapping_rule} for worker {key}"
                )

        # Ensure the parameters are valid.
        _basic_worker_params_check(key, worker_obj)

        if not self._running_started:
            # Add worker during the [Initialization Phase].
            self._add_worker_incrementally(
                key=key,
                worker_obj=worker_obj,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
        else:
            # Add worker during the [Running Phase].
            deferred_task = _AddWorkerDeferredTask(
                key=key,
                worker_obj=worker_obj,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
            # Note: the execution order of topology change deferred tasks is important and is determined by the order of the calls of add_worker(), remove_worker() in one DS.
            self._topology_change_deferred_tasks.append(deferred_task)

    def add_func_as_worker(
        self,
        key: str,
        func: Callable,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        This method is used to add a function as a worker into the automa.

        The format of the parameters will follow that of the decorator @worker(...), so that the 
        behavior of the decorated function is consistent with that of normal CallableLandableWorker objects.

        Parameters
        ----------
        key : str
            The key of the function worker.
        func : Callable
            The function to be added as a worker to the automa.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        """
        # Register func as an instance of CallableWorker.
        if not isinstance(func, MethodType):
            func = MethodType(func, self)
        else:
            # TODO: validate whether self is Automa?
            ...
        func_worker = CallableWorker(func)

        self.add_worker(
            key=key,
            worker_obj=func_worker,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )

    def worker(
        self,
        *,
        key: Optional[str] = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> Callable:
        """
        This is a decorator used to mark a function as an GraphAutoma detectable Worker. Dislike the 
        global decorator @worker(...), it is usally used after an GraphAutoma instance is initialized.

        The format of the parameters will follow that of the decorator @worker(...), so that the 
        behavior of the decorated function is consistent with that of normal CallableLandableWorker objects.

        Parameters
        ----------
        key : str
            The key of the worker. If not provided, the key of the decorated callable will be used.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        args_mapping_rule : str
            The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
        """
        def wrapper(func: Callable):
            self.add_func_as_worker(
                key=(key or func.__name__),
                func=func,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )

        return wrapper
    
    def remove_worker(self, key: str) -> None:
        """
        Remove a worker from the Automa. This method can be called at any time to remove a worker from the Automa.

        When a worker is removed, all dependencies related to this worker, including all the dependencies of the worker itself and the dependencies between the worker and its successor workers, will be also removed.

        Parameters
        ----------
        key : str
            The key of the worker to be removed.

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by key does not exist in the Automa, this exception will be raised.
        """
        if not self._running_started:
            # remove immediately
            self._remove_worker_incrementally(key)
        else:
            deferred_task = _RemoveWorkerDeferredTask(
                key=key,
            )
            # Note: the execution order of topology change deferred tasks is important and is determined by the order of the calls of add_worker(), remove_worker() in one DS.
            self._topology_change_deferred_tasks.append(deferred_task)

    def set_output_worker(self, output_worker_key: str):
        """
        This method is used to set the output worker of the automa dynamically.
        """
        if not self._running_started:
            self._output_worker_key = output_worker_key
        else:
            deferred_task = _SetOutputWorkerDeferredTask(
                output_worker_key=output_worker_key,
            )
            # Note: Only the last _SetOutputWorkerDeferredTask is valid if set_output_worker() is called multiple times in one DS.
            self._set_output_worker_deferred_task = deferred_task

    def __getattribute__(self, key):
        """
        Used to get the worker object by its key (self.<key>).
        """
        workers = super().__getattribute__('_workers')
        if key in workers:
            return workers[key]
        return super().__getattribute__(key)

    def _validate_canonical_graph(self):
        """
        This method is used to validate that DDG graph is canonical.
        """
        for worker_key, worker_obj in self._workers.items():
            for dependency_key in worker_obj.dependencies:
                if dependency_key not in self._workers:
                    raise AutomaCompilationError(
                        f"the dependency {dependency_key} of worker {worker_key} does not exist"
                    )
        assert set(self._workers.keys()) == set(self._workers_dynamic_states.keys())
        for worker_key, worker_dynamic_state in self._workers_dynamic_states.items():
            for dependency_key in worker_dynamic_state.dependency_triggers:
                assert dependency_key in self._workers[worker_key].dependencies

        for worker_key, worker_obj in self._workers.items():
            for dependency_key in worker_obj.dependencies:
                assert worker_key in self._worker_forwards[dependency_key]
        for worker_key, successor_keys in self._worker_forwards.items():
            for successor_key in successor_keys:
                assert worker_key in self._workers[successor_key].dependencies

    def _compile_graph_and_detect_risks(self):
        """
        This method should be called at the very beginning of self.run() to ensure that:
        1. The whole graph is built out of all of the following worker sources:
            - Pre-defined workers, such as:
                - Methods decorated with @worker(...)
            - Post-added workers, such as:
                - Functions decorated with @automa_obj.worker(...)
                - Workers added via automa_obj.add_func_as_worker(...)
                - Workers added via automa_obj.add_worker(...)
        2. The dependencies of each worker are confirmed to satisfy the DAG constraints.
        """

        # Validate the canonical graph.
        self._validate_canonical_graph()
        # Validate the DAG constraints.
        GraphAutomaMeta.validate_dag_constraints(self._worker_forwards)
        # TODO: More validations can be added here...

        self._current_kickoff_workers = [
            _KickoffInfo(
                worker_key=worker_key,
                last_kickoff="__automa__"
            ) for worker_key, worker_obj in self._workers.items()
            if getattr(worker_obj, "is_start", False)
        ]

        # Validate if the output worker exists.
        if self._output_worker_key and self._output_worker_key not in self._workers:
            raise AutomaCompilationError(
                f"the output worker is not found: "
                f"output_worker_key={self._output_worker_key}"
            )

        # Penetrate the running options to all levels below.
        self._penetrate_running_options()

        # Find all connected components of the whole automa graph.
        self._find_connected_components()

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        """
        Handoff control flow to the specified worker, passing along any arguments as needed.
        The specified worker will always start to run asynchronously in the next event loop, regardless of its dependencies.

        Parameters
        ----------
        worker_key : str
            The key of the worker to run.
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.
        """
        # TODO: check worker_key is valid
        kickoff_worker_key: str = self._get_kickoff_worker_from_stack()
        deferred_task = _FerryDeferredTask(
            ferry_to_worker_key=worker_key,
            kickoff_worker_key=kickoff_worker_key,
            args=args,
            kwargs=kwargs,
        )
        self._ferry_deferred_tasks.append(deferred_task)

    async def process_async(self, *args, **kwargs) -> Any:
        """
        The entry point of the automa.

        The GraphAutoma will orchestrate all workers to run, manage dependencies and dynamically handoff control flows (ferry_to) between workers. Arguments can be mapped and passed automatically if needed.

        Parameters
        ----------
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.

        TODO: add some code snippet examples here

        Returns
        -------
        Any
            The output of the output-worker if it is specified, otherwise None.
        """

        def _execute_topology_change_deferred_tasks(tc_tasks: List[Union[_AddWorkerDeferredTask, _RemoveWorkerDeferredTask]]):
            for topology_task in tc_tasks:
                if topology_task.task_type == "add_worker":
                    self._add_worker_incrementally(
                        key=topology_task.key,
                        worker_obj=topology_task.worker_obj,
                        dependencies=topology_task.dependencies,
                        is_start=topology_task.is_start,
                        args_mapping_rule=topology_task.args_mapping_rule,
                    )
                elif topology_task.task_type == "remove_worker":
                    self._remove_worker_incrementally(topology_task.key)

        if not self._running_started:
            # Here is the last chance to compile and check the DDG in the end of the [Initialization Phase] (phase 1 just before the first DS).
            self._compile_graph_and_detect_risks()
            self._running_started = True

        if self.running_options.debug:
            printer.print(f"\nGraphAutoma-[{self.name}] is getting started.", color="green")

        # Task loop divided into many dynamic steps (DS).
        while self._current_kickoff_workers:
            if self.running_options.debug:
                kickoff_worker_keys = [kickoff_info.worker_key for kickoff_info in self._current_kickoff_workers]
                printer.print(f"[DS][Before Tasks Started] kickoff workers: {kickoff_worker_keys}", color="purple")
            for kickoff_info in self._current_kickoff_workers:
                # Args mapping:
                # TODO: add top-down args mapping here...
                if kickoff_info.last_kickoff == "__automa__":
                    next_args, next_kwargs = args, kwargs
                elif kickoff_info.from_ferry:
                    next_args, next_kwargs = kickoff_info.args, kickoff_info.kwargs
                else:
                    next_args, next_kwargs = self._mapping_args(
                        kickoff_worker_key_or_name=kickoff_info.last_kickoff,
                        current_worker_key=kickoff_info.worker_key,
                    )

                if self.running_options.debug:
                    kickoff_name = kickoff_info.last_kickoff
                    if kickoff_name == "__automa__":
                        kickoff_name = f"{kickoff_name}:({self.name})"
                    printer.print(f"[{kickoff_name}] will kick off [{kickoff_info.worker_key}]", color="cyan")

                # Schedule task for each kickoff worker.
                task = asyncio.create_task(
                    # TODO1: process_async() may need to be wrapped to support better interrupt...
                    # TODO2: paralellization support...
                    self._workers[kickoff_info.worker_key].process_async(
                        *next_args, **next_kwargs
                    ),
                    name=f"Task-{kickoff_info.worker_key}"
                )
                self._running_tasks.append(_RunnningTask(
                    worker_key=kickoff_info.worker_key,
                    task=task,
                ))
            
            # Wait until all of the tasks are finished.
            while True:
                undone_tasks = [t.task for t in self._running_tasks if not t.task.done()]
                if not undone_tasks:
                    break
                await undone_tasks[0]
            

            # Process graph topology change deferred tasks triggered by add_worker() and remove_worker().
            _execute_topology_change_deferred_tasks(self._topology_change_deferred_tasks)

            # Perform post-task follow-up operations.
            for task in self._running_tasks:
                # task.task.result must be called here! It will raise an exception if task failed.
                task_result = task.task.result()
                if task.worker_key in self._workers:
                    # The current running task may be removed.
                    worker_obj = self._workers[task.worker_key]
                    # Collect results of the finished tasks.
                    worker_obj.output_buffer = task_result 
                    # reset dynamic states of finished workers.
                    self._workers_dynamic_states[task.worker_key].dependency_triggers = set(getattr(worker_obj, "dependencies", []))
                    # Update the dynamic states of successor workers.
                    for successor_key in self._worker_forwards.get(task.worker_key, []):
                        self._workers_dynamic_states[successor_key].dependency_triggers.remove(task.worker_key)

            # Graph topology validation and risk detection.
            # Guarantee the graph topology is valid and consistent after each DS.
            # 1. Validate the canonical graph.
            self._validate_canonical_graph()
            # 2. Validate the DAG constraints.
            GraphAutomaMeta.validate_dag_constraints(self._worker_forwards)
            # TODO: more validations can be added here...

            # Process set_output_worker() deferred task.
            if self._set_output_worker_deferred_task and self._set_output_worker_deferred_task.output_worker_key in self._workers:
                self._output_worker_key = self._set_output_worker_deferred_task.output_worker_key

            # TODO: Ferry-related risk detection may be added here...

            # Find next kickoff workers and rebuild _current_kickoff_workers
            self._current_kickoff_workers = []
            # New kickoff workers can be triggered by two ways:
            # 1. The ferry_to() operation is called during current worker execution.
            # 2. The dependencies are eliminated after all predecessor workers are finished.
            # So,
            # First add kickoff workers triggered by ferry_to();
            for ferry_task in self._ferry_deferred_tasks:
                self._current_kickoff_workers.append(_KickoffInfo(
                    worker_key=ferry_task.ferry_to_worker_key,
                    last_kickoff=ferry_task.kickoff_worker_key,
                    from_ferry=True,
                    args=ferry_task.args,
                    kwargs=ferry_task.kwargs,
                ))
            # Then add kickoff workers triggered by dependencies elimination.
            # Merge successor keys of all finished tasks.
            successor_keys = set()
            for task in self._running_tasks:
                for successor_key in self._worker_forwards.get(task.worker_key, []):
                    if successor_key not in successor_keys:
                        dependency_triggers = self._workers_dynamic_states[successor_key].dependency_triggers
                        if not dependency_triggers:
                            self._current_kickoff_workers.append(_KickoffInfo(
                                worker_key=successor_key,
                                last_kickoff=task.worker_key,
                            ))
                        successor_keys.add(successor_key)
            if self.running_options.debug:
                deferred_ferrys = [ferry_task.ferry_to_worker_key for ferry_task in self._ferry_deferred_tasks]
                printer.print(f"[DS][After Tasks Finished] successor workers: {successor_keys}, deferred ferrys: {deferred_ferrys}", color="purple")

            # Clear running tasks after all finished.
            self._running_tasks.clear()
            self._ferry_deferred_tasks.clear()
            self._topology_change_deferred_tasks.clear()
            self._set_output_worker_deferred_task = None

            # TODO: Can serialize and interrupt here...

        if self.running_options.debug:
            printer.print(f"GraphAutoma-[{self.name}] is finished.", color="green")

        # If the output-worker is specified, return its output as the return value of the automa.
        if self._output_worker_key:
            return self._workers[self._output_worker_key].output_buffer
        else:
            return None

    def _get_worker_dependencies(self, worker_key: str) -> List[str]:
        """
        Get the worker keys of all dependencies of the worker.
        """
        deps = self._workers[worker_key].dependencies
        return [] if deps is None else deps

    def _find_connected_components(self):
        """
        Find all of the connected components in the whole automa graph described by self._workers.
        """
        visited = set()
        component_list = []
        component_idx = {}

        def dfs(worker: str, component: List[str]):
            visited.add(worker)
            component.append(worker)
            for target in self._worker_forwards.get(worker, []):
                if target not in visited:
                    dfs(target, component)

        for worker in self._workers.keys():
            if worker not in visited:
                component_list.append([])
                current_idx = len(component_list) - 1
                current_component = component_list[current_idx]

                dfs(worker, current_component)

                for worker in current_component:
                    component_idx[worker] = current_idx

        # self._component_list, self._component_idx = component_list, component_idx
        # TODO: check how to use _component_list and _component_idx...

    def _get_kickoff_worker_from_stack(self) -> Optional[str]:
        """
        Retrieve the key of the kickoff worker for the specified worker.

        Returns
        -------
        str
            The key of the kickoff worker, or None if not found.
        """
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if frame_info.function == "process_async" and 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if isinstance(self_obj, WithKeyMixin):
                    return getattr(self_obj, "key", self.name)
        return None # TODO: how to process?

    def _mapping_args(
        self, 
        kickoff_worker_key_or_name: str,
        current_worker_key: str,
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """
        Prepare arguments for the current worker based on dependency relationships. This method is invoked 
        after all its dependencies are satisfied and before the current worker is executed.

        The arguments mapping mechanism is as follows:
        1. For workers with a single dependency: The trigger worker's output is automatically mapped as the 
        input of the current worker, using the "as_list" rule by default (unless overridden by another rule).
        2. For workers with multiple dependencies: No automatic mapping occurs (equivalent to "depressed" rule).

        Parameters
        ----------
        kickoff_worker_key_or_name : str
            The key or name of the kickoff worker.
        current_worker_key : str
            The key of the current worker.
        """
        kickoff_worker_obj = self._workers[kickoff_worker_key_or_name]
        current_worker_obj = self._workers[current_worker_key]
        args_mapping_rule = current_worker_obj.args_mapping_rule
        dep_workers_keys = self._get_worker_dependencies(current_worker_key)
        assert kickoff_worker_key_or_name in dep_workers_keys

        def as_is_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD)
            positional_param_names_without_default = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY, exclude_default=True) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True)
            var_positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.VAR_POSITIONAL)

            if len(results) == 1 and results[0] is None and len(positional_param_names) == 0 and len(var_positional_param_names) == 0:
                # The special case of returning None and no arguments are expected.
                next_args, next_kwargs = (), {}
            else:
                # Note: The order of dependent workers is important here.
                if len(results) >= len(positional_param_names_without_default) and (len(results) <= len(positional_param_names) or len(var_positional_param_names) > 0):
                    next_args, next_kwargs = tuple(results), {}
                else:
                    raise WorkerArgsMappingError(
                        f"For args_mapping_rule=\"{ArgsMappingRule.AS_IS}\", "
                        f"the worker \"{current_worker_key}\" expects at least {len(positional_param_names_without_default)} arguments "
                        f"and at most {len(positional_param_names)} arguments, "
                        f"but the dependency workers {dep_workers_keys} returned {len(results)} {'values' if len(results) > 1 else 'value'}.\n"
                        f"The returned {'values' if len(results) > 1 else 'value'} are: {results}"
                    )
            return next_args, next_kwargs

        def unpack_return_value(result: Any) -> Tuple[Tuple, Dict[str, Any]]:
            if dep_workers_keys != [kickoff_worker_key_or_name]:
                raise WorkerArgsMappingError(
                    f"The worker \"{current_worker_key}\" must has exatly one dependency for the args_mapping_rule=\"{ArgsMappingRule.UNPACK}\", "
                    f"but got {len(dep_workers_keys)} dependencies: {dep_workers_keys}"
                )
            # result is not allowed to be None, since None can not be unpacked.
            if isinstance(result, (List, Tuple)):
                # Similar args mapping logic to as_is_return_values()
                positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD)
                positional_param_names_without_default = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY, exclude_default=True) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True)
                var_positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.VAR_POSITIONAL)

                if len(result) >= len(positional_param_names_without_default) and (len(result) <= len(positional_param_names) or len(var_positional_param_names) > 0):
                    next_args, next_kwargs = tuple(result), {}
                else:
                    raise WorkerArgsMappingError(
                        f"For args_mapping_rule=\"{ArgsMappingRule.UNPACK}\", "
                        f"the worker \"{current_worker_key}\" expects at least {len(positional_param_names_without_default)} arguments "
                        f"and at most {len(positional_param_names)} arguments, "
                        f"but the kickoff worker \"{kickoff_worker_key_or_name}\" returned a list/tuple of length {len(result)}.\n"
                        f"The returned list/tuple are: {result}"
                    )
            elif isinstance(result, Mapping):
                keyword_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD) + get_param_names_by_kind(current_worker_obj.func, Parameter.KEYWORD_ONLY)
                var_keyword_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.VAR_KEYWORD)
                if len(var_keyword_param_names) > 0:
                    # A kwargs exists.
                    next_args, next_kwargs = (), {**result}
                else:
                    # Only map for the known parameters.
                    next_args, next_kwargs = (), {k: v for k, v in result.items() if k in keyword_param_names}
            else:
                # Other types, including None, are not unpackable.
                raise WorkerArgsMappingError(
                    f"args_mapping_rule=\"{ArgsMappingRule.UNPACK}\" is only valid for "
                    f"tuple/list, or dict. But the worker\"{current_worker_obj}\" got type \"{type(result)}\" from the kickoff worker \"{kickoff_worker_key_or_name}\"."
                )
            return next_args, next_kwargs

        def merge_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            if len(dep_workers_keys) < 2:
                raise WorkerArgsMappingError(
                    f"The worker \"{current_worker_key}\" must has at least 2 dependencies for the args_mapping_rule=\"{ArgsMappingRule.MERGE}\", "
                    f"but got {len(dep_workers_keys)} dependencies: {dep_workers_keys}"
                )

            positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD)
            positional_param_names_without_default = get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_ONLY, exclude_default=True) + get_param_names_by_kind(current_worker_obj.func, Parameter.POSITIONAL_OR_KEYWORD, exclude_default=True)
            var_positional_param_names = get_param_names_by_kind(current_worker_obj.func, Parameter.VAR_POSITIONAL)

            if len(positional_param_names_without_default) <= 1 and (len(positional_param_names) >= 1 or len(var_positional_param_names) > 0):
                next_args, next_kwargs = tuple([results]), {}
            else:
                raise WorkerArgsMappingError(
                    f"For args_mapping_rule=\"{ArgsMappingRule.MERGE}\", "
                    f"the worker \"{current_worker_key}\" must have at least "
                    "one positional parameter to receive the merged return values. "
                    f"Please check the signature of {current_worker_obj.func}."
                )
            return next_args, next_kwargs

        if args_mapping_rule == ArgsMappingRule.AS_IS:
            next_args, next_kwargs = as_is_return_values([self._workers[dep_worker_key].output_buffer for dep_worker_key in dep_workers_keys])
        elif args_mapping_rule == ArgsMappingRule.UNPACK:
            next_args, next_kwargs = unpack_return_value(kickoff_worker_obj.output_buffer)
        elif args_mapping_rule == ArgsMappingRule.MERGE:
            next_args, next_kwargs = merge_return_values([self._workers[dep_worker_key].output_buffer for dep_worker_key in dep_workers_keys])
        elif args_mapping_rule == ArgsMappingRule.SUPPRESSED:
            next_args, next_kwargs = (), {}

        return next_args, next_kwargs

    def __repr__(self) -> str:
        # TODO : It's good to make __repr__() of Automa compatible with eval().
        # This feature depends on the implementation of __repr__() of workers.
        class_name = self.__class__.__name__
        workers_str = self._workers.__repr__()
        return f"{class_name}(workers={workers_str})"

    def __str__(self) -> str:
        d = {}
        for k, v in self._workers.items():
            d[k] = f"{v} depends on {getattr(v, 'dependencies', [])}"
        return json.dumps(d, ensure_ascii=False, indent=4)
