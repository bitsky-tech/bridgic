import asyncio
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
import inspect
from inspect import Parameter, _ParameterKind
import json
import threading
import uuid
from typing import _ProtocolMeta
from typing import Any, List, Dict, Set, Mapping, Callable, Tuple, Optional, Literal, Union, ClassVar
from types import MethodType
from collections import defaultdict, deque
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import override
from bridgic.core.utils.console import printer
from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import *
from bridgic.core.utils.inspect_tools import get_param_names_by_kind
from bridgic.core.utils.args_map import safely_map_args
from bridgic.core.automa import Automa
from bridgic.core.automa.worker_decorator import packup_worker_decorator_rumtime_args, get_worker_decorator_default_paramap, ArgsMappingRule
from bridgic.core.types.common import AutomaType
from bridgic.core.automa.worker.callable_worker import CallableWorker
from bridgic.core.automa.interaction import Event, FeedbackSender, EventHandlerType, InteractionFeedback, Feedback, Interaction, InteractionException
from bridgic.core.automa.serialization import Snapshot
from bridgic.core.automa.arguments_descriptor import RuntimeContext, injector
from bridgic.core.utils import msgpackx

class _GraphAdaptedWorker(Worker):
    """
    A decorated worker used for GraphAutoma orchestration and scheduling. Not intended for external use.
    Follows the `Decorator` design pattern:
    - https://web.archive.org/web/20031204182047/http://patterndigest.com/patterns/Decorator.html
    - https://en.wikipedia.org/wiki/Decorator_pattern
    New Behavior Added: In addition to the original Worker functionality, maintains configuration and state variables related to dynamic scheduling and graph topology.
    """
    key: str
    dependencies: List[str]
    is_start: bool
    args_mapping_rule: str
    _decorated_worker: Worker

    def __init__(
        self,
        *,
        key: Optional[str] = None,
        worker: Optional[Worker] = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ):
        super().__init__()
        self.key = key or f"autokey-{uuid.uuid4().hex[:8]}"
        self.dependencies = dependencies
        self.is_start = is_start
        self.args_mapping_rule = args_mapping_rule
        self._decorated_worker = worker

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["key"] = self.key
        state_dict["dependencies"] = self.dependencies
        state_dict["is_start"] = self.is_start
        state_dict["args_mapping_rule"] = self.args_mapping_rule
        state_dict["decorated_worker"] = self._decorated_worker
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.key = state_dict["key"]
        self.dependencies = state_dict["dependencies"]
        self.is_start = state_dict["is_start"]
        self.args_mapping_rule = state_dict["args_mapping_rule"]
        self._decorated_worker = state_dict["decorated_worker"]

    #
    # Delegate all the properties and methods of _GraphAdaptedWorker to the decorated worker.
    # TODO: Maybe 'Worker' should be a Protocol.
    #

    @override
    async def arun(self, *args, **kwargs) -> Any:
        return await self._decorated_worker.arun(*args, **kwargs)

    @override
    def get_input_param_names(self) -> Dict[_ParameterKind, List[Tuple[str, Any]]]:
        return self._decorated_worker.get_input_param_names()

    @property
    def parent(self) -> "Automa":
        return self._decorated_worker.parent

    @parent.setter
    def parent(self, value: "Automa"):
        self._decorated_worker.parent = value

    @override
    def __str__(self) -> str:
        # TODO: need some refactoring
        return str(self._decorated_worker)
    
    @override
    def __eq__(self, other):
        if self is other:
            return True
        return self._decorated_worker == other
    
    def is_automa(self) -> bool:
        return isinstance(self._decorated_worker, Automa)

    def get_decorated_worker(self) -> Worker:
        return self._decorated_worker

@dataclass
class _RunnningTask:
    """
    States of the current running task.
    The instances of this class do not need to be serialized.
    """
    worker_key: str
    task: asyncio.Task

class _AutomaInputBuffer(BaseModel):
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}

class _KickoffInfo(BaseModel):
    # The key of the worker that is going to be kicked off.
    worker_key: str
    # Worker key or the container "__automa__"
    last_kickoff: Optional[str]
    # Whether the kickoff is triggered by ferry_to() initiated by developers.
    from_ferry: bool = False
    # Whether the run is finished.
    # Finished workers may be kicked off again after a human interaction and thus should be skipped.
    run_finished: bool = False
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = {}

class _WorkerDynamicState(BaseModel):
    # Dynamically record the dependency workers keys of each worker.
    # Will be reset to the topology edges/dependencies of the worker
    # once the task is finished or the topology is changed.
    dependency_triggers: Set[str]

class _AddWorkerDeferredTask(BaseModel):
    task_type: Literal["add_worker"] = Field(default="add_worker")
    worker_key: str
    worker_obj: Worker # Note: Not a pydantic model!!
    dependencies: List[str] = []
    is_start: bool = False
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS

    model_config = ConfigDict(arbitrary_types_allowed=True)

class _RemoveWorkerDeferredTask(BaseModel):
    task_type: Literal["remove_worker"] = Field(default="remove_worker")
    worker_key: str

class _AddDependencyDeferredTask(BaseModel):
    task_type: Literal["add_dependency"] = Field(default="add_dependency")
    worker_key: str
    dependency: str

class _SetOutputWorkerDeferredTask(BaseModel):
    output_worker_key: str

class _FerryDeferredTask(BaseModel):
    ferry_to_worker_key: str
    kickoff_worker_key: Optional[str]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class GraphAutomaMeta(_ProtocolMeta):
    """
    This metaclass is used to:
        - Correctly handle inheritence of pre-defined workers during GraphAutoma class inheritence, 
        particularly for multiple inheritence scenarios, enabling easy extension of GraphAutoma
        - Maintain the static tables of trigger-workers and forward-workers for each worker
        - Verify that all of the pre-defined worker dependencies satisfy the DAG constraints
    """

    def __new__(mcls, name, bases, dct):
        cls = super().__new__(mcls, name, bases, dct)

        # Inherit the graph structure from the parent classes and maintain the related data structures.
        registered_worker_funcs: Dict[str, Callable] = {}
        worker_static_forwards: Dict[str, List[str]] = {}
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(
                    cls, 
                    cls.worker_decorator_type(), 
                    worker_kwargs
                )
                default_paramap = get_worker_decorator_default_paramap(AutomaType.Graph)
                func = attr_value
                setattr(func, "__is_worker__", True)
                setattr(func, "__worker_key__", complete_args.get("key", default_paramap["key"]))
                setattr(func, "__dependencies__", complete_args.get("dependencies", default_paramap["dependencies"]))
                setattr(func, "__is_start__", complete_args.get("is_start", default_paramap["is_start"]))
                setattr(func, "__args_mapping_rule__", complete_args.get("args_mapping_rule", default_paramap["args_mapping_rule"]))
        
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
    
    def worker_decorator_type(cls) -> AutomaType:
        return AutomaType.Graph

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

class _InteractionEventException(Exception):
    """
    Exception raised when the `interact_with_human` method is called.
    For internal use only.
    `Interaction` obects are stored in `self.args` of the exception.
    """

class _InteractionAndFeedback(BaseModel):
    interaction: Interaction
    feedback: Optional[InteractionFeedback] = None

class GraphAutoma(Automa, metaclass=GraphAutomaMeta):
    """
    Dynamic Directed Graph (abbreviated as DDG) implementation of Automa. `GraphAutoma` manages 
    the running control flow between workers automatically, via `dependencies` and `ferry_to`.
    Outputs of workers can be mapped and passed to their successor workers in the runtime, 
    following `args_mapping_rule`.

    Parameters
    ----------
    name : Optional[str]
        The name of the automa.

    output_worker_key : Optional[str]
        The key of the output worker whose output will be returned by the automa.

    thread_pool : Optional[ThreadPoolExecutor]
        The thread pool for parallel running of I/O-bound tasks.

        - If not provided, a default thread pool will be used.
        The maximum number of threads in the default thread pool dependends on the number of CPU cores. Please refer to 
        the [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor) for detail.

        - If provided, all workers (including all nested Automa instances) will be run in it. In this case, the 
        application layer code is responsible to create it and shut it down.

    Examples
    --------

    The following example shows how to use `GraphAutoma` to create a simple graph automa that prints "Hello, Bridgic".

    ```python
    import asyncio
    from bridgic.core.automa import GraphAutoma, worker, ArgsMappingRule

    class MyGraphAutoma(GraphAutoma):
        @worker(is_start=True)
        async def greet(self) -> list[str]:
            return ["Hello", "Bridgic"]

        @worker(dependencies=["greet"], args_mapping_rule=ArgsMappingRule.AS_IS)
        async def output(self, message: list[str]):
            print("Echo: " + " ".join(message))

    async def main():
        automa_obj = MyGraphAutoma(name="my_graph_automa", output_worker_key="output")
        await automa_obj.arun()

    asyncio.run(main())
    ```
    """

    # The initial topology defined by @worker functions.
    _registered_worker_funcs: ClassVar[Dict[str, Callable]] = {}

    # IMPORTANT: The entire states of a GraphAutoma instance include 2 part:
    # 
    # Part-1 (for the states of topology structure):
    #   1. Inner worker instances: self._workers
    #   2. Relations between worker: self._worker_forwards
    #   3. Dynamic states that serve as trigger of execution of workers: self._workers_dynamic_states
    #   4. Execution result of inner workers: self._worker_output
    #   5. Configurations of this automa instance: self._output_worker_key
    # 
    # Part-2 (for the states of running states):
    #   1. Records of Workers that are going to be kicked off: self._current_kickoff_workers
    #   2. Records of running or deferred tasks:
    #      - self._running_tasks
    #      - self._topology_change_deferred_tasks
    #      - self._ferry_deferred_tasks
    #      - self._set_output_worker_deferred_task
    #   3. Buffers of automa inputs: self._input_buffer
    #   4. Ongoing human interactions: self._ongoing_interactions
    #   ...

    _workers: Dict[str, _GraphAdaptedWorker]
    _worker_output: Dict[str, Any]
    _worker_forwards: Dict[str, List[str]]
    _output_worker_key: Optional[str]

    _current_kickoff_workers: List[_KickoffInfo]
    _input_buffer: _AutomaInputBuffer
    _workers_dynamic_states: Dict[str, _WorkerDynamicState]

    # The whole running process of the DDG is divided into two main phases:
    # 1. [Initialization Phase] The first phase (when _automa_running is False): the initial topology of DDG was constructed.
    # 2. [Running Phase] The second phase (when _automa_running is True): the DDG is running, and the workers are executed in a dynamic step-by-step manner (DS loop).
    _automa_running: bool

    # Ongoing human interactions triggered by the `interact_with_human()` call from workers of the current Automa.
    # worker_key -> list of interactions.
    _ongoing_interactions: Dict[str, List[_InteractionAndFeedback]]

    #########################################################
    #### The following fields need not to be serialized. ####
    #########################################################
    _running_tasks: List[_RunnningTask]

    # TODO: The following deferred task structures need to be thread-safe.
    # TODO: Need to be refactored when parallelization features are added.
    _topology_change_deferred_tasks: List[Union[_AddWorkerDeferredTask, _RemoveWorkerDeferredTask]]
    _set_output_worker_deferred_task: _SetOutputWorkerDeferredTask
    _ferry_deferred_tasks: List[_FerryDeferredTask]

    _event_handlers: Dict[str, EventHandlerType]
    _default_event_handler: EventHandlerType

    # worker_key -> count
    _worker_interaction_indices: Dict[str, int]

    # The main event loop of the Automa, which is usually provided by the application layer.
    _main_loop: asyncio.AbstractEventLoop

    # The thread id of the main thread in which the main event loop is running.
    _main_thread_id: int

    # The thread pool for parallel running of I/O-bound tasks. It can be None.
    _thread_pool: ThreadPoolExecutor

    def __init__(
        self,
        name: Optional[str] = None,
        output_worker_key: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Parameters
        ----------
        name : Optional[str]
            The name of the automa.

        output_worker_key : Optional[str]
            The key of the output worker whose output will be returned by the automa.

        thread_pool : Optional[ThreadPoolExecutor]
            The thread pool for parallel running of I/O-bound tasks.

            - If not provided, a default thread pool will be used.
            The maximum number of threads in the default thread pool dependends on the number of CPU cores. Please refer to 
            the [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor) for detail.

            - If provided, all workers (including all nested Automa instances) will be run in it. In this case, the 
            application layer code is responsible to create it and shut it down.

        state_dict : Optional[Dict[str, Any]]
            A dictionary for initializing the automa's runtime states. This parameter is designed for framework use only.
        """
        self._workers = {}
        self._worker_outputs = {}
        super().__init__(name=name)
        self._automa_running = False

        # Initialize the states that need to be serialized.
        self._normal_init(output_worker_key)

        ########
        # The following states need not to be serialized.
        ########
        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        # deferred tasks
        self._topology_change_deferred_tasks = []
        self._set_output_worker_deferred_task = None
        self._ferry_deferred_tasks = []
        # event handling and human interactions
        self._event_handlers = {}
        self._default_event_handler = None
        self._worker_interaction_indices = {}
        self._thread_pool = thread_pool

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
        self._worker_output = {}
        self._workers_dynamic_states = {}

        if cls.worker_decorator_type() == AutomaType.Graph:
            # The _registered_worker_funcs data are from @worker decorators.
            for worker_key, worker_func in cls._registered_worker_funcs.items():
                # The decorator based mechanism (i.e. @worker) is based on the add_worker() interface.
                # Parameters check and other implementation details can be unified.
                self._add_func_as_worker_internal(
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
        # -- Automa input buffer.
        self._input_buffer = _AutomaInputBuffer()
        # -- Ongoing human interactions
        self._ongoing_interactions = {}


        ###############################################################################
        # Initialization of [Part Two: Task-Related Runtime States] ####### End #######
        ###############################################################################

    ###############################################################
    ########## [Bridgic Serialization Mechanism] starts ###########
    ###############################################################

    # The version of the serialization format.
    SERIALIZATION_VERSION: str = "1.0"

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()

        state_dict["name"] = self.name
        state_dict["automa_running"] = self._automa_running
        state_dict["output_worker_key"] = self._output_worker_key

        # States related to workers.
        state_dict["workers"] = self._workers
        state_dict["worker_forwards"] = self._worker_forwards
        state_dict["workers_dynamic_states"] = self._workers_dynamic_states
        state_dict["worker_output"] = self._worker_output

        # States related to interruption recovery.
        state_dict["current_kickoff_workers"] = self._current_kickoff_workers
        state_dict["input_buffer"] = self._input_buffer
        state_dict["ongoing_interactions"] = self._ongoing_interactions

        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)

        self.name = state_dict["name"]
        self._automa_running = state_dict["automa_running"]
        self._output_worker_key = state_dict["output_worker_key"]

        # States related to workers.
        self._workers = state_dict["workers"]
        for worker in self._workers.values():
            worker.parent = self
        self._worker_forwards = state_dict["worker_forwards"]
        self._workers_dynamic_states = state_dict["workers_dynamic_states"]
        self._worker_output = state_dict["worker_output"]

        # States related to interruption recovery.
        self._current_kickoff_workers = state_dict["current_kickoff_workers"]
        self._input_buffer = state_dict["input_buffer"]
        self._ongoing_interactions = state_dict["ongoing_interactions"]

        ########
        # The following states need not to be serialized but need to be initialized.
        ########
        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        # deferred tasks
        self._topology_change_deferred_tasks = []
        self._set_output_worker_deferred_task = None
        self._ferry_deferred_tasks = []
        # event handling and human interactions
        self._event_handlers = {}
        self._default_event_handler = None
        self._worker_interaction_indices = {}
        self._thread_pool = None

    @classmethod
    def load_from_snapshot(
        cls, 
        snapshot: Snapshot,
        thread_pool: Optional[ThreadPoolExecutor] = None,
    ) -> "GraphAutoma":
        # Here you can compare snapshot.serialization_version with SERIALIZATION_VERSION, and handle any necessary version compatibility issues if needed.
        automa = msgpackx.load_bytes(snapshot.serialized_bytes)
        if thread_pool:
            automa.thread_pool = thread_pool
        return automa

    ###############################################################
    ########### [Bridgic Serialization Mechanism] ends ############
    ###############################################################

    @property
    def thread_pool(self) -> Optional[ThreadPoolExecutor]:
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, executor: ThreadPoolExecutor) -> None:
        """
        Set the thread pool for parallel running of I/O-bound tasks.

        If an Automa is nested within another Automa, the thread pool of the top-level Automa will be used, rather than the thread pool of the nested Automa.
        """
        self._thread_pool = executor

    def _add_worker_incrementally(
        self,
        key: str,
        worker: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        Incrementally add a worker into the automa. For internal use only.
        This method is one of the very basic primitives of DDG for dynamic topology changes. 
        """
        if key in self._workers:
            raise AutomaRuntimeError(
                f"duplicate workers with the same key '{key}' are not allowed to be added!"
            )
        
        # Note: the dependencies argument must be a new copy of the list, created with list(dependencies).
        # Refer to the Python documentation for more details:
        # 1. https://docs.python.org/3/reference/compound_stmts.html#function-definitions
        # "Default parameter values are evaluated from left to right when the function definition is executed"
        # 2. https://docs.python.org/3/tutorial/controlflow.html#default-argument-values
        # "The default values are evaluated at the point of function definition in the defining scope"
        # "Important warning: The default value is evaluated only once."
        new_worker_obj = _GraphAdaptedWorker(
            key=key,
            worker=worker,
            dependencies=list(dependencies),
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
        for trigger in dependencies:
            if trigger not in self._worker_forwards:
                self._worker_forwards[trigger] = []
            self._worker_forwards[trigger].append(key)

        # TODO : Validation may be needed in appropriate time later, because we are not able to guarantee 
        # the existence of the trigger-workers in self._workers of the final automa graph after dynamically changing.

    def _remove_worker_incrementally(
        self,
        key: str
    ) -> None:
        """
        Incrementally remove a worker from the automa. For internal use only.
        This method is one of the very basic primitives of DDG for dynamic topology changes.
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

        # Remove from the forwards list of all dependencies worker.
        for trigger in worker_to_remove.dependencies:
            self._worker_forwards[trigger].remove(key)
        if key in self._worker_interaction_indices:
            del self._worker_interaction_indices[key]
        if key in self._ongoing_interactions:
            del self._ongoing_interactions[key]

        # clear the output worker key if needed.
        if key == self._output_worker_key:
            self._output_worker_key = None

    def _add_dependency_incrementally(
        self,
        key: str,
        dependency: str,
    ) -> None:
        """
        Incrementally add a dependency from `key` to `depends`. For internal use only.
        This method is one of the very basic primitives of DDG for dynamic topology changes.
        """
        if key not in self._workers:
            raise AutomaRuntimeError(
                f"fail to add dependency from a worker that does not exist: `{key}`!"
            )
        if dependency not in self._workers:
            raise AutomaRuntimeError(
                f"fail to add dependency to a worker that does not exist: `{dependency}`!"
            )
        if dependency in self._workers[key].dependencies:
            raise AutomaRuntimeError(
                f"dependency from '{key}' to '{dependency}' already exists!"
            )

        self._workers[key].dependencies.append(dependency)
        # Note this detail here for dynamic states change:
        # The new dependency added here may be removed right away if the dependency is just the next kickoff worker. This is a valid behavior.
        self._workers_dynamic_states[key].dependency_triggers.add(dependency)

        if dependency not in self._worker_forwards:
            self._worker_forwards[dependency] = []
        self._worker_forwards[dependency].append(key)

    def _add_worker_internal(
        self,
        key: str,
        worker: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        The private version of the method `add_worker()`.
        """

        def _basic_worker_params_check(key: str, worker_obj: Worker):
            if not isinstance(worker_obj, Worker):
                raise TypeError(
                    f"worker_obj to be registered must be a Worker, "
                    f"but got {type(worker_obj)} for worker '{key}'"
                )

            if not asyncio.iscoroutinefunction(worker_obj.arun):
                raise WorkerSignatureError(
                    f"arun of Worker must be an async method, "
                    f"but got {type(worker_obj.arun)} for worker '{key}'"
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
        _basic_worker_params_check(key, worker)

        if not self._automa_running:
            # Add worker during the [Initialization Phase].
            self._add_worker_incrementally(
                key=key,
                worker=worker,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
        else:
            # Add worker during the [Running Phase].
            deferred_task = _AddWorkerDeferredTask(
                worker_key=key,
                worker_obj=worker,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
            # Note1: the execution order of topology change deferred tasks is important and is determined by the order of the calls of add_worker(), remove_worker() and add_dependency() in one DS.
            # Note2: add_worker() and remove_worker() may be called in a new thread. But _topology_change_deferred_tasks is not necessary to be thread-safe due to Visibility Guarantees of the Bridgic Concurrency Model.
            self._topology_change_deferred_tasks.append(deferred_task)

    def _add_func_as_worker_internal(
        self,
        key: str,
        func: Callable,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        The private version of the method `add_func_as_worker()`.
        """
        # Register func as an instance of CallableWorker.
        if not isinstance(func, MethodType):
            func = MethodType(func, self)
        else:
            # Validate: bounded __self__ of `func` must be self when add_func_as_worker() is called.
            if func.__self__ is not self:
                raise AutomaRuntimeError(
                    f"the bounded instance of `func` must be the same as the instance of the GraphAutoma, "
                    f"but got {func.__self__}"
                )
        func_worker = CallableWorker(func)

        self._add_worker_internal(
            key=key,
            worker=func_worker,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )

    def all_workers(self) -> List[str]:
        """
        Gets a list containing the keys of all workers registered in this Automa.

        Returns
        -------
        List[str]
            A list of worker keys.
        """
        return list(self._workers.keys())

    def add_worker(
        self,
        key: str,
        worker: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
    ) -> None:
        """
        This method is used to add a worker dynamically into the automa.

        If this method is called during the [Initialization Phase], the worker will be added immediately. If this method is called during the [Running Phase], the worker will be added as a deferred task which will be executed in the next DS.

        The dependencies can be added together with a worker. However, you can add a worker without any dependencies.

        Note: The args_mapping_rule can only be set together with adding a worker, even if the worker has no any dependencies.

        Parameters
        ----------
        key : str
            The key of the worker.
        worker : Worker
            The worker instance to be registered.
        dependencies : List[str]
            A list of worker keys that the worker depends on.
        is_start : bool
            Whether the worker is a start worker.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        """
        self._add_worker_internal(
            key=key,
            worker=worker,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )

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
        self._add_func_as_worker_internal(
            key=key,
            func=func,
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
            The key of the worker. If not provided, the name of the decorated callable will be used.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        args_mapping_rule : str
            The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
        """
        def wrapper(func: Callable):
            self._add_func_as_worker_internal(
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
        if not self._automa_running:
            # remove immediately
            self._remove_worker_incrementally(key)
        else:
            deferred_task = _RemoveWorkerDeferredTask(
                worker_key=key,
            )
            # Note: the execution order of topology change deferred tasks is important and is determined by the order of the calls of add_worker(), remove_worker() and add_dependency() in one DS.
            self._topology_change_deferred_tasks.append(deferred_task)

    def add_dependency(
        self,
        key: str,
        dependency: str,
    ) -> None:
        """
        This method is used to dynamically add a dependency from `key` to `dependency`.

        Note: args_mapping_rule is not allowed to be set by this method, instead it should be set together with add_worker() or add_func_as_worker().

        Parameters
        ----------
        key : str
            The key of the worker that will depend on the worker with key `dependency`.
        dependency : str
            The key of the worker on which the worker with key `key` will depend.
        """
        ...
        if not self._automa_running:
            # add the dependency immediately
            self._add_dependency_incrementally(key, dependency)
        else:
            deferred_task = _AddDependencyDeferredTask(
                worker_key=key,
                dependency=dependency,
            )
            # Note: the execution order of topology change deferred tasks is important and is determined by the order of the calls of add_worker(), remove_worker() and add_dependency() in one DS.
            self._topology_change_deferred_tasks.append(deferred_task)

    @property
    def output_worker_key(self) -> Optional[str]:
        return self._output_worker_key
    
    @output_worker_key.setter
    def output_worker_key(self, worker_key: str):
        """
        This method is used to set the output worker of the automa dynamically.
        """
        if not self._automa_running:
            self._output_worker_key = worker_key
        else:
            deferred_task = _SetOutputWorkerDeferredTask(
                output_worker_key=worker_key,
            )
            # Note: Only the last _SetOutputWorkerDeferredTask is valid if self.output_worker_key is set multiple times in one DS.
            self._set_output_worker_deferred_task = deferred_task

    def _validate_canonical_graph(self):
        """
        This method is used to validate that DDG graph is canonical.
        """
        for worker_key, worker_obj in self._workers.items():
            for dependency_key in worker_obj.dependencies:
                if dependency_key not in self._workers:
                    raise AutomaCompilationError(
                        f"the dependency `{dependency_key}` of worker `{worker_key}` does not exist"
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

        # Validate if the output worker exists.
        if self._output_worker_key and self._output_worker_key not in self._workers:
            raise AutomaCompilationError(
                f"the output worker is not found: "
                f"output_worker_key={self._output_worker_key}"
            )

        # Find all connected components of the whole automa graph.
        self._find_connected_components()

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        """
        Defer the invocation to the specified worker, passing any provided arguments. This creates a 
        delayed call, ensuring the worker will be scheduled to run asynchronously in the next event loop, 
        independent of its dependencies.

        This primitive is commonly used for:

        1. Implementing dynamic branching based on runtime conditions.
        2. Creating logic that forms cyclic graphs.

        Parameters
        ----------
        worker_key : str
            The key of the worker to run.
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.

        Examples
        --------
        ```python
        class MyGraphAutoma(GraphAutoma):
            @worker(is_start=True)
            def start_worker(self):
                number = random.randint(0, 1)
                if number == 0:
                    self.ferry_to("cond_1_worker", number=number)
                else:
                    self.ferry_to("cond_2_worker")

            @worker()
            def cond_1_worker(self, number: int):
                print(f'Got {{number}}!')

            @worker()
            def cond_2_worker(self):
                self.ferry_to("start_worker")

        automa = MyGraphAutoma()
        await automa.arun()

        # Output: Got 0!
        ```
        """
        # TODO: check worker_key is valid, maybe deferred check...
        running_options = self._get_top_running_options()
        # if debug is enabled, trace back the kickoff worker key from stacktrace.
        kickoff_worker_key: str = self._trace_back_kickoff_worker_key_from_stack() if running_options.debug else None
        deferred_task = _FerryDeferredTask(
            ferry_to_worker_key=worker_key,
            kickoff_worker_key=kickoff_worker_key,
            args=args,
            kwargs=kwargs,
        )
        # Note: ferry_to() may be called in a new thread.
        # But _ferry_deferred_tasks is not necessary to be thread-safe due to Visibility Guarantees of the Bridgic Concurrency Model.
        self._ferry_deferred_tasks.append(deferred_task)
    

    def get_local_space(self, runtime_context: RuntimeContext) -> Dict[str, Any]:
        """
        Get the local space, if you want to clean the local space after automa.arun(), you can override the should_reset_local_space() method.

        Parameters
        ----------
        runtime_context : RuntimeContext
            The runtime context.

        Returns
        -------
        Dict[str, Any]
            The local space.
        """
        worker_key = runtime_context.worker_key
        worker_obj = self._workers[worker_key]
        return worker_obj.local_space

    def _clean_all_worker_local_space(self):
        """
        Clean the local space of all workers.
        """
        for worker_obj in self._workers.values():
            worker_obj.local_space = {}

    def should_reset_local_space(self) -> bool:
        """
        This method indicates whether to reset the local space at the end of the arun method of GraphAutoma. 
        By default, it returns True, standing for resetting. Otherwise, it means doing nothing.
        
        Examples:
        --------
        ```python
        class MyAutoma(GraphAutoma):
            def should_reset_local_space(self) -> bool:
                return False
        ```
        """
        return True

    async def arun(
        self, 
        *args: Tuple[Any, ...],
        interaction_feedback: Optional[InteractionFeedback] = None,
        interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        The entry point for running the constructed `GraphAutoma` instance.

        Parameters
        ----------
        args : optional
            Positional arguments to be passed.

        interaction_feedback : Optional[InteractionFeedback]
            Feedback that is received from a human interaction. Only one of interaction_feedback or 
            interaction_feedbacks should be provided at a time.

        interaction_feedbacks : Optional[List[InteractionFeedback]]
            Feedbacks that are received from multiple interactions occurred simultaneously before the Automa 
            was paused. Only one of interaction_feedback or interaction_feedbacks should be provided at a time.

        kwargs : optional
            Keyword arguments to be passed.

        Returns
        -------
        Any
            The execution result of the output-worker if `output_worker_key` is specified, otherwise None.

        Raises
        ------
        InteractionException
            If the Automa is the top-level Automa and the `interact_with_human()` method is called by 
            one or more workers, this exception will be raised to the application layer.

        _InteractionEventException
            If the Automa is not the top-level Automa and the `interact_with_human()` method is called by 
            one or more workers, this exception will be raised to the upper level Automa.
        """

        def _reinit_current_kickoff_workers_if_needed():
            # Note: After deserialization, the _current_kickoff_workers must not be empty!
            # Therefore, _current_kickoff_workers will only be reinitialized when the Automa is run for the first time or rerun.
            # It is guaranteed that _current_kickoff_workers will not be reinitialized when the Automa is resumed after deserialization.
            if not self._current_kickoff_workers:
                self._current_kickoff_workers = [
                    _KickoffInfo(
                        worker_key=worker_key,
                        last_kickoff="__automa__"
                    ) for worker_key, worker_obj in self._workers.items()
                    if getattr(worker_obj, "is_start", False)
                ]
                # Each time the Automa re-runs, buffer the input arguments here.
                self._input_buffer.args = args
                self._input_buffer.kwargs = kwargs

        def _execute_topology_change_deferred_tasks(tc_tasks: List[Union[_AddWorkerDeferredTask, _RemoveWorkerDeferredTask, _AddDependencyDeferredTask]]):
            for topology_task in tc_tasks:
                if topology_task.task_type == "add_worker":
                    self._add_worker_incrementally(
                        key=topology_task.worker_key,
                        worker=topology_task.worker_obj,
                        dependencies=topology_task.dependencies,
                        is_start=topology_task.is_start,
                        args_mapping_rule=topology_task.args_mapping_rule,
                    )
                elif topology_task.task_type == "remove_worker":
                    self._remove_worker_incrementally(topology_task.worker_key)
                elif topology_task.task_type == "add_dependency":
                    self._add_dependency_incrementally(topology_task.worker_key, topology_task.dependency)

        def _set_worker_run_finished(worker_key: str):
            for kickoff_info in self._current_kickoff_workers:
                if kickoff_info.worker_key == worker_key:
                    kickoff_info.run_finished = True
                    break

        def _check_and_normalize_interaction_params(
            interaction_feedback: Optional[InteractionFeedback] = None,
            interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
        ):
            if interaction_feedback and interaction_feedbacks:
                raise AutomaRuntimeError(
                    f"Only one of interaction_feedback or interaction_feedbacks can be used. "
                    f"But received interaction_feedback={interaction_feedback} and \n"
                    f"interaction_feedbacks={interaction_feedbacks}"
                )
            if interaction_feedback:
                rx_feedbacks = [interaction_feedback]
            else:
                rx_feedbacks = interaction_feedbacks
            return rx_feedbacks

        def _match_ongoing_interaction_and_feedbacks(rx_feedbacks:List[InteractionFeedback]):
            match_left_feedbacks = []
            for feedback in rx_feedbacks:
                matched = False
                for interaction_and_feedbacks in self._ongoing_interactions.values():
                    for interaction_and_feedback in interaction_and_feedbacks:
                        if interaction_and_feedback.interaction.interaction_id == feedback.interaction_id:
                            matched = True
                            # Note: Only one feedback is allowed for each interaction. Here we assume that only the first feedback is valid, which is a choice of implementation.
                            if interaction_and_feedback.feedback is None:
                                # Set feedback to self._ongoing_interactions
                                interaction_and_feedback.feedback = feedback
                            break
                    if matched:
                        break
                if not matched:
                    match_left_feedbacks.append(feedback)
            return match_left_feedbacks

        running_options = self._get_top_running_options()

        self._main_loop = asyncio.get_running_loop()
        self._main_thread_id = threading.get_ident()

        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(thread_name_prefix="bridgic-thread")

        if not self._automa_running:
            # Here is the last chance to compile and check the DDG in the end of the [Initialization Phase] (phase 1 just before the first DS).
            self._compile_graph_and_detect_risks()
            self._automa_running = True

        # An Automa needs to be re-run with _current_kickoff_workers reinitialized.
        _reinit_current_kickoff_workers_if_needed()

        rx_feedbacks = _check_and_normalize_interaction_params(interaction_feedback, interaction_feedbacks)
        if rx_feedbacks:
            rx_feedbacks = _match_ongoing_interaction_and_feedbacks(rx_feedbacks)

        if running_options.debug:
            printer.print(f"\n{type(self).__name__}-[{self.name}] is getting started.", color="green")

        # Task loop divided into many dynamic steps (DS).
        while self._current_kickoff_workers:
            # A new DS started.
            if running_options.debug:
                kickoff_worker_keys = [kickoff_info.worker_key for kickoff_info in self._current_kickoff_workers]
                printer.print(f"[DS][Before Tasks Started] kickoff workers: {kickoff_worker_keys}", color="purple")

            for kickoff_info in self._current_kickoff_workers:
                if kickoff_info.run_finished:
                    # Skip finished workers. Here is the case that the Automa is resumed after a human interaction.
                    if running_options.debug:
                        printer.print(f"[{kickoff_info.worker_key}] will be skipped - run finished", color="blue")
                    continue

                if running_options.debug:
                    kickoff_name = kickoff_info.last_kickoff
                    if kickoff_name == "__automa__":
                        kickoff_name = f"{kickoff_name}:({self.name})"
                    printer.print(f"[{kickoff_name}] will kick off [{kickoff_info.worker_key}]", color="cyan")

                # First, Arguments Mapping:
                if kickoff_info.last_kickoff == "__automa__":
                    next_args, next_kwargs = self._input_buffer.args, {}
                elif kickoff_info.from_ferry:
                    next_args, next_kwargs = kickoff_info.args, kickoff_info.kwargs
                else:
                    next_args, next_kwargs = self._mapping_args(
                        kickoff_worker_key=kickoff_info.last_kickoff,
                        current_worker_key=kickoff_info.worker_key,
                    )
                # Second, Inputs Propagation:
                next_args, next_kwargs = self._propagate_inputs(
                    current_worker_key=kickoff_info.worker_key,
                    next_args=next_args,
                    next_kwargs=next_kwargs,
                    input_kwargs=self._input_buffer.kwargs,
                )
                # Third, Resolve data injection.
                worker_obj = self._workers[kickoff_info.worker_key]
                next_args, next_kwargs = injector.inject(
                    current_worker_key=kickoff_info.worker_key, 
                    current_worker_sig=worker_obj.get_input_param_names(), 
                    worker_dict=self._workers, 
                    worker_output=self._worker_output,
                    next_args=next_args, 
                    next_kwargs=next_kwargs
                )

                # Schedule task for each kickoff worker.
                if worker_obj.is_automa():
                    coro = worker_obj.arun(
                        *next_args, 
                        interaction_feedbacks=rx_feedbacks, 
                        **next_kwargs
                    )
                else:
                    coro = worker_obj.arun(*next_args, **next_kwargs)

                task = asyncio.create_task(
                    # TODO1: arun() may need to be wrapped to support better interrupt...
                    coro,
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
                try:
                    await undone_tasks[0]
                except Exception as e:
                    ...
                    # The same exception will be raised again in the following task.result().
                    # Note: A Task is done when the wrapped coroutine either returned a value, raised an exception, or the Task was cancelled.
                    # Refer to: https://docs.python.org/3/library/asyncio-task.html#task-object
            
            # Process graph topology change deferred tasks triggered by add_worker() and remove_worker().
            _execute_topology_change_deferred_tasks(self._topology_change_deferred_tasks)

            # Perform post-task follow-up operations.
            interaction_exceptions: List[_InteractionEventException] = []
            for task in self._running_tasks:
                # task.task.result must be called here! It will raise an exception if task failed.
                try:
                    task_result = task.task.result()
                    _set_worker_run_finished(task.worker_key)
                    if task.worker_key in self._workers:
                        # The current running worker may be removed.
                        worker_obj = self._workers[task.worker_key]
                        # Collect results of the finished tasks.
                        self._worker_output[task.worker_key] = task_result
                        # reset dynamic states of finished workers.
                        self._workers_dynamic_states[task.worker_key].dependency_triggers = set(getattr(worker_obj, "dependencies", []))
                        # Update the dynamic states of successor workers.
                        for successor_key in self._worker_forwards.get(task.worker_key, []):
                            self._workers_dynamic_states[successor_key].dependency_triggers.remove(task.worker_key)
                        # Each time a worker is finished running, the ongoing interaction states should be cleared. Once it is re-run, the human interactions in the worker can be triggered again.
                        if task.worker_key in self._worker_interaction_indices:
                            del self._worker_interaction_indices[task.worker_key]
                        if task.worker_key in self._ongoing_interactions:
                            del self._ongoing_interactions[task.worker_key]

                except _InteractionEventException as e:
                    interaction_exceptions.append(e)
                    if task.worker_key in self._workers and not self._workers[task.worker_key].is_automa():
                        if task.worker_key not in self._ongoing_interactions:
                            self._ongoing_interactions[task.worker_key] = []
                        interaction=e.args[0]
                        # Make sure the interaction_id is unique for each human interaction.
                        found = False
                        for iaf in self._ongoing_interactions[task.worker_key]:
                            if iaf.interaction.interaction_id == interaction.interaction_id:
                                found = True
                                break
                        if not found:
                            self._ongoing_interactions[task.worker_key].append(_InteractionAndFeedback(
                                interaction=interaction,
                            ))

            if len(self._topology_change_deferred_tasks) > 0:
                # Graph topology validation and risk detection. Only needed when topology changes.
                # Guarantee the graph topology is valid and consistent after each DS.
                # 1. Validate the canonical graph.
                self._validate_canonical_graph()
                # 2. Validate the DAG constraints.
                GraphAutomaMeta.validate_dag_constraints(self._worker_forwards)
                # TODO: more validations can be added here...

            # Process the output_worker_key setting deferred task.
            if self._set_output_worker_deferred_task and self._set_output_worker_deferred_task.output_worker_key in self._workers:
                self._output_worker_key = self._set_output_worker_deferred_task.output_worker_key

            # TODO: Ferry-related risk detection may be added here...

            # Just after post-task operations (several deferred tasks) and before finding next kickoff workers, collect and process the interaction exceptions.
            if len(interaction_exceptions) > 0:
                all_interactions: List[Interaction] = [interaction for e in interaction_exceptions for interaction in e.args]
                if self.parent is None:
                    # This is the top-level Automa. Serialize the Automa and raise InteractionException to the application layer.
                    serialized_automa = msgpackx.dump_bytes(self)
                    snapshot = Snapshot(
                        serialized_bytes=serialized_automa,
                        serialization_version=GraphAutoma.SERIALIZATION_VERSION,
                    )
                    raise InteractionException(
                        interactions=all_interactions,
                        snapshot=snapshot,
                    )
                else:
                    # Continue raise exception to the upper level Automa.
                    raise _InteractionEventException(*all_interactions)

            # Find next kickoff workers and rebuild _current_kickoff_workers
            run_finished_worker_keys: List[str] = [kickoff_info.worker_key for kickoff_info in self._current_kickoff_workers if kickoff_info.run_finished]
            assert len(run_finished_worker_keys) == len(self._current_kickoff_workers)
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
            for worker_key in run_finished_worker_keys:
                # Note: The `worker_key` worker may have been removed from the Automa.
                for successor_key in self._worker_forwards.get(worker_key, []):
                    if successor_key not in successor_keys:
                        dependency_triggers = self._workers_dynamic_states[successor_key].dependency_triggers
                        if not dependency_triggers:
                            self._current_kickoff_workers.append(_KickoffInfo(
                                worker_key=successor_key,
                                last_kickoff=worker_key,
                            ))
                        successor_keys.add(successor_key)
            if running_options.debug:
                deferred_ferrys = [ferry_task.ferry_to_worker_key for ferry_task in self._ferry_deferred_tasks]
                printer.print(f"[DS][After Tasks Finished] successor workers: {successor_keys}, deferred ferrys: {deferred_ferrys}", color="purple")

            # Clear running tasks after all finished.
            self._running_tasks.clear()
            self._ferry_deferred_tasks.clear()
            self._topology_change_deferred_tasks.clear()
            self._set_output_worker_deferred_task = None

        if running_options.debug:
            printer.print(f"{type(self).__name__}-[{self.name}] is finished.", color="green")

        # After a complete run, reset all necessary states to allow the automa to re-run.
        self._input_buffer = _AutomaInputBuffer()
        if self.should_reset_local_space():
            self._clean_all_worker_local_space()
        self._ongoing_interactions.clear()
        self._worker_interaction_indices.clear()
        self._automa_running = False

        # If the output-worker is specified, return its output as the return value of the automa.
        if self._output_worker_key:
            return self._worker_output[self._output_worker_key]
        else:
            return None

    ###############################################################
    ########## [Bridgic Event Handling Mechanism] starts ##########
    ###############################################################

    def register_event_handler(self, event_type: Optional[str], event_handler: EventHandlerType) -> None:
        """
        Register an event handler for the specified event type.

        Note: Only event handlers registered on the top-level Automa will be invoked to handle events.

        Parameters
        ----------
        event_type: Optional[str]
            The type of event to be handled. If set to None, the event handler will be registered as the default handler and will be used to handle all event types.
        event_handler: EventHandlerType
            The event handler to be registered.
        """
        if event_type is None:
            self._default_event_handler = event_handler
        else:
            self._event_handlers[event_type] = event_handler

    def unregister_event_handler(self, event_type: Optional[str]) -> None:
        """
        Unregister an event handler for the specified event type.

        Parameters
        ----------
        event_type: Optional[str]
            The type of event to be unregistered. If set to None, the default event handler will be unregistered.
        """
        if event_type in self._event_handlers:
            del self._event_handlers[event_type]
        if event_type is None:
            self._default_event_handler = None

    def unregister_all_event_handlers(self) -> None:
        """
        Unregister all event handlers.
        """
        self._event_handlers.clear()
        self._default_event_handler = None

    class _FeedbackSender(FeedbackSender):
        def __init__(
                self, 
                future: asyncio.Future[Feedback],
                post_loop: AbstractEventLoop,
                ):
            self._future = future
            self._post_loop = post_loop
        
        def send(self, feedback: Feedback) -> None:
            try:
                current_loop = asyncio.get_running_loop()
            except Exception:
                current_loop = None
            try:
                if current_loop is self._post_loop:
                    self._future.set_result(feedback)
                else:
                    self._post_loop.call_soon_threadsafe(self._future.set_result, feedback)
            except asyncio.InvalidStateError:
                # Suppress the InvalidStateError to be raised, maybe due to timeout.
                import warnings
                warnings.warn(f"Feedback future already set. feedback: {feedback}", FutureWarning)

    @override
    def post_event(self, event: Event) -> None:
        """
        Post an event to the application layer outside the Automa.

        The event handler implemented by the application layer will be called in the same thread as the worker (maybe the main thread or a new thread from the thread pool).
        
        Note that `post_event` can be called in a non-async method or an async method.

        The event will be bubbled up to the top-level Automa, where it will be processed by the event handler registered with the event type.

        Parameters
        ----------
        event: Event
            The event to be posted.
        """
        def _handler_need_feedback_sender(handler: EventHandlerType):
            positional_param_names = get_param_names_by_kind(handler, Parameter.POSITIONAL_ONLY) + get_param_names_by_kind(handler, Parameter.POSITIONAL_OR_KEYWORD)
            var_positional_param_names = get_param_names_by_kind(handler, Parameter.VAR_POSITIONAL)
            return len(var_positional_param_names) > 0 or len(positional_param_names) > 1

        if self.parent is not None:
            # Bubble up the event to the top-level Automa.
            return self.parent.post_event(event)

        # Here is the top-level Automa.
        # Call event handlers
        if event.event_type in self._event_handlers:
            if _handler_need_feedback_sender(self._event_handlers[event.event_type]):
                self._event_handlers[event.event_type](event, feedback_sender=None)
            else:
                self._event_handlers[event.event_type](event)
        if self._default_event_handler is not None:
            if _handler_need_feedback_sender(self._default_event_handler):
                self._default_event_handler(event, feedback_sender=None)
            else:
                self._default_event_handler(event)

    def request_feedback(
        self, 
        event: Event,
        timeout: Optional[float] = None
    ) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        Note that `post_event` should only be called from within a non-async method running in the new thread of the Automa thread pool.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        timeout: Optional[float]
            A float or int number of seconds to wait for if the feedback is not received. If None, then there is no limit on the wait time.

        Returns
        -------
        Feedback
            The feedback received from the application layer.

        Raises
        ------
        TimeoutError
            If the feedback is not received before the timeout. Note that the raised exception is the built-in `TimeoutError` exception, instead of asyncio.TimeoutError or concurrent.futures.TimeoutError!
        """
        if threading.get_ident() == self._main_thread_id:
            raise AutomaRuntimeError(
                f"`request_feedback` should only be called in a different thread from the main thread of the {self.name}. "
            )
        return asyncio.run_coroutine_threadsafe(
            self.request_feedback_async(event, timeout),
            self._main_loop
        ).result()

    async def request_feedback_async(
        self, 
        event: Event,
        timeout: Optional[float] = None
    ) -> Feedback:
        """
        Request feedback for the specified event from the application layer outside the Automa. This method blocks the caller until the feedback is received.

        The event handler implemented by the application layer will be called in the next event loop, in the main thread.

        Parameters
        ----------
        event: Event
            The event to be posted to the event handler implemented by the application layer.
        timeout: Optional[float]
            A float or int number of seconds to wait for if the feedback is not received. If None, then there is no limit on the wait time.

        Returns
        -------
        Feedback
            The feedback received from the application layer.

        Raises
        ------
        TimeoutError
            If the feedback is not received before the timeout. Note that the raised exception is the built-in `TimeoutError` exception, instead of asyncio.TimeoutError!
        """
        if self.parent is not None:
            # Bubble up the event to the top-level Automa.
            return await self.parent.request_feedback_async(event, timeout)
        
        # Here is the top-level Automa.
        event_loop = asyncio.get_running_loop()
        future = event_loop.create_future()
        feedback_sender = self._FeedbackSender(future, event_loop)
        # Call event handlers
        if event.event_type in self._event_handlers:
            self._event_handlers[event.event_type](event, feedback_sender)
        if self._default_event_handler is not None:
            self._default_event_handler(event, feedback_sender)

        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError as e:
            # When python >= 3.11 here.
            raise TimeoutError(f"No feedback is received before timeout in Automa[{self.name}]") from e
        except asyncio.TimeoutError as e:
            # Version compatibility resolution: asyncio.wait_for raises asyncio.TimeoutError before python 3.11.
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.wait_for
            raise TimeoutError(f"No feedback is received before timeout in Automa[{self.name}]") from e

    ###############################################################
    ########### [Bridgic Event Handling Mechanism] ends ###########
    ###############################################################

    ###############################################################
    ######## [Bridgic Human Interaction Mechanism] starts #########
    ###############################################################

    def interact_with_human(self, event: Event) -> InteractionFeedback:
        kickoff_worker_key: str = self._trace_back_kickoff_worker_key_from_stack()
        if kickoff_worker_key:
            return self.interact_with_human_from_worker_key(event, kickoff_worker_key)
        raise AutomaRuntimeError(
            f"Get kickoff worker failed in Automa[{self.name}] "
            f"when trying to interact with human with event: {event}"
        )

    def interact_with_human_from_worker(
        self,
        event: Event,
        worker: Worker
    ) -> InteractionFeedback:
        worker_key = self._loopup_worker_key_by_instance(worker)
        if worker_key:
            return self.interact_with_human_from_worker_key(event, worker_key)
        raise AutomaRuntimeError(
            f"Not found worker[{worker}] in Automa[{self.name}] "
            f"when trying to interact with human with event: {event}"
        )

    def interact_with_human_from_worker_key(
        self,
        event: Event,
        worker_key: str
    ) -> InteractionFeedback:
        # Match interaction_feedback to see if it matches
        matched_feedback: _InteractionAndFeedback = None
        cur_interact_index = self._get_and_increment_interaction_index(worker_key)
        if worker_key in self._ongoing_interactions:
            interaction_and_feedbacks = self._ongoing_interactions[worker_key]
            if cur_interact_index < len(interaction_and_feedbacks):
                matched_feedback = interaction_and_feedbacks[cur_interact_index]
                # Check the event type
                if event.event_type != matched_feedback.interaction.event.event_type:
                    raise AutomaRuntimeError(
                        f"Event type mismatch! Automa[{self.name}-worker[{worker_key}]]. "
                        f"interact_with_human passed-in event: {event}\n"
                        f"ongoing interaction && feedback: {matched_feedback}\n"
                    )
        if matched_feedback is None or matched_feedback.feedback is None:
            # Important: The interaction_id should be unique for each human interaction.
            interaction_id = uuid.uuid4().hex if matched_feedback is None else matched_feedback.interaction.interaction_id
            # Match interaction_feedback failed, raise an exception to go into the human interactioin process.
            raise _InteractionEventException(Interaction(
                interaction_id=interaction_id,
                event=event,
            ))
        else:
            # Match interaction_feedback succeeded, return it.
            return matched_feedback.feedback

    def _get_and_increment_interaction_index(self, worker_key: str) -> int:
        if worker_key not in self._worker_interaction_indices:
            cur_index = 0
            self._worker_interaction_indices[worker_key] = 0
        else:
            cur_index = self._worker_interaction_indices[worker_key]
        self._worker_interaction_indices[worker_key] += 1
        return cur_index

    ###############################################################
    ######### [Bridgic Human Interaction Mechanism] ends ##########
    ###############################################################

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

    def _trace_back_kickoff_worker_key_from_stack(self) -> Optional[str]:
        worker = self._get_current_running_worker_instance_by_stacktrace()
        if worker:
            return self._loopup_worker_key_by_instance(worker)
        return None

    def _get_current_running_worker_instance_by_stacktrace(self) -> Optional[Worker]:
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if isinstance(self_obj, Worker) and (not isinstance(self_obj, Automa)) and (frame_info.function == "arun" or frame_info.function == "run"):
                    return self_obj
        return None

    def _loopup_worker_key_by_instance(self, worker: Worker) -> Optional[str]:
        for worker_key, worker_obj in self._workers.items():
            if worker_obj == worker:
                # Note: _GraphAdaptedWorker.__eq__() is overridden to support the '==' operator.
                return worker_key
        return None

    def _mapping_args(
        self, 
        kickoff_worker_key: str,
        current_worker_key: str,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Resolve arguments mapping between workers that have dependency relationships.

        Parameters
        ----------
        kickoff_worker_key : str
            The key of the kickoff worker.
        current_worker_key : str
            The key of the current worker.
        
        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            The mapped positional arguments and keyword arguments.
        """
        current_worker_obj = self._workers[current_worker_key]
        args_mapping_rule = current_worker_obj.args_mapping_rule
        dep_workers_keys = self._get_worker_dependencies(current_worker_key)
        assert kickoff_worker_key in dep_workers_keys

        def as_is_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            next_args, next_kwargs = tuple(results), {}
            return next_args, next_kwargs

        def unpack_return_value(result: Any) -> Tuple[Tuple, Dict[str, Any]]:
            if dep_workers_keys != [kickoff_worker_key]:
                raise WorkerArgsMappingError(
                    f"The worker \"{current_worker_key}\" must has exactly one dependency for the args_mapping_rule=\"{ArgsMappingRule.UNPACK}\", "
                    f"but got {len(dep_workers_keys)} dependencies: {dep_workers_keys}"
                )
            # result is not allowed to be None, since None can not be unpacked.
            if isinstance(result, (List, Tuple)):
                # Similar args mapping logic to as_is_return_values()
                next_args, next_kwargs = tuple(result), {}
            elif isinstance(result, Mapping):
                next_args, next_kwargs = (), {**result}

            else:
                # Other types, including None, are not unpackable.
                raise WorkerArgsMappingError(
                    f"args_mapping_rule=\"{ArgsMappingRule.UNPACK}\" is only valid for "
                    f"tuple/list, or dict. But the worker \"{current_worker_key}\" got type \"{type(result)}\" from the kickoff worker \"{kickoff_worker_key}\"."
                )
            return next_args, next_kwargs

        def merge_return_values(results: List[Any]) -> Tuple[Tuple, Dict[str, Any]]:
            next_args, next_kwargs = tuple([results]), {}
            return next_args, next_kwargs

        if args_mapping_rule == ArgsMappingRule.AS_IS:
            next_args, next_kwargs = as_is_return_values([self._worker_output[dep_worker_key] for dep_worker_key in dep_workers_keys])
        elif args_mapping_rule == ArgsMappingRule.UNPACK:
            next_args, next_kwargs = unpack_return_value(self._worker_output[kickoff_worker_key])
        elif args_mapping_rule == ArgsMappingRule.MERGE:
            next_args, next_kwargs = merge_return_values([self._worker_output[dep_worker_key] for dep_worker_key in dep_workers_keys])
        elif args_mapping_rule == ArgsMappingRule.SUPPRESSED:
            next_args, next_kwargs = (), {}

        return next_args, next_kwargs

    def _propagate_inputs(
        self, 
        current_worker_key: str,
        next_args: Tuple[Any, ...],
        next_kwargs: Dict[str, Any],
        input_kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Resolve inputs propagation from the input buffer of the container Automa to every worker within the Automa.

        Parameters
        ----------
        current_worker_key : str
            The key of the current worker.
        next_args : Tuple[Any, ...]
            The positional arguments to be mapped.
        next_kwargs : Dict[str, Any]
            The keyword arguments to be mapped.
        input_kwargs : Dict[str, Any]
            The keyword arguments to be propagated from the input buffer of the container Automa.

        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            The mapped positional arguments and keyword arguments.
        """
        current_worker_obj = self._workers[current_worker_key]
        input_kwargs = {k:v for k,v in input_kwargs.items() if k not in next_kwargs}
        next_kwargs = {**next_kwargs, **input_kwargs}
        rx_param_names_dict = current_worker_obj.get_input_param_names()
        next_args, next_kwargs = safely_map_args(next_args, next_kwargs, rx_param_names_dict)
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
        