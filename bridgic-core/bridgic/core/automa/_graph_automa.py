import asyncio
import inspect
import json
import threading
import uuid

from inspect import Parameter, _ParameterKind
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Set, Mapping, Callable, Tuple, Optional, Literal, Union, ClassVar, TYPE_CHECKING
from typing_extensions import override
from types import MethodType
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict

from bridgic.core.utils._console import printer
from bridgic.core.utils._msgpackx import dump_bytes
from bridgic.core.types._error import *
from bridgic.core.types._common import AutomaType, ArgsMappingRule
from bridgic.core.utils._args_map import safely_map_args
from bridgic.core.automa import Automa, Snapshot
from bridgic.core.automa.worker import CallableWorker, Worker
from bridgic.core.automa.interaction import Interaction, InteractionFeedback, InteractionException
from bridgic.core.automa._automa import _InteractionAndFeedback, _InteractionEventException, RunningOptions
from bridgic.core.automa.worker._worker_callback import WorkerCallback, WorkerCallbackBuilder, try_handle_error_with_callbacks
from bridgic.core.config import GlobalSetting
from bridgic.core.automa._graph_meta import GraphMeta
from bridgic.core.automa.args._args_descriptor import injector

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
    is_output: bool
    args_mapping_rule: str
    _decorated_worker: Worker
    _worker_callbacks: List[WorkerCallback]

    def __init__(
        self,
        *,
        key: Optional[str] = None,
        worker: Optional[Worker] = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
    ):
        super().__init__()
        self.key = key or f"autokey-{uuid.uuid4().hex[:8]}"
        self.dependencies = dependencies
        self.is_start = is_start
        self.is_output = is_output
        self.args_mapping_rule = args_mapping_rule
        self._decorated_worker = worker
        self._worker_callbacks = [cb.build() for cb in callback_builders]

    @override
    def get_report_info(self) -> Dict[str, Any]:
        report_info = super().get_report_info()
        report_info["key"] = self.key
        report_info["dependencies"] = self.dependencies
        report_info["is_start"] = self.is_start
        report_info["is_output"] = self.is_output
        report_info["args_mapping_rule"] = self.args_mapping_rule
        return report_info
    
    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["key"] = self.key
        state_dict["dependencies"] = self.dependencies
        state_dict["is_start"] = self.is_start
        state_dict["is_output"] = self.is_output
        state_dict["args_mapping_rule"] = self.args_mapping_rule
        state_dict["decorated_worker"] = self._decorated_worker
        state_dict["worker_callbacks"] = self._worker_callbacks
        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)
        self.key = state_dict["key"]
        self.dependencies = state_dict["dependencies"]
        self.is_start = state_dict["is_start"]
        self.is_output = state_dict["is_output"]
        self.args_mapping_rule = state_dict["args_mapping_rule"]
        self._decorated_worker = state_dict["decorated_worker"]
        self._worker_callbacks = state_dict["worker_callbacks"]
    #
    # Delegate all the properties and methods of _GraphAdaptedWorker to the decorated worker.
    # TODO: Maybe 'Worker' should be a Protocol.
    #
    @override
    async def arun(self, *args, **kwargs) -> Any:
        for callback in self._worker_callbacks:
            await callback.on_worker_start(
                key=self.key,
                is_top_level=False,
                parent=self.parent,
                arguments={"args": args, "kwargs": kwargs},
            )

        try:
            result = await self._decorated_worker.arun(*args, **kwargs)
        except Exception as e:
            # Try to handle the exception with callbacks
            handled = await try_handle_error_with_callbacks(
                callbacks=self._worker_callbacks,
                key=self.key,
                is_top_level=False,
                parent=self.parent,
                arguments={"args": args, "kwargs": kwargs},
                error=e,
            )

            # If no callback handled the exception, re-raise it
            if not handled:
                raise

            # If exception was handled, set result to None
            result = None

        for callback in self._worker_callbacks:
            await callback.on_worker_end(
                key=self.key,
                is_top_level=False,
                parent=self.parent,
                arguments={"args": args, "kwargs": kwargs},
                result=result,
            )
        return result

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
    is_output: bool = False
    args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS
    callback_builders: List[WorkerCallbackBuilder] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

class _RemoveWorkerDeferredTask(BaseModel):
    task_type: Literal["remove_worker"] = Field(default="remove_worker")
    worker_key: str

class _AddDependencyDeferredTask(BaseModel):
    task_type: Literal["add_dependency"] = Field(default="add_dependency")
    worker_key: str
    dependency: str

class _FerryDeferredTask(BaseModel):
    ferry_to_worker_key: str
    kickoff_worker_key: Optional[str]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

class GraphAutoma(Automa, metaclass=GraphMeta):
    """
    Dynamic Directed Graph (abbreviated as DDG) implementation of Automa. `GraphAutoma` manages 
    the running control flow between workers automatically, via `dependencies` and `ferry_to`.
    Outputs of workers can be mapped and passed to their successor workers in the runtime, 
    following `args_mapping_rule`.

    Parameters
    ----------
    name : Optional[str]
        The name of the automa.

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

        @worker(dependencies=["greet"], args_mapping_rule=ArgsMappingRule.AS_IS, is_output=True)
        async def output(self, message: list[str]):
            print("Echo: " + " ".join(message))

    async def main():
        automa_obj = MyGraphAutoma(name="my_graph_automa")
        await automa_obj.arun()

    asyncio.run(main())
    ```
    """

    # Automa type.
    AUTOMA_TYPE: ClassVar[AutomaType] = AutomaType.Graph

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

    _current_kickoff_workers: List[_KickoffInfo]
    _input_buffer: _AutomaInputBuffer
    _workers_dynamic_states: Dict[str, _WorkerDynamicState]

    # The whole running process of the DDG is divided into two main phases:
    # 1. [Initialization Phase] The first phase (when _automa_running is False): the initial topology of DDG was constructed.
    # 2. [Running Phase] The second phase (when _automa_running is True): the DDG is running, and the workers are executed in a dynamic step-by-step manner (DS loop).
    _automa_running: bool

    #########################################################
    #### The following fields need not to be serialized. ####
    #########################################################
    _running_tasks: List[_RunnningTask]

    # TODO: The following deferred task structures need to be thread-safe.
    # TODO: Need to be refactored when parallelization features are added.
    _topology_change_deferred_tasks: List[Union[_AddWorkerDeferredTask, _RemoveWorkerDeferredTask]]
    _ferry_deferred_tasks: List[_FerryDeferredTask]

    def __init__(
        self,
        name: Optional[str] = None,
        thread_pool: Optional[ThreadPoolExecutor] = None,
        running_options: Optional[RunningOptions] = None,
    ):
        """
        Parameters
        ----------
        name : Optional[str]
            The name of the automa.

        thread_pool : Optional[ThreadPoolExecutor]
            The thread pool for parallel running of I/O-bound tasks.

            - If not provided, a default thread pool will be used.
            The maximum number of threads in the default thread pool dependends on the number of CPU cores. Please refer to 
            the [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor) for detail.

            - If provided, all workers (including all nested Automa instances) will be run in it. In this case, the 
            application layer code is responsible to create it and shut it down.

        running_options : Optional[RunningOptions]
            Running options for this Automa instance, including callback_builders.
            If None, uses default RunningOptions.

        state_dict : Optional[Dict[str, Any]]
            A dictionary for initializing the automa's runtime states. This parameter is designed for framework use only.
        """
        super().__init__(name=name, thread_pool=thread_pool, running_options=running_options)

        self._workers = {}
        self._worker_outputs = {}
        self._automa_running = False

        # Initialize the states that need to be serialized.
        self._normal_init()

        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        # deferred tasks
        self._topology_change_deferred_tasks = []
        self._ferry_deferred_tasks = []

    def _normal_init(self):
        ###############################################################################
        # Initialization of [Part One: Topology-Related Runtime States] #### Strat ####
        ###############################################################################

        cls = type(self)

        # _workers, _worker_forwards and _workers_dynamic_states will be initialized incrementally by add_worker()...
        self._worker_forwards = {}
        self._worker_output = {}
        self._workers_dynamic_states = {}

        if cls.AUTOMA_TYPE == AutomaType.Graph:
            # The _registered_worker_funcs data are from @worker decorators.
            for worker_key, worker_func in cls._registered_worker_funcs.items():
                # The decorator based mechanism (i.e. @worker) is based on the add_worker() interface.
                # Parameters check and other implementation details can be unified.
                self._add_func_as_worker_internal(
                    key=worker_key,
                    func=worker_func,
                    dependencies=worker_func.__dependencies__,
                    is_start=worker_func.__is_start__,
                    is_output=worker_func.__is_output__,
                    args_mapping_rule=worker_func.__args_mapping_rule__,
                    callback_builders=worker_func.__callback_builders__,
                )

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

        # States related to workers.
        state_dict["workers"] = self._workers
        state_dict["worker_forwards"] = self._worker_forwards
        state_dict["workers_dynamic_states"] = self._workers_dynamic_states
        state_dict["worker_output"] = self._worker_output

        # States related to interruption recovery.
        state_dict["current_kickoff_workers"] = self._current_kickoff_workers
        state_dict["input_buffer"] = self._input_buffer

        return state_dict

    @override
    def load_from_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_from_dict(state_dict)

        self.name = state_dict["name"]
        self._automa_running = state_dict["automa_running"]

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

        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        # Deferred tasks
        self._topology_change_deferred_tasks = []
        self._set_output_worker_deferred_task = None
        self._ferry_deferred_tasks = []

    ###############################################################
    ########### [Bridgic Serialization Mechanism] ends ############
    ###############################################################

    def _add_worker_incrementally(
        self,
        key: str,
        worker: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
    ) -> None:
        """
        Incrementally add a worker into the automa. For internal use only.
        This method is one of the very basic primitives of DDG for dynamic topology changes. 
        """
        if key in self._workers:
            raise AutomaRuntimeError(
                f"duplicate workers with the same key '{key}' are not allowed to be added!"
            )

        # Merge callback builders: Global -> Ancestor Automa(s) -> Current Automa -> Nested Automa (if worker is automa) -> Worker
        effective_callback_builders = []
        effective_callback_builders.extend(GlobalSetting.read().callback_builders)
        # Collect callback builders from all ancestor automas in the ancestor chain (from top-level to current)
        effective_callback_builders.extend(self._collect_ancestor_callback_builders())
        # If the worker itself is an automa, include its own RunningOptions callback builders
        if isinstance(worker, Automa):
            effective_callback_builders.extend(worker._running_options.callback_builders)
        # Include the callback builders from the worker itself.
        effective_callback_builders.extend(callback_builders)

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
            is_output=is_output,
            args_mapping_rule=args_mapping_rule,
            callback_builders=effective_callback_builders,
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

        # If the added worker is an automa, recursively propagate callbacks to inner workers.
        if new_worker_obj.is_automa():
            nested_automa = new_worker_obj.get_decorated_worker()
            if isinstance(nested_automa, GraphAutoma):
                # Collect callback builders from all ancestor automas in the ancestor chain (from top-level to current)
                ancestor_callback_builders = self._collect_ancestor_callback_builders()
                # Append ancestor callbacks to the _cached_callbacks of the nested automa instance.
                nested_automa._cached_callbacks = nested_automa._get_automa_callbacks() + [cb.build() for cb in ancestor_callback_builders]
                # Recursively propagate ancestor callbacks to inner workers.
                self._propagate_callbacks_to_nested_automa(
                    nested_automa=nested_automa,
                    callback_builders=ancestor_callback_builders,
                )

    def _propagate_callbacks_to_nested_automa(
        self,
        nested_automa: "GraphAutoma",
        callback_builders: List[WorkerCallbackBuilder],
    ) -> None:
        """
        Recursively propagate callback builders to all workers in a nested automa.

        This method ensures that callbacks from all ancestor automas in the ancestor chain
        are applied to all workers in nested automa instances, including deeply nested ones.

        Parameters
        ----------
        nested_automa : GraphAutoma
            The nested automa instance to propagate callbacks to.
        callback_builders : List[WorkerCallbackBuilder]
            The callback builders from all ancestor automas in the ancestor chain 
            (from top-level to current) to propagate.
        """
        for worker_key in nested_automa.all_workers():
            nested_worker = nested_automa._workers[worker_key]

            # Add callback instances built from all ancestor automas' callback builders in the ancestor chain.
            new_callbacks = [cb.build() for cb in callback_builders]
            nested_worker._worker_callbacks += new_callbacks

            # Check if the nested worker is also an automa, and recursively propagate.
            if nested_worker.is_automa():
                deeper_nested_automa = nested_worker.get_decorated_worker()
                if isinstance(deeper_nested_automa, GraphAutoma):
                    # Recursively propagate to deeper nested automas.
                    # Include current nested automa's callbacks in the propagation chain,
                    # so that deeper nested workers get callbacks from all ancestor automas.
                    self._propagate_callbacks_to_nested_automa(
                        nested_automa=deeper_nested_automa,
                        callback_builders=callback_builders,
                    )

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
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
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
                is_output=is_output,
                args_mapping_rule=args_mapping_rule,
                callback_builders=callback_builders,
            )
        else:
            # Add worker during the [Running Phase].
            deferred_task = _AddWorkerDeferredTask(
                worker_key=key,
                worker_obj=worker,
                dependencies=dependencies,
                is_start=is_start,
                is_output=is_output,
                args_mapping_rule=args_mapping_rule,
                callback_builders=callback_builders,
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
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
    ) -> None:
        """
        The private version of the method `add_func_as_worker()`.
        """
        if not isinstance(func, MethodType) and key in self._registered_worker_funcs:
            func = MethodType(func, self)

        # Validate: if func is a method, its bounded __self__ must be self when add_func_as_worker() is called.
        if hasattr(func, "__self__") and func.__self__ is not self:
            raise AutomaRuntimeError(
                f"the bounded instance of `func` must be the same as the instance of the GraphAutoma, "
                f"but got {func.__self__}"
            )

        # Register func as an instance of CallableWorker.
        func_worker = CallableWorker(func)

        self._add_worker_internal(
            key=key,
            worker=func_worker,
            dependencies=dependencies,
            is_start=is_start,
            is_output=is_output,
            args_mapping_rule=args_mapping_rule,
            callback_builders=callback_builders,
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
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
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
        is_output : bool
            Whether the worker is an output worker.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        callback_builders : List[WorkerCallbackBuilder]
            A list of worker callback builders to be registered.
            Callback instances will be created from builders when the worker is instantiated.
        """
        self._add_worker_internal(
            key=key,
            worker=worker,
            dependencies=dependencies,
            is_start=is_start,
            is_output=is_output,
            args_mapping_rule=args_mapping_rule,
            callback_builders=callback_builders,
        )

    def add_func_as_worker(
        self,
        key: str,
        func: Callable,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
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
        is_output : bool
            Whether the decorated callable is an output worker. True means it is, while False means it is not.
        args_mapping_rule : ArgsMappingRule
            The rule of arguments mapping.
        callback_builders : List[WorkerCallbackBuilder]
            A list of worker callback builders to be registered.
            Callback instances will be created from builders when the worker is instantiated.
        """
        self._add_func_as_worker_internal(
            key=key,
            func=func,
            dependencies=dependencies,
            is_start=is_start,
            is_output=is_output,
            args_mapping_rule=args_mapping_rule,
            callback_builders=callback_builders,
        )

    def worker(
        self,
        *,
        key: Optional[str] = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        is_output: bool = False,
        args_mapping_rule: ArgsMappingRule = ArgsMappingRule.AS_IS,
        callback_builders: List[WorkerCallbackBuilder] = [],
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
        is_output : bool
            Whether the decorated callable is an output worker. True means it is, while False means it is not.
        args_mapping_rule : str
            The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
        callback_builders : List[WorkerCallbackBuilder]
            A list of worker callback builders to be registered.
            Callback instances will be created from builders when the worker is instantiated.
        """
        def wrapper(func: Callable):
            self._add_func_as_worker_internal(
                key=(key or func.__name__),
                func=func,
                dependencies=dependencies,
                is_start=is_start,
                is_output=is_output,
                args_mapping_rule=args_mapping_rule,
                callback_builders=callback_builders,
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
        GraphMeta.validate_dag_constraints(self._worker_forwards)
        # TODO: More validations can be added here...

        # Find all connected components of the whole automa graph.
        self._find_connected_components()

    def ferry_to(self, key: str, /, *args, **kwargs):
        """
        Defer the invocation to the specified worker, passing any provided arguments. This creates a 
        delayed call, ensuring the worker will be scheduled to run asynchronously in the next event loop, 
        independent of its dependencies.

        This primitive is commonly used for:

        1. Implementing dynamic branching based on runtime conditions.
        2. Creating logic that forms cyclic graphs.

        Parameters
        ----------
        key : str
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
            ferry_to_worker_key=key,
            kickoff_worker_key=kickoff_worker_key,
            args=args,
            kwargs=kwargs,
        )
        # Note: ferry_to() may be called in a new thread.
        # But _ferry_deferred_tasks is not necessary to be thread-safe due to Visibility Guarantees of the Bridgic Concurrency Model.
        self._ferry_deferred_tasks.append(deferred_task)

    def _clean_all_worker_local_space(self):
        """
        Clean the local space of all workers.
        """
        for worker_obj in self._workers.values():
            worker_obj.local_space = {}

    async def arun(
        self,
        *args: Tuple[Any, ...],
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        The entry point for running the constructed `GraphAutoma` instance.

        This method serves as the entry point for both initial execution and resumption after 
        interruption of an automa instance. It automatically drives the execution of workers 
        based on their `dependencies` and explicit `ferry_to()` calls. Each execution will be 
        wrapped in an `asyncio.Task` to ensure context isolation.

        **Automatic Scheduling**

        The scheduling behavior in `GraphAutoma` is automatically driven by:

        - Worker dependencies: Workers are scheduled to run only after all their necessary 
          dependencies are satisfied. The dependencies automatically drive the execution order.

        - Calling ferry_to: During execution, a worker can explicitly trigger another worker 
          by calling `ferry_to()`, which enables dynamic flow control and conditional branching.

        - Dynamic topology changes: When the graph topology is modified at runtime (such as 
          adding or removing workers or dependencies), the scheduling system seamlessly updates 
          to reflect the latest structure, ensuring that worker execution always follows the 
          current graph.

        **Human Interaction Mechanism**

        Workers can request human input by calling `interact_with_human()` during execution. 
        When this occurs:

        - The execution will be paused after the running workers finish their execution.
        - The Automa's state will be serialized into a `Snapshot` object.
        - An `InteractionException` will be raised to the application layer. It contains both the 
          list of pending `Interaction` objects and the `Snapshot` object.
        - The application layer may persist the `Snapshot` properly to resume the execution later.
        - To resume execution, the application layer should reload the Automa state using 
          `load_from_snapshot()` with the saved `Snapshot` object and call `arun()` again with 
          `feedback_data` containing the user's feedback(s) to finish a complete interactions.

        Parameters
        ----------
        args : optional
            Positional arguments to be passed.
        feedback_data : Optional[Union[InteractionFeedback, List[InteractionFeedback]]]
            Feedbacks that are received from one or multiple human interactions occurred before the
            Automa was paused. This argument may be of type `InteractionFeedback` or 
            `List[InteractionFeedback]`. If only one interaction occurred, `feedback_data` should be
            of type `InteractionFeedback`. If multiple interactions occurred simultaneously, 
            `feedback_data` should be of type `List[InteractionFeedback]`.
        kwargs : optional
            Keyword arguments which may be further propagated to contained workers.

        Returns
        -------
        Any
            The execution result of the output-worker that has the setting `is_output=True`,
            otherwise None.

        Raises
        ------
        InteractionException
            If the Automa is the top-level Automa and the `interact_with_human()` method is called
            by one or more workers within the lastest event loop iteration, this exception will be
            raised to the application layer.
        """
        if self.is_top_level():
            # For top-level automa, wrap in a task to ensure context isolation
            task = asyncio.create_task(
                self._arun_internal(*args, feedback_data=feedback_data, **kwargs),
                name=f"GraphAutoma-{self.name}-arun"
            )
            return await task
        else:
            # For nested automa, directly call _arun_internal to avoid redundant task creation
            return await self._arun_internal(*args, feedback_data=feedback_data, **kwargs)

    async def _arun_internal(
        self,
        *args: Tuple[Any, ...],
        feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
        **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Internal implementation of `arun()` for `GraphAutoma`.

        The scheduling behavior in `GraphAutoma` is automatically driven by:

        1. **Worker dependencies**: Workers are scheduled to run only after all their necessary 
        dependencies are satisfied. The dependencies automatically drives the execution order.

        2. **Calling ferry_to**: During execution, a worker can explicitly trigger another worker 
        with calling `ferry_to()`, which enables dynamic flow control and conditional branching.

        3. **Dynamic topology changes**: When the graph topology is modified at runtime (such as adding 
        or removing workers or dependencies), the scheduling system seamlessly updates to reflect 
        the latest structure, ensuring that worker execution always follows the current graph.
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
                        is_output=topology_task.is_output,
                        args_mapping_rule=topology_task.args_mapping_rule,
                        callback_builders=topology_task.callback_builders,
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
            feedback_data: Optional[Union[InteractionFeedback, List[InteractionFeedback]]] = None,
            interaction_feedback: Optional[InteractionFeedback] = None,
            interaction_feedbacks: Optional[List[InteractionFeedback]] = None,
        ):
            if feedback_data:
                if isinstance(feedback_data, list):
                    rx_feedbacks = feedback_data
                else:
                    rx_feedbacks = [feedback_data]
                return rx_feedbacks
            # For backward compatibility with old parameter names. To be removed in the future.
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

        is_top_level = self.is_top_level()

        # If this is the top-level automa, execute its callbacks separately.
        if is_top_level:
            automa_callbacks = self._get_automa_callbacks()

            for callback in automa_callbacks:
                await callback.on_worker_start(
                    key=self.name,
                    is_top_level=True,
                    parent=self.parent,
                    arguments={
                        "args": self._input_buffer.args,
                        "kwargs": self._input_buffer.kwargs,
                        "feedback_data": feedback_data,
                    },
                )

        # For backward compatibility with old parameter names. To be removed in the future.
        interaction_feedback = kwargs.get("interaction_feedback")
        interaction_feedbacks = kwargs.get("interaction_feedbacks")
        rx_feedbacks = _check_and_normalize_interaction_params(feedback_data, interaction_feedback, interaction_feedbacks)
        if rx_feedbacks:
            rx_feedbacks = _match_ongoing_interaction_and_feedbacks(rx_feedbacks)

        if running_options.debug:
            printer.print(f"\n{type(self).__name__}-[{self.name}] is getting started.", color="green")

        # Task loop divided into many dynamic steps (DS).
        is_output_worker_keys = set()
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
                    current_automa=self,
                    next_args=next_args, 
                    next_kwargs=next_kwargs
                )

                # Collect the output worker keys.
                if self._workers[kickoff_info.worker_key].is_output:
                    is_output_worker_keys.add(kickoff_info.worker_key)
                    if len(is_output_worker_keys) > 1:
                        raise AutomaRuntimeError(
                            f"It is not allowed to have more than one worker with `is_output=True` and "
                            f"they are all considered as output-worker when the automa terminates and returns."
                            f"The current output-worker keys are: {is_output_worker_keys}."
                            f"If you want to collect the results of multiple workers simultaneously, "
                            f"it is recommended that you add one worker to gather them."
                        )

                # Schedule task for each kickoff worker.
                if worker_obj.is_automa():
                    coro = worker_obj.arun(
                        *next_args,
                        feedback_data=rx_feedbacks,
                        **next_kwargs,
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

            # Handle exceptions raised by all running tasks.
            interaction_exceptions: List[_InteractionEventException] = []
            non_interaction_exceptions: List[Exception] = []

            for task in self._running_tasks:
                try:
                    # It will raise an exception if task failed.
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
                except Exception as e:
                    if isinstance(e, _InteractionEventException):
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
                    else:
                        non_interaction_exceptions.append(e)

            if len(self._topology_change_deferred_tasks) > 0:
                # Graph topology validation and risk detection. Only needed when topology changes.
                # Guarantee the graph topology is valid and consistent after each DS.
                # 1. Validate the canonical graph.
                self._validate_canonical_graph()
                # 2. Validate the DAG constraints.
                GraphMeta.validate_dag_constraints(self._worker_forwards)
                # TODO: more validations can be added here...

            # TODO: Ferry-related risk detection may be added here...

            # Handle exceptions with callbacks at the top-level automa before re-raising them.
            if is_top_level:
                # Get cached callbacks for top-level automa
                automa_callbacks = self._get_automa_callbacks()

                # Process interaction exceptions with callbacks (they cannot be suppressed, but callbacks can observe them)
                for e in interaction_exceptions + non_interaction_exceptions:
                    await try_handle_error_with_callbacks(
                        callbacks=automa_callbacks,
                        key=self.name,
                        is_top_level=True,
                        parent=self.parent,
                        arguments={
                            "args": self._input_buffer.args,
                            "kwargs": self._input_buffer.kwargs,
                            "feedback_data": feedback_data,
                        },
                        error=e,
                    )

            # For inner interaction exceptions, collect them and throw an InteractionException as a whole.
            if len(interaction_exceptions) > 0:
                all_interactions: List[Interaction] = [interaction for e in interaction_exceptions for interaction in e.args]
                if self.is_top_level():
                    # This is the top-level Automa. Serialize the Automa and raise InteractionException to the application layer.
                    serialized_automa = dump_bytes(self)
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

            # For non-interaction exceptions, immediately raise the first one directly, since none of them are meant to be suppressed.
            if len(non_interaction_exceptions) > 0:
                raise non_interaction_exceptions[0]

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

        if running_options.debug:
            printer.print(f"{type(self).__name__}-[{self.name}] is finished.", color="green")

        # After a complete run, reset all necessary states to allow the automa to re-run.
        self._input_buffer = _AutomaInputBuffer()
        if self.should_reset_local_space():
            self._clean_all_worker_local_space()
        self._ongoing_interactions.clear()
        self._worker_interaction_indices.clear()
        self._automa_running = False

        # Get result before calling callbacks
        if is_output_worker_keys:
            result = self._worker_output.get(list(is_output_worker_keys)[0], None)
        else:
            result = None

        # If this is the top-level automa, execute its callbacks separately.
        if is_top_level:
            automa_callbacks = self._get_automa_callbacks()
            for callback in automa_callbacks:
                await callback.on_worker_end(
                    key=self.name,
                    is_top_level=True,
                    parent=self.parent,
                    arguments={
                        "args": self._input_buffer.args,
                        "kwargs": self._input_buffer.kwargs,
                        "feedback_data": feedback_data,
                    },
                    result=result,
                )

        return result

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

    @override
    def _get_worker_key(self, worker: Worker) -> Optional[str]:
        for worker_key, worker_obj in self._workers.items():
            if worker_obj == worker:
                # Note: _GraphAdaptedWorker.__eq__() is overridden to support the '==' operator.
                return worker_key
        return None

    @override
    def _get_worker_instance(self, worker_key: str) -> Worker:
        return self._workers[worker_key]

    @override
    def _locate_interacting_worker(self) -> Optional[str]:
        return self._trace_back_kickoff_worker_key_from_stack()

    def _trace_back_kickoff_worker_key_from_stack(self) -> Optional[str]:
        worker = self._get_current_running_worker_instance_by_stacktrace()
        if worker:
            return self._get_worker_key(worker)
        return None

    def _get_current_running_worker_instance_by_stacktrace(self) -> Optional[Worker]:
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if isinstance(self_obj, Worker) and (not isinstance(self_obj, Automa)) and (frame_info.function == "arun" or frame_info.function == "run"):
                    return self_obj
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
