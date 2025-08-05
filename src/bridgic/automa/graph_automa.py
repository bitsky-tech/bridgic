import asyncio
import concurrent.futures
import traceback
import inspect
import json
import uuid
import concurrent

from abc import ABCMeta
from typing import Any, List, Dict, Set, Mapping, Callable, Tuple, Generic, TypeVar, Optional
from types import MethodType
from threading import Thread
from collections import defaultdict, deque
from pydantic import BaseModel, Field

from bridgic.utils.console import printer, colored
from bridgic.automa.worker import Worker
from bridgic.types.error import *
from bridgic.types.mixin import WithKeyMixin, AdaptableMixin, LandableMixin, CallableMixin
from bridgic.consts.args_mapping_rule import *
from bridgic.utils.inspect_tools import get_arg_names
from bridgic.automa import Automa
from bridgic.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType

class _GraphBuiltinWorker(WithKeyMixin, LandableMixin, CallableMixin, Worker):
    def __init__(
        self,
        *,
        key: str = None,
        func: Callable,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_SUPPRESSED,
        **kwargs,
    ):
        super().__init__(
            key=key,
            func=func,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
            **kwargs,
        )

    async def process_async(self, *args, **kwargs) -> Any:
        if self.is_coro:
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

W = TypeVar("W", bound=Worker)

class _GraphAdaptedWorker(Generic[W], WithKeyMixin, AdaptableMixin, LandableMixin, Worker):
    def __init__(
        self,
        *,
        key: str,
        worker: W,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_SUPPRESSED,
        **kwargs,
    ):
        super().__init__(
            key=key,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
            **kwargs,
        )
        self.core_worker = worker

    def __getattr__(self, key):
        return getattr(self.core_worker, key)

    async def process_async(self, *args, **kwargs) -> Any:
        return await self.core_worker.process_async(*args, **kwargs)

class WorkerState(BaseModel):
    is_running: bool = Field(default=False)
    kickoff_worker_key_or_name: str = Field(default=None)
    dependency_triggers: Set[str] = Field(default_factory=set)

def _validated_dependencies(worker_key: str, dependencies: List[str]) -> List[str]:
    if not dependencies:
        dependencies = []
    if not all([isinstance(d, str) for d in dependencies]):
        raise ValueError(
            f"dependencies must be a List of str, "
            f"but got {dependencies} for worker {worker_key}"
        )
    return dependencies

def _validated_args_mapping_rule(worker_key: str, dependencies: List[str], args_mapping_rule: str) -> str:
    if args_mapping_rule not in all_args_mapping_rules:
        raise ValueError(
            f"args_mapping_rule must be one of the following: {all_args_mapping_rules},"
            f"but got {args_mapping_rule} for worker {worker_key}"
        )
    if len(dependencies) <= 1:
        if args_mapping_rule == ARGS_MAPPING_RULE_AUTO:
            args_mapping_rule = ARGS_MAPPING_RULE_AS_LIST
    else:
        if args_mapping_rule == ARGS_MAPPING_RULE_AUTO:
            args_mapping_rule = ARGS_MAPPING_RULE_SUPPRESSED
        if args_mapping_rule != ARGS_MAPPING_RULE_SUPPRESSED:
            raise WorkerArgsMappingError(
                f"args_mapping_rule must be set to \"{ARGS_MAPPING_RULE_SUPPRESSED}\" "
                f"when the number of dependencies is greater than 1: {len(dependencies)} "
                f"for worker {worker_key}"
            )
    return args_mapping_rule

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
        worker_static_triggers: Dict[str, List[str]] = {}
        worker_static_forwards: Dict[str, List[str]] = {}
        
        for attr_name, attr_value in dct.items():
            worker_kwargs = getattr(attr_value, "__worker_kwargs__", None)
            if worker_kwargs is not None:
                complete_args = packup_worker_decorator_rumtime_args(WorkerDecoratorType.GraphAutomaMethod, worker_kwargs)
                # Convert values in complete_args to __is_worker__, __dependencies__ and other attributes used in the old metaclass version
                # TODO: reactoring may be needed
                func = attr_value
                worker_key = complete_args["key"]
                dependencies = _validated_dependencies(worker_key, complete_args["dependencies"])
                args_mapping_rule = _validated_args_mapping_rule(worker_key, dependencies, complete_args["args_mapping_rule"])
                setattr(func, "__is_worker__", True)
                setattr(func, "__worker_key__", worker_key)
                setattr(func, "__dependencies__", dependencies)
                setattr(func, "__is_start__", complete_args["is_start"])
                setattr(func, "__args_mapping_rule__", args_mapping_rule)
        
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
        mcls.validate_dag_constraints(worker_static_forwards)

        setattr(cls, "_registered_worker_funcs", registered_worker_funcs)
        setattr(cls, "_worker_static_forwards", worker_static_forwards)
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
            raise AutomaDeclarationError(
                f"the graph automa does not meet the DAG constraints, because the "
                f"following workers are in cycle: {nodes_in_cycle}"
            )

class GraphAutoma(Automa, metaclass=GraphAutomaMeta):
    _registered_worker_funcs: Dict[str, Callable] = {}
    _worker_static_forwards: Dict[str, List[str]] = {}

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
        super().__init__(name=name or f"graph-{uuid.uuid4().hex[:8]}")
        cls = type(self)

        # Start to initialize the runtime states of the Automa.
        # [IMPORTANT] 
        # The runtime states of the Automa are divided into two main parts:
        #
        # [Part One] The (dynamic) graph topology of the Automa.
        # -- Nodes: workers (A Worker can be another Automa)
        # -- Edges: Dependencies between workers
        #
        # [Part Two] The runtime states of all the workers.
        # -- Starting workers list.
        # -- Running Task list.
        # -- Dynamic in-degree of each workers.
        # -- Other runtime states...

        # Initialize [Part One]: topology.
        self._workers: Dict[str, Worker] = {}
        self._worker_triggers: Dict[str, List[str]] = {} # TODO: redundant
        self._worker_forwards: Dict[str, List[str]] = {}

        if state_dict:
            pass
            # TODO: deserialize from the state_dict
        else:
            # Workers in the class definition all can be transformed into _GraphBuiltinWorker objects.
            for worker_key, worker_func in cls._registered_worker_funcs.items():
                # Register the worker_func as a built-in worker.
                worker_obj = _GraphBuiltinWorker(
                    key=worker_key,
                    func=MethodType(worker_func, self),
                    dependencies=worker_func.__dependencies__,
                    is_start=worker_func.__is_start__,
                    args_mapping_rule=worker_func.__args_mapping_rule__,
                )
                worker_obj.parent = self
                self._workers[worker_key] = worker_obj

        # Define the related data structures that are used to compile the whole automa.
        self._component_list: List[List[str]] = []
        self._component_idx: Dict[str, int] = {}

        # Define the runtime data structures that are used when running the whole automa.
        self._worker_states: Dict[str, WorkerState] = {}
        self._worker_running_snapshot: Dict[str, bool] = None

        # Record the key of the output-worker.
        self._output_worker_key: str = output_worker_key

    def add_worker(
        self,
        key: str,
        worker_obj: Worker,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_AUTO,
    ):
        """
        This method is used to add a worker into the automa.

        Parameters
        ----------
        key : str
            The key of the worker.
        worker_obj : Worker
            The worker instance to be registered.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        args_mapping_rule : str
            The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
        """
        if not isinstance(worker_obj, Worker):
            raise TypeError(
                f"worker_obj to be registered must be a Worker, "
                f"but got {type(worker_obj)}, worker_key={key}"
            )

        if key in self._workers:
            raise AutomaDeclarationError(
                f"duplicate worker keys are not allowed: "
                f"worker_key={key}"
            )

        if not asyncio.iscoroutinefunction(worker_obj.process_async):
            raise WorkerSignatureError(
                f"process_async of Worker must be an async method, "
                f"but got {type(worker_obj.process_async)}: worker_key={key}"
            )

        if (
            isinstance(worker_obj, Worker)
            and isinstance(worker_obj, WithKeyMixin)
            and isinstance(worker_obj, LandableMixin)
        ):
            # If the worker_obj is able to be registered, directly register and rename it.
            worker_obj.key = key
            worker_obj.parent = self
            self._workers[worker_obj.key] = worker_obj
        else:
            # Validate the parameters.
            dependencies = _validated_dependencies(key, dependencies)
            args_mapping_rule = _validated_args_mapping_rule(key, dependencies, args_mapping_rule)

            worker_obj.parent = self

            new_worker_obj = _GraphAdaptedWorker[type(worker_obj)](
                key=key,
                worker=worker_obj,
                dependencies=dependencies,
                is_start=is_start,
                args_mapping_rule=args_mapping_rule,
            )
            new_worker_obj.parent = self

            # Register the worker_obj.
            self._workers[new_worker_obj.key] = new_worker_obj

    def add_func_as_worker(
        self,
        key: str,
        func: Callable,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_AUTO,
    ):
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
        args_mapping_rule : str
            The rule of arguments mapping. The options are: "auto", "as_list", "as_dict", "suppressed".
        """
        # Validate the parameters.
        dependencies = _validated_dependencies(key, dependencies)
        args_mapping_rule = _validated_args_mapping_rule(key, dependencies, args_mapping_rule)

        # Register func as an instance of _GraphBuiltinWorker.
        worker_obj = _GraphBuiltinWorker(
            key=key,
            func=MethodType(func, self),
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )
        worker_obj.parent = self
        self._workers[worker_obj.key] = worker_obj

    def worker(
        self,
        *,
        key: str = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        args_mapping_rule: str = ARGS_MAPPING_RULE_AUTO,
    ):
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

    def __getattribute__(self, key):
        workers = super().__getattribute__('_workers')
        if key in workers:
            return workers[key]
        return super().__getattribute__(key)

    def __repr__(self) -> str:
        # TODO : It's good to make __repr__() of Automa compatible with eval().
        # This feature depends on the implementation of __repr__() of workers.
        class_name = self.__class__.__name__
        workers_str = self._workers.__repr__()
        return f"{class_name}(workers={workers_str})"

    def __str__(self) -> str:
        d = {}
        for k, v in self._workers.items():
            w_type = v.__class__.__name__
            if isinstance(v, _GraphBuiltinWorker):
                w_type += "[" + v.func.__name__ + "]"
            if isinstance(v, _GraphAdaptedWorker):
                w_type += "[" + v.core_worker.__class__.__name__ + "]"
            d[k] = f"{w_type} depends on {getattr(v, 'dependencies', [])}"
        return json.dumps(d, ensure_ascii=False, indent=4)

    def ferry_to(self, worker_key: str, /, *args, **kwargs):
        """
        Designate a worker to run asynchronously, enabling subsequent workers to run after its completion. 

        This operation will inject a new control flow into the automa through the designated worker, 
        and the subsequent part of the automa may be automatically triggered by this control flow.
        Whether a subsequent worker will actually run depends on whether its dependencies are satisfied 
        or whether there is an explicit "ferry_to" operation calling to it.

        Parameters
        ----------
        worker_key : str
            The key of the worker to execute.
        args : optional
            Positional arguments to be passed.
        kwargs : optional
            Keyword arguments to be passed.

        Note
        ----
        The "ferry_to" operation serves as the ONLY entry point for worker execution. If a worker 
        is marked as a start worker, or its dependencies are satisfied, a "ferry_to" operation will be 
        automatically called on it by the framework behind the scene. By using the "ferry_to" method, 
        developers can customize more complex execution logic in a flexible way and achieve more powerful 
        worker orchestration.
        """
        kickoff_worker_key: str = self._get_kickoff_worker(worker_key)

        if self.running_options.debug:
            printer.print(f"[{kickoff_worker_key}] will kick off [{worker_key}]")

        # Check if the target worker exists.
        if worker_key not in self._workers:
            raise AutomaRuntimeError(
                f"the target worker that is ferried to is not found: "
                f"worker_key={worker_key}"
            )

        worker = self._workers[worker_key]
        worker_state = self._worker_states[worker_key]

        # To keep control flow as unique as possible, we do not allow a worker to be started again while it is running.
        if worker_state.is_running:
            raise AutomaRuntimeError(
                f"a worker should not be requested to execute again while it is already running: "
                f"worker_key={worker_key}"
            )

        # Lock the running state to ensure that the worker is not requested to run again while it is running.
        worker_state.is_running = True

        # Record the future of the execution of this worker.
        future = self._submit_to_backend(
            worker_key,
            *args,
            **kwargs
        )
        self._future_list.append(future)

    async def process_async(self, *args, **kwargs) -> Any:
        """
        Drives the execution of the entire automa by starting all start workers inside it.
        When there is no worker in the automa that can run, the whole process will end.
        If the output-worker is specified, the output of the output-worker will be returned.
        Otherwise, None will be returned.

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
        # Compile the whole automa graph, taking into account both statically defined and dynamically added workers.
        self._compile_automa()

        if self.running_options.debug:
            printer.print(f"GraphAutoma-[{self.name}] is getting started.")

        self._loop = asyncio.get_running_loop()
        self._finish_event = asyncio.Event()

        # Initialize the runtime states of all workers.
        for worker_key in self._workers.keys():
            self._refresh_worker_state(worker_key)
            self._refresh_worker_buffers(worker_key)

        # Collect all of the start nodes.
        start_workers: List[str] = [
            worker_key for worker_key, worker_obj in self._workers.items()
            if getattr(worker_obj, "is_start", False)
        ]

        # Kick off all of the start workers via "ferry_to". It will submit the designated worker
        # to the backend and all start workers will be executed in parallel.
        for worker_key in start_workers:
            # The kickoff_worker of the nested worker is the automa itself.
            self._set_kickoff_worker(worker_key, self.name)
            self.ferry_to(worker_key, *args, **kwargs)

        await self._finish_event.wait()

        for future in self._future_list:
            if future.done():
                exc = future.exception()
                if exc is not None:
                    raise exc

        if self.running_options.debug:
            printer.print(f"GraphAutoma-[{self.name}] is finished.")

        # If the output-worker is specified, return its output as the return value of the automa.
        if self._output_worker_key:
            return self._workers[self._output_worker_key].output_buffer
        else:
            return None

    def _compile_automa(self):
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
        cls = type(self)

        def validate_worker(worker_key: str):
            if worker_key not in self._workers:
                raise AutomaCompilationError(
                    f'there exists an uninstantiated worker "{worker_key}" in automa "{cls.__name__}"'
                )

        # It is necessary to strictly ensure that all occurrences of the worker key correspond to an instance.
        for current, target_list in cls._worker_static_forwards.items():
            validate_worker(current)
            for target in target_list:
                validate_worker(target)

        # Refresh the related data structures of the automa instance.
        self._worker_triggers, self._worker_forwards = {}, {}

        # Read from self._workers to confirm their dependencies and maintain the related data structures.
        for worker_key, worker_obj in self._workers.items():
            dependencies = (getattr(worker_obj, "dependencies", []))

            for trigger in dependencies:
                self._worker_triggers[worker_key] = self._worker_triggers.get(worker_key, [])
                self._worker_triggers[worker_key].append(trigger)
                self._worker_forwards[trigger] = self._worker_forwards.get(trigger, [])
                self._worker_forwards[trigger].append(worker_key)

        # Validate the DAG constraints.
        GraphAutomaMeta.validate_dag_constraints(self._worker_forwards)

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

        self._component_list, self._component_idx = component_list, component_idx

    def _try_to_start_successors(self, worker_key: str):
        """
        Try to start the successors of the trigger worker.

        Parameters
        ----------
        worker_key : str
            The key of the trigger worker.
        """
        for target in self._worker_forwards.get(worker_key, []):
            worker_state = self._worker_states[target]
            worker_state.dependency_triggers.discard(worker_key)

            if len(worker_state.dependency_triggers) == 0:
                # Map the arguments of the next worker according to the mapping rule.
                next_args, next_kwargs = self._mapping_args(
                    kickoff_worker_key_or_name=worker_key,
                    current_worker_key=target,
                )

                self._set_kickoff_worker(target, worker_key)
                self.ferry_to(target, *next_args, **next_kwargs)

    def _refresh_worker_state(self, worker_key: str):
        """
        Refresh the runtime state of the worker.

        Parameters
        ----------
        worker_key : str
            The key of the worker to refresh.
        """
        if worker_key not in self._worker_states:
            self._worker_states[worker_key] = WorkerState()

        worker_state = self._worker_states[worker_key]
        worker_state.is_running = False
        worker_state.kickoff_worker_key_or_name = None
        worker_state.dependency_triggers = set(self._worker_triggers.get(worker_key, []))

    def _set_kickoff_worker(self, worker_key: str, kickoff_worker_key_or_name: str):
        """
        Set the kickoff worker name of the worker, which is part of the worker running state.

        Parameters
        ----------
        worker_key : str
            The key of the worker to set the kickoff_worker_name for.
        kickoff_worker_key_or_name : str
            The key or name of the kickoff worker.
        """
        worker_state = self._worker_states[worker_key]
        worker_state.kickoff_worker_key_or_name = kickoff_worker_key_or_name

    def _get_kickoff_worker(self, worker_key: str) -> str:
        """
        Retrieve the key of the kickoff worker for the specified worker.

        Parameters
        ----------
        worker_key : str
            The key of the worker whose kickoff worker is to be retrieved.

        Returns
        -------
        str
            The key or name of the kickoff worker.
        """
        worker_state = self._worker_states[worker_key]
        if worker_state.kickoff_worker_key_or_name is not None:
            return worker_state.kickoff_worker_key_or_name
        else:
            for frame_info in inspect.stack():
                frame = frame_info.frame
                if frame_info.function == "process_async" and 'self' in frame.f_locals:
                    self_obj = frame.f_locals['self']
                    if isinstance(self_obj, WithKeyMixin):
                        return getattr(self_obj, "key", self.name)
            return self.name

    def _refresh_worker_buffers(self, worker_key: str):
        """
        Refresh the output buffer and local space of the worker.

        Parameters
        ----------
        worker_key : str
            The key of the worker to refresh.
        """
        worker_obj = self._workers[worker_key]
        worker_obj.output_buffer = None
        worker_obj.local_space.clear()

    def _submit_to_backend(
        self,
        worker_key: str,
        *args,
        **kwargs
    ) -> asyncio.Future:
        """
        Submit the coroutine to the backend and return its future.

        Parameters
        ----------
        worker_key : str
            The key of the worker to submit.

        Returns
        -------
        asyncio.Future
            The future of the submitted coroutine.
        """
        worker_obj = self._workers[worker_key]

        async def run_coro():
            if self.running_options.debug:
                printer.print(f"[{worker_key}]", "running")

            exc: Exception = None

            # Run the worker coroutine and set back the output to the corresponding output buffer.
            try:
                worker_obj.output_buffer = await worker_obj.process_async(*args, **kwargs)
            except Exception as e:
                exc = e

            if self.running_options.debug:
                printer.print(f"[{worker_key}]", "done" if exc is None else "failed")

            # Exception handling strategy by default:
            # 1. If the current worker fails, the entire automa execution will be terminated.
            # 2. If the current worker succeeds, it will proceed to trigger its successor worker(s).
            # TODO : It may be useful to provide callback api for developers to define how to handle exceptions.

            if exc is not None:
                stack_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                printer.print(f"Error occurred in [{worker_key}]:\n\n{stack_str}")
                self._finish_event.set()
                raise exc
            else:
                # The successors of the current worker may have a chance to run.
                self._try_to_start_successors(worker_key)

                # Release the running state of the worker.
                self._refresh_worker_state(worker_key)

                # Check and terminate the entire automa if no worker could be kickoffed.
                self._check_and_terminate_if_possible()

        return self._loop.create_task(run_coro())

    def _is_in_terminated_state(self) -> bool:
        """
        Check if the automa is in a terminated state (i.e., all workers are not running).
        """
        running_array = [not worker_state.is_running for worker_state in self._worker_states.values()]

        if self.running_options.debug:
            running_snapshot = {k: v.is_running for k, v in self._worker_states.items()}
            if running_snapshot != self._worker_running_snapshot:
                self._worker_running_snapshot = running_snapshot
                d = []
                for k, v in self._worker_running_snapshot.items():
                    key = colored(k, "blue")
                    state = "Running" if v else "Inactive"
                    state = f"{state:<9}"
                    state = colored(state, "green") if v else colored(state, "red")
                    d.append(f'\n    {state}: {key}')
                printer.print(
                    colored(f"Current status of [{self.name}] =>", "yellow"),
                    " ".join(d),
                )

        return all(running_array)

    def _check_and_terminate_if_possible(self):
        """
        Check whether all workers in the automa are inactive, and if so, terminate the automa.
        """
        if self._is_in_terminated_state():
            self._finish_event.set()

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

        def convert_return_in_list_rule(result: Any) -> Tuple[Tuple, Dict[str, Any]]:
            if not result:
                args, kwargs = (), {}
            else:
                if isinstance(result, (List, Tuple)):
                    args, kwargs = tuple(item for item in result), {}
                else:
                    args, kwargs = tuple([result]), {}
            return args, kwargs

        def convert_return_in_dict_rule(result: Any) -> Tuple[Tuple, Dict[str, Any]]:
            if not result:
                args, kwargs = (), {}
            else:
                if isinstance(result, Mapping):
                    if isinstance(current_worker_obj, _GraphBuiltinWorker):
                        arg_names = get_arg_names(current_worker_obj.func)
                    else:
                        arg_names = get_arg_names(current_worker_obj.process_async)
                    # Only map for the known parameters.
                    args, kwargs = (), {k: v for k, v in result.items() if k in arg_names}
                else:
                    raise AutomaRuntimeError(
                        f"since the args_mapping_rule is set to \"{ARGS_MAPPING_RULE_AS_DICT}\" "
                        f"for the worker \"{current_worker_key}\", the return type of the previous worker "
                        f"must be Mapping, but got {type(result)}"
                    )
            return args, kwargs

        next_args, next_kwargs = (), {}
        args_mapping_rule = current_worker_obj.args_mapping_rule

        if self._worker_triggers.get(current_worker_key, []) == [kickoff_worker_key_or_name]:
            # For one-to-one dependency relationship, map the return value of the previous worker
            # as the arguments of the current worker, according to the args_mapping_rule.
            if args_mapping_rule == ARGS_MAPPING_RULE_AS_LIST:
                next_args, next_kwargs = convert_return_in_list_rule(kickoff_worker_obj.output_buffer)
            elif args_mapping_rule == ARGS_MAPPING_RULE_AS_DICT:
                next_args, next_kwargs = convert_return_in_dict_rule(kickoff_worker_obj.output_buffer)
            elif args_mapping_rule == ARGS_MAPPING_RULE_SUPPRESSED:
                next_args, next_kwargs = (), {}
        else:
            # For multi-to-one dependency relationship, only "suppressed" rule is allowed.
            assert args_mapping_rule == ARGS_MAPPING_RULE_SUPPRESSED

        return next_args, next_kwargs
