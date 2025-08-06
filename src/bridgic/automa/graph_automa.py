import asyncio
import traceback
import inspect
import json
import uuid

from abc import ABCMeta
from typing import Any, List, Dict, Set, Mapping, Callable, Tuple, Optional
from types import MethodType
from collections import defaultdict, deque
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing_extensions import override

from bridgic.utils.console import printer, colored
from bridgic.automa.worker import Worker
from bridgic.types.error import *
from bridgic.types.mixin import WithKeyMixin, AdaptableMixin, LandableMixin
from bridgic.consts.args_mapping_rule import *
from bridgic.utils.inspect_tools import get_arg_names
from bridgic.automa import Automa
from bridgic.automa.worker_decorator import packup_worker_decorator_rumtime_args, WorkerDecoratorType
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

class _FerryDeferredTask(BaseModel):
    ferry_to_worker_key: str
    kickoff_worker_key: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

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

    # [IMPORTANT] 
    # The whole states of the Automa are divided into two main parts:
    #
    # [Part One: Topology] The (dynamic) graph topology of the Automa.
    # -- Nodes: workers (A Worker can be another Automa): {_workers}
    # -- Edges: Dependencies between workers: {dependencies in _workers, _worker_forwards}
    # -- Configurations: Start workers, output worker, args_mapping_rule, etc. {_output_worker_key, is_start and args_mapping_rule in _workers}
    #
    # [Part Two: Runtime States] The runtime states of all the workers.
    # -- Current kickoff workers list. {_current_kickoff_workers}
    # -- Running Task list. {_running_tasks, _ferry_deferred_tasks}
    # -- Dynamic in-degree of each workers. {_workers_dynamic_states}
    # -- Output buffer and local space for each worker. {in _workers}
    # -- Other runtime states...

    # Note: Just declarations, NOT instances!!!
    # So do not initialize them here!!!
    _workers: Dict[str, Worker]
    _worker_forwards: Dict[str, List[str]]
    _output_worker_key: str

    _current_kickoff_workers: List[_KickoffInfo]
    _workers_dynamic_states: Dict[str, _WorkerDynamicState]
    # The following need not to be serialized.
    _running_tasks: List[_RunnningTask]
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
        super().__init__(name=name or f"graph-{uuid.uuid4().hex[:8]}")
        self._workers = {}

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
        ############################################################
        # Initialization of [Part One: Topology] ******* Strat *****
        ############################################################

        # self._workers: Dict[str, Worker] = {} #TODO: __getattribute__() refactoring
        self._worker_forwards = {} # TODO: will be moved here from _compile_automa()...
        self._output_worker_key = output_worker_key

        cls = type(self)
        for worker_key, worker_func in cls._registered_worker_funcs.items():
            func_worker = CallableWorker(MethodType(worker_func, self))
            # Register the worker_func as a built-in worker.
            worker_obj = _GraphAdaptedWorker(
                key=worker_key,
                worker=func_worker,
                dependencies=worker_func.__dependencies__,
                is_start=worker_func.__is_start__,
                args_mapping_rule=worker_func.__args_mapping_rule__,
            )
            worker_obj.parent = self
            self._workers[worker_key] = worker_obj

        ############################################################
        # Initialization of [Part One: Topology] ******* End *****
        ############################################################

        ##################################################################
        # Initialization of [Part Two: Runtime States] ******* Strat *****
        ##################################################################

        # -- Current kickoff workers list.
        # The key list of the workers that are ready to be immediately executed in the next DS (Dynamic Step).
        self._current_kickoff_workers = [
            _KickoffInfo(
                worker_key=worker_key,
                last_kickoff="__automa__"
            ) for worker_key, worker_obj in self._workers.items()
            if getattr(worker_obj, "is_start", False)
        ]
        # -- Running Task list.
        # The list of the tasks that are currently being executed.
        self._running_tasks = []
        self._ferry_deferred_tasks = []
        # -- Dynamic in-degree of each workers.
        self._workers_dynamic_states = {}
        for worker_key, worker_obj in self._workers.items():
            self._workers_dynamic_states[worker_key] = _WorkerDynamicState(
                dependency_triggers=set(getattr(worker_obj, "dependencies", []))
            )

        # Define the related data structures that are used to compile the whole automa.
        self._component_list: List[List[str]] = []
        self._component_idx: Dict[str, int] = {}
        # TODO: check how to use _component_list and _component_idx...

        ##################################################################
        # Initialization of [Part Two: Runtime States] ******* End *****
        ##################################################################

    def _init_from_state_dict(
            self,
            state_dict: Dict[str, Any],
    ):
        ...
        # TODO: deserialize from the state_dict

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

        # Validate the parameters.
        dependencies = _validated_dependencies(key, dependencies)
        args_mapping_rule = _validated_args_mapping_rule(key, dependencies, args_mapping_rule)

        new_worker_obj = _GraphAdaptedWorker(
            key=key,
            worker=worker_obj,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )
        new_worker_obj.parent = self

        # Register the worker_obj.
        self._workers[new_worker_obj.key] = new_worker_obj
        # Incrementally update the dynamic states of the workers.
        self._workers_dynamic_states[new_worker_obj.key] = _WorkerDynamicState(
            dependency_triggers=set(getattr(new_worker_obj, "dependencies", []))
        )


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

        # Register func as an instance of CallableWorker.
        func_worker = CallableWorker(MethodType(func, self))
        worker_obj = _GraphAdaptedWorker(
            key=key,
            worker=func_worker,
            dependencies=dependencies,
            is_start=is_start,
            args_mapping_rule=args_mapping_rule,
        )
        worker_obj.parent = self
        self._workers[worker_obj.key] = worker_obj
        self._workers_dynamic_states[worker_obj.key] = _WorkerDynamicState(
            dependency_triggers=set(getattr(worker_obj, "dependencies", []))
        )

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
            d[k] = f"{v} depends on {getattr(v, 'dependencies', [])}"
        return json.dumps(d, ensure_ascii=False, indent=4)

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
        # Compile the whole automa graph, taking into account both statically defined and dynamically added workers.
        # TODO: will be removed later...
        self._compile_automa()

        if self.running_options.debug:
            printer.print(f"GraphAutoma-[{self.name}] is getting started.", color="green")

        # Task loop divided into many dynamic steps (DS).
        while self._current_kickoff_workers:
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
                    printer.print(f"[{kickoff_info.last_kickoff}] will kick off [{kickoff_info.worker_key}]", color="cyan")

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
            
            # Carry out post-task follow-up work
            for task in self._running_tasks:
                worker_obj = self._workers[task.worker_key]
                # Collect results of the finished tasks.
                worker_obj.output_buffer = task.task.result() # Will raise an exception if task failed.
                # reset dynamic states of finished workers.
                self._workers_dynamic_states[task.worker_key].dependency_triggers = set(getattr(worker_obj, "dependencies", []))
                # Update the dynamic states of successor workers.
                for successor_key in self._worker_forwards.get(task.worker_key, []):
                    self._workers_dynamic_states[successor_key].dependency_triggers.discard(task.worker_key)

            # TODO: Graph topology may be changed here...
            # TODO: Graph topology risk detection may be added here...
            # TODO: Ferry-related risk detection may be added here...

            # Find next kickoff workers and rebuild _current_kickoff_workers
            self._current_kickoff_workers = []
            # New kickoff workers can be triggered by two ways:
            # 1. The ferry_to() operation is called during current worker execution.
            # 2. The dependencies are eliminated after all predecessor workers are finished.
            # So,
            # First add kickoff workers triggered by ferry_to();
            for deferred_task in self._ferry_deferred_tasks:
                self._current_kickoff_workers.append(_KickoffInfo(
                    worker_key=deferred_task.ferry_to_worker_key,
                    last_kickoff=deferred_task.kickoff_worker_key,
                    from_ferry=True,
                    args=deferred_task.args,
                    kwargs=deferred_task.kwargs,
                ))
            # Then add kickoff workers triggered by dependencies elimination.
            for task in self._running_tasks:
                for successor_key in self._worker_forwards.get(task.worker_key, []):
                    dependency_triggers = self._workers_dynamic_states[successor_key].dependency_triggers
                    if not dependency_triggers:
                        self._current_kickoff_workers.append(_KickoffInfo(
                            worker_key=successor_key,
                            last_kickoff=task.worker_key,
                        ))

            # Clear running tasks after all finished.
            self._running_tasks.clear()
            self._ferry_deferred_tasks.clear()

            # TODO: Can serialize and interrupt here...

        if self.running_options.debug:
            printer.print(f"GraphAutoma-[{self.name}] is finished.", color="green")

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
        self._worker_forwards = {}

        # Read from self._workers to confirm their dependencies and maintain the related data structures.
        for worker_key, worker_obj in self._workers.items():
            dependencies = (getattr(worker_obj, "dependencies", []))
            for trigger in dependencies:
                if trigger not in self._worker_forwards:
                    self._worker_forwards[trigger] = []
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

        self._component_list, self._component_idx = component_list, component_idx

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
                    arg_names = get_arg_names(current_worker_obj.func)
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

        if self._get_worker_dependencies(current_worker_key) == [kickoff_worker_key_or_name]:
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
