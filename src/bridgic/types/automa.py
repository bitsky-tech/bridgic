import asyncio
import functools
import inspect
import traceback
import json

from typing import Any, List, Dict, Set, Callable, Union, Coroutine
from types import MethodType
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from collections import defaultdict, deque
from pydantic import BaseModel, Field

from bridgic.utils.console import printer, colored
from bridgic.types.worker import Worker
from bridgic.types.error import AutomaDeclarationError, AutomaCompilationError, WorkerSignatureError, AutomaRuntimeError

class _LandableMixin:
    def __init__(self, dependencies: List[str] = [], is_start: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dependencies = dependencies
        self.is_start = is_start

class _CallableMixin:
    def __init__(self, func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.is_coro = inspect.iscoroutinefunction(func)

class _ThreadableMixin:
    def __init__(self, as_thread: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.as_thread = as_thread

class _AutomaBuiltinWorker(_LandableMixin, _CallableMixin, _ThreadableMixin, Worker):
    def __init__(
        self,
        *,
        name: str = None,
        func: Callable,
        dependencies: List[str] = [],
        is_start: bool = False,
        as_thread: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            func=func,
            dependencies=dependencies,
            is_start=is_start,
            as_thread=as_thread,
            **kwargs,
        )

    async def process_async(self, *args, **kwargs) -> Any:
        if self.is_coro:
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

class WorkerState(BaseModel):
    is_running: bool = Field(default=False)
    dependency_triggers: Set[str] = Field(default_factory=set)

def worker(
    *,
    name: str = None,
    dependencies: List[str] = [],
    is_start: bool = False,
    as_thread: bool = False,
):
    """
    This is a decorator used to mark a callable object (such as a function or method) as an Automa 
    detectable Worker. Its core function is to add specific magic properties (such as __is_worker__) 
    to the decorated Callable so that it can be transformed into a CallableLandableWorker object.

    Parameters
    ----------
    name : str
        The name of the worker. If not provided, the name of the decorated callable will be used.
    dependencies : List[str]
        A list of worker names that the decorated callable depends on.
    is_start : bool
        Whether the decorated callable is a start worker. True means it is, while False means it is not.
    """
    if not dependencies:
        dependencies = []
    if not all([isinstance(d, str) for d in dependencies]):
        raise ValueError(f"dependencies must be a List of str")

    def validate_params(func: Callable) -> bool:
        sig = inspect.signature(func)
        has_args = any(param.kind == param.VAR_POSITIONAL for param in sig.parameters.values())
        has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
        if not (has_args and has_kwargs):
            raise WorkerSignatureError(f"parameters of function {func.__name__} must contain *args and **kwargs")
    
    def injected(func: Callable):
        validate_params(func)
        setattr(func, "__is_worker__", True)
        setattr(func, "__worker_name__", name)
        setattr(func, "__dependencies__", dependencies)
        setattr(func, "__is_start__", is_start)
        setattr(func, "__as_thread__", as_thread)
        return func

    def wrapper(func: Callable):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def inner_async(*args, **kwargs):
            return await func(*args, **kwargs)

        return injected(inner) if not inspect.iscoroutinefunction(func) else injected(inner_async)

    return wrapper

class AutomaMeta(type):
    def __new__(mcls, name, bases, dct):
        """
        This metaclass is used to:
            - Correctly handle inheritence of pre-defined workers during Automa class inheritence, 
            particularly for multiple inheritence scenarios, enabling easy extension of Automa
            - Maintain the static tables of trigger-workers and forward-workers for each worker
            - Verify that all of the pre-defined worker dependencies satisfy the DAG constraints
        """
        cls = super().__new__(mcls, name, bases, dct)

        # Inherit the graph structure from the parent classes and maintain the related data structures.
        registered_worker_funcs: Dict[str, Callable] = {}
        worker_static_triggers: Dict[str, List[str]] = {}
        worker_static_forwards: Dict[str, List[str]] = {}

        for base in bases:
            for worker_name, worker_func in getattr(base, "_registered_worker_funcs", {}).items():
                if worker_name not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_name] = worker_func
                else:
                    raise AutomaDeclarationError(
                        f"worker is defined in multiple base classes: "
                        f"base={base}, worker={worker_name}"
                    )

            for current, trigger_list in getattr(base, "_worker_static_triggers", {}).items():
                if current not in worker_static_triggers.keys():
                    worker_static_triggers[current] = trigger_list

            for current, forward_list in getattr(base, "_worker_static_forwards", {}).items():
                if current not in worker_static_forwards.keys():
                    worker_static_forwards[current] = []
                worker_static_forwards[current].extend(forward_list)

        for attr_name, attr_value in dct.items():
            # Attributes with __is_worker__ will be registered as workers.
            if hasattr(attr_value, "__is_worker__"):
                worker_name = getattr(attr_value, "__worker_name__", None) or attr_name
                dependencies = list(set(attr_value.__dependencies__))

                # Update the registered workers for current class.
                if worker_name not in registered_worker_funcs.keys():
                    registered_worker_funcs[worker_name] = attr_value
                else:
                    raise AutomaDeclarationError(
                        f"Duplicate worker names are not allowed: "
                        f"worker={worker_name}"
                    )

                # Update the table of static triggers.
                if worker_name not in worker_static_triggers.keys():
                    worker_static_triggers[worker_name] = []
                worker_static_triggers[worker_name].extend(dependencies)

                # Update the table of static forwards.
                for trigger in dependencies:
                    if trigger not in worker_static_forwards.keys():
                        worker_static_forwards[trigger] = []
                    worker_static_forwards[trigger].append(worker_name)

        # Validate if the DAG constraint is met.
        mcls.validate_dag_constraints(worker_static_forwards)

        setattr(cls, "_registered_worker_funcs", registered_worker_funcs)
        setattr(cls, "_worker_static_triggers", worker_static_triggers)
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
                f"the automa graph does not meet the DAG constraints, because the "
                f"following workers are in cycle: {nodes_in_cycle}"
            )

class Automa(_AutomaBuiltinWorker, metaclass=AutomaMeta):
    _registered_worker_funcs: Dict[str, Callable] = {}
    _worker_static_triggers: Dict[str, List[str]] = {}
    _worker_static_forwards: Dict[str, List[str]] = {}

    def __init__(
        self,
        name: str = None,
        parallel_num: int = 2,
        workers: Dict[str, Worker] = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str (default = None, then a generated name will be provided)
            The name of the automa.
        parallel_num : int (default = 2)
            The capacity of the built-in thread pool of the automa (excluding the main thread).
        workers : Dict[str, Worker] (default = {})
            A dictionary that maps the worker name to the worker instance.
        """
        # Automa will own its own thread, so it does not need to be assigned a thread.
        if kwargs.get("as_thread", False):
            raise ValueError("Automa does not need to be assigned a thread. It will have it own thread.")

        super().__init__(name=name, func=None, as_thread=False)
        cls = type(self)

        # Initialize the instances of pre-declared workers.
        self._workers: Dict[str, Worker] = {}
        if workers:
            for worker_name, worker_obj in workers.items():
                if not isinstance(worker_obj, (_LandableMixin, _ThreadableMixin)):
                    raise TypeError(
                        f"the type of worker_obj must inherit from __LandableMixin and __ThreadableMixin, "
                        f"but got {type(worker_obj)}: worker_name={worker_name}"
                    )
                self._workers[worker_name] = worker_obj
        else:
            # Workers in the class definition all can be transformed into CallableLandableWorker objects.
            for worker_name, worker_func in cls._registered_worker_funcs.items():
                # Register the worker_func as a built-in worker.
                worker_obj = _AutomaBuiltinWorker(
                    name=worker_name,
                    func=MethodType(worker_func, self),
                    dependencies=worker_func.__dependencies__,
                    is_start=worker_func.__is_start__,
                    as_thread=worker_func.__as_thread__,
                )
                self._workers[worker_name] = worker_obj

        # Define the related data structures that are used to compile the whole automa.
        self._worker_triggers: Dict[str, List[str]] = {}
        self._worker_forwards: Dict[str, List[str]] = {}
        self._component_list: List[List[str]] = []
        self._component_idx: Dict[str, int] = {}

        # Define the runtime data structures that are used when running the whole automa.
        self._worker_states: Dict[str, WorkerState] = {}
        self._worker_running_snapshot: Dict[str, bool] = None

        # Initialize the thread pool for the current automa's execution.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=parallel_num)

        # Define the main thread and its event loop for the current automa's execution.
        self._loop: asyncio.AbstractEventLoop = None
        self._thread: Thread = None

    def add_worker(
        self,
        worker_obj: Worker,
        /,
        *,
        dependencies: List[str] = [],
        is_start: bool = False,
        as_thread: bool = False,
    ):
        """
        This method is used to add a worker into the automa.

        Parameters
        ----------
        worker_obj : Worker
            The worker instance to be registered.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        as_thread : bool (default = False)
            If True, the worker will be run as a new thread.
            If False, the worker will be run as a new event.
        """
        if not isinstance(worker_obj, Worker):
            raise TypeError(
                f"worker_obj to be registered must be a Worker, "
                f"but got {type(worker_obj)}, worker_name={worker_obj.name}"
            )

        if worker_obj.name in self._workers:
            raise AutomaDeclarationError(
                f"duplicate worker names are not allowed: "
                f"worker_name={worker_obj.name}"
            )

        if not asyncio.iscoroutinefunction(worker_obj.process_async):
            raise WorkerSignatureError(
                f"process_async must be an async method, "
                f"but got {type(worker_obj.process_async)}: worker_name={worker_obj.name}"
            )

        # Validate the parameters of the "process_async" method of the worker_obj.
        worker(
            name=worker_obj.name,
            dependencies=dependencies,
            is_start=is_start,
            as_thread=as_thread,
        )(worker_obj.process_async)

        # Adapt the class of worker_obj to __LandableMixin and __ThreadableMixin type.
        class AdaptedWorker(type(worker_obj), _LandableMixin, _ThreadableMixin):
            pass

        # Create an adapted object.
        new_worker_obj = AdaptedWorker.__new__(AdaptedWorker)
        new_worker_obj.__dict__ = worker_obj.__dict__.copy()
        new_worker_obj.parent_automa = self
        new_worker_obj.dependencies = dependencies
        new_worker_obj.is_start = is_start
        new_worker_obj.as_thread = as_thread

        # Register the worker_obj.
        self._workers[new_worker_obj.name] = new_worker_obj

    def add_func_as_worker(
        self,
        /,
        *,
        name: str,
        func: Callable,
        dependencies: List[str] = [],
        is_start: bool = False,
        as_thread: bool = False,
    ):
        """
        This method is used to add a function as a worker into the automa.

        The format of the parameters will follow that of the decorator @worker(...), so that the 
        behavior of the decorated function is consistent with that of normal CallableLandableWorker objects.

        Parameters
        ----------
        name : str
            The name of the function worker.
        func : Callable
            The function to be added as a worker to the automa.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        as_thread : bool
            If True, the function that registered as a worker will be run as a new thread.
            If False, the function that registered as a worker will be run as a new event.
        """
        # Validate the parameters that the user provided.
        worker(name=name, dependencies=dependencies, is_start=is_start, as_thread=as_thread)(func)

        # Register func as a built-in worker.
        worker_obj = _AutomaBuiltinWorker(
            name=name,
            func=MethodType(func, self),
            dependencies=dependencies,
            is_start=is_start,
            as_thread=as_thread,
        )
        self._workers[worker_obj.name] = worker_obj

    def worker(
        self,
        *,
        name: str = None,
        dependencies: List[str] = [],
        is_start: bool = False,
        as_thread: bool = False,
    ):
        """
        This is a decorator used to mark a function as an Automa detectable Worker. Dislike the 
        global decorator @worker(...), it is usally used after an Automa instance is initialized.

        The format of the parameters will follow that of the decorator @worker(...), so that the 
        behavior of the decorated function is consistent with that of normal CallableLandableWorker objects.

        Parameters
        ----------
        name : str
            The name of the worker. If not provided, the name of the decorated callable will be used.
        dependencies : List[str]
            A list of worker names that the decorated callable depends on.
        is_start : bool
            Whether the decorated callable is a start worker. True means it is, while False means it is not.
        """
        def wrapper(func: Callable):
            self.add_func_as_worker(
                name=(name or func.__name__),
                func=func,
                dependencies=dependencies,
                is_start=is_start,
                as_thread=as_thread,
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
        d = {k: f"{v.__class__.__name__} depends on {getattr(v, 'dependencies', [])}" for k, v in self._workers.items()}
        return json.dumps(d, ensure_ascii=False, indent=4)

    def ferry_to(self, worker_name: str, /, *args, **kwargs):
        """
        Designate a worker to run asynchronously, enabling subsequent workers to run after its completion. 

        This operation will inject a new control flow into the automa through the designated worker, 
        and the subsequent part of the automa may be automatically triggered by this control flow.
        Whether a subsequent worker will actually run depends on whether its dependencies are satisfied 
        or whether there is an explicit "ferry_to" operation calling to it.

        Parameters
        ----------
        worker_name : str
            The name of the worker to execute.
        args : Any, optional
            Positional arguments to be passed.
        kwargs : Any, optional
            Keyword arguments to be passed.

        Note
        ----
        The "ferry_to" operation serves as the ONLY entry point for worker execution. If a worker 
        is marked as a start worker, or its dependencies are satisfied, a "ferry_to" operation will be 
        automatically called on it by the framework behind the scene. By using the "ferry_to" method, 
        developers can customize more complex execution logic in a flexible way and achieve more powerful 
        worker orchestration.
        """
        debug: bool = kwargs.get("debug", False)
        kickoff_worker: str = kwargs.get("kickoff_worker", self.name)

        if debug:
            printer.print(f"[{kickoff_worker}] will kick off [{worker_name}]")

        # Check if the target worker exists.
        if worker_name not in self._workers:
            raise AutomaRuntimeError(
                f"the target worker that is ferried to is not found: "
                f"worker_name={worker_name}"
            )

        worker = self._workers[worker_name]
        worker_state = self._worker_states[worker_name]

        # To keep control flow as unique as possible, we do not allow a worker to be started again while it is running.
        if worker_state.is_running:
            raise AutomaRuntimeError(
                f"a worker should not be requested to execute again while it is already running: "
                f"worker_name={worker_name}"
            )

        # Lock the running state to ensure that the worker is not requested to run again while it is running.
        worker_state.is_running = True

        # The kick-off-worker of next step is the current worker.
        kwargs["kickoff_worker"] = worker_name

        self._submit_to_backend(
            worker_name=worker_name,
            worker_coro=worker.process_async(*args, **kwargs),
            as_thread=worker.as_thread,
            debug=debug,
        )

    async def process_async(self, *, debug: bool = False, **kwargs) -> Any:
        # Compile the whole automa graph, taking into account both statically defined and dynamically added workers.
        self._compile_automa()

        printer.print(f"Automa-[{self.name}] is getting started.")

        # Main thread function of the current automa.
        def run_until_end(main_loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(main_loop)

            async def wait_for_termination():
                while not self._is_in_terminated_state(debug=debug):
                    await asyncio.sleep(0.01)

            self._loop.run_until_complete(wait_for_termination())

            if debug:
                printer.print("Event loop is closing.")

        # Initialize the runtime states of all workers.
        for worker_name in self._workers.keys():
            self._refresh_worker_state(worker_name)
            self._refresh_worker_buffers(worker_name)

        # Before executing the automa, the main thread and its event loop need to be initialized.
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=run_until_end, args=(self._loop,), daemon=True)

        # Collect all of the start nodes.
        start_workers: List[str] = [
            worker_name for worker_name, worker_obj in self._workers.items()
            if getattr(worker_obj, "is_start", False)
        ]

        # Kick off all of the start workers. Because "ferry_to" will submit the designated worker
        # to the backend, all start workers will be executed in parallel. As an implicit convention,
        # all start workers should have no additional input parameters and no dependencies.
        for worker_name in start_workers:
            self.ferry_to(worker_name, debug=debug, kickoff_worker=self.name, **kwargs)

        # Start the main event loop until the automa is in terminated state.
        self._thread.start()
        self._thread.join()

        printer.print(f"Automa-[{self.name}] completed execution.")

    def _compile_automa(self):
        """
        This method needs to be called at the very beginning of self.run() to ensure that:
        1. The whole graph is built out of all of the following worker sources:
            - pre-init-added workers (e.g., methods decorated by @worker(...))
            - init-added workers (e.g., using self.add_worker(...) in self.__init__)
            - post-init-added workers (e.g., functions decorated by @automa_obj.worker(...))
        2. The dependencies of each worker are confirmed to satisfy the DAG constraints
        """
        cls = type(self)

        def validate_worker(worker_name: str):
            if worker_name not in self._workers:
                raise AutomaCompilationError(
                    f'there exists an uninstantiated worker "{worker_name}" in automa "{cls.__name__}"'
                )

        # It is necessary to strictly ensure that all occurrences of the worker name correspond to an instance.
        for current, target_list in cls._worker_static_forwards.items():
            validate_worker(current)
            for target in target_list:
                validate_worker(target)

        # Refresh the related data structures of the automa instance.
        self._worker_triggers, self._worker_forwards = {}, {}

        # Read from self._workers to confirm their dependencies and maintain the related data structures.
        for worker_name, worker_obj in self._workers.items():
            dependencies = (getattr(worker_obj, "dependencies", []))

            for trigger in dependencies:
                self._worker_triggers[worker_name] = self._worker_triggers.get(worker_name, [])
                self._worker_triggers[worker_name].append(trigger)
                self._worker_forwards[trigger] = self._worker_forwards.get(trigger, [])
                self._worker_forwards[trigger].append(worker_name)

        # Validate the DAG constraints.
        AutomaMeta.validate_dag_constraints(self._worker_forwards)

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

    def _try_to_start_successors(self, worker_name: str, debug: bool = False):
        """
        Try to start the successors of the trigger worker.

        Parameters
        ----------
        worker_name : str
            The name of the trigger worker.
        """
        for target in self._worker_forwards.get(worker_name, []):
            worker_state = self._worker_states[target]
            worker_state.dependency_triggers.discard(worker_name)
            if len(worker_state.dependency_triggers) == 0:
                self.ferry_to(target, debug=debug, kickoff_worker=worker_name)

        if debug:
            printer.print(f"[{worker_name}] handover done")

    def _refresh_worker_state(self, worker_name: str):
        """
        Refresh the runtime state of the worker.

        Parameters
        ----------
        worker_name : str
            The name of the worker to refresh.
        """
        if worker_name not in self._worker_states:
            self._worker_states[worker_name] = WorkerState()

        worker_state = self._worker_states[worker_name]
        worker_state.is_running = False
        worker_state.dependency_triggers = set(self._worker_triggers.get(worker_name, []))

    def _refresh_worker_buffers(self, worker_name: str):
        """
        Refresh the output buffer and local space of the worker.

        Parameters
        ----------
        worker_name : str
            The name of the worker to refresh.
        """
        worker_obj = self._workers[worker_name]
        worker_obj.output_buffer = None
        worker_obj.local_space.clear()

    def _submit_to_backend(
        self,
        *,
        worker_name: str,
        worker_coro: Coroutine,
        as_thread: bool = False,
        debug: bool = False,
    ) -> asyncio.Future:
        """
        Submit the coroutine to the backend and return its future.

        Parameters
        ----------
        worker_name : str
            The name of the worker to submit.
        worker_coro : Coroutine
            The coroutine to be submitted.
        as_thread : bool (default = False)
            Whether to submit the coroutine as a new thread. If True, the coroutine will be submitted 
            to the thread pool. If False, the coroutine will be submitted to the event loop.

        Returns
        -------
        asyncio.Future
            The future of the submitted coroutine.
        """
        worker_obj = self._workers[worker_name]

        async def run_coro():
            if debug:
                printer.print(f"[{worker_name}]", "running")

            # Run the worker coroutine and set back the output to the corresponding output buffer.
            try:
                worker_obj.output_buffer = await worker_coro
            except Exception as e:
                printer.print(
                    f"worker {worker_name} failed with error: {e}"
                    f"\n{traceback.format_exc()}"
                )

            # The successors of the current worker may have a chance to run.
            self._try_to_start_successors(worker_name, debug=debug)

            # Release the running state of the worker.
            self._refresh_worker_state(worker_name)

            if debug:
                printer.print(f"[{worker_name}]", "done")

        if as_thread:
            # Submit a new thread.
            loop = asyncio.new_event_loop()
            future = self._loop.run_in_executor(self._executor, lambda: loop.run_until_complete(run_coro()))
            return future
        else:
            # Submit a new event to the main thread.
            future = asyncio.run_coroutine_threadsafe(run_coro(), self._loop)
            return asyncio.wrap_future(future)

    def _is_in_terminated_state(self, debug: bool = False) -> bool:
        """
        Check if the automa is in a terminated state (i.e., all workers are not running).
        """
        running_array = [not worker_state.is_running for worker_state in self._worker_states.values()]

        if debug:
            running_snapshot = {k: v.is_running for k, v in self._worker_states.items()}
            if running_snapshot != self._worker_running_snapshot:
                self._worker_running_snapshot = running_snapshot
                d = []
                for k, v in self._worker_running_snapshot.items():
                    name = colored(k, "blue")
                    state = colored("Running", "green") if v else colored("Inactive", "red")
                    d.append(f'({name}: {state})')
                printer.print(colored(f"Automa-[{self.name}] current status =>", "blue"), " ".join(d))

        return all(running_array)
