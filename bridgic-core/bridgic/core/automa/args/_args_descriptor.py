import re
from inspect import _ParameterKind
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict, TYPE_CHECKING

from bridgic.core.automa.worker import Worker
from bridgic.core.types._error import WorkerArgsInjectionError
from bridgic.core.utils._args_map import safely_map_args

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma


class InjectorNone: 
    """
    Marker object for Injector.inject() when the default value is None.
    """
    ...

class ArgsDescriptor:
    """
    A descriptor for arguments that can be injected.
    """
    ...

@dataclass
class From(ArgsDescriptor):
    """
    Implementing arguments injection for worker parameters with default value.

    When a worker needs the output of another worker but does not directly depend on 
    it in execution, you can use `From` to declare an arguments injection in 
    its parameters.

    Attributes
    ----------
    key : str
        The key of the worker to inject arguments from.
    default : Optional[Any]
        The default value of the arguments.

    Examples
    --------
    ```python
    class MyAutoma(GraphAutoma):
        @worker(is_start=True)
        def worker_0(self, user_input: int) -> int:
            return user_input + 1
        
        @worker(dependencies=["worker_0"])
        def worker_1(self, worker_0_output: int) -> int:
            return worker_0_output + 1
        
        @worker(dependencies=["worker_1"], is_output=True)
        def worker_2(self, worker_1_output: int, worker_0_output: int = From("worker_0", 1)) -> int:
            # needs the output of worker_0 but does not directly depend on it in execution
            print(f'worker_0_output: {worker_0_output}')
            return worker_1_output + 1
    ```

    Returns
    -------
    Any
        The output of the worker specified by the key.

    Raises
    ------
    WorkerArgsInjectionError
        If the worker specified by the key does not exist and no default value is set.

    Note:
    ------
    1. Can set a default value for a `From` declaration, which will be returned when the specified worker does not exist.
    2. Will raise `WorkerArgsInjectionError` if the worker specified by the key does not exist and no default value is set.
    """
    key: str
    default: Optional[Any] = InjectorNone()

def resolve_from(dep: From, worker_output: Dict[str, Any]) -> Any:
    inject_res = worker_output.get(dep.key, dep.default)
    if isinstance(inject_res, InjectorNone):
        raise WorkerArgsInjectionError(
            f"the worker: `{dep.key}` is not found in the automa or `{dep.key}` is already removed. "
            "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
        )
    return inject_res

@dataclass
class System(ArgsDescriptor):
    """
    Implementing system-level arguments injection for worker parameters.
        
    System provides access to automa-level resources and context through arguments 
    injection. It supports pattern matching for different types of system resources.

    Attributes
    ----------
    key : str
        The system resource key to inject. Supported keys:
        - "runtime_context": Runtime context for data persistence across worker executions.
        - "automa": Current automa instance.
        - "automa:worker_key": Sub-automa instance in current automa.

    Examples
    --------
    ```python
    def worker_1(x: int, current_automa = System("automa")) -> int:
        # Access current automa instance
        current_automa.add_worker(
            key="sub_automa",
            worker=SubAutoma(),
            dependencies=["worker_1"]
        )
        return x + 1

    class SubAutoma(GraphAutoma):
        @worker(is_start=True)
        def worker_0(self, user_input: int) -> int:
            return user_input + 1

    class MyAutoma(GraphAutoma):
        @worker(is_start=True)
        def worker_0(self, user_input: int, rtx = System("runtime_context")) -> int:
            # Access runtime context for data persistence
            local_space = self.get_local_space(rtx)
            count = local_space.get("count", 0)
            local_space["count"] = count + 1

            self.add_func_as_worker(
                key="worker_1",
                func=worker_1,
                dependencies=["worker_0"]
            )

            return user_input + count
            
        @worker(dependencies=["worker_1"])
        def worker_2(self, worker_1_output: int, sub_automa = System("automa:sub_automa")) -> int:
            # Access sub-automa from worker_1
            sub_automa.add_worker(
                key="worker_3",
                worker=SubAutoma(),
                dependencies=["worker_2"],
                is_output=True,
            )
            return worker_1_output + 1
    ```

    Returns
    -------
    Any
        The system resource specified by the key:
        - RuntimeContext: For "runtime_context"
        - AutomaInstance: For current automa instance or a sub-automa instance from the current automa.

    Raises
    ------
    WorkerArgsInjectionError
        - If the key pattern is not supported.
        - If the specified resource does not exist.
        - If the specified resource is not an Automa.

    Note
    ----
    1. "runtime_context" provides a `RuntimeContext` instance for data persistence
    2. "automa" provides access to the current automa instance
    3. "automa:worker_key" provides access to a sub-automa from the specified worker key
    """
    key: str
    
    def __post_init__(self):
        allowed_patterns = [
            r"^runtime_context$",
            r"^automa:.*$",
            r"^automa$",
        ]
        
        if not any(re.match(pattern, self.key) for pattern in allowed_patterns):
            raise WorkerArgsInjectionError(
                f"Key '{self.key}' is not supported. Supported keys: \n"
                f"- `runtime_context`: a context for data persistence of the current worker.\n"
                f"- `automa:<worker_key>`: a sub-automa in current automa.\n"
                f"- `automa`: the current automa instance.\n"
            )

JSON_SCHEMA_IGNORE_ARG_TYPES = (System, From)

def resolve_system(dep: System, current_worker_key: str, worker_dict: Dict[str, Worker], current_automa: "GraphAutoma") -> Any:
    if dep.key == "runtime_context":
        return RuntimeContext(worker_key=current_worker_key)
    elif dep.key.startswith("automa:"):
        worker_key = dep.key[7:]

        inject_res = worker_dict.get(worker_key, InjectorNone())
        if isinstance(inject_res, InjectorNone):
            raise WorkerArgsInjectionError(
                f"the sub-atoma: `{dep.key}` is not found in current automa. "
            )

        if not inject_res.is_automa():
            raise WorkerArgsInjectionError(
                f"the `{dep.key}` instance is not an Automa. "
            )
        
        return inject_res.get_decorated_worker()
    elif dep.key == "automa":
        return current_automa

class RuntimeContext(BaseModel):
    worker_key: str

class WorkerInjector:
    """
    Worker Dependency injection container for resolving dependency data injection of workers.

    This class manages workers for dependency injection, allowing you to inject 
    dependencies into function parameters based on their default values. 
    It is used with the `From` and `System` classes to implement dependency injection of workers.
    """
    
    def dump_to_dict(self) -> Dict[str, Any]:
        """
        Serialize WorkerInjector to dictionary.
        Since WorkerInjector is stateless, we just return an empty dict.
        """
        return {"type": "WorkerInjector"}
    
    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "WorkerInjector":
        """
        Deserialize WorkerInjector from dictionary.
        Since WorkerInjector is stateless, we just return a new instance.
        """
        return cls()

    def inject(
        self, 
        current_worker_key: str,
        current_worker_sig: Dict[_ParameterKind, List],
        current_automa: "GraphAutoma",
        next_args: Tuple[Any, ...],
        next_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Inject dependencies into parameters whose default value is a `From` or `System`.

        Parameters
        ----------
        current_worker_key : str
            The key of the current worker being processed.
        current_worker_sig : Dict[_ParameterKind, List]
            Dictionary mapping parameters to their signature information of the current worker.
        current_automa : GraphAutoma
            The current automa instance.
        next_args : Tuple[Any, ...]
            Positional arguments to be passed to the current worker.
        next_kwargs : Dict[str, Any]
            Keyword arguments to be passed to the current worker.
            
        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing the processed positional and keyword arguments for the current worker.
        """
        worker_dict = current_automa._workers
        worker_output = current_automa._worker_output

        param_list = [
            param
            for _, param_list in current_worker_sig.items()
            for param in param_list
        ]

        from_inject_kwargs = {}
        system_inject_kwargs = {}
        for name, default_value in param_list:
            if isinstance(default_value, From):
                value = resolve_from(default_value, worker_output)
                from_inject_kwargs[name] = value
            elif isinstance(default_value, System):
                value = resolve_system(default_value, current_worker_key, worker_dict, current_automa)
                system_inject_kwargs[name] = value

        if len(param_list) <= len(next_args) and (len(from_inject_kwargs) or len(system_inject_kwargs)):
            raise WorkerArgsInjectionError(
                f"The number of parameters is less than or equal to the number of positional arguments, "
                f"but got {len(param_list)} parameters and {len(next_args)} positional arguments"
            )

        # kwargs will cover priority follows: original kwargs < from inject kwargs < system inject kwargs
        current_kwargs = {}
        for k, v in next_kwargs.items():
            current_kwargs[k] = v
        for k, v in from_inject_kwargs.items():
            current_kwargs[k] = v
        for k, v in system_inject_kwargs.items():
            current_kwargs[k] = v

        next_args, next_kwargs = safely_map_args(next_args, current_kwargs, current_worker_sig)
        return next_args, next_kwargs


injector = WorkerInjector()
