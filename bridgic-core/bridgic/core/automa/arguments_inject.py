from enum import Enum
from inspect import Parameter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import AutomaDataInjectionError
from bridgic.core.utils.args_map import safely_map_args
from pydantic import BaseModel


class InjectorNone: 
    """
    Marker object for Injector.inject() when the default value is None.
    """

@dataclass
class From:
    """
    A declarative object for dependency injection.
    """
    key: str
    default: Optional[Any] = InjectorNone()

class SystemType(Enum):
    """
    The type of the system injection.
    """
    RUNTIME_CONTEXT = "runtime_context"

@dataclass
class System:
    """
    A declarative object for system injection.
    """
    type: str

class RuntimeContext(BaseModel):
    worker_key: str

class WorkerInjector:
    """
    Worker Dependency injection container for resolving dependency data injection of workers.

    This class manages workers for dependency injection, allowing you to  inject 
    dependencies into function parameters based on their default values. 
    It is used with the `From` class to implement dependency injection of workers.

    Main Methods
    ------------
    resolve(dep: From, worker_dict: Dict[str, Worker]) -> Any
        Resolve and return the result for a given dependency key.

    inject(param_list: List[Tuple[str, Optional[Any]]], worker_dict: Dict[str, Worker]) -> dict
        Inject dependencies into parameters whose default value is a `From` instance.
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
    
    def _resolve_from(self, dep: From, worker_dict: Dict[str, Worker]) -> Any:
        inject_res = worker_dict.get(dep.key, dep.default)
        if isinstance(inject_res, InjectorNone):
            raise AutomaDataInjectionError(
                f"the worker: `{dep.key}` is not found in the worker dictionary. "
                "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
            )

        if isinstance(inject_res, Worker):
            return inject_res.output_buffer
        else:
            return inject_res

    def _resolve_system(self, dep: System, current_worker_key: str, worker_dict: Dict[str, Worker]) -> Any:
        system_type = dep.type
        if system_type == SystemType.RUNTIME_CONTEXT:
            inject_res = RuntimeContext(worker_key=current_worker_key)
        else:
            raise AutomaDataInjectionError(
                f"the system type: `{system_type}` is not supported. "
                "You may need to set the default value of the parameter to a `System` instance with the key of the system."
            )
        
        return inject_res

    def inject(
        self, 
        current_worker_key: str,
        current_worker_sig: Dict[Parameter, List],
        worker_dict: Dict[str, Worker],
        next_args: Tuple[Any, ...],
        next_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Inject dependencies into parameters whose default value is a `From` instance.
        """
        param_list = [
            param
            for _, param_list in current_worker_sig.items()
            for param in param_list
        ]

        from_inject_kwargs = {}
        system_inject_kwargs = {}
        for name, default_value in param_list:
            if isinstance(default_value, From):
                value = self._resolve_from(default_value, worker_dict)
                from_inject_kwargs[name] = value
            elif isinstance(default_value, System):
                value = self._resolve_system(default_value, current_worker_key, worker_dict)
                system_inject_kwargs[name] = value

        if len(param_list) <= len(next_args) and (len(from_inject_kwargs) or len(system_inject_kwargs)):
            raise AutomaDataInjectionError(
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