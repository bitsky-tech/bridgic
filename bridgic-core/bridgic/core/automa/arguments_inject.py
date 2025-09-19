from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple, Union
from inspect import Parameter

from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import WorkerArgsMappingError


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

@dataclass
class System:
    """
    A declarative object for framework arguments injection.
    """
    key: str

class SystemType(Enum):
    """
    A declarative object for framework arguments type.
    """
    RUNTIME_CONTEXT = "runtime_context"

class WorkerInjector:
    """
    Worker Dependency injection container for resolving dependency data injection of workers.

    This class manages workers for dependency injection, allowing you to  inject 
    dependencies into function parameters based on their default values. 
    It is used with the `From` class to implement dependency injection of workers.

    Main Methods
    ------------
    resolve(dep: From | System, worker_dict: Dict[str, Worker]) -> Any
        Resolve and return the result for a given dependency key.

    inject(param_list: List[Tuple[str, Optional[Any]]], worker_dict: Dict[str, Worker]) -> dict
        Inject dependencies into parameters whose default value is a `From` or `System` instance.
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

    def _resolve_system(self, dep: System, worker_key: Optional[str]) -> Any:
        """
        Resolve and return the result for a given system key.
        """
        system_key = dep.key
        if system_key == SystemType.RUNTIME_CONTEXT:
            return {
                'worker_key': worker_key,
            }

        raise WorkerArgsMappingError(
            f"the system key: `{system_key}` is not supported. "
        )

    def _resolve_from(self, dep: From, worker_dict: Dict[str, Worker]) -> Any:
        """
        Resolve and return the result for a given from key.
        """
        inject_res = worker_dict.get(dep.key, dep.default)
        if isinstance(inject_res, InjectorNone):
            raise WorkerArgsMappingError(
                f"the worker: `{dep.key}` is not found in the worker dictionary. "
                "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
            )

        if isinstance(inject_res, Worker):
            return inject_res.output_buffer
        else:
            return inject_res

    def resolve(
        self, 
        dep: Union[From, System], 
        worker_dict: Optional[Dict[str, Worker]] = None,
        worker_key: Optional[str] = None,
    ) -> Any:
        """
        Resolve and return the result for a given dependency key.
        """
        if isinstance(dep, System):
            return self._resolve_system(dep, worker_key)
        elif isinstance(dep, From):
            return self._resolve_from(dep, worker_dict)

    def inject(
        self, 
        *,
        sig: Dict[Parameter, List],
        next_args: Tuple[Any, ...],
        worker_dict: Optional[Dict[str, Worker]] = None,
        worker_key: Optional[str] = None,
    ) -> Any:
        """
        Inject dependencies into parameters whose default value is a `From` instance.
        """
        print(f"worker_dict: {worker_dict}")

        param_list = [
            param
            for _, param_list in sig.items()
            for param in param_list
        ]
        
        inject_kwargs = {}
        for name, default_value in param_list:
            if isinstance(default_value, From) or isinstance(default_value, System):
                value = self.resolve(default_value, worker_dict, worker_key)
                inject_kwargs[name] = value

        # If the number of parameters is less than or equal to the number of positional arguments, raise an error.
        # TODO: add more details errors
        if len(param_list) <= len(next_args) and len(inject_kwargs):
            raise WorkerArgsMappingError(
                f"The number of parameters is less than or equal to the number of positional arguments, "
                f"but got {len(param_list)} parameters and {len(next_args)} positional arguments"
            )

        return inject_kwargs