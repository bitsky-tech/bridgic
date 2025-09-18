from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

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

    def resolve(
            self, 
            dep: From, 
            worker_dict: Dict[str, Worker]
    ) -> Any:
        """
        Resolve and return the result for a given dependency key.
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

    def inject(
            self, 
            param_list: List[Tuple[str, Optional[Any]]],
            worker_dict: Dict[str, Worker],
    ) -> Any:
        """
        Inject dependencies into parameters whose default value is a `From` instance.
        """
        inject_kwargs = {}
        for name, default_value in param_list:
            if isinstance(default_value, From):
                value = self.resolve(default_value, worker_dict)
                inject_kwargs[name] = value
        return inject_kwargs