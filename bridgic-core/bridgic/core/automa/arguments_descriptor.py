import re
from inspect import _ParameterKind
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

from bridgic.core.automa.worker import Worker
from bridgic.core.types.error import AutomaDataInjectionError
from bridgic.core.utils.args_map import safely_map_args
from bridgic.core.automa.automa import Automa


class InjectorNone: 
    """
    Marker object for Injector.inject() when the default value is None.
    """

class ArgumentsDescriptor:
    """
    A descriptor for arguments that can be injected.
    """
    ...

@dataclass
class From(ArgumentsDescriptor):
    """
    worker dependency data from other workers.
    """
    key: str
    default: Optional[Any] = InjectorNone()

@dataclass
class System(ArgumentsDescriptor):
    """
    worker dependency data from the automa with pattern matching support.
    """
    key: str
    
    def __post_init__(self):
        allowed_patterns = [
            r"^runtime_context$",
            r"^automa:.*$",
        ]
        
        if not any(re.match(pattern, self.key) for pattern in allowed_patterns):
            raise AutomaDataInjectionError(
                f"Key '{self.key}' is not supported. Supported keys: "
                f"`runtime_context`: a context for data persistence of the current worker."
                f"`automa:.*`: a sub-automa in current automa."
            )

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
    
    def _resolve_from(self, dep: From, worker_output: Dict[str, Any]) -> Any:
        inject_res = worker_output.get(dep.key, dep.default)
        if isinstance(inject_res, InjectorNone):
            raise AutomaDataInjectionError(
                f"the worker: `{dep.key}` is not found in the worker dictionary. "
                "You may need to set the default value of the parameter to a `From` instance with the key of the worker."
            )

        if isinstance(inject_res, Worker):
            return inject_res
        else:
            return inject_res

    def _resolve_system(self, dep: System, current_worker_key: str, worker_dict: Dict[str, Worker]) -> Any:
        if dep.key == "runtime_context":
            return RuntimeContext(worker_key=current_worker_key)
        elif dep.key.startswith("automa:"):
            worker_key = dep.key[7:]

            inject_res = worker_dict.get(worker_key, InjectorNone())
            if isinstance(inject_res, InjectorNone):
                raise AutomaDataInjectionError(
                    f"the sub-atoma: `{dep.key}` is not found in current automa. "
                )
  
            if not inject_res.is_automa():
                raise AutomaDataInjectionError(
                    f"the `{dep.key}` instance is not an Automa. "
                )
           
            return inject_res.get_decorated_worker()

    def inject(
        self, 
        current_worker_key: str,
        current_worker_sig: Dict[_ParameterKind, List],
        worker_dict: Dict[str, Worker],
        worker_output: Dict[str, Any],
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
        worker_dict : Dict[str, Worker]
            Dictionary containing all available workers in the automa.
        worker_output : Dict[str, Any]
            Dictionary containing the output of all workers in the automa.
        next_args : Tuple[Any, ...]
            Positional arguments to be passed to the current worker.
        next_kwargs : Dict[str, Any]
            Keyword arguments to be passed to the current worker.
            
        Returns
        -------
        Tuple[Tuple[Any, ...], Dict[str, Any]]
            A tuple containing the processed positional and keyword arguments for the current worker.
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
                value = self._resolve_from(default_value, worker_output)
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


injector = WorkerInjector()
