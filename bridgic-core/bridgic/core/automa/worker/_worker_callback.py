from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma


class WorkerCallback:
    """
    Callback for worker execution hooks in the automa framework.

    This class defines callbacks that are invoked before and after worker
    execution. Callbacks are intended for monitoring, logging, validation,
    and other side effects. They cannot modify worker execution logic,
    arguments, or output results.

    Methods
    -------
    pre_worker_execute(key, parent, arguments)
        Hook invoked before worker execution.
    post_worker_execute(key, parent, arguments, result)
        Hook invoked after worker execution.
        
    Notes
    -----
    All callback methods are asynchronous and support async operations.
    If a callback raises an exception, the automa treats it as the
    worker itself raised the exception. 
    """
    async def pre_worker_execute(
        self, 
        key: str, 
        parent: "GraphAutoma",
        arguments: Dict[str, Any], 
    ) -> None:
        """
        Hook invoked before worker execution.

        Called immediately before the worker runs. Use for arguments
        validation, logging, or monitoring. Cannot modify execution
        arguments or logic.

        Parameters
        ----------
        key : str
            Worker identifier.
        parent : GraphAutoma
            Parent automa instance containing this worker.
        arguments : Dict[str, Any]
            Execution parameters with keys "args" and "kwargs".

        Notes
        -----
        This method cannot modify worker execution logic or arguments.
        """
        pass

    async def post_worker_execute(
        self, 
        key: str, 
        parent: "GraphAutoma",
        arguments: Dict[str, Any], 
        result: Any,
    ) -> None:
        """
        Hook invoked after worker execution.

        Called immediately after the worker completes. Use for result
        monitoring, logging, event publishing, or validation. Cannot
        modify execution results or logic.

        Parameters
        ----------
        key : str
            Worker identifier.
        parent : GraphAutoma
            Parent automa instance containing this worker.
        arguments : Dict[str, Any]
            Execution arguments with keys "args" and "kwargs".
        result : Any
            Worker execution result.

        Notes
        -----
        This method cannot modify worker execution logic or arguments.
        """
        pass

    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "WorkerCallback":
        return cls()