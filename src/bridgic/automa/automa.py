import asyncio
import uuid

from typing import List, Any
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from bridgic.automa.worker import Worker
from bridgic.types.mixin import AdaptableMixin

class RunningOptions(BaseModel):
    debug: bool = False

class Automa(Worker, metaclass=ABCMeta):
    def __init__(self, name: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the name of the Automa instance.
        self.name = name or f"automa-{uuid.uuid4().hex[:8]}"

        # Define the shared running options.
        self.running_options: RunningOptions = RunningOptions()

        self._loop: asyncio.AbstractEventLoop = None
        self._finish_event: asyncio.Event = None
        self._future_list: List[asyncio.Future] = []

    async def process_async(self, *args, **kwargs) -> Any:
        raise NotImplementedError("process_async() is not implemented for Automa")

    def is_top_level(self) -> bool:
        """
        Check if the current automa is the top-level automa.

        Returns
        -------
        bool
            True if the current automa is the top-level automa, False otherwise.
        """
        return self.parent is None

    def all_workers(self) -> List[str]:
        """
        Gets a list containing the keys of all workers registered in this Automa.

        Returns
        -------
        List[str]
            A list of worker keys.
        """
        raise NotImplementedError("process_async() is not implemented for Automa")

    def set_running_options(self, debug: bool = None):
        """
        Set running options for this Automa instance, and ensure these options propagate through (penetrate) all nested 
        Automa instances. For each option, if its value is None, the original value will be retained and not overwritten.

        When an option is set multiple times across different nested Automa instances, the setting from the outermost 
        (top-level) Automa will override the settings of all inner (nested) Automa instances.

        For example, if the top-level Automa instance sets `debug = True` and the nested instances sets `debug = False`, 
        then the nested Automa instance will run in debug mode, when the top-level Automa instance is executed.

        Parameters
        ----------
        debug : bool, optional
            Whether to enable debug mode. If not set, the effect is the same as setting `debug = False` by default.
        """
        if debug is not None:
            self.running_options.debug = debug

    def _penetrate_running_options(self):
        for worker_obj in self._workers.values():
            if isinstance(worker_obj, AdaptableMixin) and isinstance(worker_obj.core_worker, Automa):
                worker_obj.core_worker.set_running_options(**(self.running_options.model_dump()))
                worker_obj.core_worker._penetrate_running_options()

    def _get_top_level_automa(self) -> "Automa":
        """
        Get the top-level automa instance reference.
        """
        top_level_automa = self
        while not top_level_automa.is_top_level():
            top_level_automa = top_level_automa.parent
        return top_level_automa

class GoalOrientedAutoma(Automa):
    @abstractmethod
    def remove_worker(self, name: str) -> None:
        """
        Remove a worker from the Automa. In GoalOrientedAutoma and its subclasses, this method can be called at any time to remove a worker from the Automa.

        Parameters
        ----------
        name : str
            The name of the worker to be removed.

        Returns
        -------
        None

        Raises
        ------
        AutomaDeclarationError
            If the worker specified by worker_name does not exist in the Automa, this exception will be raised.
        """
        ...
