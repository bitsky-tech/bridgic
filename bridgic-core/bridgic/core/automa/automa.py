import asyncio
import uuid

from typing import List, Any, Optional, Dict
from typing_extensions import override
from abc import ABCMeta, abstractmethod
from pydantic import BaseModel
from bridgic.core.automa.worker import Worker

class RunningOptions(BaseModel):
    debug: bool = False

class Automa(Worker, metaclass=ABCMeta):
    _running_options: RunningOptions

    def __init__(self, name: str = None, state_dict: Optional[Dict[str, Any]] = None):
        super().__init__(state_dict=state_dict)

        # Set the name of the Automa instance.
        if state_dict is None:
            self.name = name or f"automa-{uuid.uuid4().hex[:8]}"
        else:
            self.name = state_dict["name"]

        # Initialize the shared running options.
        self._running_options = RunningOptions()

    @override
    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = super().dump_to_dict()
        state_dict["name"] = self.name
        return state_dict

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
        return list(self._workers.keys())

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
            self._running_options.debug = debug

    def _get_top_running_options(self) -> RunningOptions:
        if self.parent is None:
            # Here we are at the top-level automa.
            return self._running_options
        return self.parent._get_top_running_options()

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
        # TODO: may be removed later...
