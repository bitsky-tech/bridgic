import copy
from asyncio import Future
from typing_extensions import override
from typing import Any, Dict, get_type_hints, TYPE_CHECKING, Optional, Tuple
from bridgic.automa.interaction import Event, InteractionFeedback, Feedback
from bridgic.serialization import Serializable
from bridgic.types.error import WorkerRuntimeError

if TYPE_CHECKING:
    from bridgic.automa.automa import Automa

class Worker:
    __output_buffer: Any
    __output_setted: bool
    __local_space: Dict[str, Any]

    def __init__(self, state_dict: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        state_dict : Optional[Dict[str, Any]] (default = None)
            A dictionary for initializing the worker's runtime state. This parameter is intended for internal framework use only, specifically for deserialization, and should not be used by developers.
        """
        self.__parent: Automa = None
        if state_dict is None:
            self.__output_setted = False
            self.__local_space = {}
        else:
            self.__output_setted = state_dict["output_setted"]
            if self.__output_setted:
                self.__output_buffer = state_dict["output_buffer"]
            self.__local_space = state_dict["local_space"]

    async def process_async(self, *args: Optional[Tuple[Any]], **kwargs: Optional[Dict[str, Any]]) -> Any:
        raise NotImplementedError(f"process_async is not implemented in {type(self)}")

    @property
    def parent(self) -> "Automa":
        return self.__parent

    @parent.setter
    def parent(self, value: "Automa"):
        self.__parent = value

    @property
    def output_buffer(self) -> Any:
        if not self.__output_setted:
            raise RuntimeError(f"output of worker is not ready yet")
        return copy.deepcopy(self.__output_buffer)
    
    @output_buffer.setter
    def output_buffer(self, value: Any):
        self.__output_buffer = value
        self.__output_setted = True

    @property
    def local_space(self) -> Dict[str, Any]:
        return self.__local_space
    
    @local_space.setter
    def local_space(self, value: Dict[str, Any]):
        self.__local_space = value

    def dump_to_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["output_setted"] = self.__output_setted
        if self.__output_setted:
            state_dict["output_buffer"] = self.__output_buffer
        state_dict["local_space"] = self.__local_space
        return state_dict

    @classmethod
    def load_from_dict(cls, state_dict: Dict[str, Any]) -> "Worker":
        return cls(state_dict=state_dict)

    def post_event(self, event: Event) -> Future[Feedback]:
        """
        Post an event to the application layer outside the Automa.

        Parameters
        ----------
        event: Event
            The event to be posted.
        """
        if self.parent is None:
            raise WorkerRuntimeError(f"`post_event` method can only be called by a worker inside an Automa")
        return self.parent.post_event(event)

    def interact_with_human(self, event: Event) -> InteractionFeedback:
        if self.parent is None:
            raise WorkerRuntimeError(f"`interact_with_human` method can only be called by a worker inside an Automa")
        return self.parent.interact_with_human_from_worker(event, self)
