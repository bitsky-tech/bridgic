import copy
from asyncio import Future

from typing import Any, Dict, get_type_hints, TYPE_CHECKING, Optional, Tuple
from bridgic.automa.interaction import Event, InteractionFeedback, Feedback

from bridgic.types.error import WorkerRuntimeError

if TYPE_CHECKING:
    from bridgic.automa.automa import Automa

class Worker:
    def __init__(self, *args, **kwargs):
        self.__parent: Automa = None
        self.__output_buffer: Any = None
        self.__output_setted: bool = False
        self.__local_space: Dict[str, Any] = {}

    async def process_async(self, *args: Optional[Tuple[Any]], **kwargs: Optional[Dict[str, Any]]) -> Any:
        raise NotImplementedError(f"process_async is not implemented in {type(self)}")

    @property
    def return_type(self) -> type:
        return get_type_hints(self.process_async).get('return', Any)

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
        ...
        #TODO: implement...