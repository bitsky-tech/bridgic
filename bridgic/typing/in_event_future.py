from typing import Any
from bridgic.typing.event import InEvent

class InEventFuture:
    def emit(self, event: InEvent) -> None:
        pass

    async def get(self) -> InEvent:
        pass