from typing import Any
from bridgic.typing.event.event import InEvent

class InEventEmiter:
    def emit(self, event: InEvent) -> None:
        pass

    async def get(self) -> InEvent:
        pass