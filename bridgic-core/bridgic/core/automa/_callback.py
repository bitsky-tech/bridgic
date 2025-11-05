from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from bridgic.core.automa._graph_automa import GraphAutoma


class WorkerCallback:
    async def pre_worker_execute(
        self, 
        key: str, 
        automa: "GraphAutoma",
        params: Dict[str, Any], 
    ) -> None:
        pass

    async def post_worker_execute(
        self, 
        key: str, 
        automa: "GraphAutoma",
        params: Dict[str, Any], 
        result: Any,
    ) -> None:
        pass

    def dump_to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> "WorkerCallback":
        return cls()