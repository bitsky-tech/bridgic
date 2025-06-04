from bridgic.core.worker import Worker
from bridgic.automa.meta_class.meta import AutoMaMeta
from bridgic.core.worker import MethodWorker
from bridgic.automa.bridge.decorator.bridge_info import _BridgeInfo
from typing import cast
from typing import Any
from bridgic.typing.event.event import InEvent, OutEvent
from bridgic.typing.event.in_event_emiter import InEventEmiter
import asyncio
from typing import Callable
from bridgic.core.worker.data_model import TaskResult
from types import NoneType

# 
# AutoMa表达一种智能体和精确处理逻辑的混合系统。
# 它的职责是：
# 1，负责编排和组织Processor。Processor分两种，一种是Agent，提供智能处理，可以处理模糊的输入数据，另一种是精确控制的Processor。
# 2，它统一管理对外的conduit；输出progress和接收human feedback（以事件的形式）。
# 3，它和bridge包配合，解决fuzzy和logic的交互；
# 4，负责AutoMa层面的guardrail机制（而Agent层面的guardrail机制，是Agent的职责）；
# 
class AutoMa(Worker, metaclass=AutoMaMeta):
    _workers: dict[str, Worker]
    _worker_bridges: list[_BridgeInfo]

    def register_conduit_hook(self, hook: Callable[[OutEvent, InEventEmiter], None]) -> None:
        # TODO:
        pass

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_workers", {})

        # Create Worker instances based on self._processor_bridges
        if len(self._worker_bridges) > 0:
            workers = self.__dict__.get("_workers")
            for bridge in self._worker_bridges:
                bound_func = bridge.func.__get__(self, type(self))
                worker = MethodWorker(bound_func)
                workers[bridge.func.__name__] = worker

    def process(self, *args, **kwargs) -> Any:
        pass

    def post_out_event(self, event: OutEvent, event_emiter: InEventEmiter) -> None:
        pass
    
    def __setattr__(self, name: str, value: Worker | None) -> None:
        workers = self.__dict__.get("_workers")
        if workers is None:
            raise AttributeError(
                        "cannot assign worker before AutoMa.__init__() call"
                    )

        if isinstance(value, Worker):
            workers[name] = value
        elif value is None and name in workers:
                del workers[name]

    def __getattr__(self, name: str) -> Worker | None:
        workers = self.__dict__.get("_workers")
        if workers is None:
            raise AttributeError(
                        "cannot get worker before AutoMa.__init__() call"
                    )
        return workers.get(name)

    async def process_async(self, *args, **kwargs) -> Any:
        # TODO: check anything here
        
        # 基于decorator-based机制编排好的self._worker_bridges静态信息，在这里依照依赖关系动态执行Worker实例
        end_processed = False
        while not end_processed:
            workers = self.__dict__.get("_workers")
            for bridge in self._worker_bridges:
                assert bridge.predecessor_count >= 0
                if bridge.predecessor_count == 0:
                    worker = workers.get(bridge.func.__name__)
                    assert worker is not None
                    result = await worker.process_async(*args, **kwargs)
                    # 把后续节点的predecessor_count减1
                    successors = self._find_successors(bridge)
                    for successor in successors:
                        successor.predecessor_count -= 1
                    # TODO: maybe need to consider the expected argument types of the successor
                    args, kwargs = self._convert_result_to_args(result)
                    if bridge.is_end:
                        end_processed = True
                        break

                    # TODO: Guardrails
        return result

    def _find_successors(self, current: _BridgeInfo) -> list[_BridgeInfo]:
        successors = []
        for bridge in self._worker_bridges:
            if bridge.listen == current.func_wrapper:
                assert bridge.predecessor_count > 0
                successors.append(bridge)
        return successors
    
    def _convert_result_to_args(self, result: Any) -> tuple[Any, Any]:
        if isinstance(result, tuple):
            if len(result) == 1:
                args, kwargs = self._convert_non_tuple_result_to_args(result[0])
            elif len(result) > 1:
                args = result
                kwargs = {}
            else:
                # len(result) == 0
                args = ()
                kwargs = {}
        else:
            args, kwargs = self._convert_non_tuple_result_to_args(result)
        return args, kwargs

    def _convert_non_tuple_result_to_args(self, single_result: Any) -> tuple[Any, Any]:
        if isinstance(single_result, TaskResult):
            args = ()
            kwargs = single_result.model_dump()
        elif isinstance(single_result, dict):
            args = ()
            kwargs = single_result
        elif isinstance(single_result, NoneType):
            args = ()
            kwargs = {}
        else:
            args = (single_result, )
            kwargs = {}
        
        return args, kwargs

