from bridgic.core.worker import Worker
from bridgic.core.worker.data_model import ProcessorData, Task
from bridgic.automa.meta_class.meta import AutoMaMeta
from bridgic.core.worker import MethodProcessor
from bridgic.automa.bridge.decorator.bridge_info import _BridgeInfo
from typing import cast
from bridgic.typing.event import InEvent, OutEvent
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

# 
# AutoMa表达一种智能体和精确处理逻辑的混合系统。
# 它的职责是：
# 1，负责编排和组织Processor。Processor分两种，一种是Agent，提供智能处理，可以处理模糊的输入数据，另一种是精确控制的Processor。
# 2，它统一管理对外的conduit；输出progress和接收human feedback（以事件的形式）。
# 3，它和bridge包配合，解决fuzzy和logic的交互；
# 4，负责AutoMa层面的guardrail机制（而Agent层面的guardrail机制，是Agent的职责）；
# 
class AutoMa(Worker, metaclass=AutoMaMeta):
    _processors: dict[str, Worker]
    _processor_bridges: list[_BridgeInfo]

    def register_conduit_hook(self, hook: Callable[[OutEvent, xxx_future], None]) -> None:
        pass

    def __init__(self, *args, **kwargs) -> None:
        super().__setattr__("_processors", {})

        # 帮助子类根据self._processor_bridges的信息来创建Processor实例
        if len(self._processor_bridges) > 0:
            processors = self.__dict__.get("_processors")
            for bridge in self._processor_bridges:
                bound_func = bridge.func.__get__(self, type(self))
                processor = MethodProcessor(bound_func)
                processors[bridge.func.__name__] = processor

    def process(self, *args, **kwargs) -> ProcessorData:
        pass

    def post_out_event(self, event: OutEvent) -> None:
        pass
    
    def create_awaitable_event(self) -> InEvent:
        pass


    def __setattr__(self, name: str, value: Worker | None) -> None:
        processors = self.__dict__.get("_processors")
        if processors is None:
            raise AttributeError(
                        "cannot assign processor before AutoMa.__init__() call"
                    )

        if isinstance(value, Worker):
            processors[name] = value
        elif value is None and name in processors:
                del processors[name]

    def __getattr__(self, name: str) -> Worker | None:
        processors = self.__dict__.get("_processors")
        if processors is None:
            raise AttributeError(
                        "cannot get processor before AutoMa.__init__() call"
                    )
        return processors.get(name)

    async def process_async(self, *args, **kwargs) -> ProcessorData:
        data: ProcessorData
        if len(args) > 0 and isinstance(args[0], ProcessorData):
            data = cast(ProcessorData, args[0])
        else:
            # TODO: 需要一个系统的converion机制
            data = Task(**kwargs)

        # TODO: check here
        
        # 基于decorator-based机制编排好的self._processor_bridges静态信息，在这里依照依赖关系动态执行Processor实例
        end_processed = False
        while not end_processed:
            processors = self.__dict__.get("_processors")
            for bridge in self._processor_bridges:
                assert bridge.predecessor_count >= 0
                if bridge.predecessor_count == 0:
                    processor = processors.get(bridge.func.__name__)
                    assert processor is not None
                    data = await processor.process(data)
                    # 把后续节点的predecessor_count减1
                    successors = self._find_successors(bridge)
                    for successor in successors:
                        successor.predecessor_count -= 1
                    # TODO: 返回值及状态传递
                    if bridge.is_end:
                        end_processed = True
                        break

                    # TODO: Guardrails
        return data

    def _find_successors(self, current: _BridgeInfo) -> list[_BridgeInfo]:
        successors = []
        for bridge in self._processor_bridges:
            if bridge.listen == current.func_wrapper:
                assert bridge.predecessor_count > 0
                successors.append(bridge)
        return successors