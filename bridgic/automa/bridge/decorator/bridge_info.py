from pydantic import BaseModel
from typing import Callable, Optional

class _BridgeInfo(BaseModel):
    func: Callable
    func_wrapper: Callable
    listen: Optional[Callable] = None
    is_start: bool = False
    is_end: bool = False

    # TODO: check一下listen、is_start、is_end至少一个不为空，这个check逻辑怎么写？

    # 下面是动态信息，在运行时计算
    predecessor_count: int = 1 #前序节点计数，用于判断是否可以执行