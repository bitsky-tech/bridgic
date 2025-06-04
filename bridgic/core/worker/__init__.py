# bridgic.core.processor这个包的作用：
# 定义核心的执行器概念： Processor。
# Processor是上层编排层操纵的基础单元。

from .worker import Worker
from .function_worker import FunctionWorker
from .method_worker import MethodWorker

__all__ = ["Worker", "FunctionWorker", "MethodWorker"]
