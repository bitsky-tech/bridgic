# bridgic.core.processor这个包的作用：
# 定义核心的执行器概念： Processor。
# Processor是上层编排层操纵的基础单元。

from .worker import Worker
from .callable_worker import CallableWorker

__all__ = ["Worker", "CallableWorker"]
